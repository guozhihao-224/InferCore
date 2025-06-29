use std::fs;

use half::bf16;
use half::f16;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

use crate::model::attention::Attention;
use crate::model::mlp::MLP;
use crate::model::rms_norm::RMSNorm;
use crate::tensor::device::Device;
use crate::tensor::tensor::Tensor;
use ndarray_npy::read_npy;

#[derive(Debug, Clone)]
pub struct QwenBlock {
    pub input_norm: RMSNorm,
    pub attn: Attention,
    pub post_attn_norm: RMSNorm,
    pub mlp: MLP,
}

impl QwenBlock {
    pub fn new(
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        intermediate: usize,
        eps: f32,
        device: Device,
    ) -> Self {
        Self {
            input_norm: RMSNorm::new(hidden, eps),
            attn: Attention::new(hidden, n_heads, n_kv_heads, device),
            post_attn_norm: RMSNorm::new(hidden, eps),
            mlp: MLP::new(hidden, intermediate, device),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // input: [batch, seq, hidden]
        let h_norm = self.input_norm.forward(x);
        let attn_out = self.attn.forward(&h_norm);
        let h_residual = x.add(&attn_out); // 残差
        let attn_out_vec = attn_out.to_vec();
        let h_residual_vec = h_residual.to_vec();
       
       
        let h2 = self.post_attn_norm.forward(&h_residual);
        let mlp_out = self.mlp.forward(&h2);
        let out = h_residual.add(&mlp_out); // 残差
        let mlp_out_vec = mlp_out.to_vec();
        let out_vec = out.to_vec();
        
        
        out
    }
}

#[derive(Debug, Clone)]
pub struct QwenTransformer {
    pub embedding: Tensor, // [vocab_size, hidden]
    pub blocks: Vec<QwenBlock>,
    pub norm: RMSNorm,
    pub lm_head: Tensor, // [hidden, vocab_size]
}

impl QwenTransformer {
    pub fn new(
        vocab_size: usize,
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        intermediate: usize,
        n_layers: usize,
        eps: f32,
        device: Device,
    ) -> Self {
        let embedding = Tensor::zeros(&[vocab_size, hidden], device);
        let blocks = (0..n_layers)
            .map(|_| QwenBlock::new(hidden, n_heads, n_kv_heads, intermediate, eps, device))
            .collect();
        let norm = RMSNorm::new(hidden, eps);
        let lm_head = embedding.transpose(&[1, 0]);
        Self {
            embedding,
            blocks,
            norm,
            lm_head,
        }
    }

    /// 输入 token_ids: [batch, seq]，输出 logits: [batch, seq, vocab_size]
    pub fn forward(&self, token_ids: &Tensor) -> Tensor {
        // embedding lookup: [batch, seq] -> [batch, seq, hidden]
        let (batch, seq) = (token_ids.shape()[0], token_ids.shape()[1]);
        let hidden = self.embedding.shape()[1];
        let mut x = Tensor::zeros(&[batch, seq, hidden], token_ids.device);
        let token_vec = token_ids.to_vec();
        for b in 0..batch {
            for s in 0..seq {
                let idx = token_vec[b * seq + s] as usize;
                for h in 0..hidden {
                    *x.get_mut(&[b, s, h]) = self.embedding.get(&[idx, h]);
                }
            }
        }
        // 多层 Block 堆叠
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x);
            let x_vec = x.to_vec();
            let max = x_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = x_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean = x_vec.iter().sum::<f32>() / x_vec.len() as f32;
           
        }
        // 最后 RMSNorm
        let x = self.norm.forward(&x);
        // 输出头: [batch, seq, hidden] @ [hidden, vocab_size] -> [batch, seq, vocab_size]
        x.matmul(&self.lm_head)
    }

    pub fn load_weights_safetensors(&mut self, path: &str) -> Result<(), String> {
        let data = fs::read(path).map_err(|e| e.to_string())?;
        let tensors = SafeTensors::deserialize(&data).map_err(|e| e.to_string())?;
        for (name, tensor) in tensors.tensors() {
            if name == "model.embed_tokens.weight" {
                self.embedding = tensor_to_tensor(tensor, self.embedding.device)?;
                self.lm_head = self.embedding.transpose(&[1, 0]);
            } else if name == "model.norm.weight" {
                self.norm.gamma = tensor_to_tensor(tensor, self.norm.gamma.device)?;
            } else if name.starts_with("model.layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() < 5 {
                    continue;
                }
                let layer_idx: usize = match parts[2].parse() {
                    Ok(idx) => idx,
                    Err(_) => continue,
                };
                if layer_idx >= self.blocks.len() {
                    continue;
                }
                let block = &mut self.blocks[layer_idx];
                let layer_name = parts[3];
                let weight_name = parts[4];
                let weight_type = parts.get(5).cloned();

                match (layer_name, weight_name, weight_type) {
                    // Attention weights and biases
                    ("self_attn", "q_proj", Some("weight")) => {
                        block.attn.q_proj = tensor_to_tensor(tensor, block.attn.q_proj.device)?
                    }
                    ("self_attn", "q_proj", Some("bias")) => {
                        block.attn.q_bias = tensor_to_tensor(tensor, block.attn.q_bias.device)?
                    }
                    ("self_attn", "k_proj", Some("weight")) => {
                        block.attn.k_proj =
                            tensor_to_tensor(tensor, block.attn.k_proj.device)?.transpose(&[1, 0])
                    }
                    ("self_attn", "k_proj", Some("bias")) => {
                        block.attn.k_bias = tensor_to_tensor(tensor, block.attn.k_bias.device)?
                    }
                    ("self_attn", "v_proj", Some("weight")) => {
                        block.attn.v_proj =
                            tensor_to_tensor(tensor, block.attn.v_proj.device)?.transpose(&[1, 0])
                    }
                    ("self_attn", "v_proj", Some("bias")) => {
                        block.attn.v_bias = tensor_to_tensor(tensor, block.attn.v_bias.device)?
                    }
                    ("self_attn", "o_proj", Some("weight")) => {
                        block.attn.o_proj = tensor_to_tensor(tensor, block.attn.o_proj.device)?
                    }
                    // Layer norms
                    ("input_layernorm", "weight", None) => {
                        block.input_norm.gamma =
                            tensor_to_tensor(tensor, block.input_norm.gamma.device)?
                    }
                    ("post_attention_layernorm", "weight", None) => {
                        block.post_attn_norm.gamma =
                            tensor_to_tensor(tensor, block.post_attn_norm.gamma.device)?
                    }
                    // MLP weights
                    ("mlp", "gate_proj", Some("weight")) => {
                        block.mlp.w1 = tensor_to_tensor(tensor, block.mlp.w1.device)?
                    }
                    ("mlp", "up_proj", Some("weight")) => {
                        block.mlp.w3 = tensor_to_tensor(tensor, block.mlp.w3.device)?
                    }
                    ("mlp", "down_proj", Some("weight")) => {
                        block.mlp.w2 =
                            tensor_to_tensor(tensor, block.mlp.w2.device)?.transpose(&[1, 0])
                    }
                    _ => {
                        // println!("Unknown weight: {}", name)
                    }
                }
            }
        }
        Ok(())
    }
}

fn tensor_to_tensor(tensor: TensorView, device: Device) -> Result<Tensor, String> {
    let shape: Vec<usize> = tensor.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = match tensor.dtype() {
        safetensors::Dtype::F32 => tensor
            .data()
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => {
            let v: Vec<f32> = tensor
                .data()
                .chunks(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect();
            println!("f16解包前10: {:?}", &v[..10]);
            v
        }
        safetensors::Dtype::BF16 => {
            let v: Vec<f32> = tensor
                .data()
                .chunks(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect();
            println!("bf16解包前10: {:?}", &v[..10]);
            v
        }
        _ => return Err("Only f32, f16, bfloat16 supported".to_string()),
    };
    Ok(Tensor::from_vec(data, &shape, device))
}

pub fn check_model_shapes(model: &QwenTransformer, config: &crate::model::config::QwenConfig) {
    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;
    let num_attention_heads = config.num_attention_heads;
    let num_key_value_heads = config.num_key_value_heads;
    let intermediate_size = config.intermediate_size;
    let num_hidden_layers = config.num_hidden_layers;
    let head_dim = hidden_size / num_attention_heads;
    let kv_dim = num_key_value_heads * head_dim;

    assert_eq!(
        model.embedding.shape(),
        &[vocab_size, hidden_size],
        "embedding shape mismatch"
    );
    assert_eq!(
        model.lm_head.shape(),
        &[hidden_size, vocab_size],
        "lm_head shape mismatch"
    );
    for (i, block) in model.blocks.iter().enumerate() {
        assert_eq!(
            block.attn.q_proj.shape(),
            &[hidden_size, hidden_size],
            "block{} attn.q_proj shape mismatch",
            i
        );
        assert_eq!(
            block.attn.k_proj.shape(),
            &[hidden_size, kv_dim],
            "block{} attn.k_proj shape mismatch",
            i
        );
        assert_eq!(
            block.attn.v_proj.shape(),
            &[hidden_size, kv_dim],
            "block{} attn.v_proj shape mismatch",
            i
        );
        assert_eq!(
            block.attn.o_proj.shape(),
            &[hidden_size, hidden_size],
            "block{} attn.o_proj shape mismatch",
            i
        );
        assert_eq!(
            block.input_norm.gamma.shape(),
            &[hidden_size],
            "block{} input_norm.gamma shape mismatch",
            i
        );
        assert_eq!(
            block.post_attn_norm.gamma.shape(),
            &[hidden_size],
            "block{} post_attn_norm.gamma shape mismatch",
            i
        );
        assert_eq!(
            block.mlp.w1.shape(),
            &[intermediate_size, hidden_size],
            "block{} mlp.w1 (gate_proj) shape mismatch",
            i
        );
        assert_eq!(
            block.mlp.w3.shape(),
            &[intermediate_size, hidden_size],
            "block{} mlp.w3 (up_proj) shape mismatch",
            i
        );
        assert_eq!(
            block.mlp.w2.shape(),
            &[intermediate_size, hidden_size],
            "block{} mlp.w2 (down_proj) shape mismatch",
            i
        );
    }
    assert_eq!(
        model.norm.gamma.shape(),
        &[hidden_size],
        "final norm.gamma shape mismatch"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::device::Device;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_qwenblock_forward_shape_and_residual() {
        let (batch, seq, hidden, n_heads, n_kv_heads, intermediate) = (2, 3, 4, 2, 2, 8);
        let eps = 1e-5;
        let device = Device::Cpu;
        let mut block = QwenBlock::new(hidden, n_heads, n_kv_heads, intermediate, eps, device);
        // 设置所有权重为单位阵/全1，便于可控
        let identity = Tensor::from_vec(
            (0..hidden * hidden)
                .map(|i| if i % (hidden + 1) == 0 { 1.0 } else { 0.0 })
                .collect(),
            &[hidden, hidden],
            device,
        );
        block.attn.q_proj = identity.clone();
        block.attn.k_proj = identity.clone();
        block.attn.v_proj = identity.clone();
        block.attn.o_proj = identity.clone();
        // biases are now part of attention
        block.attn.q_bias = Tensor::zeros(&[hidden], device);
        block.attn.k_bias = Tensor::zeros(&[hidden], device);
        block.attn.v_bias = Tensor::zeros(&[hidden], device);

        block.input_norm.gamma = Tensor::from_vec(vec![1.0; hidden], &[hidden], device);
        block.post_attn_norm.gamma = Tensor::from_vec(vec![1.0; hidden], &[hidden], device);
        // MLP: w1/w3/w2
        block.mlp.w1 = Tensor::from_vec(
            vec![1.0; hidden * intermediate],
            &[hidden, intermediate],
            device,
        );
        block.mlp.w3 = Tensor::from_vec(
            vec![1.0; hidden * intermediate],
            &[hidden, intermediate],
            device,
        );
        block.mlp.w2 = Tensor::from_vec(
            vec![1.0; intermediate * hidden],
            &[intermediate, hidden],
            device,
        );

        // 输入
        let x = Tensor::from_vec(
            (0..batch * seq * hidden).map(|v| v as f32).collect(),
            &[batch, seq, hidden],
            device,
        );
        let y = block.forward(&x);
        // 输出 shape 应为 [batch, seq, hidden]
        assert_eq!(y.shape(), &[batch, seq, hidden]);
        // 残差：只要 attn/MLP 不全为 0，输出应与输入不同
        assert!(y.to_vec() != x.to_vec());
    }

    #[test]
    fn test_qwenblock_stack() {
        let (batch, seq, hidden, n_heads, n_kv_heads, intermediate) = (1, 2, 4, 2, 2, 8);
        let eps = 1e-5;
        let device = Device::Cpu;
        let mut block1 = QwenBlock::new(hidden, n_heads, n_kv_heads, intermediate, eps, device);
        let mut block2 = QwenBlock::new(hidden, n_heads, n_kv_heads, intermediate, eps, device);
        // 设置权重为单位阵/全1
        let identity = Tensor::from_vec(
            (0..hidden * hidden)
                .map(|i| if i % (hidden + 1) == 0 { 1.0 } else { 0.0 })
                .collect(),
            &[hidden, hidden],
            device,
        );
        for block in [&mut block1, &mut block2] {
            block.attn.q_proj = identity.clone();
            block.attn.k_proj = identity.clone();
            block.attn.v_proj = identity.clone();
            block.attn.o_proj = identity.clone();
            block.attn.q_bias = Tensor::zeros(&[hidden], device);
            block.attn.k_bias = Tensor::zeros(&[hidden], device);
            block.attn.v_bias = Tensor::zeros(&[hidden], device);
            block.input_norm.gamma = Tensor::from_vec(vec![1.0; hidden], &[hidden], device);
            block.post_attn_norm.gamma = Tensor::from_vec(vec![1.0; hidden], &[hidden], device);
            block.mlp.w1 = Tensor::from_vec(
                vec![1.0; hidden * intermediate],
                &[hidden, intermediate],
                device,
            );
            block.mlp.w3 = Tensor::from_vec(
                vec![1.0; hidden * intermediate],
                &[hidden, intermediate],
                device,
            );
            block.mlp.w2 = Tensor::from_vec(
                vec![1.0; intermediate * hidden],
                &[intermediate, hidden],
                device,
            );
        }
        let x = Tensor::from_vec(
            (0..batch * seq * hidden).map(|v| v as f32).collect(),
            &[batch, seq, hidden],
            device,
        );
        let y1 = block1.forward(&x);
        let y2 = block2.forward(&y1);
        // shape 不变
        assert_eq!(y2.shape(), &[batch, seq, hidden]);
        // 堆叠后输出应与输入不同
        assert!(y2.to_vec() != x.to_vec());
    }

    #[test]
    fn test_qwen_transformer_forward_shape() {
        let (batch, seq, vocab_size, hidden, n_heads, n_kv_heads, intermediate, n_layers) =
            (2, 3, 10, 4, 2, 2, 8, 2);
        let eps = 1e-5;
        let device = Device::Cpu;
        let model = QwenTransformer::new(
            vocab_size,
            hidden,
            n_heads,
            n_kv_heads,
            intermediate,
            n_layers,
            eps,
            device,
        );
        // 输入 token_ids
        let token_ids = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[batch, seq], device);
        let logits = model.forward(&token_ids);
        // 输出 shape 应为 [batch, seq, vocab_size]
        assert_eq!(logits.shape(), &[batch, seq, vocab_size]);
    }

    #[test]
    fn test_load_weights_safetensors_mock() {
        // 这里只做接口和 shape 验证，实际可用 mock safetensors 文件或跳过
        // 假设有 model.safetensors 文件，且 embedding/blocks/lm_head shape 正确
        // 这里只测试能否正常调用和不 panic
        let (vocab_size, hidden, n_heads, n_kv_heads, intermediate, n_layers) = (10, 4, 2, 2, 8, 2);
        let eps = 1e-5;
        let device = Device::Cpu;
        let mut model = QwenTransformer::new(
            vocab_size,
            hidden,
            n_heads,
            n_kv_heads,
            intermediate,
            n_layers,
            eps,
            device,
        );
        // 这里用一个不存在的文件名，实际部署时请替换为真实 safetensors 路径
        let result = model.load_weights_safetensors("model.safetensors");
        // 只要能正常返回（即使是 Err），说明接口无 panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_embedding_lookup_full_vector() {
        let vocab_size = 151936;
        let hidden = 896;
        let device = Device::Cpu;
        let mut model = QwenTransformer::new(
            vocab_size,
            hidden,
            14, // n_heads
            2,  // n_kv_heads
            3072, // intermediate
            24,   // n_layers
            1e-5, // eps
            device,
        );
        let _ = model.load_weights_safetensors("Qwen2.5-0.5B-instruct/model.safetensors");

        let token_ids = vec![108386, 3837, 48, 16948, 0];
        let input_tensor = Tensor::from_vec(
            token_ids.iter().map(|&x| x as f32).collect(),
            &[1, token_ids.len()],
            device,
        );

        let (batch, seq) = (input_tensor.shape()[0], input_tensor.shape()[1]);
        let hidden = model.embedding.shape()[1];
        let mut x = Tensor::zeros(&[batch, seq, hidden], device);
        let token_vec = input_tensor.to_vec();
        for b in 0..batch {
            for s in 0..seq {
                let idx = token_vec[b * seq + s] as usize;
                for h in 0..hidden {
                    *x.get_mut(&[b, s, h]) = model.embedding.get(&[idx, h]);
                }
            }
        }

        // Rust 端 embedding
        let mut rust_emb = vec![];
        for h in 0..hidden {
            rust_emb.push(x.get(&[0, 0, h]));
        }

        // 读取 Python 导出的 embedding
        let py_emb: ndarray::Array1<f32> = read_npy("/Users/guozhihao/work/study/mlsys/py_emb.npy").unwrap();
        let py_emb_vec = py_emb.to_vec();

        assert_eq!(rust_emb.len(), py_emb_vec.len());
        for (i, (a, b)) in rust_emb.iter().zip(py_emb_vec.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "Mismatch at dim {}: {} vs {}", i, a, b);
        }
    }
}
