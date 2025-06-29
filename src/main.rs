// use std::io::stdout;
// use std::io::Write;
// use qwen3_infer::model::config::{QwenConfig, GenerationConfig};
// use qwen3_infer::model::transformer::{check_model_shapes, QwenTransformer};
// use qwen3_infer::tensor::device::Device;
// use qwen3_infer::tensor::tensor::Tensor;
// use qwen3_infer::tokenizer::tokenizer::QwenTokenizer;
// use rand::prelude::*;

// /// 构造 ChatML 格式 prompt
// fn build_chat_prompt(messages: &[(String, String)]) -> String {
//     let mut prompt = String::new();
//     for (role, content) in messages {
//         prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
//     }
//     prompt.push_str("<|im_start|>assistant\n");
//     prompt
// }

// /// 严格对齐 transformers 的 Qwen2 ChatML 模板拼接
// /// messages: [ (role, content), ... ]
// /// role 只能是 system/user/assistant
// /// 返回的字符串和 transformers apply_chat_template 完全一致
// fn build_chatml_prompt(messages: &[(String, String)]) -> String {
//     let mut prompt = String::new();
//     // 检查是否有 system 消息，没有则自动补全
//     let has_system = messages.iter().any(|(role, _)| role == "system");
//     if !has_system {
//         prompt.push_str("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n");
//     }
//     for (role, content) in messages {
//         // role 只能是 system/user/assistant
//         let role = match role.as_str() {
//             "system" | "user" | "assistant" => role,
//             _ => panic!("Invalid role: {}", role),
//         };
//         prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
//     }
//     // 结尾加 <|im_start|>assistant\n，不加 <|im_end|>
//     prompt.push_str("<|im_start|>assistant\n");
//     prompt
// }

// /// softmax 实现
// fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
//     let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
//     let exp: Vec<f32> = logits.iter().map(|&x| ((x - max) / temperature).exp()).collect();
//     let sum: f32 = exp.iter().sum();
//     exp.iter().map(|&x| x / sum).collect()
// }

// /// top_k/top_p 采样
// fn sample_from_logits(
//     logits: &[f32],
//     temperature: f32,
//     top_p: f32,
//     top_k: u32,
//     repetition_penalty: f32,
//     generated_tokens: &[u32],
// ) -> u32 {
//     let mut logits = logits.to_vec();
//     // repetition_penalty
//     for &token in generated_tokens {
//         let idx = token as usize;
//         if idx < logits.len() {
//             if logits[idx] < 0.0 {
//                 logits[idx] *= repetition_penalty;
//             } else {
//                 logits[idx] /= repetition_penalty;
//             }
//         }
//     }
//     // softmax with temperature
//     let mut probs = softmax(&logits, temperature);
//     // top_k
//     if top_k > 0 && (top_k as usize) < probs.len() {
//         let mut topk: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
//         topk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//         let kth_prob = topk[top_k as usize - 1].1;
//         for (i, p) in probs.iter_mut().enumerate() {
//             if *p < kth_prob {
//                 *p = 0.0;
//             }
//         }
//     }
//     // top_p
//     if top_p < 1.0 {
//         let mut top_p_vec: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
//         top_p_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//         let mut cum_prob = 0.0;
//         for (i, &(_, p)) in top_p_vec.iter().enumerate() {
//             cum_prob += p;
//             if cum_prob > top_p {
//                 for j in i+1..top_p_vec.len() {
//                     probs[top_p_vec[j].0] = 0.0;
//                 }
//                 break;
//             }
//         }
//     }
//     // 归一化
//     let sum: f32 = probs.iter().sum();
//     if sum > 0.0 {
//         for p in probs.iter_mut() { *p /= sum; }
//     }
//     // multinomial 采样
//     let mut rng = thread_rng();
//     let between = rand::distributions::Uniform::new(0.0, 1.0);
//     let mut acc = 0.0;
//     let sample: f32 = rng.sample(between);
//     for (i, &p) in probs.iter().enumerate() {
//         acc += p;
//         if sample < acc {
//             return i as u32;
//         }
//     }
//     // fallback: argmax
//     probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
// }

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // 1. 加载 config
//     let config = QwenConfig::from_file("Qwen2.5-0.5B-instruct/config.json")?;
//     let gen_config = GenerationConfig::from_file("Qwen2.5-0.5B-instruct/generation_config.json")?;
//     // 2. 初始化模型
//     let mut model = QwenTransformer::new(
//         config.vocab_size,
//         config.hidden_size,
//         config.num_attention_heads,
//         config.num_key_value_heads,
//         config.intermediate_size,
//         config.num_hidden_layers,
//         config.rms_norm_eps,
//         Device::Cpu,
//     );
//     model.load_weights_safetensors("Qwen2.5-0.5B-instruct/model.safetensors")?;
//     check_model_shapes(&model, &config);
//     // 3. 加载分词器
//     let tokenizer = QwenTokenizer::from_file("Qwen2.5-0.5B-instruct/tokenizer.json").unwrap();
//     // 4. 构造多轮对话
//     let mut messages = vec![
//         ("system".to_string(), "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.".to_string()),
//         ("user".to_string(), "what's your name?".to_string()),
//     ];

//     let prompt = build_chatml_prompt(&messages);

//     println!("输出的prompt {:?}", prompt);

//     let mut token_ids = tokenizer.encode(&prompt)?;
//     token_ids.insert(0, gen_config.bos_token_id);
//     let mut generated_token_ids = token_ids.clone();
//     let max_gen_len = 64;
//     print!("模型回复: ");
//     stdout().flush().unwrap();
//     // 5. 生成
//     for _ in 0..max_gen_len {
//         let input_tensor = Tensor::from_vec(
//             generated_token_ids.iter().map(|&x| x as f32).collect(),
//             &[1, generated_token_ids.len()],
//             Device::Cpu,
//         );
//         let logits = model.forward(&input_tensor);
//         let logits_vec = logits.to_vec();
//         let shape = logits.shape();
//         let vocab_size = shape[2];
//         let last_logits = &logits_vec[(generated_token_ids.len() - 1) * vocab_size..generated_token_ids.len() * vocab_size];
//         let next_token_id = if gen_config.do_sample {
//             sample_from_logits(
//                 last_logits,
//                 gen_config.temperature,
//                 gen_config.top_p,
//                 gen_config.top_k,
//                 gen_config.repetition_penalty,
//                 &generated_token_ids,
//             )
//         } else {
//             last_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
//         };
//         if gen_config.eos_token_id.contains(&next_token_id) {
//             break;
//         }
//         generated_token_ids.push(next_token_id);
//         // 增量decode并输出新token
//         let new_token = tokenizer.decode(&[next_token_id])?;
//         print!("{}", new_token);
//         stdout().flush().unwrap();
//     }
//     println!();
//     // // 6. 一次性 decode 新生成的 token
//     // let response = tokenizer.decode(&generated_token_ids[token_ids.len()..])?;
//     // println!("模型回复: {}", response);
//     Ok(())
// }

use candle_core::Tensor;

// variable name: model.layers.17.mlp.gate_proj.weight
// variable name: model.layers.13.self_attn.q_proj.bias
// variable name: model.layers.2.input_layernorm.weight
// variable name: model.layers.7.mlp.up_proj.weight
// variable name: model.layers.14.self_attn.k_proj.bias
// variable name: model.layers.8.self_attn.v_proj.weight
// variable name: model.layers.12.self_attn.k_proj.weight
// variable name: model.layers.7.self_attn.v_proj.bias
// variable name: model.layers.13.self_attn.o_proj.weight
// variable name: model.layers.4.self_attn.v_proj.weight
// variable name: model.layers.15.mlp.down_proj.weight
// variable name: model.layers.0.self_attn.k_proj.weight
// variable name: model.layers.2.mlp.gate_proj.weight
// variable name: model.layers.22.self_attn.v_proj.weight
// variable name: model.layers.0.self_attn.v_proj.weight
// variable name: model.layers.9.mlp.down_proj.weight
// variable name: model.layers.19.mlp.gate_proj.weight
// variable name: model.layers.20.self_attn.k_proj.weight
// variable name: model.layers.2.self_attn.q_proj.bias
// variable name: model.layers.1.post_attention_layernorm.weight
// variable name: model.layers.13.input_layernorm.weight
// variable name: model.layers.7.self_attn.v_proj.weight
// variable name: model.layers.10.self_attn.o_proj.weight
// variable name: model.layers.13.self_attn.k_proj.bias
// variable name: model.layers.4.self_attn.o_proj.weight
// variable name: model.layers.5.input_layernorm.weight
// variable name: model.layers.7.self_attn.k_proj.bias
// variable name: model.layers.10.mlp.up_proj.weight
// variable name: model.layers.23.mlp.gate_proj.weight
// variable name: model.layers.1.self_attn.k_proj.weight
// variable name: model.layers.21.mlp.up_proj.weight
// variable name: model.layers.21.post_attention_layernorm.weight
// variable name: model.layers.15.post_attention_layernorm.weight
// variable name: model.layers.0.mlp.up_proj.weight
// variable name: model.layers.19.self_attn.q_proj.bias
// variable name: model.layers.12.mlp.gate_proj.weight
// variable name: model.layers.7.input_layernorm.weight
// variable name: model.layers.6.self_attn.k_proj.weight
// variable name: model.layers.15.self_attn.q_proj.bias
// variable name: model.layers.3.self_attn.k_proj.weight
// variable name: model.layers.20.self_attn.q_proj.weight
// variable name: model.layers.17.post_attention_layernorm.weight
// variable name: model.layers.7.post_attention_layernorm.weight
// variable name: model.layers.23.self_attn.v_proj.bias
// variable name: model.layers.16.mlp.gate_proj.weight
// variable name: model.layers.4.post_attention_layernorm.weight
// variable name: model.layers.6.mlp.gate_proj.weight
// variable name: model.layers.16.mlp.up_proj.weight
// variable name: model.layers.4.self_attn.k_proj.bias
// variable name: model.layers.18.mlp.gate_proj.weight
// variable name: model.layers.14.mlp.gate_proj.weight
// variable name: model.layers.4.self_attn.v_proj.bias
// variable name: model.layers.23.self_attn.o_proj.weight
// variable name: model.layers.13.mlp.down_proj.weight
// variable name: model.layers.3.mlp.gate_proj.weight
// variable name: model.layers.10.mlp.down_proj.weight
// variable name: model.layers.11.mlp.up_proj.weight
// variable name: model.layers.22.self_attn.q_proj.weight
// variable name: model.layers.9.self_attn.v_proj.weight
// variable name: model.layers.9.mlp.up_proj.weight
// variable name: model.layers.18.self_attn.v_proj.bias
// variable name: model.layers.14.mlp.down_proj.weight
// variable name: model.layers.21.self_attn.q_proj.bias
// variable name: model.layers.13.post_attention_layernorm.weight
// variable name: model.layers.1.mlp.down_proj.weight
// variable name: model.layers.3.post_attention_layernorm.weight
// variable name: model.layers.3.self_attn.v_proj.bias
// variable name: model.layers.4.self_attn.q_proj.weight
// variable name: model.layers.17.mlp.down_proj.weight
// variable name: model.layers.1.self_attn.o_proj.weight
// variable name: model.layers.9.self_attn.q_proj.weight
// variable name: model.layers.18.mlp.down_proj.weight
// variable name: model.layers.4.mlp.up_proj.weight
// variable name: model.layers.12.self_attn.v_proj.weight
// variable name: model.layers.17.mlp.up_proj.weight
// variable name: model.layers.19.self_attn.q_proj.weight
// variable name: model.layers.9.mlp.gate_proj.weight
// variable name: model.layers.14.mlp.up_proj.weight
// variable name: model.layers.20.mlp.up_proj.weight
// variable name: model.layers.15.self_attn.k_proj.bias
// variable name: model.layers.14.self_attn.v_proj.bias
// variable name: model.layers.15.self_attn.v_proj.bias
// variable name: model.layers.22.self_attn.k_proj.weight
// variable name: model.layers.11.self_attn.q_proj.weight
// variable name: model.layers.15.self_attn.q_proj.weight
// variable name: model.layers.9.self_attn.k_proj.weight
// variable name: model.layers.19.self_attn.v_proj.bias
// variable name: model.layers.2.self_attn.k_proj.bias
// variable name: model.layers.21.self_attn.v_proj.bias
// variable name: model.layers.7.mlp.gate_proj.weight
// variable name: model.layers.3.mlp.up_proj.weight
// variable name: model.layers.5.post_attention_layernorm.weight
// variable name: model.layers.0.self_attn.q_proj.weight
// variable name: model.layers.22.self_attn.v_proj.bias
// variable name: model.layers.5.self_attn.v_proj.bias
// variable name: model.layers.12.input_layernorm.weight
// variable name: model.layers.1.self_attn.k_proj.bias
// variable name: model.layers.6.self_attn.q_proj.bias
// variable name: model.layers.23.mlp.up_proj.weight
// variable name: model.layers.8.self_attn.k_proj.bias
// variable name: model.layers.12.self_attn.q_proj.weight
// variable name: model.layers.20.self_attn.v_proj.bias
// variable name: model.layers.11.self_attn.v_proj.bias
// variable name: model.layers.23.post_attention_layernorm.weight
// variable name: model.layers.19.self_attn.k_proj.weight
// variable name: model.layers.10.self_attn.q_proj.weight
// variable name: model.layers.3.self_attn.q_proj.weight
// variable name: model.layers.11.post_attention_layernorm.weight
// variable name: model.layers.12.self_attn.q_proj.bias
// variable name: model.layers.18.self_attn.o_proj.weight
// variable name: model.layers.4.self_attn.q_proj.bias
// variable name: model.layers.8.self_attn.v_proj.bias
// variable name: model.layers.7.self_attn.o_proj.weight
// variable name: model.layers.14.self_attn.q_proj.weight
// variable name: model.layers.14.self_attn.q_proj.bias
// variable name: model.layers.23.self_attn.k_proj.bias
// variable name: model.layers.11.mlp.gate_proj.weight
// variable name: model.layers.19.self_attn.k_proj.bias
// variable name: model.layers.20.self_attn.v_proj.weight
// variable name: model.layers.23.input_layernorm.weight
// variable name: model.layers.14.self_attn.o_proj.weight
// variable name: model.layers.7.self_attn.q_proj.weight
// variable name: model.layers.12.self_attn.k_proj.bias
// variable name: model.layers.16.self_attn.v_proj.bias
// variable name: model.layers.17.self_attn.k_proj.bias
// variable name: model.layers.14.input_layernorm.weight
// variable name: model.layers.9.self_attn.q_proj.bias
// variable name: model.layers.15.input_layernorm.weight
// variable name: model.layers.5.self_attn.k_proj.bias
// variable name: model.layers.4.mlp.gate_proj.weight
// variable name: model.layers.1.mlp.up_proj.weight
// variable name: model.layers.6.input_layernorm.weight
// variable name: model.layers.11.mlp.down_proj.weight
// variable name: model.layers.16.self_attn.q_proj.weight
// variable name: model.layers.22.mlp.down_proj.weight
// variable name: model.layers.22.mlp.gate_proj.weight
// variable name: model.layers.19.self_attn.o_proj.weight
// variable name: model.layers.5.self_attn.q_proj.bias
// variable name: model.layers.9.input_layernorm.weight
// variable name: model.layers.5.self_attn.o_proj.weight
// variable name: model.layers.15.self_attn.o_proj.weight
// variable name: model.layers.10.self_attn.k_proj.bias
// variable name: model.layers.8.post_attention_layernorm.weight
// variable name: model.layers.13.mlp.up_proj.weight
// variable name: model.layers.23.self_attn.k_proj.weight
// variable name: model.layers.15.self_attn.k_proj.weight
// variable name: model.layers.16.input_layernorm.weight
// variable name: model.layers.1.self_attn.q_proj.weight
// variable name: model.layers.1.self_attn.v_proj.bias
// variable name: model.layers.22.self_attn.q_proj.bias
// variable name: model.layers.8.self_attn.o_proj.weight
// variable name: model.layers.2.self_attn.o_proj.weight
// variable name: model.layers.22.post_attention_layernorm.weight
// variable name: model.layers.23.self_attn.v_proj.weight
// variable name: model.layers.16.self_attn.o_proj.weight
// variable name: model.layers.4.input_layernorm.weight
// variable name: model.layers.5.mlp.up_proj.weight
// variable name: model.layers.19.mlp.up_proj.weight
// variable name: model.layers.3.self_attn.v_proj.weight
// variable name: model.layers.17.self_attn.k_proj.weight
// variable name: model.layers.7.self_attn.k_proj.weight
// variable name: model.layers.20.self_attn.k_proj.bias
// variable name: model.layers.10.self_attn.k_proj.weight
// variable name: model.layers.21.self_attn.k_proj.weight
// variable name: model.layers.5.self_attn.q_proj.weight
// variable name: model.layers.17.self_attn.q_proj.bias
// variable name: model.layers.3.self_attn.o_proj.weight
// variable name: model.layers.0.self_attn.k_proj.bias
// variable name: model.layers.2.self_attn.q_proj.weight
// variable name: model.layers.20.post_attention_layernorm.weight
// variable name: model.layers.6.mlp.down_proj.weight
// variable name: model.layers.11.self_attn.o_proj.weight
// variable name: model.layers.12.mlp.up_proj.weight
// variable name: model.layers.8.input_layernorm.weight
// variable name: model.layers.13.self_attn.v_proj.bias
// variable name: model.layers.21.mlp.gate_proj.weight
// variable name: model.layers.7.self_attn.q_proj.bias
// variable name: model.layers.6.mlp.up_proj.weight
// variable name: model.layers.0.self_attn.q_proj.bias
// variable name: model.layers.2.mlp.down_proj.weight
// variable name: model.layers.23.self_attn.q_proj.weight
// variable name: model.layers.18.self_attn.k_proj.bias
// variable name: model.layers.9.self_attn.k_proj.bias
// variable name: model.layers.15.mlp.up_proj.weight
// variable name: model.layers.21.self_attn.k_proj.bias
// variable name: model.layers.21.self_attn.v_proj.weight
// variable name: model.layers.18.self_attn.q_proj.weight
// variable name: model.layers.17.self_attn.v_proj.weight
// variable name: model.layers.10.self_attn.v_proj.weight
// variable name: model.layers.17.input_layernorm.weight
// variable name: model.layers.14.post_attention_layernorm.weight
// variable name: model.layers.19.input_layernorm.weight
// variable name: model.layers.0.input_layernorm.weight
// variable name: model.layers.22.input_layernorm.weight
// variable name: model.layers.8.self_attn.q_proj.bias
// variable name: model.layers.2.mlp.up_proj.weight
// variable name: model.layers.6.self_attn.q_proj.weight
// variable name: model.layers.10.self_attn.q_proj.bias
// variable name: model.layers.21.self_attn.q_proj.weight
// variable name: model.layers.8.self_attn.k_proj.weight
// variable name: model.layers.10.post_attention_layernorm.weight
// variable name: model.layers.16.mlp.down_proj.weight
// variable name: model.layers.8.mlp.gate_proj.weight
// variable name: model.layers.13.self_attn.k_proj.weight
// variable name: model.layers.11.self_attn.q_proj.bias
// variable name: model.layers.6.post_attention_layernorm.weight
// variable name: model.layers.22.self_attn.o_proj.weight
// variable name: model.layers.13.self_attn.q_proj.weight
// variable name: model.layers.22.self_attn.k_proj.bias
// variable name: model.layers.2.self_attn.k_proj.weight
// variable name: model.layers.2.post_attention_layernorm.weight
// variable name: model.layers.3.self_attn.k_proj.bias
// variable name: model.layers.15.mlp.gate_proj.weight
// variable name: model.layers.12.self_attn.v_proj.bias
// variable name: model.embed_tokens.weight
// variable name: model.layers.3.input_layernorm.weight
// variable name: model.layers.9.self_attn.v_proj.bias
// variable name: model.layers.8.self_attn.q_proj.weight
// variable name: model.layers.6.self_attn.o_proj.weight
// variable name: model.layers.21.mlp.down_proj.weight
// variable name: model.layers.16.self_attn.v_proj.weight
// variable name: model.layers.20.self_attn.q_proj.bias
// variable name: model.layers.18.self_attn.v_proj.weight
// variable name: model.layers.12.self_attn.o_proj.weight
// variable name: model.layers.20.self_attn.o_proj.weight
// variable name: model.layers.18.self_attn.q_proj.bias
// variable name: model.layers.11.self_attn.k_proj.bias
// variable name: model.norm.weight
// variable name: model.layers.20.input_layernorm.weight
// variable name: model.layers.20.mlp.down_proj.weight
// variable name: model.layers.9.self_attn.o_proj.weight
// variable name: model.layers.16.self_attn.q_proj.bias
// variable name: model.layers.1.self_attn.q_proj.bias
// variable name: model.layers.4.self_attn.k_proj.weight
// variable name: model.layers.1.input_layernorm.weight
// variable name: model.layers.18.mlp.up_proj.weight
// variable name: model.layers.12.mlp.down_proj.weight
// variable name: model.layers.6.self_attn.k_proj.bias
// variable name: model.layers.18.post_attention_layernorm.weight
// variable name: model.layers.5.self_attn.k_proj.weight
// variable name: model.layers.0.mlp.down_proj.weight
// variable name: model.layers.18.input_layernorm.weight
// variable name: model.layers.19.mlp.down_proj.weight
// variable name: model.layers.13.mlp.gate_proj.weight
// variable name: model.layers.4.mlp.down_proj.weight
// variable name: model.layers.14.self_attn.k_proj.weight
// variable name: model.layers.18.self_attn.k_proj.weight
// variable name: model.layers.19.post_attention_layernorm.weight
// variable name: model.layers.5.self_attn.v_proj.weight
// variable name: model.layers.12.post_attention_layernorm.weight
// variable name: model.layers.9.post_attention_layernorm.weight
// variable name: model.layers.14.self_attn.v_proj.weight
// variable name: model.layers.22.mlp.up_proj.weight
// variable name: model.layers.21.self_attn.o_proj.weight
// variable name: model.layers.1.self_attn.v_proj.weight
// variable name: model.layers.17.self_attn.q_proj.weight
// variable name: model.layers.8.mlp.down_proj.weight
// variable name: model.layers.0.self_attn.v_proj.bias
// variable name: model.layers.6.self_attn.v_proj.bias
// variable name: model.layers.0.mlp.gate_proj.weight
// variable name: model.layers.10.self_attn.v_proj.bias
// variable name: model.layers.19.self_attn.v_proj.weight
// variable name: model.layers.20.mlp.gate_proj.weight
// variable name: model.layers.21.input_layernorm.weight
// variable name: model.layers.7.mlp.down_proj.weight
// variable name: model.layers.8.mlp.up_proj.weight
// variable name: model.layers.2.self_attn.v_proj.bias
// variable name: model.layers.16.post_attention_layernorm.weight
// variable name: model.layers.10.mlp.gate_proj.weight
// variable name: model.layers.3.mlp.down_proj.weight
// variable name: model.layers.16.self_attn.k_proj.bias
// variable name: model.layers.10.input_layernorm.weight
// variable name: model.layers.6.self_attn.v_proj.weight
// variable name: model.layers.16.self_attn.k_proj.weight
// variable name: model.layers.23.self_attn.q_proj.bias
// variable name: model.layers.5.mlp.down_proj.weight
// variable name: model.layers.0.post_attention_layernorm.weight
// variable name: model.layers.23.mlp.down_proj.weight
// variable name: model.layers.17.self_attn.v_proj.bias
// variable name: model.layers.5.mlp.gate_proj.weight
// variable name: model.layers.0.self_attn.o_proj.weight
// variable name: model.layers.11.input_layernorm.weight
// variable name: model.layers.13.self_attn.v_proj.weight
// variable name: model.layers.11.self_attn.v_proj.weight
// variable name: model.layers.2.self_attn.v_proj.weight
// variable name: model.layers.15.self_attn.v_proj.weight
// variable name: model.layers.1.mlp.gate_proj.weight
// variable name: model.layers.11.self_attn.k_proj.weight
// variable name: model.layers.3.self_attn.q_proj.bias
// variable name: model.layers.17.self_attn.o_proj.weight

fn main() {
    let m = candle_core::safetensors::load(
        "Qwen2.5-0.5B-instruct/model.safetensors",
        &candle_core::Device::Cpu,
    )
    .unwrap();
    for (name, t) in m.iter() {
        if name.eq("model.layers.14.mlp.up_proj.weight") {
            println!("{:?}", t.shape())
        }
    }
}
