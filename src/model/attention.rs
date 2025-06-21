use crate::tensor::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Attention {
    pub wq: Tensor, // [hidden, hidden]
    pub wk: Tensor, // [hidden, hidden]
    pub wv: Tensor, // [hidden, hidden]
    pub n_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn qkv(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        // x: [batch, seq, hidden]
        // wq/wk/wv: [hidden, hidden]
        let q = x.matmul(&self.wq); // [batch, seq, hidden]
        let k = x.matmul(&self.wk); // [batch, seq, hidden]
        let v = x.matmul(&self.wv); // [batch, seq, hidden]
        (q, k, v)
    }

    pub fn split_heads(&self, x: &Tensor) -> Tensor {
        // x: [batch, seq, hidden]
        let (batch, seq, hidden) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(hidden, self.n_heads * self.head_dim);

        // 先 reshape 为 [batch, seq, n_heads, head_dim]
        let reshaped = x.reshape(&[batch, seq, self.n_heads, self.head_dim]);
        // 再转置为 [batch, n_heads, seq, head_dim]
        reshaped.transpose(&[0, 2, 1, 3])
    }

    pub fn merge_heads(&self, x: &Tensor) -> Tensor {
        // x: [batch, n_heads, seq, head_dim]
        let (batch, n_heads, seq, head_dim) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        assert_eq!(n_heads, self.n_heads);
        assert_eq!(head_dim, self.head_dim);

        // 先转置为 [batch, seq, n_heads, head_dim]
        let transposed = x.transpose(&[0, 2, 1, 3]);
        // 再 reshape 为 [batch, seq, hidden]
        transposed.reshape(&[batch, seq, n_heads * head_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{device::Device, tensor::Tensor};

    #[test]
    fn test_qkv_projection() {
        // 假设 hidden=4
        let wq = Tensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &[4, 4],
            Device::Cpu,
        );
        let wk = wq.clone();
        let wv = wq.clone();

        let attn = Attention {
            wq,
            wk,
            wv,
            n_heads: 1,
            head_dim: 1,
        };
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], Device::Cpu);

        let (q, k, v) = attn.qkv(&x);
        assert_eq!(q.to_vec(), x.to_vec());
        assert_eq!(k.to_vec(), x.to_vec());
        assert_eq!(v.to_vec(), x.to_vec());
    }

    #[test]
    fn test_split_and_merge_heads() {
        let n_heads = 2;
        let head_dim = 2;
        let hidden = n_heads * head_dim;
        let batch = 1;
        let seq = 2;
        let attn = Attention {
            wq: Tensor::zeros(&[hidden, hidden], Device::Cpu),
            wk: Tensor::zeros(&[hidden, hidden], Device::Cpu),
            wv: Tensor::zeros(&[hidden, hidden], Device::Cpu),
            n_heads,
            head_dim,
        };
        let x = Tensor::from_vec(
            (0..(batch * seq * hidden)).map(|v| v as f32).collect(),
            &[batch, seq, hidden],
            Device::Cpu,
        );

        let split = attn.split_heads(&x);
        assert_eq!(split.shape(), &[batch, n_heads, seq, head_dim]);

        let merged = attn.merge_heads(&split);
        assert_eq!(merged.shape(), &[batch, seq, hidden]);
        assert_eq!(merged.to_vec(), x.to_vec());
    }

    #[test]
    fn test_qkv_projection_multihead() {
        let n_heads = 2;
        let head_dim = 2;
        let hidden = n_heads * head_dim;
        let batch = 1;
        let seq = 2;

        // 构造单位矩阵权重，便于验证
        let wq = Tensor::from_vec(
            (0..hidden * hidden)
                .map(|i| if i % (hidden + 1) == 0 { 1.0 } else { 0.0 })
                .collect(),
            &[hidden, hidden],
            Device::Cpu,
        );
        let wk = wq.clone();
        let wv = wq.clone();

        let attn = Attention {
            wq,
            wk,
            wv,
            n_heads,
            head_dim,
        };

        // 输入 shape: [batch, seq, hidden]
        let x = Tensor::from_vec(
            (0..(batch * seq * hidden)).map(|v| v as f32).collect(),
            &[batch, seq, hidden],
            Device::Cpu,
        );

        // Q/K/V projection
        let (q, k, v) = attn.qkv(&x);

        // 多头拆分
        let q_split = attn.split_heads(&q);
        let k_split = attn.split_heads(&k);
        let v_split = attn.split_heads(&v);

        // 检查 shape
        assert_eq!(q_split.shape(), &[batch, n_heads, seq, head_dim]);
        assert_eq!(k_split.shape(), &[batch, n_heads, seq, head_dim]);
        assert_eq!(v_split.shape(), &[batch, n_heads, seq, head_dim]);

        // 检查内容（单位矩阵权重，输出应等于输入）
        let q_merged = attn.merge_heads(&q_split);
        assert_eq!(q_merged.to_vec(), x.to_vec());
    }
}
