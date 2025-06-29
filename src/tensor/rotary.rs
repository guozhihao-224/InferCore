use crate::tensor::tensor::Tensor;

/// 对 Q/K 张量应用 rotary embedding
/// x: [batch, seq_len, n_heads, head_dim]
pub fn apply_rotary_embedding(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let (batch, seq_len, n_heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    assert!(head_dim % 2 == 0, "head_dim 必须为偶数");

    // 1. 生成 inv_freq
    let inv_freq: Vec<f32> = (0..head_dim / 2)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    // 2. 生成每个 token 位置的 cos/sin
    let mut cos = vec![0.0; seq_len * head_dim / 2];
    let mut sin = vec![0.0; seq_len * head_dim / 2];
    for pos in 0..seq_len {
        for i in 0..head_dim / 2 {
            let theta = pos as f32 * inv_freq[i];
            cos[pos * head_dim / 2 + i] = theta.cos();
            sin[pos * head_dim / 2 + i] = theta.sin();
        }
    }

    // 3. 对 x 的每个 batch, seq, head, head_dim 做旋转
    let mut out = x.clone();
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..n_heads {
                for i in 0..head_dim / 2 {
                    let idx0 = 2 * i;
                    let idx1 = 2 * i + 1;
                    let q0 = x.get(&[b, s, h, idx0]);
                    let q1 = x.get(&[b, s, h, idx1]);
                    let c = cos[s * head_dim / 2 + i];
                    let s_ = sin[s * head_dim / 2 + i];
                    *out.get_mut(&[b, s, h, idx0]) = q0 * c - q1 * s_;
                    *out.get_mut(&[b, s, h, idx1]) = q0 * s_ + q1 * c;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::device::Device;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_rotary_embedding_shape_and_value() {
        // 构造一个 shape = [1, 2, 1, 4] 的张量
        // batch=1, seq_len=2, n_heads=1, head_dim=4
        // 数据为递增 0.0, 1.0, 2.0, ...
        let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let x = Tensor::from_vec(data.clone(), &[1, 2, 1, 4], Device::Cpu);
        let y = apply_rotary_embedding(&x);
        // shape 不变
        assert_eq!(y.shape(), &[1, 2, 1, 4]);
        // 手工计算第一个 token 的旋转
        // head_dim=4, inv_freq=[1, 1/10000]
        let inv_freq = vec![1.0, 0.0001];
        // pos=0
        let theta0 = 0.0f32 * inv_freq[0];
        let theta1 = 0.0f32 * inv_freq[1];
        let cos0 = theta0.cos();
        let sin0 = theta0.sin();
        let cos1 = theta1.cos();
        let sin1 = theta1.sin();
        // x[0,0,0,0]=0.0, x[0,0,0,1]=1.0, x[0,0,0,2]=2.0, x[0,0,0,3]=3.0
        let v0 = 0.0 * cos0 - 1.0 * sin0;
        let v1 = 0.0 * sin0 + 1.0 * cos0;
        let v2 = 2.0 * cos1 - 3.0 * sin1;
        let v3 = 2.0 * sin1 + 3.0 * cos1;
        let y_vec = y.to_vec();
        assert!((y_vec[0] - v0).abs() < 1e-6);
        assert!((y_vec[1] - v1).abs() < 1e-6);
        assert!((y_vec[2] - v2).abs() < 1e-6);
        assert!((y_vec[3] - v3).abs() < 1e-6);
    }
}
