use crate::tensor::tensor::Tensor;

/// 泛型多头自注意力结构体，字段命名对齐 Qwen2.5 权重
pub struct Attention<T: crate::tensor::tensor::Tensor<Elem = f32>> {
    pub q_proj: T, // Q 投影权重 [hidden, hidden]
    pub q_bias: Option<T>,
    pub k_proj: T, // K 投影权重 [hidden, hidden]
    pub k_bias: Option<T>,
    pub v_proj: T, // V 投影权重 [hidden, hidden]
    pub v_bias: Option<T>,
    pub o_proj: T, // 输出投影权重 [hidden, hidden]
    pub o_bias: Option<T>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<T: crate::tensor::tensor::Tensor<Elem = f32>> Attention<T> {
    pub fn new(
        q_proj: T, q_bias: Option<T>,
        k_proj: T, k_bias: Option<T>,
        v_proj: T, v_bias: Option<T>,
        o_proj: T, o_bias: Option<T>,
        num_heads: usize, head_dim: usize
    ) -> Self {
        Self { q_proj, q_bias, k_proj, k_bias, v_proj, v_bias, o_proj, o_bias, num_heads, head_dim }
    }

    /// [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
    fn split_heads(&self, x: &T) -> T {
        let d = x.dims();
        let (batch, seq, hidden) = (d[0], d[1], d[2]);
        assert_eq!(hidden, self.num_heads * self.head_dim);
        let x = x.reshape(&[batch, seq, self.num_heads, self.head_dim]);
        x.permute(&[0, 2, 1, 3])
    }
    /// [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
    fn merge_heads(&self, x: &T) -> T {
        let d = x.dims();
        let (batch, num_heads, seq, head_dim) = (d[0], d[1], d[2], d[3]);
        let x = x.permute(&[0, 2, 1, 3]);
        x.reshape(&[batch, seq, num_heads * head_dim])
    }

    /// x: [batch, seq, hidden]
    /// mask: Option<[batch, seq, seq]> (可选)
    /// 返回: [batch, seq, hidden]
    pub fn forward(&self, x: &T, mask: Option<&T>) -> T {
        // 1. Q/K/V 投影
        let q = {
            let mut t = x.matmul(&self.q_proj);
            if let Some(ref b) = self.q_bias {
                let b = b.broadcast_as(t.dims());
                t = t.add(&b);
            }
            t
        };
        let k = {
            let mut t = x.matmul(&self.k_proj);
            if let Some(ref b) = self.k_bias {
                let b = b.broadcast_as(t.dims());
                t = t.add(&b);
            }
            t
        };
        let v = {
            let mut t = x.matmul(&self.v_proj);
            if let Some(ref b) = self.v_bias {
                let b = b.broadcast_as(t.dims());
                t = t.add(&b);
            }
            t
        };
        // 2. 多头分解 [batch, num_heads, seq, head_dim]
        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);
        
        // 3. QK^T / sqrt(d)
        let k_t = k.permute(&[0, 1, 3, 2]); // [batch, num_heads, head_dim, seq]
        let scale = (self.head_dim as f32).sqrt().ln();
        let attn_scores = q.matmul(&k_t).add_scalar(-scale);

        // 4. mask（可选）
        let attn_scores = if let Some(mask) = mask {
            attn_scores.add(mask)
        } else {
            attn_scores
        };
        // 5. softmax 最后一个轴
        let attn_probs = attn_scores.softmax(-1);
        // 6. attention output
        let attn_out = attn_probs.matmul(&v); // [batch, num_heads, seq, head_dim]
        // 7. 合并头 [batch, seq, hidden]
        let attn_out = self.merge_heads(&attn_out);
        // 8. 输出投影
        let mut y = attn_out.matmul(&self.o_proj);
        if let Some(ref b) = self.o_bias {
            let b = b.broadcast_as(y.dims());
            y = y.add(&b);
        }
        y
    }

}


pub fn apply_qwen2_rotary_embedding(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let (batch, seq_len, n_heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    assert!(head_dim % 2 == 0, "head_dim 必须为偶数");
    let half_dim = head_dim / 2;

    // 1. 前半部分频率
    let inv_freq1: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / half_dim as f32))
        .collect();
    // 2. 后半部分频率（可与前半不同，Qwen2.5官方实现通常用更高频率或不同间隔）
    let inv_freq2: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / half_dim as f32))
        .collect();

    // 3. 生成 cos/sin
    let mut cos1 = vec![0.0; seq_len * half_dim];
    let mut sin1 = vec![0.0; seq_len * half_dim];
    let mut cos2 = vec![0.0; seq_len * half_dim];
    let mut sin2 = vec![0.0; seq_len * half_dim];
    for pos in 0..seq_len {
        for i in 0..half_dim {
            let theta1 = pos as f32 * inv_freq1[i];
            let theta2 = pos as f32 * inv_freq2[i];
            cos1[pos * half_dim + i] = theta1.cos();
            sin1[pos * half_dim + i] = theta1.sin();
            cos2[pos * half_dim + i] = theta2.cos();
            sin2[pos * half_dim + i] = theta2.sin();
        }
    }

    // 4. 对 x 的每个 batch, seq, head, head_dim 做分半旋转
    let mut out = x.clone();
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..n_heads {
                // 前半
                for i in 0..(half_dim / 2) {
                    let idx0 = 2 * i;
                    let idx1 = 2 * i + 1;
                    let q0 = x.get(&[b, s, h, idx0]);
                    let q1 = x.get(&[b, s, h, idx1]);
                    let c = cos1[s * half_dim + idx0 / 2];
                    let s_ = sin1[s * half_dim + idx0 / 2];
                    *out.get_mut(&[b, s, h, idx0]) = q0 * c - q1 * s_;
                    *out.get_mut(&[b, s, h, idx1]) = q0 * s_ + q1 * c;
                }
                // 后半
                for i in 0..(half_dim / 2) {
                    let idx0 = half_dim + 2 * i;
                    let idx1 = half_dim + 2 * i + 1;
                    let q0 = x.get(&[b, s, h, idx0]);
                    let q1 = x.get(&[b, s, h, idx1]);
                    let c = cos2[s * half_dim + idx0 / 2 - half_dim / 2];
                    let s_ = sin2[s * half_dim + idx0 / 2 - half_dim / 2];
                    *out.get_mut(&[b, s, h, idx0]) = q0 * c - q1 * s_;
                    *out.get_mut(&[b, s, h, idx1]) = q0 * s_ + q1 * c;
                }
            }
        }
    }
    out
}



// --- 用法示例 ---
// let attn = Attention::new(q_proj, None, k_proj, None, v_proj, None, o_proj, None, num_heads, head_dim);
// let y = attn.forward(&x, None);

// --- 测试建议 ---
// 1. shape 检查：输入输出 shape 是否一致
// 2. 与 PyTorch 对齐：数值精度
// 3. mask 支持：mask shape、广播、极端值
// 4. 多 batch/多头/不同 head_dim
