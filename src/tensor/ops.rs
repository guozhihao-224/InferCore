use ndarray::Axis;
use rayon::prelude::*;

use crate::tensor::tensor::Tensor;

/// 对 attention scores 应用 causal mask - 优化版本
/// scores: [batch, n_heads, seq, seq]
pub fn apply_causal_mask(scores: &mut Tensor) {
    let shape = scores.shape();
    let (batch, n_heads, seq, _) = (shape[0], shape[1], shape[2], shape[3]);
    
    // 使用简单的循环，避免并行化的借用问题
    for b in 0..batch {
        for h in 0..n_heads {
            for i in 0..seq {
                for j in (i + 1)..seq {
                    *scores.get_mut(&[b, h, i, j]) = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// 优化的softmax实现
pub fn softmax(x: &Tensor, axis: i32) -> Tensor {
    let last_axis = x.shape.len() - 1;
    let true_axis = if axis < 0 {
        (x.shape.len() as i32 + axis) as usize
    } else {
        axis as usize
    };
    assert_eq!(
        true_axis, last_axis,
        "Simplified softmax only supports the last axis"
    );

    let mut new_data = x.data.clone();

    // 并行处理每一行
    new_data.axis_iter_mut(Axis(last_axis)).par_bridge().for_each(|mut row| {
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut exp_sum = 0.0;
        row.mapv_inplace(|v| {
            let exp_val = (v - max_val).exp();
            exp_sum += exp_val;
            exp_val
        });

        row.mapv_inplace(|v| v / exp_sum);
    });

    Tensor {
        data: new_data,
        shape: x.shape.clone(),
        device: x.device,
        dtype: x.dtype,
    }
}

/// 优化的SiLU激活函数
pub fn silu(x: &Tensor) -> Tensor {
    // 并行计算sigmoid和乘法
    let data: Vec<f32> = x
        .to_vec()
        .into_par_iter()
        .map(|v| v * (1.0 / (1.0 + (-v).exp())))
        .collect();
    Tensor::from_vec(data, x.shape(), x.device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::device::Device;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_apply_causal_mask() {
        // 构造一个 shape = [1, 1, 4, 4] 的分数张量，值为递增 0~15
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let mut scores = Tensor::from_vec(data.clone(), &[1, 1, 4, 4], Device::Cpu);
        apply_causal_mask(&mut scores);
        let masked = scores.to_vec();
        // 期望下三角（含对角线）为原值，上三角为 -inf
        let mut expected = data.clone();
        for i in 0..4 {
            for j in (i + 1)..4 {
                expected[i * 4 + j] = f32::NEG_INFINITY;
            }
        }
        for k in 0..16 {
            if expected[k].is_infinite() {
                assert!(
                    masked[k].is_infinite() && masked[k].is_sign_negative(),
                    "位置{} 应为 -inf",
                    k
                );
            } else {
                assert!((masked[k] - expected[k]).abs() < 1e-6, "位置{} 值不符", k);
            }
        }
    }
}
