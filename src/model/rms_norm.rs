use crate::tensor::{device::Device, tensor::Tensor};
use std::f32;

#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub gamma: Tensor, // shape: [hidden_size]
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            gamma: Tensor::ones(&[hidden_size], Device::Cpu),
            eps,
        }
    }

    /// RMSNorm 归一化公式：
    ///     y = x / (RMS(x) + eps) * gamma
    ///
    /// 其中：
    ///     RMS(x) = sqrt(mean(x^2))
    ///
    /// - x: 输入向量
    /// - eps: 极小值，防止除零
    /// - gamma： 可学习的缩放参数
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        debug_assert!(!shape.is_empty());
        let hidden_size = *shape.last().unwrap();
        let mut output = input.data.clone();

        output.outer_iter_mut().for_each(|mut row| {
            let rms =
                (row.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32 + self.eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = *v / rms * self.gamma.data[i];
            }
        });

        Tensor {
            data: output,
            dtype: input.dtype,
            device: input.device,
            shape: input.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::device::Device;

    #[test]
    fn test_rms_norm_basic() {
        let rms = RMSNorm::new(4, 1e-5);
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], Device::Cpu);
        let output = rms.forward(&input);
        assert_eq!(output.shape(), &[1, 4]);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let rms_val = (mean_square + eps).sqrt();
        let expected: Vec<f32> = x
            .iter()
            .zip(gamma.iter())
            .map(|(xi, gi)| xi / rms_val * gi)
            .collect();

        let out_vec = output.to_vec();
        for (o, e) in out_vec.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "output: {}, expected: {}", o, e);
        }
    }

    #[test]
    fn test_rmsnorm_with_known_result() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], Device::Cpu);
        let gamma = Tensor::from_vec(vec![1.0, 2.0, 0.5, -1.0], &[4], Device::Cpu);
        let rms = RMSNorm { gamma, eps: 1e-5 };
        let output = rms.forward(&input);
        let expected = vec![0.36514813, 1.46059251, 0.54772219, -1.46059251];
        for (o, e) in output.to_vec().iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "output: {}, expected: {}", o, e);
        }
    }

    #[test]
    fn test_rms_norm_eps_stability() {
        // 测试极小 eps 下的数值稳定性
        let rms = RMSNorm::new(4, 1e-12);
        let input = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[1, 4], Device::Cpu);
        let output = rms.forward(&input);
        // 归一化后应为全 0
        for v in output.to_vec() {
            assert!(v.abs() < 1e-6);
        }
    }
}
