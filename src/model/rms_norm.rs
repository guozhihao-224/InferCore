use candle_core::Tensor;

pub struct RMSNorm {
    pub gamma: Tensor, // shape: [hidden]
    pub eps: f32,      // 一般为 1e-6
}

impl RMSNorm {
    pub fn new(gamma: Tensor, eps: f32) -> Self {
        Self { gamma, eps }
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // 1. x^2
        let x2 = x.sqr()?;
        // 2. mean(x^2) over last dim, keepdim
        let mean = x2.mean_keepdim(x2.rank() - 1)?;
        // 3. sqrt(mean + epa)
        let eps_tensor = Tensor::full(self.eps, mean.shape(), mean.device())?;
        let denom = mean.add(&eps_tensor)?.sqrt()?;
        // 4. x / denom
        let normed = x.broadcast_div(&denom)?;
        // 5. normed * gamma
        normed.broadcast_mul(&self.gamma)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use candle_core::Device;
    use half::f16;

    use super::*;
    use crate::util;

    fn make_tensor(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_rmsnorm_pytorch_alignment() {
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = make_tensor(&[1.0, 2.0, 0.5, -1.0], &[4]);
        let eps = 1e-5;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        let expected = vec![vec![0.36514813, 1.46059251, 0.54772219, -1.46059251]];
        for (row, exp_row) in y_vec.iter().zip(expected.iter()) {
            for (v, e) in row.iter().zip(exp_row.iter()) {
                assert_abs_diff_eq!(v, e, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_rmsnorm_batch_broadcast() {
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let gamma = make_tensor(&[1.0, 2.0, 0.5], &[3]);
        let eps = 1e-5;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        assert_eq!(y_vec.len(), 2);
        assert_eq!(y_vec[0].len(), 3);
        assert_eq!(y_vec[1].len(), 3);
    }

    #[test]
    fn test_rmsnorm_all_zeros() {
        let x = make_tensor(&[0.0; 8], &[2, 4]);
        let gamma = make_tensor(&[1.0; 4], &[4]);
        let eps = 1e-6;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        for row in y_vec {
            for v in row {
                assert_abs_diff_eq!(v, 0.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_rmsnorm_large_numbers() {
        let x = make_tensor(&[1e6, 2e6, 3e6, 4e6], &[1, 4]);
        let gamma = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let eps = 1e-5;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        assert!(y_vec[0].iter().all(|v| v.abs() < 2e6));
    }

    #[test]
    fn test_rmsnorm_small_eps() {
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let eps = 1e-12;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        for v in &y_vec[0] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_rmsnorm_gamma_nontrivial() {
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = make_tensor(&[2.0, 0.5, -1.0, 1.0], &[4]);
        let eps = 1e-5;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        assert_eq!(y_vec[0].len(), 4);
        for v in &y_vec[0] {
            assert!(v.abs() > 0.0);
        }
    }

    #[test]
    fn test_rmsnorm_shape_broadcast() {
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let gamma = make_tensor(&[1.0, 1.0], &[2]);
        let eps = 1e-5;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec3::<f32>().unwrap();
        assert_eq!(y_vec.len(), 2);
        assert_eq!(y_vec[0].len(), 2);
        assert_eq!(y_vec[0][0].len(), 2);
    }

    #[test]
    fn test_rmsnorm_float16_weight() {
        let gamma_f16: Vec<f16> = vec![f16::from_f32(1.0); 4];
        let gamma = Tensor::from_vec(
            gamma_f16.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            &[4],
            &Device::Cpu,
        )
        .unwrap();
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let eps = 1e-5;
        let rms = RMSNorm::new(gamma, eps);
        let y = rms.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        assert_eq!(y_vec[0].len(), 4);
    }

    #[test]
    fn test_rmsnorm_against_mlx() {
        use util::assert_allclose;
        use util::load_tensor_from_npy;
        let device = Device::Cpu;
        let test_cases = ["large_2d", "medium_2d", "small_2d", "single_row"];
        let base_path = "tests/rms_norm_test_data/";
        for case_id in test_cases {
            // 加载测试数据
            let data =
                load_tensor_from_npy(&format!("{}rms_data_{}.npy", base_path, case_id), &device)
                    .unwrap();
            let weight =
                load_tensor_from_npy(&format!("{}rms_weight_{}.npy", base_path, case_id), &device)
                    .unwrap();
            let y_ref =
                load_tensor_from_npy(&format!("{}rms_output_{}.npy", base_path, case_id), &device)
                    .unwrap();
            let eps_tensor =
                load_tensor_from_npy(&format!("{}rms_eps_{}.npy", base_path, case_id), &device)
                    .unwrap();
            let eps = eps_tensor.to_vec1::<f32>().unwrap()[0];

            // 构建 RMSNorm
            let rmsnorm = RMSNorm::new(weight, eps);

            // 前向计算
            let y = rmsnorm.forward(&data).unwrap();

            // allclose 检查
            assert_allclose(&y, &y_ref, candle_core::DType::F32, None, None);

            println!("Case {} passed!", case_id);
        }
    }
}
