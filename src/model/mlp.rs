use candle_core::Tensor;

pub struct SwiGluMlp {
    pub gate_proj: Tensor,
    pub gate_bias: Option<Tensor>,
    pub up_proj: Tensor,
    pub up_bias: Option<Tensor>,
    pub down_proj: Tensor,
    pub down_bias: Option<Tensor>,
}

impl SwiGluMlp {
    pub fn new(
        gate_proj: Tensor,
        gate_bias: Option<Tensor>,
        up_proj: Tensor,
        up_bias: Option<Tensor>,
        down_proj: Tensor,
        down_bias: Option<Tensor>,
    ) -> Self {
        Self {
            gate_proj,
            gate_bias,
            up_proj,
            up_bias,
            down_proj,
            down_bias,
        }
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // 1. gate = x @ gate_proj + gate_bias
        let mut gate = x.broadcast_matmul(&self.gate_proj.t()?)?;
        if let Some(ref bias) = self.gate_bias {
            gate = { gate + bias }?;
        }

        // 2. up = x @ up_proj + up_bias
        let mut up = x.broadcast_matmul(&self.up_proj.t()?)?;
        if let Some(ref bias) = self.up_bias {
            up = { up + bias }?;
        }
        // 3. h = silu(gate) * up
        let h = { gate.silu()? * up }?;

        // 4. y = h @ down_proj + down_bias
        let mut y = h.broadcast_matmul(&self.down_proj.t()?)?;
        if let Some(ref bias) = self.down_bias {
            y = { y + bias }?;
        }

        Ok(y)
    }
}

#[cfg(test)]
mod compare_tests {
    use candle_core::Device;
    use ndarray_npy::read_npy;

    use super::*;
    use crate::util::assert_allclose;

    fn load_tensor_from_npy(path: &str, device: &Device) -> anyhow::Result<Tensor> {
        let arr: ndarray::ArrayD<f32> = read_npy(path)?;
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f32> = arr.iter().cloned().collect();
        Ok(Tensor::from_vec(data, shape, device)?)
    }

    #[test]
    fn test_mlp() {
        let base_path = "tests/mlp_test_data/";
        let device = Device::Cpu;
        let groups = ["small_dims", "large_dims", "single_token"];
        for group in groups {
            let x =
                load_tensor_from_npy(&format!("{}x_{}.npy", base_path, group), &device).unwrap();
            let w_gate =
                load_tensor_from_npy(&format!("{}w_gate_{}.npy", base_path, group), &device)
                    .unwrap();
            let w_up =
                load_tensor_from_npy(&format!("{}w_up_{}.npy", base_path, group), &device).unwrap();
            let w_down =
                load_tensor_from_npy(&format!("{}w_down_{}.npy", base_path, group), &device)
                    .unwrap();
            let y_ref = load_tensor_from_npy(
                &format!("{}reference_output_{}.npy", base_path, group),
                &device,
            )
            .unwrap();

            let mlp = SwiGluMlp::new(w_gate, None, w_up, None, w_down, None);
            let y = mlp.forward(&x).unwrap();

            let rtol = 1e-5;
            let atol = 1e-8;
            assert_allclose(&y, &y_ref, candle_core::DType::F32, Some(rtol), Some(atol));
        }
    }
}
