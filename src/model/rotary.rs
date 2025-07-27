use candle_core::DType;
use candle_core::Device;
use candle_core::Result;
use candle_core::Tensor;

#[allow(dead_code)]
pub struct RotaryEmbedding {
    dims: usize,
    seq_len: usize,
    half_dims: usize,
    cos_freqs: Tensor, // (seq_len, half_dims)
    sin_freqs: Tensor, // (seq_len, half_dims)
    traditional: bool,
}

impl RotaryEmbedding {
    pub fn new(dims: usize, seq_len: usize, base: f32, traditional: bool) -> Result<Self> {
        // 1. 确保维度是偶数，因为RoPE总是成对进行处理
        assert!(dims % 2 == 0, "dims must be even");
        let half_dims = dims / 2;

        // 2. 计算频率
        let inner = Tensor::arange(0, half_dims as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .broadcast_div(&Tensor::full(half_dims as f32, (), &Device::Cpu)?)?;

        let base_tensor = Tensor::full(base, (half_dims,), &Device::Cpu)?;
        let freqs = base_tensor.pow(&(inner * -1.0)?)?;

        // 3. 创建位置和频率的组合
        //   t = [0, 1, 2, ..., seq_len-1]
        let t = Tensor::arange(0, seq_len as u32, &Device::Cpu)?.to_dtype(DType::F32)?;
        let freqs = t
            .reshape((seq_len, 1))?
            .matmul(&freqs.reshape((1, half_dims))?)?;

        let cos_freqs = freqs.cos()?;
        let sin_freqs = freqs.sin()?;

        Ok(Self {
            dims,
            seq_len,
            half_dims,
            cos_freqs,
            sin_freqs,
            traditional,
        })
    }

    pub fn apply_rotary(&self, x: &Tensor, offset: Option<usize>) -> Result<Tensor> {
        // x的shape是 [batch, seq_len, num_heads, head_dim]
        let (batch, seq_len, num_heads, head_dim) = x.dims4()?;
        assert_eq!(head_dim, self.dims, "head_dim must match dims");

        // 1. 根据偏移量获取当前的cos和sin
        let mut cos_basis = if let Some(offset) = offset {
            self.cos_freqs.narrow(0, offset, seq_len)?
        } else {
            self.cos_freqs.narrow(0, 0, seq_len)?
        };

        let mut sin_basis = if let Some(offset) = offset {
            self.sin_freqs.narrow(0, offset, seq_len)?
        } else {
            self.sin_freqs.narrow(0, 0, seq_len)?
        };

        // 2. 分割输入的x
        let (x1, x2) = {
            if self.traditional {
                let x_reshaped = x.reshape((batch, seq_len, num_heads, self.half_dims, 2))?;
                (
                    x_reshaped.narrow(4, 0, 1)?.squeeze(4)?,
                    x_reshaped.narrow(4, 1, 1)?.squeeze(4)?,
                )
            } else {
                (
                    x.narrow(3, 0, self.half_dims)?,
                    x.narrow(3, self.half_dims, self.half_dims)?,
                )
            }
        };

        // 3. 调整 cos/sin 基的形状以进行广播
        cos_basis = cos_basis.reshape((1, seq_len, 1, self.half_dims))?;
        sin_basis = sin_basis.reshape((1, seq_len, 1, self.half_dims))?;
        cos_basis = cos_basis.expand((batch, seq_len, num_heads, self.half_dims))?;
        sin_basis = sin_basis.expand((batch, seq_len, num_heads, self.half_dims))?;

        // 4. 应用旋转（核心计算）
        //    这等价于复数乘法 (x1 + i*x2) * (cos + i*sin)
        //    或者说，将二维向量 [x1, x2] 左乘一个旋转矩阵 [[cos, -sin], [sin, cos]]
        let real = ((&x1 * &cos_basis)? - (&x2 * &sin_basis)?)?;
        let imag = ((&x2 * &cos_basis)? + (&x1 * &sin_basis)?)?;

        // 5. 合并结果
        let y = {
            if self.traditional {
                // 将旋转后的 real 和 imag 交错堆叠回去
                // Python: mx.stack([real, imag], axis=-1)
                let real_expanded = real.unsqueeze(4)?; // 添加最后一个维度
                let imag_expanded = imag.unsqueeze(4)?; // 添加最后一个维度
                let y = Tensor::cat(&[&real_expanded, &imag_expanded], 4)?; // 在最后一个维度拼接
                y.reshape((batch, seq_len, num_heads, head_dim))?
            } else {
                // 将旋转后的两部分拼接起来
                // Python: mx.concat([real, imag], axis=-1)
                let y = Tensor::cat(&[&real, &imag], 3)?; // 在第3维（最后一个维度）拼接
                y.reshape((batch, seq_len, num_heads, head_dim))?
            }
        };

        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use candle_core::Device;
    use ndarray::Array1;
    use ndarray::Array4;
    use ndarray_npy::read_npy;

    use super::*;
    use crate::util::assert_allclose;

    /// 将ndarray转换为Tensor
    fn array_to_tensor(arr: Array4<f32>, device: &Device) -> Tensor {
        let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = arr.iter().cloned().collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    /// 加载单个样本 (input/reference/offset)
    fn load_case(prefix: &str) -> (Array4<f32>, Array4<f32>, usize) {
        let x: Array4<f32> = read_npy(format!("{}_input.npy", prefix)).unwrap();
        let reference: Array4<f32> = read_npy(format!("{}_reference.npy", prefix)).unwrap();
        let offset: Array1<i32> = read_npy(format!("{}_offset.npy", prefix)).unwrap();
        (x, reference, offset[0] as usize)
    }

    #[test]
    fn test_rope_qwen25() {
        let device = Device::Cpu;
        let base_path = "tests/rope_qwen25_data/";

        // 检查测试数据目录是否存在
        if !std::path::Path::new(base_path).exists() {
            println!("测试数据目录不存在: {}", base_path);
            println!("请先运行Python脚本生成测试数据");
            return;
        }

        for entry in fs::read_dir(base_path).unwrap() {
            let path = entry.unwrap().path();
            let path_str = path.to_string_lossy();
            if path_str.ends_with("_input.npy") {
                let prefix = path_str.replace("_input.npy", "");

                // 1️⃣ 读取 Python 生成的数据
                let (x, reference, offset) = load_case(&prefix);

                // 2️⃣ 转 ndarray -> Tensor
                let x_tensor = array_to_tensor(x.clone(), &device);
                let reference_tensor = array_to_tensor(reference.clone(), &device);

                // 3️⃣ 创建ROPE实例并应用
                let head_dim = x.shape()[3];
                let max_seq_len = 20; // 根据Python脚本中的设置
                let base = 1_000_000.0; // Qwen2.5的base值

                let rope = RotaryEmbedding::new(head_dim, max_seq_len, base, false);
                assert_eq!(rope.is_ok(), true);
                let x_rot = rope.unwrap().apply_rotary(&x_tensor, Some(offset));
                assert_eq!(x_rot.is_ok(), true);

                assert_allclose(
                    &reference_tensor,
                    &x_rot.unwrap(),
                    DType::F32,
                    None,
                    Some(5e-6),
                );
            }
        }
    }
}
