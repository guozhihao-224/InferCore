use std::sync::Arc;
use tch::{Tensor as TchTensor, Kind, Device as TchDevice};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;

/// 高性能Tensor实现，使用PyTorch后端
#[derive(Debug, Clone)]
pub struct HighPerfTensor {
    pub data: TchTensor,
    pub dtype: DType,
    pub device: Device,
    pub shape: Vec<i64>,
}

impl HighPerfTensor {
    pub fn zeros(shape: &[usize], device: Device) -> Self {
        let tch_device = match device {
            Device::Cpu => TchDevice::Cpu,
            Device::Cuda(_) => TchDevice::Cuda(0), // 简化处理
        };
        
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let data = TchTensor::zeros(&shape_i64, (Kind::Float, tch_device));
        
        Self {
            data,
            dtype: DType::F32,
            device,
            shape: shape_i64,
        }
    }

    pub fn ones(shape: &[usize], device: Device) -> Self {
        let tch_device = match device {
            Device::Cpu => TchDevice::Cpu,
            Device::Cuda(_) => TchDevice::Cuda(0),
        };
        
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let data = TchTensor::ones(&shape_i64, (Kind::Float, tch_device));
        
        Self {
            data,
            dtype: DType::F32,
            device,
            shape: shape_i64,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize], device: Device) -> Self {
        let tch_device = match device {
            Device::Cpu => TchDevice::Cpu,
            Device::Cuda(_) => TchDevice::Cuda(0),
        };
        
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let data_tensor = TchTensor::of_slice(&data).to_device(tch_device);
        let data_tensor = data_tensor.reshape(&shape_i64);
        
        Self {
            data: data_tensor,
            dtype: DType::F32,
            device,
            shape: shape_i64,
        }
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.flatten(0, -1).into()
    }

    pub fn add(&self, other: &HighPerfTensor) -> HighPerfTensor {
        let result = &self.data + &other.data;
        Self {
            data: result,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }

    pub fn mul(&self, other: &HighPerfTensor) -> HighPerfTensor {
        let result = &self.data * &other.data;
        Self {
            data: result,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }

    /// 高性能矩阵乘法，使用PyTorch的优化实现
    pub fn matmul(&self, other: &HighPerfTensor) -> HighPerfTensor {
        let result = self.data.matmul(&other.data);
        let result_shape = result.size();
        
        Self {
            data: result,
            dtype: self.dtype,
            device: self.device,
            shape: result_shape,
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> HighPerfTensor {
        let new_shape_i64: Vec<i64> = new_shape.iter().map(|&x| x as i64).collect();
        let reshaped = self.data.reshape(&new_shape_i64);
        
        Self {
            data: reshaped,
            dtype: self.dtype,
            device: self.device,
            shape: new_shape_i64,
        }
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> HighPerfTensor {
        let transposed = self.data.transpose(dim0, dim1);
        let new_shape = transposed.size();
        
        Self {
            data: transposed,
            dtype: self.dtype,
            device: self.device,
            shape: new_shape,
        }
    }

    pub fn div_scalar(&self, scalar: f32) -> HighPerfTensor {
        let result = &self.data / scalar;
        Self {
            data: result,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }

    /// 高性能softmax实现
    pub fn softmax(&self, dim: i64) -> HighPerfTensor {
        let result = self.data.softmax(dim, Kind::Float);
        
        Self {
            data: result,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }

    /// 高性能SiLU激活函数
    pub fn silu(&self) -> HighPerfTensor {
        let result = self.data.silu();
        
        Self {
            data: result,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }

    /// 转换为标准Tensor格式
    pub fn to_standard_tensor(&self) -> crate::tensor::tensor::Tensor {
        let data_vec = self.to_vec();
        let shape_usize: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        crate::tensor::tensor::Tensor::from_vec(data_vec, &shape_usize, self.device)
    }

    /// 从标准Tensor创建
    pub fn from_standard_tensor(tensor: &crate::tensor::tensor::Tensor) -> Self {
        let data_vec = tensor.to_vec();
        Self::from_vec(data_vec, tensor.shape(), tensor.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::device::Device;

    #[test]
    fn test_high_perf_tensor_basic_ops() {
        let device = Device::Cpu;
        
        // 测试基本创建
        let t1 = HighPerfTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device);
        let t2 = HighPerfTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], device);
        
        // 测试加法
        let sum = t1.add(&t2);
        let sum_vec = sum.to_vec();
        assert_eq!(sum_vec, vec![6.0, 8.0, 10.0, 12.0]);
        
        // 测试乘法
        let prod = t1.mul(&t2);
        let prod_vec = prod.to_vec();
        assert_eq!(prod_vec, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_high_perf_matmul() {
        let device = Device::Cpu;
        
        let a = HighPerfTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device);
        let b = HighPerfTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], device);
        
        let c = a.matmul(&b);
        let c_vec = c.to_vec();
        
        // 期望结果: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c_vec, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_high_perf_softmax() {
        let device = Device::Cpu;
        
        let t = HighPerfTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], device);
        let softmax_result = t.softmax(-1);
        let result_vec = softmax_result.to_vec();
        
        // 验证softmax结果
        let sum: f32 = result_vec.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result_vec.iter().all(|&x| x > 0.0));
    }
} 