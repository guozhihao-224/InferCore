use crate::tensor::{device::Device, dtype::DType};
use ndarray::{ArrayD, Axis, Ix2, IxDyn};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>, // TODO 后续进行泛化
    pub dtype: DType,
    pub device: Device,
    pub shape: Vec<usize>, // 冗余存储，便于快速访问
                           // 可扩展 name、requires_grad 等字段
}

impl Tensor {
    pub fn zeros(shape: &[usize], device: Device) -> Self {
        let data = ArrayD::<f32>::zeros(IxDyn(shape));
        Self {
            data,
            dtype: DType::F32,
            device,
            shape: shape.to_vec(),
        }
    }

    pub fn ones(shape: &[usize], device: Device) -> Self {
        let data = ArrayD::<f32>::ones(IxDyn(shape));
        Self {
            data,
            dtype: DType::F32,
            device,
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize], device: Device) -> Self {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();
        Self {
            data: arr,
            dtype: DType::F32,
            device,
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().cloned().collect()
    }

    pub fn astype(&self, dtype: DType) -> Tensor {
        match dtype {
            DType::F32 => self.clone(),
            DType::I32 => {
                let v: Vec<i32> = self.data.iter().map(|x| *x as i32).collect();
                Tensor {
                    data: ArrayD::from_shape_vec(
                        IxDyn(&self.shape),
                        v.iter().map(|&x| x as f32).collect(),
                    )
                    .unwrap(),
                    dtype: DType::I32,
                    device: self.device,
                    shape: self.shape.clone(),
                }
            }
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: &self.data + &other.data,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Case 1: [batch, seq, hidden] @ [hidden, out_hidden] => [batch, seq, out_hidden]
        if self.shape.len() == 3 && other.shape.len() == 2 {
            let batch = self.shape[0];
            let seq = self.shape[1];
            let hidden = self.shape[2];
            let out_hidden = other.shape[1];
            assert_eq!(other.shape[0], hidden, "matmul: shape mismatch");

            // reshape self to [batch*seq, hidden] and convert to Array2
            let a = self
                .data
                .clone()
                .into_shape((batch * seq, hidden))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();
            let b = other.data.clone().into_dimensionality::<Ix2>().unwrap();
            let c = a.dot(&b); // [batch*seq, out_hidden]
            let c = c.into_shape((batch, seq, out_hidden)).unwrap();

            return Tensor {
                data: c.into_dyn(),
                dtype: self.dtype,
                device: self.device,
                shape: vec![batch, seq, out_hidden],
            };
        }

        // Case 2: [batch, hidden] @ [hidden, out_hidden] => [batch, out_hidden]
        if self.shape.len() == 2 && other.shape.len() == 2 {
            let batch = self.shape[0];
            let hidden = self.shape[1];
            let out_hidden = other.shape[1];
            assert_eq!(other.shape[0], hidden, "matmul: shape mismatch");

            let a = self.data.clone().into_dimensionality::<Ix2>().unwrap();
            let b = other.data.clone().into_dimensionality::<Ix2>().unwrap();
            let c = a.dot(&b); // [batch, out_hidden]

            return Tensor {
                data: c.into_dyn(),
                dtype: self.dtype,
                device: self.device,
                shape: vec![batch, out_hidden],
            };
        }

        // Case 3: [hidden] @ [hidden, out_hidden] => [out_hidden]
        if self.shape.len() == 1 && other.shape.len() == 2 {
            let hidden = self.shape[0];
            let out_hidden = other.shape[1];
            assert_eq!(other.shape[0], hidden, "matmul: shape mismatch");

            let a = self
                .data
                .clone()
                .into_shape((1, hidden))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();
            let b = other.data.clone().into_dimensionality::<Ix2>().unwrap();
            let c = a.dot(&b); // [1, out_hidden]
            let c = c.into_shape((out_hidden,)).unwrap();

            return Tensor {
                data: c.into_dyn(),
                dtype: self.dtype,
                device: self.device,
                shape: vec![out_hidden],
            };
        }

        // 其他情况暂未实现
        panic!(
            "matmul: unsupported shapes {:?} @ {:?}",
            self.shape, other.shape
        );
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        Tensor {
            data: self.data.clone().into_shape(IxDyn(new_shape)).unwrap(),
            dtype: self.dtype,
            device: self.device,
            shape: new_shape.to_vec(),
        }
    }

    pub fn transpose(&self, axes: &[usize]) -> Tensor {
        assert_eq!(
            axes.len(),
            self.shape.len(),
            "Axes length must match tensor rank"
        );
        let permuted = self.data.clone().permuted_axes(axes);
        let new_shape = axes.iter().map(|&i| self.shape[i]).collect();
        Tensor {
            data: permuted,
            dtype: self.dtype,
            device: self.device,
            shape: new_shape,
        }
    }

    /// 将张量除以一个标量
    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data / scalar,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }

    /// Softmax 归一化公式：
    ///
    ///     y_i = exp(x_i) / sum_j exp(x_j)
    ///
    /// 其中：
    /// - x: 输入向量或张量在指定 axis 上的切片
    /// - y: softmax 输出
    /// - sum_j: 对 axis 维度上的所有元素求和
    ///
    /// 常用于将一组实数归一化为概率分布（如分类、注意力权重）。
    pub fn softmax(&self, axis: isize) -> Tensor {
        let axis = if axis < 0 {
            (self.shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };
        let data = self.data.clone();

        // Compute max along axis, keep dims for broadcasting
        // 如果 x_i 很大，那么 e^(x_i) 会迅速溢出（变成无穷大，导致NaN）
        // 为防止这种情况，常用trick是先减去最大值
        let max = data
            .map_axis(Axis(axis), |row| {
                row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            })
            .insert_axis(Axis(axis));
        
        // 3. 每个元素减去最大值后取指数, max会自动广播
        let exp = (&data - &max).mapv(f32::exp);

        // 4. 计算每个“行”的指数和，保持维度用于广播
        let sum = exp.sum_axis(Axis(axis)).insert_axis(Axis(axis));

        // 5. 每个元素除以该行的指数和， sum会自动广播
        let softmax = &exp / &sum;

        Tensor {
            data: softmax,
            dtype: self.dtype,
            device: self.device,
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::device::Device;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3], Device::Cpu);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&[2, 2], Device::Cpu);
        assert_eq!(t.shape(), &[2, 2]);
        assert!(t.to_vec().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_vec_and_to_vec() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec(v.clone(), &[2, 2], Device::Cpu);
        assert_eq!(t.to_vec(), v);
    }

    #[test]
    fn test_add() {
        let a = Tensor::ones(&[2, 2], Device::Cpu);
        let b = Tensor::ones(&[2, 2], Device::Cpu);
        let c = a.add(&b);
        assert!(c.to_vec().iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::Cpu);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], Device::Cpu);
        let c = a.matmul(&b);
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::ones(&[2, 3], Device::Cpu);
        let t2 = t.reshape(&[3, 2]);
        assert_eq!(t2.shape(), &[3, 2]);
        assert!(t2.to_vec().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_astype() {
        let t = Tensor::from_vec(vec![1.2, 2.7, 3.5], &[3], Device::Cpu);
        let t2 = t.astype(DType::I32);
        assert_eq!(t2.dtype, DType::I32);
        assert_eq!(t2.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reshape_and_transpose() {
        let t = Tensor::from_vec((0..6).map(|v| v as f32).collect(), &[2, 3], Device::Cpu);
        let t2 = t.reshape(&[3, 2]);
        assert_eq!(t2.shape(), &[3, 2]);
        let t3 = t2.transpose(&[1, 0]);
        assert_eq!(t3.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_scalar() {
        let t = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2], Device::Cpu);
        let t2 = t.div_scalar(2.0);
        assert_eq!(t2.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_softmax_last_axis() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::Cpu);
        let t2 = t.softmax(-1);
        // 计算 softmax
        let s1 = (1.0f32).exp() + (2.0f32).exp();
        let s2 = (3.0f32).exp() + (4.0f32).exp();
        let expected = vec![
            (1.0f32).exp() / s1,
            (2.0f32).exp() / s1,
            (3.0f32).exp() / s2,
            (4.0f32).exp() / s2,
        ];
        for (a, b) in t2.to_vec().iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "softmax output: {}, expected: {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_softmax_axis0() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::Cpu);
        let t2 = t.softmax(0);
        // 计算 softmax
        let s1 = (1.0f32).exp() + (3.0f32).exp();
        let s2 = (2.0f32).exp() + (4.0f32).exp();
        let expected = vec![
            (1.0f32).exp() / s1,
            (2.0f32).exp() / s2,
            (3.0f32).exp() / s1,
            (4.0f32).exp() / s2,
        ];
        for (a, b) in t2.to_vec().iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "softmax output: {}, expected: {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_softmax_2d_axis1() {
        // 2D tensor, softmax along axis 1
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::Cpu);
        let t2 = t.softmax(1);
        // Row 1: [1,2,3], Row 2: [4,5,6]
        let s1 = (1.0f32).exp() + (2.0f32).exp() + (3.0f32).exp();
        let s2 = (4.0f32).exp() + (5.0f32).exp() + (6.0f32).exp();
        let expected = vec![
            (1.0f32).exp() / s1,
            (2.0f32).exp() / s1,
            (3.0f32).exp() / s1,
            (4.0f32).exp() / s2,
            (5.0f32).exp() / s2,
            (6.0f32).exp() / s2,
        ];
        for (a, b) in t2.to_vec().iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "softmax output: {}, expected: {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_softmax_3d_axis2() {
        // 3D tensor, softmax along axis 2
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], Device::Cpu);
        let t2 = t.softmax(2);
        // For each [i, j, :] do softmax
        let s1 = (1.0f32).exp() + (2.0f32).exp() + (3.0f32).exp();
        let s2 = (4.0f32).exp() + (5.0f32).exp() + (6.0f32).exp();
        let expected = vec![
            (1.0f32).exp() / s1,
            (2.0f32).exp() / s1,
            (3.0f32).exp() / s1,
            (4.0f32).exp() / s2,
            (5.0f32).exp() / s2,
            (6.0f32).exp() / s2,
        ];
        for (a, b) in t2.to_vec().iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "softmax output: {}, expected: {}",
                a,
                b
            );
        }
    }
}
