// use crate::tensor::device::Device;
// use crate::tensor::dtype::DType;
use candle_core::Tensor as CandleTensor;

pub trait Tensor: Sized {
    type Elem: Copy + 'static + std::fmt::Debug;

    fn shape(&self) -> Vec<usize>;
    fn matmul(&self, other: &Self) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn to_vec(&self) -> Vec<Self::Elem>;
    fn sqr(&self) -> Self;
    fn mean_keepdim(&self, axis: isize) -> Self;
    fn sqrt(&self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn clone(&self) -> Self;
    fn softmax(&self, axis: isize) -> Self;
    fn add_scalar(&self, v: Self::Elem) -> Self;
    fn broadcast_div(&self, other: &Self) -> Self;
    fn broadcast_mul(&self, other: &Self) -> Self;
    fn silu(&self) -> Self;
    fn dims(&self) -> &[usize];
    fn broadcast_as(&self, shape: &[usize]) -> Self;
    fn reshape(&self, shape: &[usize]) -> Self;
    fn transpose(&self, dim1: usize, dim2: usize) -> Self;
    fn permute(&self, axes: &[usize]) -> Self;
}

impl Tensor for CandleTensor {
    type Elem = f32;

    fn shape(&self) -> Vec<usize> {
        self.dims().iter().map(|&d| d as usize).collect()
    }

    fn matmul(&self, other: &Self) -> Self {
        self.matmul(other).unwrap()
    }

    fn add(&self, other: &Self) -> Self {
        self.add(other).unwrap()
    }

    fn to_vec(&self) -> Vec<Self::Elem> {
        self.to_vec1::<f32>().unwrap()
    }

    fn sqr(&self) -> Self {
        self.sqr().unwrap()
    }

    fn mean_keepdim(&self, axis: isize) -> Self {
        use candle_core::D;
        if axis == -1 {
            return self.mean_keepdim(D::Minus1).unwrap();
        }
        self.mean_keepdim(axis as usize).unwrap()
    }

    fn sqrt(&self) -> Self {
        self.sqrt().unwrap()
    }

    fn div(&self, other: &Self) -> Self {
        self.div(other).unwrap()
    }

    fn broadcast_div(&self, other: &Self) -> Self {
        self.broadcast_div(other).unwrap()
    }

    fn mul(&self, other: &Self) -> Self {
        self.mul(other).unwrap()
    }

    fn broadcast_mul(&self, other: &Self) -> Self {
        self.broadcast_mul(other).unwrap()
    }

    fn clone(&self) -> Self {
        <candle_core::Tensor as Clone>::clone(self)
    }

    fn add_scalar(&self, v: Self::Elem) -> Self {
        let scalar_tensor = candle_core::Tensor::full(v, self.dims(), self.device()).unwrap();
        self.add(&scalar_tensor).unwrap()
    }

    fn softmax(&self, axis: isize) -> Self {
        if axis < 0 {
            return candle_nn::ops::softmax_last_dim(self).unwrap();
        }
        candle_nn::ops::softmax(self, axis as usize).unwrap()
    }

    fn silu(&self) -> Self {
        candle_nn::ops::silu(self).unwrap()
    }

    fn dims(&self) -> &[usize] {
        self.dims()
    }

    fn broadcast_as(&self, shape: &[usize]) -> Self {
        self.broadcast_as(shape).unwrap()
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        self.reshape(shape).unwrap()
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        self.transpose(dim1, dim2).unwrap()
    }

    fn permute(&self, axes: &[usize]) -> Self {
        match axes.len() {
            2 => self.permute((axes[0], axes[1])).unwrap(),
            3 => self.permute((axes[0], axes[1], axes[2])).unwrap(),
            4 => self.permute((axes[0], axes[1], axes[2], axes[3])).unwrap(),
            _ => panic!("Unsupported axes dims for permute: {:?} (only 2/3/4 supported)", axes),
        }
    }

}

#[cfg(test)]
mod trait_tests {
    use super::*;
    use candle_core::Device;
    use candle_core::Tensor as CandleTensor;

    fn make_tensor(data: &[f32], shape: &[usize]) -> CandleTensor {
        let device = Device::Cpu;
        CandleTensor::from_vec(data.to_vec(), shape, &device).unwrap()
    }

    #[test]
    fn test_trait_transpose_candletensor() {
        // 2D
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = <CandleTensor as Tensor>::transpose(&x, 0, 1);
        assert_eq!(<CandleTensor as Tensor>::dims(&y), &[2, 2]);
        let flat = y.reshape(&[4]).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat, vec![1.0, 3.0, 2.0, 4.0]);
        // 3D
        let x = make_tensor(&(0..6).map(|v| v as f32).collect::<Vec<_>>(), &[1, 2, 3]);
        let y = <CandleTensor as Tensor>::transpose(&x, 0, 2);
        assert_eq!(<CandleTensor as Tensor>::dims(&y), &[3, 2, 1]);
        // 4D
        let x = make_tensor(&(0..24).map(|v| v as f32).collect::<Vec<_>>(), &[2, 3, 2, 2]);
        let y = <CandleTensor as Tensor>::transpose(&x, 1, 3);
        assert_eq!(<CandleTensor as Tensor>::dims(&y), &[2, 2, 2, 3]);
    }

    #[test]
    fn test_trait_permute_candletensor() {
        // 2D
        let x = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = <CandleTensor as Tensor>::permute(&x, &[1, 0]);
        assert_eq!(<CandleTensor as Tensor>::dims(&y), &[2, 2]);
        let flat = y.reshape(&[4]).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat, vec![1.0, 3.0, 2.0, 4.0]);
        // 3D
        let x = make_tensor(&(0..6).map(|v| v as f32).collect::<Vec<_>>(), &[1, 2, 3]);
        let y = <CandleTensor as Tensor>::permute(&x, &[2, 1, 0]);
        assert_eq!(<CandleTensor as Tensor>::dims(&y), &[3, 2, 1]);
        // 4D
        let x = make_tensor(&(0..24).map(|v| v as f32).collect::<Vec<_>>(), &[2, 3, 2, 2]);
        let y = <CandleTensor as Tensor>::permute(&x, &[3, 2, 1, 0]);
        assert_eq!(<CandleTensor as Tensor>::dims(&y), &[2, 2, 3, 2]);
    }
}

