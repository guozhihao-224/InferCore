use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use ndarray_npy::read_npy;

/// 用于测试两个Tensor是否近似
pub fn assert_allclose(
    a: &Tensor,
    b: &Tensor,
    precision: DType,
    rtol: Option<f32>,
    atol: Option<f32>,
) {
    let (rtol, atol) = match precision {
        DType::F32 => (rtol.unwrap_or(1e-5), atol.unwrap_or(1e-8)),
        DType::F16 => (rtol.unwrap_or(3e-2), atol.unwrap_or(1e-5)),
        _ => panic!("Unsupported precision"),
    };

    // 2. shape 检查
    assert_eq!(
        a.dims(),
        b.dims(),
        "shape mismatch: {:?} vs {:?}",
        a.dims(),
        b.dims()
    );

    // 3. flatten 成 Vec<f32>
    let a_flat = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b_flat = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // 4. allclose 判断
    let mut all_close = true;
    let mut first_bad = None;
    for (i, (x, y)) in a_flat.iter().zip(b_flat.iter()).enumerate() {
        let close = (x - y).abs() <= atol + rtol * y.abs();
        if !close {
            all_close = false;
            if first_bad.is_none() {
                first_bad = Some((i, *x, *y, (x - y).abs()));
            }
        }
    }
    if !all_close {
        println!("a: {:?}", &a_flat[..std::cmp::min(10, a_flat.len())]);
        println!("b: {:?}", &b_flat[..std::cmp::min(10, b_flat.len())]);
        println!("first_bad: {:?}", first_bad);
        panic!("result mismatch");
    }
}

pub fn load_tensor_from_npy(path: &str, device: &Device) -> anyhow::Result<Tensor> {
    let arr: ndarray::ArrayD<f32> = read_npy(path)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<f32> = arr.iter().cloned().collect();
    Ok(Tensor::from_vec(data, shape, device)?)
}
