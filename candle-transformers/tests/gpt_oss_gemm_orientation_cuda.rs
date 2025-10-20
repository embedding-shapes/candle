use candle::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

const O: usize = 2;
const I: usize = 3;

#[cfg(feature = "cuda")]
#[test]
fn t_linear_gemm_orientation_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;

    // Deterministic tiny matrix W [O,I] and vector x [I]
    // W = [[1,2,3], [4,5,6]], x = [7,8,9]
    // y_ref = W · x = [50, 122]
    let w_host: [f32; O * I] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_host: [f32; I] = [7.0, 8.0, 9.0];
    let y_ref_host: [f32; O] = [50.0, 122.0];

    let w = Tensor::from_slice(&w_host, (O, I), &dev)?;
    // Use a batch dimension to avoid 1D x 2D matmul shape mismatch.
    let x = Tensor::from_slice(&x_host, (1, I), &dev)?;

    // Log dtype, device, shapes, strides, and raw inputs
    println!(
        "W: dtype={:?} device={:?} shape={:?} stride={:?} data={:?}",
        w.dtype(),
        w.device(),
        w.dims(),
        w.stride(),
        &w_host
    );
    println!(
        "x: dtype={:?} device={:?} shape={:?} stride={:?} data={:?}",
        x.dtype(),
        x.device(),
        x.dims(),
        x.stride(),
        &x_host
    );

    // Current Linear convention: y = x @ W^T (so W stored [out,in])
    let lin = Linear::new(w.clone(), None);
    let y = lin.forward(&x)?; // shape [1,O]

    // Collect GPU result to host for comparison
    let y_vec2 = y.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let y_vec = &y_vec2[0];

    // Print outputs and numeric metrics
    let mut max_abs = 0.0f32;
    let mut l2 = 0.0f32;
    for i in 0..O {
        let diff = (y_vec[i] - y_ref_host[i]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        l2 += diff * diff;
    }
    l2 = l2.sqrt();
    println!(
        "y_ref={:?} y_gpu={:?} L_inf={:.8e} L2={:.8e}",
        &y_ref_host, &y_vec, max_abs, l2
    );

    // Bit-exact on small integers with f32 on GPU can vary; use tight tol.
    assert!(max_abs <= 1e-6, "orientation mismatch: max_abs={max_abs}");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[test]
fn t_linear_gemm_orientation_cuda_skipped() {
    // Ensure test suite is explicit about CUDA requirement
    eprintln!("skipped: build without 'cuda' feature");
}
