use candle::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, Activation};

// Verifies the SwiGLU variant is SiLU(Wg x) ⊙ (Wu x) on CUDA.
// Computes both paths on GPU and checks parity at 1e-6 tolerance.
#[test]
fn t42_swiglu_cuda_parity() -> Result<()> {
    // Use CUDA:0 for this micro-check
    let dev = Device::new_cuda(0)?;

    // Shapes
    let in_dim = 4usize;
    let inter = 6usize; // out per branch
    let batch = 3usize; // tokens

    // Deterministic inputs and weights (F32)
    // x: (batch, in_dim)
    let x_vals: Vec<f32> = (0..(batch * in_dim))
        .map(|i| ((i as i32 % 7) as f32 - 3.0) / 10.0)
        .collect();
    let x = Tensor::from_vec(x_vals.clone(), (batch, in_dim), &dev)?.to_dtype(DType::F32)?;

    // Wg, Wu: (inter, in_dim)
    let wg_vals: Vec<f32> = (0..(inter * in_dim))
        .map(|i| ((i as i32 % 11) as f32 - 5.0) / (in_dim as f32))
        .collect();
    let wu_vals: Vec<f32> = (0..(inter * in_dim))
        .map(|i| ((i as i32 % 13) as f32 - 6.0) / (in_dim as f32))
        .collect();
    let wg = Tensor::from_vec(wg_vals.clone(), (inter, in_dim), &dev)?.to_dtype(DType::F32)?;
    let wu = Tensor::from_vec(wu_vals.clone(), (inter, in_dim), &dev)?.to_dtype(DType::F32)?;

    // Build a single Linear with concatenated weights [Wg; Wu] along out dim -> (2*inter, in_dim)
    let wcat = Tensor::cat(&[wg.clone(), wu.clone()], 0)?; // (2*inter, in_dim)
    let mut map = std::collections::HashMap::new();
    map.insert("weight".to_string(), wcat.clone());
    let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &dev);
    let lin = linear_no_bias(in_dim, 2 * inter, vb)?;

    // Path A (library): y_lib = Swish(GateUp(x)) where Swish==SiLU applied to first half, times second half.
    let y_cat = x.apply(&lin)?; // (batch, 2*inter)
    let y_lib = y_cat.apply(&Activation::Swiglu)?; // (batch, inter)

    // Path B (manual): y_ref = SiLU(Wg x) * (Wu x)
    let gate = x.matmul(&wg.t()?)?.silu()?; // (batch, inter)
    let up = x.matmul(&wu.t()?)?; // (batch, inter)
    let y_ref = (gate * up)?; // (batch, inter)

    // Compare
    let diff = (&y_lib - &y_ref)?.abs()?;
    let flat = diff.reshape((diff.elem_count(),))?;
    let l_inf = flat.max(candle::D::Minus1)?.to_scalar::<f32>()?;
    let l2 = diff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

    // Log evidence
    eprintln!(
        "device={:?} dtype={:?} x.shape={:?} w.shape={:?}",
        dev, y_lib.dtype(), x.shape(), wcat.shape()
    );
    eprintln!("x (first 8) = {:?}", &x_vals[..8.min(x_vals.len())]);
    let y_lib_host = y_lib.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let y_ref_host = y_ref.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    eprintln!("y_lib first8 = {:?}", &y_lib_host[0][..8.min(inter)]);
    eprintln!("y_ref first8 = {:?}", &y_ref_host[0][..8.min(inter)]);
    eprintln!("L_inf = {:.9}, L2 = {:.9}", l_inf, l2);

    assert!(l_inf <= 1e-6, "SwiGLU variant mismatch: L_inf={l_inf} > 1e-6");
    Ok(())
}
