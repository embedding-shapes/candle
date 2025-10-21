use candle::{DType, Device, Result, Tensor};

// Validates RMSNorm formula on CUDA: v / sqrt(mean(v^2) + eps) * g
// Prints detailed diagnostics: dtype, device, shapes, strides, inputs/outputs, and error metrics.

const EPS_CASES: [f32; 2] = [1e-5, 1e-6];

fn cpu_reference(v: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let n = v.len() as f32;
    let sum2: f32 = v.iter().map(|&x| x * x).sum();
    let denom = (sum2 / n + eps).sqrt();
    v.iter()
        .zip(g.iter())
        .map(|(&x, &gg)| x / denom * gg)
        .collect()
}

fn l2_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn linf_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn rmsnorm_cuda_epsilon_location_micro() -> Result<()> {
    // Device: enforce CUDA 0, fail fast if unavailable.
    let dev = Device::new_cuda(0)?;

    // Input vector with varied magnitudes and signs (tiny to large).
    // Gamma includes different scales to ensure elementwise scaling is applied.
    let v_host: Vec<f32> = vec![
        1e-6, -1e-4, 0.0, 1e-2, -0.1, 0.5, -1.0, 2.0, -5.0, 10.0, -20.0, 50.0, -100.0,
    ];
    let g_host: Vec<f32> = vec![
        1.0, 0.5, -2.0, 1.5, -0.75, 0.25, 3.0, -1.0, 0.1, 2.0, -0.5, 4.0, -1.25,
    ];
    assert_eq!(v_host.len(), g_host.len());

    // Move to CUDA f32 tensors.
    let v = Tensor::from_vec(v_host.clone(), v_host.len(), &dev)?.to_dtype(DType::F32)?;
    let g = Tensor::from_vec(g_host.clone(), g_host.len(), &dev)?.to_dtype(DType::F32)?;

    // Diagnostics: tensor metadata.
    println!("seed: fixed/static inputs");
    println!("device: {:?}", dev);
    println!(
        "v: dtype={:?} shape={:?} stride={:?}",
        v.dtype(),
        v.shape(),
        v.stride()
    );
    println!(
        "g: dtype={:?} shape={:?} stride={:?}",
        g.dtype(),
        g.shape(),
        g.stride()
    );
    println!("v_host: {:?}", v_host);
    println!("g_host: {:?}", g_host);

    for &eps in &EPS_CASES {
        let y = candle_nn::ops::rms_norm(&v, &g, eps)?;
        let y_host: Vec<f32> = y.to_vec1()?;

        // CPU reference as per spec: v / sqrt(mean(v^2) + eps) * g
        let y_ref = cpu_reference(&v_host, &g_host, eps);

        let linf = linf_diff(&y_host, &y_ref);
        let l2 = l2_diff(&y_host, &y_ref);

        println!("eps={}: y_gpu={:?}", eps, y_host);
        println!("eps={}: y_ref={:?}", eps, y_ref);
        println!("eps={}: metrics: L_inf={:.12e}, L2={:.12e}", eps, linf, l2);

        // Tolerance in f32 for this short vector and deterministic reduction.
        assert!(linf <= 1e-6, "L_inf too high: {} (> 1e-6)", linf);
        assert!(l2 <= 1e-6, "L2 too high: {} (> 1e-6)", l2);
    }

    Ok(())
}
