use candle_core::{DType, Device, Result, Tensor};

// Targeted repro for rows=17, out_dim=5760, in_dim=2880 on CUDA with MMQ
// Mirrors the shapes seen in end-to-end run just before the illegal access.
#[test]
fn test_mmq_rows17_out5760_in2880() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    if !dev.is_cuda() {
        eprintln!("CUDA not available; skipping test");
        return Ok(());
    }

    let rows = 16usize; // try 16 first (multiple of mmq_y) to smoke test
    let in_dim = 2880usize;
    let out_dim = 5760usize;
    let nblocks = in_dim / 32; // 90

    // Deterministic inputs
    // Activations [rows, in_dim] BF16
    let mut act_v = vec![0f32; rows * in_dim];
    // Simple deterministic pattern
    for i in 0..rows*in_dim {
        let v = ((i as u64 * 6364136223846793005u64 + 1u64) >> 33) as u32;
        act_v[i] = (v as f32 / 2147483648.0) * 2.0 - 1.0;
    }
    let act = Tensor::from_vec(act_v, (rows, in_dim), &dev)?.to_dtype(DType::BF16)?;

    // MXFP4 blocks [out_dim, nblocks, 16]
    let mut blocks = vec![0u8; out_dim * nblocks * 16];
    for i in 0..blocks.len() {
        let v = ((i as u64 * 2862933555777941757u64 + 3037000493u64) >> 7) as u32;
        blocks[i] = (v & 0xFF) as u8;
    }
    let blocks = Tensor::from_vec(blocks, (out_dim, nblocks, 16), &dev)?;

    // E8M0 scales [out_dim, nblocks]
    let mut scales = vec![0u8; out_dim * nblocks];
    for i in 0..scales.len() {
        let v = ((i as u64 * 1442695040888963407u64 + 12345u64) >> 11) as u32;
        scales[i] = (20 + (v % 200) as u8) as u8;
    }
    let scales = Tensor::from_vec(scales, (out_dim, nblocks), &dev)?;

    // Compute using MMQ
    std::env::set_var("CANDLE_MXFP4_USE_MMQ", "1");
    let out_mmq = candle_core::mxfp4::matmul_mxfp4_bf16(&act, &blocks, &scales)?;
    std::env::remove_var("CANDLE_MXFP4_USE_MMQ");

    // Compute using reference (non-MMQ)
    let out_ref = candle_core::mxfp4::matmul_mxfp4_bf16(&act, &blocks, &scales)?;

    let out_mmq_f32 = out_mmq.to_dtype(DType::F32)?;
    let out_ref_f32 = out_ref.to_dtype(DType::F32)?;
    let a = out_mmq_f32.to_vec2::<f32>()?;
    let b = out_ref_f32.to_vec2::<f32>()?;

    let mut max_diff = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut cnt = 0usize;
    for i in 0..rows { for j in 0..out_dim {
        let d = (a[i][j] - b[i][j]).abs();
        if d > max_diff { max_diff = d; }
        sum_sq += (d as f64) * (d as f64);
        cnt += 1;
    }}
    let l2 = (sum_sq / cnt as f64).sqrt();
    println!("rows=17 parity: max_diff={:.6}, l2={:.6}", max_diff, l2);
    assert!(max_diff <= 0.1, "max_diff={max_diff}");
    Ok(())
}
