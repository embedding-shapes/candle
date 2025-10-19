use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

fn fa_acausal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

fn fa_acausal_softcap(q: &Tensor, k: &Tensor, v: &Tensor, softcap: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    // let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = q.matmul(&k.t()?)?;
    let att = (softcap as f64 * ((att / softcap as f64)?.tanh())?)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[test]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let ys1 = fa_acausal(&q, &k, &v, 0.5)?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        candle_flash_attn::flash_attn(&q, &k, &v, 0.5, false)?.transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys1.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys1, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );

    assert_eq!(ys2.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys2, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-5);
    Ok(())
}

#[test]
fn flash_attn_acausal_softcap() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 5 * 8, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 5, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;
    let softcap = 5.0f32;

    let ys1 = fa_acausal_softcap(&q, &k, &v, softcap.clone())?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        candle_flash_attn::flash_attn_alibi_windowed_softcap(
            &q,
            &k,
            &v,
            None,            //  alibi_slopes //
            1.0,             // softmax //
            None,            // window_size_left //
            None,            // window_size_right //
            softcap.clone(), // softcap //
        )?
        .transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys1.dims(), &[3, 5, 8]);
    assert_eq!(ys2.dims(), &[3, 5, 8]);
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-3);
    Ok(())
}

#[test]
fn flash_attn_varlen() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let seqlens_q = Tensor::new(&[0u32, 2u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 2u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        candle_flash_attn::flash_attn_varlen(
            &q, &k, &v, &seqlens_q, &seqlens_k, 32, 32, 0.5, false,
        )?
        .transpose(0, 1)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );
    Ok(())
}

// T9 LSE shape sanity: For toy shapes, assert LSE shape matches input
// and causal/windowed settings for padded and varlen paths.
#[test]
fn flash_attn_lse_shape_sanity() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Padded path
    let q = Tensor::arange(0u32, 3 * 2 * 8, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 8))?; // (b=1, seqlen_q=3, heads=2, dim=8)
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let (o, lse) = candle_flash_attn::flash_attn_with_lse(&q, &k, &v, 0.5, false)?;
    assert_eq!(o.dims(), &[1, 3, 2, 8]);
    assert_eq!(lse.dims(), &[1, 2, 3]); // (b, h, seqlen_q)

    // Windowed (same shape for lse)
    let (_o2, lse2) = candle_flash_attn::flash_attn_windowed_with_lse(&q, &k, &v, 0.5, Some(1), Some(1))?;
    assert_eq!(lse2.dims(), &[1, 2, 3]);

    // Varlen path: LSE shape (heads, total_q)
    let qv = Tensor::arange(0u32, 3 * 2 * 8, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 8))?; // (seqlen_q_total=3, heads=2, dim=8)
    let kv = (&qv / 40.)?;
    let vv = (&qv / 50.)?;
    let qv = (&qv / 30.)?;
    let seqlens_q = Tensor::new(&[0u32, 3u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 3u32], &device)?;
    let (_ov, lsev) = candle_flash_attn::flash_attn_varlen_with_lse(
        &qv,
        &kv,
        &vv,
        &seqlens_q,
        &seqlens_k,
        3,
        3,
        0.5,
        false,
    )?;
    assert_eq!(lsev.dims(), &[2, 3]); // (heads, total_q)
    Ok(())
}

// T10 LSE vs eager: Build tiny q/k and compute logits on CPU to get
// lse = logsumexp(qk + mask), compare to LSE returned by FA (same softmax_scale).
#[test]
fn flash_attn_lse_vs_eager() -> Result<()> {
    let device = Device::new_cuda(0)?;
    // Small toy example: b=1, h=2, q=3, k=4, d=8
    let b = 1usize;
    let h = 2usize;
    let qlen = 3usize;
    let klen = 4usize;
    let d = 8usize;
    let scale = 0.5f32;

    // Build q/k/v in FA layout: (b, q, h, d)
    let q_bqhd = Tensor::arange(0u32, (b * h * qlen * d) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((b, qlen, h, d))?;
    let k_bkhd = Tensor::arange(0u32, (b * h * klen * d) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((b, klen, h, d))?;
    let v_bkhd = (&k_bkhd / 50.)?;
    let q = (&q_bqhd / 30.)?;
    let k = k_bkhd.clone();
    let v = v_bkhd.clone();

    let (_o, lse) = candle_flash_attn::flash_attn_with_lse(&q, &k, &v, scale, false)?;
    // Compute eager LSE on CPU: for each (b,h) compute logsumexp over K.
    let q_cpu = q.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let k_cpu = k.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

    // Reshape to (b*h, q, d) and (b*h, k, d)
    let q_bh_q_d = q_cpu.transpose(1, 2)?.reshape((b * h, qlen, d))?;
    let k_bh_k_d = k_cpu.transpose(1, 2)?.reshape((b * h, klen, d))?;
    // For each bh row: logits = scale * (Q @ K^T), lse = logsumexp over k axis
    let mut lse_rows: Vec<f32> = Vec::with_capacity(b * h * qlen);
    for bh in 0..(b * h) {
        let q_i = q_bh_q_d.i(bh)?.clone(); // (q, d)
        let k_i = k_bh_k_d.i(bh)?.clone(); // (k, d)
        let logits = (q_i.matmul(&k_i.t()?)? * scale as f64)?; // (q, k)
        let lse_i = logits.log_sum_exp(D::Minus1)?; // (q)
        let v: Vec<f32> = lse_i.to_vec1()?;
        lse_rows.extend(v);
    }
    let lse_eager = Tensor::from_vec(lse_rows, (b, h, qlen), &Device::Cpu)?;
    let lse_gpu = lse.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

    // Compare within a small tolerance
    let diff = (lse_eager - lse_gpu)?.abs()?.flatten_all()?.max(0)?;
    assert!(diff.to_vec0::<f32>()? < 1e-3);
    Ok(())
}
