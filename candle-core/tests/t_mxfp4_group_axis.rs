#![cfg(feature = "cuda")]

// Verify MXFP4 grouping axis for linear weights matches Candle's matmul contract:
// - Weights are [out, in] with K=in contiguous along the last dim.
// - MXFP4 packs per-row blocks along K (columns) with per-block E8M0 scales.
//
// Test strategy
// 1) Construct an expected BF16 matrix E of shape (M, K) where values depend on (row r, col c)
//    via block index along K: E[r,c] = 2^(r + floor(c/G)), using power-of-two scales and FP4 code for +1.0.
// 2) Build MXFP4 packed tensors using ROW grouping: blocks_row: (M, K/G, 16), scales_row: (M, K/G)
//    where every nibble encodes +1.0 and the scale exponent code is 127 + (r + b).
// 3) Dequant on CUDA to D_row and assert exact equality with E (after BF16→F32).
// 4) Build MXFP4 packed tensors using COLUMN grouping: blocks_col: (K, M/G, 16), scales_col: (K, M/G)
//    analogously with code 127 + (k + mb). Dequant to (K, M) then transpose to (M, K) as D_col.
// 5) Show D_col != E with nonzero max-abs/L2 error, proving grouping must be along K (columns) per row.

use anyhow::Result;
use candle_core::{mxfp4::dequant_mxfp4_to_bf16, DType, Device, Tensor};
use half::bf16;

const G: usize = 32; // MXFP4 group size per block

// Pack 32 FP4 E2M1 codes (all set to +1.0 => 0b0010) into 16 bytes, low nibble then high nibble per spec.
fn packed_block_all_ones() -> [u8; 16] {
    [0x22u8; 16]
}

fn l2_linf(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut l2 = 0f32;
    let mut linf = 0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        l2 += d * d;
        if d > linf {
            linf = d;
        }
    }
    (l2.sqrt(), linf)
}

#[test]
fn t_mxfp4_group_axis_linear_contract() -> Result<()> {
    // Dimensions chosen to allow both row and column grouping without tail blocks.
    let m = 64usize; // out_dim
    let k = 64usize; // in_dim, contiguous K
    let nb_k = k / G;
    let nb_m = m / G;
    assert_eq!(nb_k, 2);
    assert_eq!(nb_m, 2);

    // Expected E[r,c] = 2^(r + floor(c/G)) using FP4 code +1.0 scaled by E8M0 powers-of-two.
    // We will compare dequantized outputs against this expectation.
    let mut e_host = vec![0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            let b = c / G; // block index along K
            let exp = (r + b) as i32;
            e_host[r * k + c] = (2f32).powi(exp);
        }
    }

    // Build ROW-grouped MXFP4: blocks (m, nb_k, 16), scales (m, nb_k)
    let mut blocks_row = vec![0u8; m * nb_k * 16];
    let mut scales_row = vec![0u8; m * nb_k];
    let blk = packed_block_all_ones();
    for r in 0..m {
        for b in 0..nb_k {
            let off = (r * nb_k + b) * 16;
            blocks_row[off..off + 16].copy_from_slice(&blk);
            // scale code = 127 + (r + b)
            let code = (127 + (r + b) as i32) as u8;
            scales_row[r * nb_k + b] = code;
        }
    }

    // Build COLUMN-grouped MXFP4: blocks (k, nb_m, 16), scales (k, nb_m)
    let mut blocks_col = vec![0u8; k * nb_m * 16];
    let mut scales_col = vec![0u8; k * nb_m];
    for c in 0..k {
        // interpret as "row" in this packing
        for mb in 0..nb_m {
            let off = (c * nb_m + mb) * 16;
            blocks_col[off..off + 16].copy_from_slice(&blk);
            // scale code = 127 + (c + mb)
            let code = (127 + (c + mb) as i32) as u8;
            scales_col[c * nb_m + mb] = code;
        }
    }

    // GPU device
    let dev = Device::new_cuda(0)?;

    // Dequant ROW-grouped on CUDA -> (m, k)
    let blocks_row_t = Tensor::from_vec(blocks_row.clone(), (m, nb_k, 16), &dev)?;
    let scales_row_t = Tensor::from_vec(scales_row.clone(), (m, nb_k), &dev)?;
    let out_row_bf16 = dequant_mxfp4_to_bf16(&blocks_row_t, &scales_row_t, [m, k])?;
    assert_eq!(out_row_bf16.dtype(), DType::BF16);
    let out_row_f32 = out_row_bf16.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let out_row = out_row_f32.flatten_all()?.to_vec1::<f32>()?;

    // Dequant COLUMN-grouped on CUDA -> (k, m), then transpose to (m, k)
    let blocks_col_t = Tensor::from_vec(blocks_col.clone(), (k, nb_m, 16), &dev)?;
    let scales_col_t = Tensor::from_vec(scales_col.clone(), (k, nb_m), &dev)?;
    let out_col_bf16 = dequant_mxfp4_to_bf16(&blocks_col_t, &scales_col_t, [k, m])?;
    let out_col_f32 = out_col_bf16.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let out_col_t = Tensor::from_vec(out_col_f32.to_vec2::<f32>()?.concat(), (k, m), &Device::Cpu)?;
    let out_col_trans = out_col_t.t()?; // (m, k)
    let out_col = out_col_trans.flatten_all()?.to_vec1::<f32>()?;

    // Build E as BF16-rounded f32 to match dequant dtype
    let e_bf16: Vec<f32> = e_host.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();

    // Logs
    println!(
        "GPU ROW dequant dtype bf16->f32 shape [m={},k={}] stride K-contiguous",
        m, k
    );
    println!(
        "GPU COL dequant dtype bf16->f32 shape [k={},m={}] then transpose",
        k, m
    );
    println!(
        "blocks_row shape (m,nb_k,16)=({}, {}, 16); scales_row (m,nb_k)=({}, {})",
        m, nb_k, m, nb_k
    );
    println!(
        "blocks_col shape (k,nb_m,16)=({}, {}, 16); scales_col (k,nb_m)=({}, {})",
        k, nb_m, k, nb_m
    );
    // Show a few sample cells (r=0,1 ; c=0,31,32,63)
    let sample_idxs = [(0usize, 0usize), (0, 31), (0, 32), (0, 63), (1, 0), (1, 32)];
    for (r, c) in sample_idxs {
        let idx = r * k + c;
        println!(
            "E[{},{}]={:.6}  ROW={:.6}  COL={:.6}",
            r, c, e_bf16[idx], out_row[idx], out_col[idx]
        );
    }

    // Metrics and assertions
    let (l2_row, linf_row) = l2_linf(&e_bf16, &out_row);
    let (l2_col, linf_col) = l2_linf(&e_bf16, &out_col);
    println!("ROW vs E: L2={:.6} L∞={:.6}", l2_row, linf_row);
    println!("COL vs E: L2={:.6} L∞={:.6}", l2_col, linf_col);

    // Exact equality after BF16 rounding for row-grouped path; column-grouped must not match.
    let tol = 0.0f32;
    assert!(
        linf_row <= tol,
        "Row-grouping mismatch: L∞={} > {}",
        linf_row,
        tol
    );
    assert!(
        linf_col > 0.0,
        "Column-grouping unexpectedly matched E exactly"
    );

    Ok(())
}
