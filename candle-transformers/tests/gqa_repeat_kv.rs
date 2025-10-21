use candle::{Device, Result, Tensor, D};
use candle_transformers::utils::repeat_kv;

// Validate that repeat_kv repeats each KV head contiguously across the head axis,
// i.e. with n_rep=3: [kv0,kv1] -> [kv0,kv0,kv0, kv1,kv1,kv1].
#[test]
fn t_gqa_repeat_kv_ordering() -> Result<()> {
    let dev = Device::Cpu;
    let b = 1usize;
    let n_kv = 2usize;
    let t = 3usize;
    let d = 1usize;

    // Build input (b, n_kv, t, d) with identifiable values per kv head and timestep.
    // Value encoding: v(h, t, d) = 100*h + 10*t + d
    let mut data = Vec::new();
    for _ in 0..b {
        for h in 0..n_kv {
            for tt in 0..t {
                for dd in 0..d {
                    data.push((100 * h + 10 * tt + dd) as f32);
                }
            }
        }
    }
    let xs = Tensor::from_vec(data, (b, n_kv, t, d), &dev)?;

    let n_rep = 3usize;
    let ys = repeat_kv(xs, n_rep)?; // (b, n_kv*n_rep, t, d)
    let got = ys.squeeze(D::Minus1)?.to_vec3::<f32>()?; // drop d=1 -> (b, n_kv*n_rep, t)

    // Build expected output with contiguous repeats per kv head.
    let mut exp = Vec::new(); // (b, n_kv*n_rep, t)
    for _ in 0..b {
        let mut heads = Vec::new();
        for hq in 0..(n_kv * n_rep) {
            let kv = hq / n_rep;
            let row: Vec<f32> = (0..t).map(|tt| (100 * kv + 10 * tt) as f32).collect();
            heads.push(row);
        }
        exp.push(heads);
    }

    assert_eq!(got[0].len(), n_kv * n_rep);
    assert_eq!(got, exp);
    Ok(())
}
