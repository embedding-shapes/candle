use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize)]
struct HiddenFixture {
    layers: usize,
    hidden_size: usize,
    data: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct AttnFixture {
    output_shape: Vec<usize>,
    weights_shape: Vec<usize>,
    output_sample: Vec<f32>,
    weights_sample: Vec<Vec<f32>>, // [num_heads_sampled, seq_len]
}

#[derive(Deserialize)]
struct YarnFixture {
    head_dim: usize,
    pos: usize,
    q_in: Vec<f32>,
    k_in: Vec<f32>,
    q_rot: Vec<f32>,
    k_rot: Vec<f32>,
}

#[derive(Deserialize)]
struct Mxfp4SliceFixture {
    raw_bytes: Vec<u8>, // 16 bytes
    scale_u8: u8,
    decoded32: Vec<f32>,
}

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures")
}

#[test]
fn t50_hidden_states_steps_sanity() -> Result<()> {
    let dir = fixtures_dir();
    let f1 = std::fs::File::open(dir.join("hidden_states_step1.json"))?;
    let f2 = std::fs::File::open(dir.join("hidden_states_step2.json"))?;
    let s1: HiddenFixture = serde_json::from_reader(f1)?;
    let s2: HiddenFixture = serde_json::from_reader(f2)?;
    assert!(s1.layers > 0 && s1.hidden_size > 0);
    assert_eq!(s1.data.len(), s1.layers);
    assert!(s2.layers > 0 && s2.hidden_size > 0);
    assert_eq!(s2.data.len(), s2.layers);
    // Length consistency
    for row in &s1.data {
        assert_eq!(row.len(), s1.hidden_size);
    }
    for row in &s2.data {
        assert_eq!(row.len(), s2.hidden_size);
    }
    // Ensure there is at least one differing value between steps
    let mut diff = 0.0f32;
    for (a, b) in s1.data[0].iter().zip(&s2.data[0]) {
        diff += (a - b).abs();
    }
    assert!(
        diff > 1e-6,
        "hidden states identical across steps; expected change"
    );
    Ok(())
}

#[test]
fn t51_attn_weights_are_probabilities() -> Result<()> {
    let dir = fixtures_dir();
    for name in ["attn_case1.json", "attn_case2.json"] {
        let f = std::fs::File::open(dir.join(name))?;
        let a: AttnFixture = serde_json::from_reader(f)?;
        assert_eq!(a.output_shape.len(), 3);
        assert_eq!(a.weights_shape.len(), 4);
        // Each sampled row should sum to ~1
        for row in &a.weights_sample {
            let s: f32 = row.iter().copied().sum();
            assert!(
                (s - 1.0).abs() < 1e-4,
                "{}: weights row not normalized: {}",
                name,
                s
            );
        }
    }
    Ok(())
}

#[test]
fn t52_yarn_rotary_basic_sanity() -> Result<()> {
    let dir = fixtures_dir();
    for name in ["yarn_qk_pos0.json", "yarn_qk_pos17.json"] {
        let f = std::fs::File::open(dir.join(name))?;
        let y: YarnFixture = serde_json::from_reader(f)?;
        assert_eq!(y.q_in.len(), y.head_dim);
        assert_eq!(y.k_in.len(), y.head_dim);
        assert_eq!(y.q_rot.len(), y.head_dim);
        assert_eq!(y.k_rot.len(), y.head_dim);
        // No NaNs/Infs
        let finite = |v: &Vec<f32>| v.iter().all(|x| x.is_finite());
        assert!(finite(&y.q_in) && finite(&y.k_in) && finite(&y.q_rot) && finite(&y.k_rot));
        // Rotation changed at least one element per tensor
        let any_diff_q = y
            .q_in
            .iter()
            .zip(y.q_rot.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        let any_diff_k = y
            .k_in
            .iter()
            .zip(y.k_rot.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff_q, "{}: rotation had no effect on q", name);
        assert!(any_diff_k, "{}: rotation had no effect on k", name);
    }
    Ok(())
}

fn decode_fp4_e2m1(nibble: u8) -> f32 {
    let s = (nibble >> 3) & 0x1;
    let e = (nibble >> 1) & 0x3;
    let m = nibble & 0x1;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        let frac = (m as f32) * 0.5;
        let exp = 0.0; // 2^(1-bias) with bias=1 -> 2^0
        sign * (2.0f32).powf(exp) * frac
    } else {
        let frac = 1.0 + (m as f32) * 0.5;
        let exp = (e as i32 - 1) as f32;
        sign * (2.0f32).powf(exp) * frac
    }
}

fn pow2_e8m0(u: u8) -> f32 {
    if u == 0xFF {
        f32::NAN
    } else {
        (2.0f32).powf(u as f32 - 127.0)
    }
}

#[test]
fn t53_mxfp4_slice_roundtrip_decode() -> Result<()> {
    let dir = fixtures_dir();
    let f = std::fs::File::open(dir.join("mxfp4_row0_block0_slice.json"))?;
    let s: Mxfp4SliceFixture = serde_json::from_reader(f)?;
    assert_eq!(s.raw_bytes.len(), 16);
    let scale = pow2_e8m0(s.scale_u8);
    let mut got = vec![0f32; 32];
    for j in 0..16usize {
        let byte = s.raw_bytes[j];
        let lo = byte & 0x0F;
        let hi = (byte >> 4) & 0x0F;
        let c0 = 2 * j;
        let c1 = c0 + 1;
        got[c0] = decode_fp4_e2m1(lo) * scale;
        got[c1] = decode_fp4_e2m1(hi) * scale;
    }
    assert_eq!(s.decoded32.len(), 32);
    for (a, b) in got.iter().zip(s.decoded32.iter()) {
        assert!((a - b).abs() < 1e-6, "decoded mismatch: {} vs {}", a, b);
    }
    Ok(())
}
