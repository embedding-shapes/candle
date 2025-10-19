#[test]
fn sinks_scale_monotonic_and_limits() {
    // scale = 1 / (1 + exp(sink - lse))
    let lse = 2.0f32;
    let mut prev = 1.0f32;
    for &sink in &[-10.0f32, -5.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0] {
        let scale = 1.0f32 / (1.0 + (sink - lse).exp());
        // As sink increases, scale decreases monotonically.
        assert!(scale <= prev + 1e-7, "monotonicity broken: sink={}, scale={}, prev={}", sink, scale, prev);
        prev = scale;
        // Bounds: (0,1)
        assert!(scale > 0.0 && scale < 1.0);
    }
    // Identity-ish in extremes
    let near_one = 1.0f32 / (1.0 + (-1000.0 - lse).exp());
    let near_zero = 1.0f32 / (1.0 + (1000.0 - lse).exp());
    assert!(near_one > 0.9999);
    assert!(near_zero < 1e-4);
}

