use std::path::Path;

use anyhow::{bail, Result};
use candle::{Device, Tensor};

// Constants should be passed in from caller; keep this module generic and testable.

/// Compute resized dimensions for Pixtral-style preprocessing.
/// - Preserve aspect ratio.
/// - Set the longest edge to `max_side`.
/// - Round both dimensions to the nearest multiple of `divisor`.
pub fn compute_resized_dims_from_wh(
    orig_w: u32,
    orig_h: u32,
    max_side: usize,
    divisor: usize,
) -> (usize, usize) {
    let divisor = divisor.max(1);
    let max_side = max_side.max(divisor);

    let (orig_wf, orig_hf) = (orig_w as f32, orig_h as f32);
    let (new_wf, new_hf) = if orig_w >= orig_h {
        // Width is longer side
        let scale = max_side as f32 / orig_wf;
        (orig_wf * scale, orig_hf * scale)
    } else {
        // Height is longer side
        let scale = max_side as f32 / orig_hf;
        (orig_wf * scale, orig_hf * scale)
    };
    // Round to nearest multiple of `divisor` (e.g., patch_size * spatial_merge_size).
    // Ensure we never return zero by clamping the multiple to at least 1.
    let round_to_multiple = |x: f32, d: usize| -> usize {
        let m = (x / d as f32).round();
        let m = if m < 1.0 { 1.0 } else { m };
        (m as usize) * d
    };
    let new_w = round_to_multiple(new_wf, divisor);
    let new_h = round_to_multiple(new_hf, divisor);

    (new_h, new_w)
}

/// Compute resized dimensions using the image file's width/height.
pub fn compute_pixtral_resize_dims<P: AsRef<Path>>(
    path: P,
    max_side: usize,
    divisor: usize,
) -> Result<(usize, usize)> {
    let (w, h) = image::image_dimensions(&path).map_err(candle::Error::wrap)?;
    Ok(compute_resized_dims_from_wh(w, h, max_side, divisor))
}

/// Load and normalize an image to the requested height/width using Pixtral constants.
/// Returns a tensor with shape (3, H, W) and dtype f32 on CPU.
pub fn load_image_pixtral<P: AsRef<Path>>(
    path: P,
    height: usize,
    width: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Tensor> {
    if height == 0 || width == 0 {
        bail!("invalid target size: {}x{}", height, width);
    }
    let img = image::ImageReader::open(&path)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(mean, &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, &Device::Cpu)?.reshape((3, 1, 1))?;
    Ok((data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_multiple_rounding_and_divisibility() {
        // 1920x1080 scaled to max_side=1540, divisor=28 → (868, 1540)
        let (h, w) = compute_resized_dims_from_wh(1920, 1080, 1540, 28);
        assert_eq!((h, w), (868, 1540));
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);

        // Sanity across shapes: results must be positive and divisible by 28
        let cases = [
            (1800u32, 1024u32),
            (1024, 1800),
            (1024, 1024),
            (4000, 3000),
            (3000, 4000),
            (1921, 1081),
        ];
        for (ow, oh) in cases {
            let (hh, ww) = compute_resized_dims_from_wh(ow, oh, 1540, 28);
            assert!(hh > 0 && ww > 0, "{}x{} -> {}x{}", ow, oh, hh, ww);
            assert_eq!(hh % 28, 0, "{}x{} -> {} not divisible by 28", ow, oh, hh);
            assert_eq!(ww % 28, 0, "{}x{} -> {} not divisible by 28", ow, oh, ww);
        }
    }

    #[test]
    fn grid_and_merge_token_count() {
        // For 868x1540 with patch=14 and spatial_merge_size=2 ->
        // grid = (62, 110) and merged = (31, 55) => 1705 tokens.
        let (h, w) = (868usize, 1540usize);
        let patch = 14usize;
        let s = 2usize;
        let (gh, gw) = (h / patch, w / patch);
        let (mh, mw) = (gh / s, gw / s);
        assert_eq!((gh, gw), (62, 110));
        assert_eq!((mh, mw), (31, 55));
        assert_eq!(mh * mw, 1705);
    }
}
