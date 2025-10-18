use std::path::Path;

use anyhow::{bail, Result};
use candle::{Device, Tensor};

// Constants should be passed in from caller; keep this module generic and testable.

/// Compute resized dimensions for Pixtral-style preprocessing.
/// - Preserve aspect ratio.
/// - Set the longest edge to `max_side`.
/// - Round both dimensions down to a multiple of `patch`.
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
    // Floor to integer pixels
    let mut new_w = new_wf.floor() as usize;
    let mut new_h = new_hf.floor() as usize;
    // Ensure at least 1 pixel before patch rounding
    new_w = new_w.max(1);
    new_h = new_h.max(1);

    // Round down so both dims are multiples of `divisor` (e.g., patch_size * spatial_merge_size).
    new_w = (new_w / divisor).max(1) * divisor;
    new_h = (new_h / divisor).max(1) * divisor;

    // Longest side should equal max_side (already divisible by patch for 14|1540)
    // After rounding, one side might drop slightly below; this mirrors HF processors
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
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(mean, &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, &Device::Cpu)?.reshape((3, 1, 1))?;
    Ok(
        (data.to_dtype(candle::DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_preserve_ar_and_divisor() {
        // Example: original 1800x1024 -> scale longest to 1540, then round to multiple of 14
        // 1024 * 1540 / 1800 = 875.56 -> floor -> 875 -> round down to 868 (62 * 14)
        let (h, w) = compute_resized_dims_from_wh(1800, 1024, 1540, 28);
        assert_eq!((h, w), (868, 1540));

        // Portrait image: 1024x1800 -> (1540, 868)
        let (h2, w2) = compute_resized_dims_from_wh(1024, 1800, 1540, 28);
        assert_eq!((h2, w2), (1540, 868));

        // Square image remains square at max_side and multiple of patch
        let (h3, w3) = compute_resized_dims_from_wh(1024, 1024, 1540, 28);
        assert_eq!((h3, w3), (1540, 1540));
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
        assert_eq!(mh * mw, 1705);
    }
}
