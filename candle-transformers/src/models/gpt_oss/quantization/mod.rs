use candle::Result;
use candle::Tensor;

pub mod experts;
pub mod linear;

// Re-export for convenience
pub use experts::{load_all_experts_linear_mxfp4_grouped, load_expert_linear_mxfp4_grouped};
pub use linear::{detect_mxfp4_pair_names, load_linear_maybe_mxfp4};

// Constants/config
pub const MXFP4_BLOCK_ELEMS: usize = 32; // k=32 elements per block on the last dim.
pub const MXFP4_BLOCK_BYTES: usize = 16; // packed two nibbles per byte.

pub fn dequantize_mxfp4_linear(
    blocks: &Tensor,
    scales: &Tensor,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    candle::mxfp4::dequant_mxfp4_to_bf16(blocks, scales, [out_dim, in_dim])
}
