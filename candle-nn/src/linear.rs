//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::{Linear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use candle::{bail, Result, Tensor};
use std::fmt;
use std::sync::{Arc, OnceLock};

#[derive(Clone)]
struct Mxfp4LinearWeight {
    blocks: Tensor,
    scales: Tensor,
    in_dim: usize,
    out_dim: usize,
    dense_cache: Arc<OnceLock<Tensor>>,
}

impl fmt::Debug for Mxfp4LinearWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mxfp4LinearWeight")
            .field("blocks_shape", &self.blocks.dims().to_vec())
            .field("scales_shape", &self.scales.dims().to_vec())
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .finish()
    }
}

#[derive(Clone)]
enum LinearWeight {
    Dense(Tensor),
    Mxfp4(Mxfp4LinearWeight),
}

impl fmt::Debug for LinearWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearWeight::Dense(w) => f
                .debug_struct("Dense")
                .field("shape", &w.dims().to_vec())
                .finish(),
            LinearWeight::Mxfp4(w) => w.fmt(f),
        }
    }
}

#[derive(Clone)]
pub struct Linear {
    weight: LinearWeight,
    bias: Option<Tensor>,
}

impl fmt::Debug for Linear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Linear")
            .field("weight", &self.weight)
            .field("bias_shape", &self.bias.as_ref().map(|b| b.dims().to_vec()))
            .finish()
    }
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight: LinearWeight::Dense(weight),
            bias,
        }
    }

    pub fn from_mxfp4(
        blocks: Tensor,
        scales: Tensor,
        in_dim: usize,
        out_dim: usize,
        bias: Option<Tensor>,
    ) -> Self {
        Self {
            weight: LinearWeight::Mxfp4(Mxfp4LinearWeight {
                blocks,
                scales,
                in_dim,
                out_dim,
                dense_cache: Arc::new(OnceLock::new()),
            }),
            bias,
        }
    }

    pub fn weight(&self) -> &Tensor {
        match &self.weight {
            LinearWeight::Dense(w) => w,
            LinearWeight::Mxfp4(w) => {
                let blocks = w.blocks.clone();
                let scales = w.scales.clone();
                let in_dim = w.in_dim;
                let out_dim = w.out_dim;
                w.dense_cache.get_or_init(move || {
                    candle::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [out_dim, in_dim])
                        .expect("MXFP4 dequantization failed in Linear::weight()")
                })
            }
        }
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    fn forward_dense(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = weight.t()?;
                    Ok(x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?)
                } else {
                    let w = weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = weight.t()?;
                    Ok(x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?)
                } else {
                    let w = weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)
                }
            }
            _ => {
                let w = weight.t()?;
                x.matmul(&w)
            }
        }
    }

    fn forward_mxfp4(&self, x: &Tensor, weight: &Mxfp4LinearWeight) -> Result<Tensor> {
        let dims = x.dims();
        if dims.is_empty() {
            bail!("Linear expects input with at least one dimension")
        }
        let in_dim = *dims.last().unwrap();
        if in_dim != weight.in_dim {
            bail!(
                "Linear input dim mismatch: expected {}, got {}",
                weight.in_dim,
                in_dim
            )
        }
        let rows: usize = if dims.len() == 1 {
            1
        } else {
            dims[..dims.len() - 1].iter().product()
        };
        let flat = x.reshape((rows, in_dim))?;
        let flat = if flat.is_contiguous() {
            flat
        } else {
            flat.contiguous()?
        };
        let out2d = candle::mxfp4::matmul_mxfp4_bf16(&flat, &weight.blocks, &weight.scales)?;
        let mut out_shape = dims.to_vec();
        *out_shape.last_mut().unwrap() = weight.out_dim;
        if out_shape.len() == 1 {
            out2d.reshape((weight.out_dim,))
        } else {
            out2d.reshape(out_shape)
        }
    }
}

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let out = match &self.weight {
            LinearWeight::Dense(w) => self.forward_dense(x, w)?,
            LinearWeight::Mxfp4(w) => self.forward_mxfp4(x, w)?,
        };
        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

/// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}
