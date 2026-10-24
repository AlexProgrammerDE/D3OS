use super::model::{Error, Result};
use super::{AttentionConfig, DeltaConfig};
use d3os_tensor_file::element_count;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub linear_heads: usize,
    pub linear_key_heads: usize,
    pub tied_embeddings: bool,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    pub convolution_kernel: usize,
    pub full_attention_interval: usize,
    pub max_positions: usize,
    pub rope_theta: f32,
    pub epsilon: f32,
}

impl Config {
    pub const SMALL: Self = Self {
        hidden: 1024,
        intermediate: 3584,
        layers: 24,
        vocab: 248320,
        heads: 8,
        kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        linear_heads: 16,
        linear_key_heads: 16,
        tied_embeddings: true,
        linear_key_dim: 128,
        linear_value_dim: 128,
        convolution_kernel: 4,
        full_attention_interval: 4,
        max_positions: 262144,
        rope_theta: 10000000.,
        epsilon: 1e-6,
    };

    pub(crate) fn parse(values: &[f32]) -> Result<Self> {
        if !matches!(values.len(), 16 | 18) {
            return Err(Error::InvalidConfig);
        }
        let mut dims = [0usize; 14];
        for (dim, &value) in dims.iter_mut().zip(values) {
            if !value.is_finite() || !(1. ..=1_048_576.).contains(&value) || value != (value as usize) as f32 {
                return Err(Error::InvalidConfig);
            }
            *dim = value as usize;
        }
        let [
            hidden,
            intermediate,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            linear_heads,
            linear_key_dim,
            linear_value_dim,
            convolution_kernel,
            full_attention_interval,
            max_positions,
        ] = dims;
        let (linear_key_heads, tied_embeddings) = if values.len() == 18 {
            let heads = values[16];
            if !heads.is_finite() || !(1. ..=1_048_576.).contains(&heads) || heads != (heads as usize) as f32 {
                return Err(Error::InvalidConfig);
            }
            (
                heads as usize,
                match values[17] {
                    0. => false,
                    1. => true,
                    _ => return Err(Error::InvalidConfig),
                },
            )
        } else {
            (linear_heads, true)
        };
        let config = Self {
            linear_key_heads,
            tied_embeddings,
            hidden,
            intermediate,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            linear_heads,
            linear_key_dim,
            linear_value_dim,
            convolution_kernel,
            full_attention_interval,
            max_positions,
            rope_theta: values[14],
            epsilon: values[15],
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(self) -> Result<()> {
        let Self {
            hidden,
            intermediate,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            linear_heads,
            linear_key_heads,
            linear_key_dim,
            linear_value_dim,
            convolution_kernel,
            full_attention_interval,
            max_positions,
            ..
        } = self;
        let dimensions = [
            hidden,
            intermediate,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            linear_heads,
            linear_key_heads,
            linear_key_dim,
            linear_value_dim,
            convolution_kernel,
            full_attention_interval,
            max_positions,
        ];
        if dimensions.iter().any(|&n| n == 0 || n > 1_048_576)
            || layers > 128
            || !heads.is_multiple_of(kv_heads)
            || !linear_heads.is_multiple_of(linear_key_heads)
            || rotary_dim > head_dim
            || !rotary_dim.is_multiple_of(2)
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.
        {
            return Err(Error::InvalidConfig);
        }
        element_count(&[vocab, hidden])?;
        element_count(&[heads, head_dim, 2])?;
        element_count(&[intermediate, hidden])?;
        element_count(&[linear_heads, linear_key_dim, linear_value_dim])?;
        let keys = element_count(&[linear_key_heads, linear_key_dim, 2])?;
        let values = element_count(&[linear_heads, linear_value_dim])?;
        let channels = keys.checked_add(values).ok_or(Error::InvalidConfig)?;
        element_count(&[channels, hidden])?;
        Ok(())
    }

    pub(crate) fn attention(self) -> AttentionConfig {
        AttentionConfig {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            rope_theta: self.rope_theta,
            epsilon: self.epsilon,
        }
    }
    pub(crate) fn delta(self) -> DeltaConfig {
        DeltaConfig {
            key_heads: self.linear_key_heads,
            heads: self.linear_heads,
            key_dim: self.linear_key_dim,
            value_dim: self.linear_value_dim,
            kernel: self.convolution_kernel,
            epsilon: self.epsilon,
        }
    }
    pub(crate) fn linear_width(self) -> usize {
        self.linear_heads * self.linear_value_dim
    }
    pub(crate) fn conv_width(self) -> usize {
        2 * self.linear_key_heads * self.linear_key_dim + self.linear_heads * self.linear_value_dim
    }
}
