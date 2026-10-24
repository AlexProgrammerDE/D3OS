use d3os_ai::qwen::{
    Error,
    gguf::{Gguf, ReadAt},
};

fn string(output: &mut Vec<u8>, value: &str) {
    output.extend((value.len() as u64).to_le_bytes());
    output.extend(value.as_bytes());
}

fn model(records: &[(&str, u32, u64)], alignment: u32) -> Vec<u8> {
    let numbers = [
        ("general.alignment", alignment),
        ("qwen35.embedding_length", 8),
        ("qwen35.feed_forward_length", 12),
        ("qwen35.block_count", 1),
        ("qwen35.attention.head_count", 2),
        ("qwen35.attention.head_count_kv", 1),
        ("qwen35.attention.key_length", 4),
        ("qwen35.attention.value_length", 4),
        ("qwen35.rope.dimension_count", 2),
        ("qwen35.ssm.group_count", 1),
        ("qwen35.ssm.state_size", 2),
        ("qwen35.ssm.inner_size", 6),
        ("qwen35.ssm.time_step_rank", 2),
        ("qwen35.ssm.conv_kernel", 3),
        ("qwen35.full_attention_interval", 1),
        ("qwen35.context_length", 32),
    ];
    let mut data = b"GGUF".to_vec();
    data.extend(3u32.to_le_bytes());
    data.extend((records.len() as u64).to_le_bytes());
    data.extend((numbers.len() as u64 + 4).to_le_bytes());
    string(&mut data, "general.architecture");
    data.extend(8u32.to_le_bytes());
    string(&mut data, "qwen35");
    for (key, value) in numbers {
        string(&mut data, key);
        data.extend(4u32.to_le_bytes());
        data.extend(value.to_le_bytes());
    }
    for (key, value) in [("qwen35.rope.freq_base", 10000f32), ("qwen35.attention.layer_norm_rms_epsilon", 1e-6f32)] {
        string(&mut data, key);
        data.extend(6u32.to_le_bytes());
        data.extend(value.to_le_bytes());
    }
    string(&mut data, "tokenizer.ggml.tokens");
    data.extend(9u32.to_le_bytes());
    data.extend(8u32.to_le_bytes());
    data.extend(3u64.to_le_bytes());
    for value in ["", "hello", "世界"] {
        string(&mut data, value);
    }
    for &(name, dtype, offset) in records {
        string(&mut data, name);
        data.extend(2u32.to_le_bytes());
        data.extend(8u64.to_le_bytes());
        data.extend(24u64.to_le_bytes());
        data.extend(dtype.to_le_bytes());
        data.extend(offset.to_le_bytes());
    }
    data.resize(data.len().div_ceil(32) * 32 + 4096, 0);
    data
}

#[test]
fn metadata_skips_token_arrays_and_distinguishes_tied_output_weights() {
    let bytes = model(&[("token_embd.weight", 0, 0)], 32);
    let parsed = Gguf::parse(bytes.as_slice()).unwrap();
    assert_eq!(parsed.config().hidden, 8);
    assert_eq!(parsed.config().vocab, 24);
    assert_eq!(parsed.config().linear_key_heads, 1);
    assert_eq!(parsed.config().linear_heads, 2);
    assert!(parsed.config().tied_embeddings);
    let bytes = model(&[("token_embd.weight", 0, 0), ("output.weight", 0, 768)], 32);
    assert!(!Gguf::parse(bytes.as_slice()).unwrap().config().tied_embeddings);
}

#[test]
fn malformed_regions_and_partial_weight_blocks_are_rejected() {
    for records in [
        vec![("token_embd.weight", 0, 0), ("output.weight", 0, 0)],
        vec![("token_embd.weight", 0, 1)],
        vec![("token_embd.weight", 0, u64::MAX - 31)],
        vec![("token_embd.weight", 0, 4096)],
        vec![("token_embd.weight", 8, 0)],
        vec![("token_embd.weight", 99, 0)],
        vec![("token_embd.weight", 0, 0), ("token_embd.weight", 0, 768)],
    ] {
        assert!(Gguf::parse(model(&records, 32).as_slice()).is_err());
    }
    for alignment in [0, 3, 8192] {
        assert!(Gguf::parse(model(&[("token_embd.weight", 0, 0)], alignment).as_slice()).is_err());
    }
    let bytes = model(&[("token_embd.weight", 0, 0)], 32);
    for size in [0, 4, 24, bytes.len() - 4096] {
        assert!(Gguf::parse(&bytes[..size]).is_err());
    }
}

struct Broken;
impl ReadAt for Broken {
    fn len(&self) -> u64 {
        1024
    }
    fn read_at(&mut self, _: u64, _: &mut [u8]) -> core::result::Result<(), Error> {
        Err(Error::SourceRead)
    }
}

#[test]
fn source_failures_are_preserved() {
    assert!(matches!(Gguf::parse(Broken), Err(Error::SourceRead)));
}
