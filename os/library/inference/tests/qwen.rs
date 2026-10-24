use d3os_compute::{Backend, cpu::Cpu};
use d3os_inference::{Error, qwen::Qwen, tokenizer::Tokenizer, verify};

type Record = (String, Vec<usize>, Vec<f32>);

fn archive(records: &[Record]) -> Vec<u8> {
    let mut bytes = b"D3TENSR\0".to_vec();
    bytes.extend(1u32.to_le_bytes());
    bytes.extend((records.len() as u32).to_le_bytes());
    for (name, shape, values) in records {
        bytes.extend((name.len() as u16).to_le_bytes());
        bytes.extend((shape.len() as u16).to_le_bytes());
        bytes.extend((values.len() as u64).to_le_bytes());
        bytes.extend(name.bytes());
        for &dim in shape {
            bytes.extend((dim as u64).to_le_bytes());
        }
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
    }
    bytes
}

fn model_records() -> Vec<Record> {
    let mut records = vec![(
        "qwen3_5.config".into(),
        vec![16],
        vec![8., 12., 4., 24., 2., 1., 4., 2., 2., 2., 3., 3., 2., 32., 10000., 1e-6],
    )];
    let mut add = |name: String, shape: &[usize]| {
        let values = (0..shape.iter().product())
            .map(|index| (((index * 13 + records.len() * 7) % 23) as f32 - 11.) / 31.)
            .collect();
        records.push((name, shape.to_vec(), values));
    };
    add("embed_tokens.weight".into(), &[24, 8]);
    add("norm.weight".into(), &[8]);
    for index in 0..4 {
        for (name, shape) in [
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("mlp.gate_proj.weight", vec![12, 8]),
            ("mlp.up_proj.weight", vec![12, 8]),
            ("mlp.down_proj.weight", vec![8, 12]),
        ] {
            add(format!("layers.{index}.{name}"), &shape);
        }
        if index % 2 == 1 {
            for (name, shape) in [
                ("q_proj.weight", vec![16, 8]),
                ("k_proj.weight", vec![4, 8]),
                ("v_proj.weight", vec![4, 8]),
                ("o_proj.weight", vec![8, 8]),
                ("q_norm.weight", vec![4]),
                ("k_norm.weight", vec![4]),
            ] {
                add(format!("layers.{index}.self_attn.{name}"), &shape);
            }
        } else {
            for (name, shape) in [
                ("in_proj_qkv.weight", vec![14, 8]),
                ("in_proj_z.weight", vec![6, 8]),
                ("in_proj_a.weight", vec![2, 8]),
                ("in_proj_b.weight", vec![2, 8]),
                ("conv1d.weight", vec![14, 1, 3]),
                ("A_log", vec![2]),
                ("dt_bias", vec![2]),
                ("norm.weight", vec![3]),
                ("out_proj.weight", vec![8, 6]),
            ] {
                add(format!("layers.{index}.linear_attn.{name}"), &shape);
            }
        }
    }
    records
}

#[test]
fn prefill_and_incremental_decode_agree_and_sessions_remain_independent() {
    let model = Qwen::load(&Cpu, &archive(&model_records())).unwrap();
    let tokens = [1, 7, 12, 3];
    let mut incremental = model.session(&Cpu, 8).unwrap();
    let mut second = model.session(&Cpu, 8).unwrap();
    let mut expected = Vec::new();
    for token in tokens {
        expected = Cpu.download(&incremental.step(token).unwrap()).unwrap();
        second.step(19).unwrap();
    }
    let mut prefill = model.session(&Cpu, 8).unwrap();
    verify(&Cpu.download(&prefill.prefill(&tokens).unwrap()).unwrap(), &expected, 0., 0.).unwrap();
    assert_eq!(prefill.position(), 4);
    verify(
        &Cpu.download(&prefill.step(5).unwrap()).unwrap(),
        &Cpu.download(&incremental.step(5).unwrap()).unwrap(),
        0.,
        0.,
    )
    .unwrap();
    assert_ne!(
        Cpu.download(&second.step(5).unwrap()).unwrap(),
        Cpu.download(&prefill.step(5).unwrap()).unwrap()
    );
    assert!(model.session(&Cpu, 33).is_err());
    assert!(model.generate(&Cpu, &[], 4).is_err());
    assert!(model.generate(&Cpu, &[1], 0).is_err());
    assert_eq!(model.generate(&Cpu, &[1], 3).unwrap().tokens.len(), 3);
    let expected = model.generate(&Cpu, &[1, 7], 5).unwrap();
    let mut streamed = Vec::new();
    let actual = model
        .generate_stream(&Cpu, &[1, 7], 5, |token| {
            streamed.push(token);
            Ok(())
        })
        .unwrap();
    assert_eq!(streamed, expected.tokens);
    assert_eq!(actual.tokens, expected.tokens);
    assert_eq!(actual.stop, expected.stop);
    let mut calls = 0;
    assert!(
        model
            .generate_stream(&Cpu, &[1], 5, |_| {
                calls += 1;
                Err(d3os_inference::Error::InvalidToken)
            })
            .is_err()
    );
    assert_eq!(calls, 1);
}

#[test]
fn prompt_batches_preserve_nonzero_prefixes_chunk_boundaries_and_later_decode() {
    let mut records = model_records();
    records[0].2[13] = 80.;
    let model = Qwen::load(&Cpu, &archive(&records)).unwrap();
    let mut batched = model.session(&Cpu, 80).unwrap();
    let mut sequential = model.session(&Cpu, 80).unwrap();
    for token in [1, 7, 2] {
        batched.step(token).unwrap();
        sequential.step(token).unwrap();
    }
    let tokens: Vec<_> = (0..65).map(|i| (i * 7 % 24) as u32).collect();
    let mut invalid = tokens.clone();
    invalid[64] = 24;
    assert!(matches!(batched.prefill(&invalid), Err(Error::InvalidToken)));
    assert_eq!(batched.position(), 3);
    let mut expected = Vec::new();
    let mut traces = Vec::new();
    for &token in &tokens {
        let output = sequential
            .step_observed(token, |layer, output| {
                if layer == 0 {
                    traces.push(Vec::new());
                }
                traces.last_mut().unwrap().push(Cpu.download(output)?);
                Ok(())
            })
            .unwrap();
        expected = Cpu.download(&output).unwrap();
    }
    let mut chunks = Vec::new();
    let actual = batched
        .prefill_observed(&tokens, |start, layer, output| {
            let values = Cpu.download(output)?;
            if layer == 0 {
                chunks.push((start, output.shape()[0]));
            }
            for (row, actual) in values.chunks_exact(8).enumerate() {
                assert_eq!(actual, traces[start - 3 + row][layer]);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(Cpu.download(&actual).unwrap(), expected);
    assert_eq!(chunks, [(3, 32), (35, 32), (67, 1)]);
    assert_eq!(batched.position(), 68);
    for token in [5, 3, 9] {
        assert_eq!(
            Cpu.download(&batched.step(token).unwrap()).unwrap(),
            Cpu.download(&sequential.step(token).unwrap()).unwrap()
        );
    }
}

#[test]
fn invalid_input_preserves_cache_and_mid_step_errors_invalidate_the_session() {
    let model = Qwen::load(&Cpu, &archive(&model_records())).unwrap();
    let mut session = model.session(&Cpu, 2).unwrap();
    assert!(matches!(session.prefill(&[1, 24]), Err(Error::InvalidToken)));
    assert_eq!(session.position(), 0);
    session.step(1).unwrap();
    session.step(2).unwrap();
    assert!(matches!(session.step(3), Err(Error::ContextFull)));
    assert_eq!(session.position(), 2);
    let mut session = model.session(&Cpu, 4).unwrap();
    assert!(
        session
            .step_observed(1, |layer, _| if layer == 1 { Err(Error::VerificationFailed) } else { Ok(()) })
            .is_err()
    );
    assert!(matches!(session.step(2), Err(Error::SessionFailed)));
    let mut session = model.session(&Cpu, 4).unwrap();
    assert!(
        session
            .prefill_observed(&[1, 2], |_, layer, _| if layer == 1 { Err(Error::VerificationFailed) } else { Ok(()) })
            .is_err()
    );
    assert!(matches!(session.prefill(&[3]), Err(Error::SessionFailed)));
}

#[test]
fn loader_rejects_unsupported_configuration_and_misshaped_or_missing_weights() {
    let mut records = model_records();
    records[0].2[7] = 3.;
    assert!(matches!(Qwen::load(&Cpu, &archive(&records)), Err(Error::InvalidConfig)));
    let mut records = model_records();
    records[1].1 = vec![8, 24];
    assert!(matches!(Qwen::load(&Cpu, &archive(&records)), Err(Error::InvalidModelShape)));
    let mut records = model_records();
    records.pop();
    assert!(matches!(Qwen::load(&Cpu, &archive(&records)), Err(Error::MissingTensor(_))));
}

fn tokenizer_bytes() -> Vec<u8> {
    let extras: &[&[u8]] = &[b"aa", b"aaa", b"aaaa", b"12", "é".as_bytes()];
    let rules = [(97u32, 97u32, 256u32), (256, 97, 257), (256, 256, 258), (49, 50, 259), (195, 169, 260)];
    let mut data = b"D3BPE\0\0\0".to_vec();
    for value in [1u32, 256 + extras.len() as u32, rules.len() as u32, 0] {
        data.extend(value.to_le_bytes());
    }
    for byte in 0..256 {
        data.extend(1u32.to_le_bytes());
        data.push(byte as u8);
    }
    for token in extras {
        data.extend((token.len() as u32).to_le_bytes());
        data.extend_from_slice(token);
    }
    for (a, b, c) in rules {
        for value in [a, b, c] {
            data.extend(value.to_le_bytes());
        }
    }
    data
}

#[test]
fn tokenizer_applies_ranked_overlapping_merges_nfc_and_number_boundaries() {
    let data = tokenizer_bytes();
    let tokenizer = Tokenizer::parse(&data).unwrap();
    assert_eq!(tokenizer.encode("aaaaaaa").unwrap(), [258, 257]);
    assert_eq!(tokenizer.encode("12").unwrap(), [49, 50]);
    assert_eq!(tokenizer.encode("e\u{301}").unwrap(), [260]);
    assert_eq!(tokenizer.decode(&[258, 257], false).unwrap(), "aaaaaaa");
    assert!(tokenizer.decode(&[u32::MAX], true).is_err());
    for length in 0..data.len() {
        assert!(Tokenizer::parse(&data[..length]).is_err());
    }
    let mut duplicate = data.clone();
    duplicate.extend([0]);
    assert!(Tokenizer::parse(&duplicate).is_err());
}

#[test]
fn streaming_decoder_preserves_split_scalars_and_matches_lossy_batch_decoding() {
    let data = tokenizer_bytes();
    let tokenizer = Tokenizer::parse(&data).unwrap();
    for bytes in ["aé😀z".as_bytes(), &[0xf0, 0x9f], &[0xc3, b'a', 0xff, 0xe2, 0x82, 0xac]] {
        let ids: Vec<_> = bytes.iter().map(|&byte| byte as u32).collect();
        let mut decoder = tokenizer.decoder(true);
        let mut text = String::new();
        for &id in &ids {
            text.push_str(&decoder.push(id).unwrap());
        }
        text.push_str(&decoder.finish());
        assert_eq!(text, tokenizer.decode(&ids, true).unwrap());
    }
    let mut decoder = tokenizer.decoder(false);
    assert_eq!(decoder.push(0xc3).unwrap(), "");
    assert!(decoder.push(u32::MAX).is_err());
    assert_eq!(decoder.push(0xa9).unwrap(), "é");
}
#[test]
fn independent_output_weights_control_logits() {
    let mut records = model_records();
    records[0].1 = vec![18];
    records[0].2.extend([2., 0.]);
    records.push(("lm_head.weight".into(), vec![24, 8], vec![0.; 24 * 8]));
    let model = Qwen::load(&Cpu, &archive(&records)).unwrap();
    assert!(!model.config().tied_embeddings);
    let mut session = model.session(&Cpu, 8).unwrap();
    assert!(Cpu.download(&session.step(1).unwrap()).unwrap().iter().all(|&x| x == 0.));
    let tied = Qwen::load(&Cpu, &archive(&model_records())).unwrap();
    let mut tied = tied.session(&Cpu, 8).unwrap();
    assert!(Cpu.download(&tied.step(1).unwrap()).unwrap().iter().any(|&x| x != 0.));
}

#[test]
fn grouped_model_prefill_and_incremental_sessions_agree() {
    let mut records = model_records();
    records[0].1 = vec![18];
    records[0].2.extend([1., 1.]);
    for (name, shape, values) in &mut records {
        if name.ends_with("linear_attn.in_proj_qkv.weight") || name.ends_with("linear_attn.conv1d.weight") {
            shape[0] = 10;
            values.truncate(shape.iter().product());
        }
    }
    let model = Qwen::load(&Cpu, &archive(&records)).unwrap();
    assert_eq!(model.config().linear_key_heads, 1);
    let mut sequential = model.session(&Cpu, 8).unwrap();
    let mut expected = Vec::new();
    for token in [1, 7, 12, 3] {
        expected = Cpu.download(&sequential.step(token).unwrap()).unwrap();
    }
    let mut batched = model.session(&Cpu, 8).unwrap();
    assert_eq!(Cpu.download(&batched.prefill(&[1, 7, 12, 3]).unwrap()).unwrap(), expected);
    assert_eq!(
        Cpu.download(&batched.step(5).unwrap()).unwrap(),
        Cpu.download(&sequential.step(5).unwrap()).unwrap()
    );
}

#[test]
fn caller_owned_tensor_loading_matches_archives_and_validates_before_reading() {
    use d3os_inference::qwen::Config;
    use std::collections::BTreeMap;
    let archive_model = Qwen::load(&Cpu, &archive(&model_records())).unwrap();
    let mut records: BTreeMap<_, _> = model_records()
        .into_iter()
        .filter(|(name, _, _)| name != "qwen3_5.config")
        .map(|(name, shape, data)| (name, (shape, data)))
        .collect();
    let streamed = Qwen::load_weights(archive_model.config(), |name, expected_shape| {
        let (shape, data) = records.remove(name).unwrap();
        assert_eq!(shape, expected_shape);
        Ok(d3os_compute::Tensor::from_vec(&Cpu, &shape, data)?)
    })
    .unwrap();
    assert!(records.is_empty());
    let mut a = archive_model.session(&Cpu, 8).unwrap();
    let mut b = streamed.session(&Cpu, 8).unwrap();
    assert_eq!(
        Cpu.download(&a.prefill(&[1, 7, 12, 3]).unwrap()).unwrap(),
        Cpu.download(&b.prefill(&[1, 7, 12, 3]).unwrap()).unwrap()
    );
    assert!(matches!(
        Qwen::<Cpu>::load_weights(Config { hidden: 0, ..Config::SMALL }, |_, _| panic!(
            "invalid configuration must not read weights"
        )),
        Err(Error::InvalidConfig)
    ));
    assert!(matches!(
        Qwen::<Cpu>::load_weights(archive_model.config(), |_, _| Ok(d3os_compute::Tensor::from_vec(&Cpu, &[1], vec![0.])?)),
        Err(Error::InvalidModelShape)
    ));
}
