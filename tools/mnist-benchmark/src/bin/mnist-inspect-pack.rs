//! Decode numerical reference weights with Burn's own format reader, without a GPU.
use burn_pack::{Bytes, DType, Reader};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::{self, Read},
};

fn inspect() -> Result<(), String> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input).map_err(|error| error.to_string())?;
    let tensors = Reader::from_bytes(Bytes::from_bytes_vec(input))
        .map_err(|error| error.to_string())?
        .into_tensors()
        .map_err(|error| error.to_string())?;
    let mut output = BTreeMap::new();
    for tensor in tensors {
        let (name, dtype, shape, _, bytes) = tensor.into_parts().map_err(|error| error.to_string())?;
        if dtype != DType::F32 || bytes.len() % 4 != 0 {
            return Err(format!("expected FP32 model parameter: {name}"));
        }
        let values: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|word| f64::from(f32::from_le_bytes(*word))).collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("non-finite model parameter: {name}"));
        }
        output.insert(name, json!({"shape": shape.to_vec(), "values": values}));
    }
    println!("{}", serde_json::to_string(&output).map_err(|error| error.to_string())?);
    Ok(())
}

fn main() {
    if let Err(error) = inspect() {
        eprintln!("MNIST_INSPECT_FAIL {error}");
        std::process::exit(1);
    }
}
