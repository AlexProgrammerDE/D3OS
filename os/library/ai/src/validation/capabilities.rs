//! Declared device capabilities, kept separate from numerical conformance.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{Backend, BoolStore, DType, DTypeUsage, Dispatch},
    tensor::{
        Device,
        quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
    },
};

#[derive(Debug)]
pub struct Capability {
    pub name: String,
    pub storage: bool,
    pub arithmetic: bool,
    pub accelerated: bool,
}

fn query(device: &Device, name: String, dtype: DType) -> Capability {
    let usage = Dispatch::dtype_usage(device.as_dispatch(), dtype);
    Capability {
        name,
        storage: usage.contains(DTypeUsage::Storage),
        arithmetic: usage.contains(DTypeUsage::Arithmetic),
        accelerated: usage.contains(DTypeUsage::Accelerated),
    }
}

/// Query ordinary dtypes and representative legal quantization schemes.
/// These flags describe the backend's declarations, not proof that every
/// operation, shape, or execution mode works for the reported dtype.
pub fn declared(device: &Device) -> Vec<Capability> {
    let mut rows: Vec<_> = [
        DType::F64,
        DType::F32,
        DType::Flex32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool(BoolStore::Native),
        DType::Bool(BoolStore::U8),
        DType::Bool(BoolStore::U32),
    ]
    .into_iter()
    .map(|dtype| query(device, format!("{dtype:?}"), dtype))
    .collect();
    for value in [
        QuantValue::Q8F,
        QuantValue::Q8S,
        QuantValue::Q4F,
        QuantValue::Q4S,
        QuantValue::Q2F,
        QuantValue::Q2S,
        QuantValue::E4M3,
        QuantValue::E5M2,
        QuantValue::E2M1,
    ] {
        let mut stores = vec![QuantStore::PackedU32(0)];
        if value.size_bits() == 8 {
            stores.push(QuantStore::Native);
        }
        if value == QuantValue::E2M1 {
            stores.push(QuantStore::PackedNative(0));
        }
        for store in stores {
            for scale in [ScaleDtype::F32, ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE8M0, ScaleDtype::UE4M3] {
                for level in ["tensor", "block", "two_level"] {
                    let scheme = QuantScheme::default().with_value(value).with_store(store);
                    let scheme = match level {
                        "tensor" => scheme.per_tensor(scale),
                        "block" => scheme.per_block([16], scale),
                        _ => scheme.per_block([16], scale).per_tensor(ScaleDtype::F32),
                    };
                    rows.push(query(device, format!("quant/{value:?}/{store:?}/{scale:?}/{level}"), DType::QFloat(scheme)));
                }
            }
        }
    }
    rows
}
