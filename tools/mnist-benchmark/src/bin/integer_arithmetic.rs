//! Exact public integer arithmetic with a device-free corpus description.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::integer_arithmetic::{Operation, Options, Summary, expected_summary, run},
};

fn summary_json(s: &Summary) -> serde_json::Value {
    serde_json::json!({
        "dtypes": s.dtypes, "operations": s.operations, "cases": s.cases(),
        "tensor_cases": s.tensor_cases, "scalar_cases": s.scalar_cases,
        "unary_cases": s.unary_cases, "extension_cases": s.extension_cases,
        "mixed_exponent_cases": s.mixed_exponent_cases,
        "source_checks": s.source_checks, "guard_checks": s.guard_checks,
        "values": s.values, "exhaustive_pairs": s.exhaustive_pairs,
        "exhaustive_unary_values": s.exhaustive_unary_values,
        "unsigned_neg_cases": s.unsigned_neg_cases,
    })
}
fn main() {
    let mut options = Options::default();
    let mut describe = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--expected-summary" => describe = true,
            "--extensions" => options.extensions = true,
            _ => {
                if let Some(value) = arg.strip_prefix("--dtype=") {
                    options.dtype = Some(match value {
                        "i8" => DType::I8,
                        "i16" => DType::I16,
                        "i32" => DType::I32,
                        "i64" => DType::I64,
                        "u8" => DType::U8,
                        "u16" => DType::U16,
                        "u32" => DType::U32,
                        "u64" => DType::U64,
                        _ => panic!("unknown integer dtype: {value}"),
                    });
                } else if let Some(value) = arg.strip_prefix("--op=") {
                    options.operation = Some(match value {
                        "add" => Operation::Add,
                        "sub" => Operation::Sub,
                        "mul" => Operation::Mul,
                        "div" => Operation::Div,
                        "floor-rem" => Operation::FloorRem,
                        "powi" => Operation::Powi,
                        "neg" => Operation::Neg,
                        "abs" => Operation::Abs,
                        "sign" => Operation::Sign,
                        "and" => Operation::And,
                        "or" => Operation::Or,
                        "xor" => Operation::Xor,
                        "not" => Operation::Not,
                        "shl" => Operation::Shl,
                        "shr" => Operation::Shr,
                        _ => panic!("unknown integer operation: {value}"),
                    });
                } else {
                    panic!("unknown integer arithmetic argument: {arg}");
                }
            }
        }
    }
    // This must precede device construction and backend initialization.
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let summary = try_read_sync(run(&device, &options))
        .expect("integer arithmetic executor")
        .expect("exact integer arithmetic");
    device.sync().expect("integer arithmetic synchronization");
    println!(
        "BURN_INTEGER_ARITHMETIC_OK os=linux backend={backend} totals={} reference=u128_i128 operations=15 values=exact_bits exponent=u32_bits_32_steps scalar=checked_base_format scalar_entry=backend public_power_shortcuts=true views=offset,step,permuted,reversed,broadcast_both ranks=0,1,2,8 tails=0,1,2,3,4,5,127,128,129,257 source_preservation=true poison=retained_allocation physical_padding=not_public extensions={} mixed_exponents={} unsigned_neg={}",
        summary_json(&summary),
        options.extensions,
        if options.extensions {
            "native_extension_all8_rhs_formats"
        } else {
            "same_format_portable"
        },
        if options.extensions {
            "native_extension"
        } else {
            "excluded_native_extension"
        },
    );
}
