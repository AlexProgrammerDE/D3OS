//! Generate typed conversion operations inside one runtime-selected kernel.
use d3os_nvidia::compute::cast::Format;

fn encode(width: u32) -> String {
    let unsigned = format!("u{}", width * 8);
    let extend = if width == 8 {
        "converted_bits".to_owned()
    } else {
        "exti(converted_bits)".to_owned()
    };
    format!("let converted_bits: Tile<{unsigned}, {{ [128] }}> = bitcast(value);\n{extend}\n")
}

fn helper(name: &str, ty: &str, width: u32, float: bool) -> String {
    let mut source = format!("fn cast_{name}(bits: Tile<u64, {{ [128] }}>, destination: u32) -> Tile<u64, {{ [128] }}> {{\n");
    if float {
        if width == 8 {
            source.push_str("let x: Tile<f64, { [128] }> = bitcast(bits);\n");
        } else {
            source.push_str(&format!(
                "let raw: Tile<u{bits}, {{ [128] }}> = trunci(bits & ((broadcast_scalar(1u64, shape![128]) << broadcast_scalar({bits}u64, shape![128])) - broadcast_scalar(1u64, shape![128])), overflow::NoUnsignedWrap);\nlet x: Tile<{ty}, {{ [128] }}> = bitcast(raw);\n",
                bits = width * 8
            ));
        }
        source.push_str(&format!("if destination >= 13u32 {{\nlet magnitude = bits & ((broadcast_scalar(1u64, shape![128]) << broadcast_scalar({}u64, shape![128])) - broadcast_scalar(1u64, shape![128]));\nselect(ne_tile(magnitude, broadcast_scalar(0u64, shape![128])), broadcast_scalar(1u64, shape![128]), broadcast_scalar(0u64, shape![128]))\n}}\n",width*8-1));
    } else {
        source.push_str(&format!("let x: Tile<{ty}, {{ [128] }}> = bitcast(bits);\nif destination >= 13u32 {{ select(ne_tile(bits, broadcast_scalar(0u64, shape![128])), broadcast_scalar(1u64, shape![128]), broadcast_scalar(0u64, shape![128])) }}\n"));
    }
    for (format, target, bytes) in [
        (Format::F64, "f64", 8),
        (Format::F32, "f32", 4),
        (Format::F16, "f16", 2),
        (Format::BF16, "bf16", 2),
    ] {
        let condition = if format == Format::F32 {
            "destination == 1u32 || destination == 2u32".to_owned()
        } else {
            format!("destination == {}u32", format as u32)
        };
        source.push_str(&format!("else if {condition} {{\n"));
        if target == ty {
            source.push_str("bits\n");
        } else if target == "bf16" && ty == "f64" {
            source.push_str("cast_f64_bf16(bits)\n");
        } else if target == "bf16" && !float {
            if ty == "i64" {
                source.push_str("let negative = lt_tile(x,broadcast_scalar(0i64,shape![128]));\nlet magnitude = select(negative,broadcast_scalar(0u64,shape![128])-bits,bits);\ncast_integer_bf16(magnitude,negative)\n");
            } else {
                source.push_str("cast_integer_bf16(bits,broadcast_scalar(false,shape![128]))\n");
            }
        } else {
            let operation = if float { "ftof" } else { "itof" };
            source.push_str(&format!(
                "let value: Tile<{target}, {{ [128] }}> = {operation}(x, rounding::NearestEven);\n{{ {} }}\n",
                encode(bytes)
            ));
        }
        source.push_str("}\n");
    }
    if float {
        for (format, target, bytes) in [
            (Format::I64, "i64", 8),
            (Format::I32, "i32", 4),
            (Format::I16, "i16", 2),
            (Format::I8, "i8", 1),
            (Format::U64, "u64", 8),
            (Format::U32, "u32", 4),
            (Format::U16, "u16", 2),
            (Format::U8, "u8", 1),
        ] {
            source.push_str(&format!("else if destination == {}u32 {{\nlet value: Tile<{target}, {{ [128] }}> = unsafe {{ ftoi_saturating(x, saturating::Enabled) }};\n{{ {} }}\n}}\n",format as u32,encode(bytes)));
        }
        source.push_str("else { broadcast_scalar(0u64, shape![128]) }\n}\n");
    } else {
        source.push_str("else { bits }\n}\n");
    }
    source
}

pub fn expand(template: &str, parameters: &str, geometry: &str) -> String {
    let leading = [32u64,16,8,4,2,1].into_iter().map(|shift| {
        let advance=if shift==1 { String::new() } else { "remaining = select(present,high,remaining);\n".to_owned() };
        format!("let high = shri(remaining,broadcast_scalar({shift}u64,shape![128]));\nlet present = ne_tile(high,zero);\n{advance}top = top + select(present,broadcast_scalar({shift}u64,shape![128]),zero);\n")
    }).collect::<String>();
    let mut helpers = std::fs::read_to_string("src/burn-cast-bf16.rs.in")
        .expect("read direct BF16 rounding")
        .replace("BF16_LEADING_BIT", &leading);
    for (name, ty, width, float) in [
        ("f64", "f64", 8, true),
        ("f32", "f32", 4, true),
        ("f16", "f16", 2, true),
        ("bf16", "bf16", 2, true),
        ("i64", "i64", 8, false),
        ("u64", "u64", 8, false),
    ] {
        helpers.push_str(&helper(name, ty, width, float));
    }
    template
        .replace("CAST_HELPERS", &helpers)
        .replace("CAST_PARAMETERS", parameters)
        .replace("CAST_GEOMETRY_ARGUMENTS", geometry)
}
