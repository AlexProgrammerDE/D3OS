use std::{env, fs, path::PathBuf};
fn main() {
    let mut source = String::from("#[cutile::module] pub mod kernels { use cutile::core::*;\n");
    for file in [
        "common.rs",
        "burn-math.rs",
        "quantized.rs",
        "quad.rs",
        "elementwise.rs",
        "matrix.rs",
        "tensor.rs",
        "attention.rs",
        "delta.rs",
        "delta-split.rs",
        "raster.rs",
        "diagnostic.rs",
    ] {
        println!("cargo:rerun-if-changed=src/{file}");
        source.push_str(&fs::read_to_string(format!("src/{file}")).expect("read tile source"));
        source.push('\n');
    }
    println!("cargo:rerun-if-changed=src/burn-elementwise.rs.in");
    let elementwise = fs::read_to_string("src/burn-elementwise.rs.in").expect("read Burn elementwise template");
    let mut parameters: Vec<String> = ["count", "_rank", "operation", "scalar", "offset_a", "offset_b", "extent_a", "extent_b"]
        .map(str::to_owned)
        .into();
    parameters.extend((0..8).map(|axis| format!("dim{axis}")));
    parameters.extend((0..8).map(|axis| format!("stride_a{axis}")));
    parameters.extend((0..8).map(|axis| format!("stride_b{axis}")));
    let parameters = parameters.into_iter().map(|name| format!("{name}: u32")).collect::<Vec<_>>().join(", ");
    let addresses = (0..8)
        .rev()
        .map(|axis| {
            let advance = if axis == 0 {
                String::new()
            } else {
                format!("linear = linear / broadcast_scalar(dim{axis}, shape![128]);\n")
            };
            format!(
                "let coordinate = linear % broadcast_scalar(dim{axis}, shape![128]);\n\
         {advance}\
         let coordinate: Tile<i32, {{ [128] }}> = bitcast(coordinate);\n\
         address_a = address_a + coordinate * broadcast_scalar(signed(stride_a{axis}), shape![128]);\n\
         address_b = address_b + coordinate * broadcast_scalar(signed(stride_b{axis}), shape![128]);\n"
            )
        })
        .collect::<String>();
    for op in d3os_nvidia::compute::elementwise::Elementwise::ALL {
        source.push_str(
            &elementwise
                .replace("BURN_ENTRY", op.symbol())
                .replace("BURN_PARAMETERS", &parameters)
                .replace("BURN_ADDRESSES", &addresses)
                .replace("BURN_OP", &format!("{}u32", op as u32)),
        );
    }
    println!("cargo:rerun-if-changed=src/burn-words.rs.in");
    let words = fs::read_to_string("src/burn-words.rs.in").expect("read Burn word template");
    for op in d3os_nvidia::compute::words::WordOp::ALL {
        source.push_str(
            &words
                .replace("BURN_ENTRY", op.symbol())
                .replace("BURN_PARAMETERS", &parameters)
                .replace("BURN_ADDRESSES", &addresses)
                .replace("BURN_OP", &format!("{}u32", op as u32)),
        );
    }
    println!("cargo:rerun-if-changed=src/burn-gather.rs.in");
    let gather = fs::read_to_string("src/burn-gather.rs.in").expect("read gather template");
    let mut parameters: Vec<String> = ["_rank", "axis", "count", "extent_a", "offset_a", "extent_b", "offset_b", "axis_length"]
        .map(str::to_owned)
        .into();
    parameters.extend((0..8).map(|axis| format!("dim{axis}")));
    parameters.extend((0..8).map(|axis| format!("stride_a{axis}")));
    parameters.extend((0..8).map(|axis| format!("stride_b{axis}")));
    let parameters = parameters.into_iter().map(|name| format!("{name}: u32")).collect::<Vec<_>>().join(", ");
    let addresses = (0..8)
        .rev()
        .map(|axis| {
            let advance = if axis == 0 {
                String::new()
            } else {
                format!("linear = linear / broadcast_scalar(dim{axis}, shape![128]);\n")
            };
            format!(
                "let coordinate = linear % broadcast_scalar(dim{axis}, shape![128]);\n{advance}\
                 let coordinate: Tile<i32, {{ [128] }}> = bitcast(coordinate);\n\
                 if axis != {axis}u32 {{ address_a = address_a + coordinate * broadcast_scalar(signed(stride_a{axis}), shape![128]); }}\n\
                 address_b = address_b + coordinate * broadcast_scalar(signed(stride_b{axis}), shape![128]);\n"
            )
        })
        .collect::<String>();
    let stride = (0..7)
        .map(|axis| format!("if axis == {axis}u32 {{ stride_a{axis} }} else "))
        .collect::<String>()
        + "{ stride_a7 }";
    source.push_str(
        &gather
            .replace("GATHER_PARAMETERS", &parameters)
            .replace("GATHER_ADDRESSES", &addresses)
            .replace("GATHER_STRIDE", &stride),
    );
    println!("cargo:rerun-if-changed=src/burn-reduction.rs.in");
    let reduction = fs::read_to_string("src/burn-reduction.rs.in").expect("read Burn reduction template");
    let mut parameters: Vec<String> = ["count", "_rank", "_operation", "axis", "offset", "extent", "length", "_reserved"]
        .map(str::to_owned)
        .into();
    parameters.extend((0..8).map(|axis| format!("dim{axis}")));
    parameters.extend((0..8).map(|axis| format!("stride{axis}")));
    let parameters = parameters.into_iter().map(|name| format!("{name}: u32")).collect::<Vec<_>>().join(", ");
    let addresses = (0..8)
        .rev()
        .map(|axis| {
            let advance = if axis == 0 { String::new() } else { String::from("linear = linear / dim;\n") };
            format!(
                "let dim = if axis == {axis}u32 {{ 1u32 }} else {{ dim{axis} }};\n\
                 let coordinate = linear % dim;\n\
                 {advance}\
                 base = base + signed(coordinate) * signed(stride{axis});\n"
            )
        })
        .collect::<String>();
    let stride = (0..7).map(|axis| format!("if axis == {axis}u32 {{ stride{axis} }} else ")).collect::<String>() + "{ stride7 }";
    for op in d3os_nvidia::compute::reduction::Reduction::ALL {
        let minimum = matches!(
            op,
            d3os_nvidia::compute::reduction::Reduction::Min | d3os_nvidia::compute::reduction::Reduction::ArgMin
        );
        source.push_str(
            &reduction
                .replace("REDUCE_ENTRY", op.symbol())
                .replace("REDUCE_PARAMETERS", &parameters)
                .replace("REDUCE_ADDRESSES", &addresses)
                .replace("REDUCE_STRIDE", &stride)
                .replace(
                    "REDUCE_IDENTITY",
                    if minimum {
                        "bits_f32(0x7f800000u32)"
                    } else {
                        "bits_f32(unsigned(-8388608i32))"
                    },
                )
                .replace("REDUCE_COMPARE", if minimum { "lt_tile" } else { "gt_tile" })
                .replace("REDUCE_EXTREME", if minimum { "reduce_min" } else { "reduce_max" })
                .replace("REDUCE_OP", &format!("{}u32", op as u32)),
        );
    }
    println!("cargo:rerun-if-changed=src/quantized.rs.in");
    let quantized = fs::read_to_string("src/quantized.rs.in").expect("read quantized template");
    for width in [128, 512] {
        source.push_str(&quantized.replace("QUANT_WIDTH", &width.to_string()));
    }
    println!("cargo:rerun-if-changed=src/delta-recurrent.rs.in");
    let recurrent = fs::read_to_string("src/delta-recurrent.rs.in").expect("read recurrent head template");
    for (entry, extra, channels, query_head, key_heads) in [
        ("delta_recurrent_batch", "", "heads * (2u32 * key_dim + value_dim)", "head", "heads"),
        (
            "delta_grouped_recurrent_batch",
            ", key_heads: u32",
            "(2u32 * key_heads * key_dim + heads * value_dim)",
            "(head / (heads / key_heads))",
            "key_heads",
        ),
    ] {
        source.push_str(
            &recurrent
                .replace("RECURRENT_ENTRY", entry)
                .replace("RECURRENT_EXTRA", extra)
                .replace("RECURRENT_CHANNELS", channels)
                .replace("RECURRENT_QUERY_HEAD", query_head)
                .replace("RECURRENT_KEY_HEADS", key_heads),
        );
    }
    println!("cargo:rerun-if-changed=src/delta-sequence.rs.in");
    let delta = fs::read_to_string("src/delta-sequence.rs.in").expect("read recurrent template");
    for (history, name) in [(4, "delta_sequence_four_taps"), (8, "delta_sequence")] {
        source.push_str(
            &delta
                .replace("DELTA_SEQUENCE_HELPER", name)
                .replace("HISTORY_WIDTH", &history.to_string())
                .replace("HISTORY_VALUES", &(512 * history).to_string()),
        );
    }
    println!("cargo:rerun-if-changed=src/matrix.rs.in");
    let matrix = fs::read_to_string("src/matrix.rs.in").expect("read matrix tile template");
    for inner in [8, 16, 64] {
        source.push_str(
            &matrix
                .replace("compensated_matrix_template", &format!("compensated_matrix_{inner}"))
                .replace("TILE_INNER_U32", &format!("{inner}u32"))
                .replace("TILE_INNER", &inner.to_string())
                .replace("TILE_A_VALUES", &(16 * inner).to_string())
                .replace("TILE_B_VALUES", &(32 * inner).to_string()),
        );
    }
    println!("cargo:rerun-if-changed=src/vector.rs.in");
    let vector = fs::read_to_string("src/vector.rs.in").expect("read vector tile template");
    for columns in [4, 32] {
        let specialized = vector
            .replace("vector_project_template", &format!("vector_project_{columns}"))
            .replace("TILE_COLUMNS", &columns.to_string())
            .replace("FLAT_VALUES", &(columns * 128).to_string());
        source.push_str(&specialized);
    }
    println!("cargo:rerun-if-changed=src/paired.rs.in");
    let paired = fs::read_to_string("src/paired.rs.in").expect("read paired BF16 template");
    source.push_str(
        &paired
            .replace("paired_project_template", "paired_project_32")
            .replace("TILE_COLUMNS", "32")
            .replace("PAIR_VALUES", "2048"),
    );
    println!("cargo:rerun-if-changed=src/texture.rs.in");
    let texture = fs::read_to_string("src/texture.rs.in").expect("read texture tile template");
    for width in [1, 32] {
        source.push_str(
            &texture
                .replace("texture_tile_template", &format!("texture_tile_{width}"))
                .replace("[B]", &format!("[{width}]"))
                .replace("BYTES, B", &format!("BYTES, {width}")),
        );
    }
    println!("cargo:rerun-if-changed=src/ordered.rs.in");
    let ordered = fs::read_to_string("src/ordered.rs.in").expect("read ordered memory template");
    for (element, width) in [("u32", 1), ("u8", 1), ("f32", 128)] {
        source.push_str(
            &ordered
                .replace("ordered_read_template", &format!("ordered_read_{element}_{width}"))
                .replace("[B]", &format!("[{width}]"))
                .replace("<E,", &format!("<{element},"))
                .replace("padding: E", &format!("padding: {element}"))
                .replace("*const E", &format!("*const {element}")),
        );
    }
    for (format, name) in [(8, "q8_0"), (11, "q3k"), (12, "q4k"), (13, "q5k"), (14, "q6k")] {
        source.push_str(&format!("#[cutile::entry()] unsafe fn vector_project_{name}(a:*const f32,b:*const f32,out:*mut f32,_m:u32,k:u32,n:u32,_format:u32){{unsafe{{ vector_project_4::<false,false,{format}>(a,b,b,out,k,n) }} }}\n"));
        for (prefix, base, arguments, body) in [
            (
                "project",
                "project_quantized",
                "a:*const f32,b:*const f32,out:*mut f32,m:u32,k:u32,n:u32,_format:u32",
                format!("compensated_matrix_16::<true,false,false,{format}>(a,b,out,m,k,n,1u32)"),
            ),
            (
                "dequantize",
                "dequantize_weights",
                "a:*const f32,out:*mut f32,n:u32,_format:u32",
                format!("unpack_quantized::<{format}>(a,out,n)"),
            ),
            (
                "embedding",
                "embedding_quantized",
                "a:*const f32,ids:*const u32,out:*mut f32,vocab:u32,width:u32,rows:u32,_format:u32",
                format!("gather_quantized::<{format}>(a,ids,out,vocab,width,rows)"),
            ),
        ] {
            let entry = if format == 8 { base.to_owned() } else { format!("{prefix}_{name}") };
            source.push_str(&format!("#[cutile::entry()] unsafe fn {entry}({arguments}){{unsafe{{{body}}}}}\n"));
        }
    }
    source.push('}');
    fs::write(PathBuf::from(env::var_os("OUT_DIR").expect("build directory")).join("programs.rs"), source).expect("write tile module");
}
