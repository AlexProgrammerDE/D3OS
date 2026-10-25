use std::{env, fs, path::PathBuf};
fn main() {
    let mut source = String::from("#[cutile::module] pub mod kernels { use cutile::core::*;\n");
    for file in [
        "common.rs",
        "burn-math.rs",
        "burn-erf.rs",
        "burn-integer-matrix.rs",
        "burn-scale-common.rs",
        "burn-range.rs",
        "burn-random-common.rs",
        "burn-pooling-common.rs",
        "quantized.rs",
        "quantized-vector.rs",
        "quad.rs",
        "elementwise.rs",
        "optimizer.rs",
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
    println!("cargo:rerun-if-changed=src/burn-quantization.rs.in");
    let quantization = fs::read_to_string("src/burn-quantization.rs.in").expect("read quantization source");
    let (address, entry) = quantization.split_once("#[cutile::entry()]").expect("quantization entry");
    let addresses = (0..8)
        .rev()
        .map(|axis| {
            format!(
                "let dim = unsafe {{ read_tile(metadata, broadcast_scalar({axis}u32, shape![128]), broadcast_scalar(true, shape![128]), 1u32) }};\n\
         let dim = select(eq_tile(dim, broadcast_scalar(0u32, shape![128])), broadcast_scalar(1u32, shape![128]), dim);\n\
         let stride = unsafe {{ read_tile(metadata, broadcast_scalar({stride}u32, shape![128]), broadcast_scalar(true, shape![128]), 0u32) }};\n\
         let stride: Tile<i32, {{ [128] }}> = bitcast(stride);\n\
         let block = unsafe {{ read_tile(metadata, broadcast_scalar({block}u32, shape![128]), broadcast_scalar(true, shape![128]), 1u32) }};\n\
         let block = select(eq_tile(block, broadcast_scalar(0u32, shape![128])), broadcast_scalar(1u32, shape![128]), block);\n\
         let scale_stride = unsafe {{ read_tile(metadata, broadcast_scalar({scale}u32, shape![128]), broadcast_scalar(true, shape![128]), 0u32) }};\n\
         let coordinate = linear % dim;\n\
         {advance}\n\
         scale = scale + (coordinate / block) * scale_stride;\n\
         let coordinate: Tile<i32, {{ [128] }}> = bitcast(coordinate);\n\
         address = address + coordinate * stride;\n",
                stride = 8 + axis,
                block = 16 + axis,
                scale = 24 + axis,
                advance = if axis == 0 { "" } else { "linear = linear / dim;" },
            )
        })
        .collect::<String>();
    source.push_str(&address.replace("QUANTIZATION_ADDRESSES", &addresses));
    for operation in [0, 1, 2] {
        let operation = d3os_nvidia::compute::quantization::Operation::from_raw(operation).expect("quantization operation");
        source.push_str("#[cutile::entry()]");
        source.push_str(
            &entry
                .replace("QUANTIZATION_ENTRY", operation.symbol())
                .replace("QUANTIZATION_MODE", &format!("{}u32", operation as u32)),
        );
    }
    println!("cargo:rerun-if-changed=src/burn-scale.rs.in");
    let scales = fs::read_to_string("src/burn-scale.rs.in").expect("read scale source");
    for operation in [0, 1, 2] {
        let operation = d3os_nvidia::compute::quantization::ScaleOperation::from_raw(operation).expect("scale operation");
        source.push_str(
            &scales
                .replace("SCALE_ENTRY", operation.symbol())
                .replace("SCALE_MODE", &format!("{}u32", operation as u32)),
        );
    }
    println!("cargo:rerun-if-changed=src/burn-random.rs.in");
    let random = fs::read_to_string("src/burn-random.rs.in").expect("read random source");
    for mode in d3os_nvidia::compute::random::Mode::ALL {
        source.push_str(
            &random
                .replace("RANDOM_ENTRY", mode.symbol())
                .replace("RANDOM_MODE", &format!("{}u32", mode as u32)),
        );
    }
    println!("cargo:rerun-if-changed=src/burn-elementwise.rs.in");
    let pool_names = [
        "batch", "channels", "depth", "height", "width", "od", "oh", "ow", "kd", "kh", "kw", "sd", "sh", "sw", "pd", "ph", "pw", "ed", "eh", "ew", "dd", "dh",
        "dw", "flags", "extent", "offset", "stride_n", "stride_c", "stride_d", "stride_h", "stride_w", "_mode",
    ];
    for backward in [false, true] {
        let file = if backward { "burn-pooling-backward.rs.in" } else { "burn-pooling.rs.in" };
        println!("cargo:rerun-if-changed=src/{file}");
        let template = fs::read_to_string(format!("src/{file}")).expect("read pooling source");
        let parameters = pool_names
            .iter()
            .map(|&name| {
                format!(
                    "{}: u32",
                    if backward && ["dd", "dh", "dw"].contains(&name) {
                        format!("_{name}")
                    } else {
                        name.to_owned()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        for mode in d3os_nvidia::compute::pooling::Mode::ALL {
            if backward && mode == d3os_nvidia::compute::pooling::Mode::Max {
                continue;
            }
            source.push_str(
                &template
                    .replace("POOL_ENTRY", mode.symbol(backward))
                    .replace("POOL_PARAMETERS", &parameters)
                    .replace("POOL_MODE", &format!("{}u32", mode as u32)),
            );
        }
    }
    println!("cargo:rerun-if-changed=src/burn-convolution.rs.in");
    let convolution = fs::read_to_string("src/burn-convolution.rs.in").expect("read convolution source");
    let geometry = [
        "batch", "channels", "height", "width", "kh", "kw", "oh", "ow", "sh", "sw", "ph", "pw", "dh", "dw", "groups", "_end_h", "_end_w", "outputs",
    ];
    let mut window = geometry.map(str::to_owned).to_vec();
    window.extend(["extent_x", "offset_x", "stride_x0", "stride_x1", "stride_x2", "stride_x3"].map(str::to_owned));
    let mut full = window.clone();
    full.extend(["extent_w", "offset_w", "stride_w0", "stride_w1", "stride_w2", "stride_w3"].map(str::to_owned));
    full.push("bias_enabled".into());
    window[17] = "_outputs".into();
    let mut geometry = geometry.map(str::to_owned);
    geometry[17] = "_outputs".into();
    let parameters = |names: &[String]| names.iter().map(|n| format!("{n}: u32")).collect::<Vec<_>>().join(", ");
    source.push_str(
        &convolution
            .replace("CONV_PARAMETERS", &parameters(&full))
            .replace("WINDOW_PARAMETERS", &parameters(&window))
            .replace("GEOMETRY_PARAMETERS", &parameters(&geometry)),
    );
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
    println!("cargo:rerun-if-changed=src/burn-updates.rs.in");
    let updates = fs::read_to_string("src/burn-updates.rs.in").expect("read update source");
    source.push_str(
        &updates
            .replace(
                "BURN_PARAMETERS",
                &parameters.replace("operation: u32", "_operation: u32").replace("scalar: u32", "_scalar: u32"),
            )
            .replace("BURN_ADDRESSES", &addresses),
    );
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
    println!("cargo:rerun-if-changed=src/burn-scatter.rs.in");
    println!("cargo:rerun-if-changed=src/burn-scatter-f32.rs.in");
    let scatter = fs::read_to_string("src/burn-scatter.rs.in").expect("read scatter template");
    let float_add = fs::read_to_string("src/burn-scatter-f32.rs.in").expect("read floating-point scatter accumulation");
    for operation in d3os_nvidia::compute::indexing::Scatter::ALL {
        use d3os_nvidia::compute::indexing::Scatter;
        let update = match operation {
            Scatter::FloatAdd => float_add.clone(),
            Scatter::IntAdd | Scatter::BoolOr => format!(
                "let (_old, _token): (Tile<u32, {{ [128] }}>, Token) = unsafe {{\n\
                 atomic_rmw_tko(pointer, value, atomic::{}, ordering::Relaxed, scope::Device, Some(valid), None)\n}};",
                if operation == Scatter::IntAdd { "Add" } else { "Or" },
            ),
        };
        source.push_str(
            &scatter
                .replace("SCATTER_ENTRY", operation.symbol())
                .replace("GATHER_PARAMETERS", &parameters)
                .replace("GATHER_ADDRESSES", &addresses)
                .replace("GATHER_STRIDE", &stride)
                .replace("SCATTER_UPDATE", &update),
        );
    }
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
    println!("cargo:rerun-if-changed=src/burn-scan.rs.in");
    let scans = fs::read_to_string("src/burn-scan.rs.in").expect("read scan template");
    for op in d3os_nvidia::compute::scan::Scan::ALL {
        use d3os_nvidia::compute::scan::Scan;
        let identity = match op {
            Scan::Sum | Scan::IntSum => "0u32",
            Scan::Product => "0x3f800000u32",
            Scan::IntProduct => "1u32",
            Scan::Min => "0x7f800000u32",
            Scan::Max => "0x7f800000u32 | (1u32 << 31u32)",
            Scan::IntMin => "0x7fffffffu32",
            Scan::IntMax => "1u32 << 31u32",
        };
        let minimum = matches!(op, Scan::Min | Scan::IntMin);
        source.push_str(
            &scans
                .replace("SCAN_ENTRY", op.symbol())
                .replace("SCAN_PARAMETERS", &parameters.replace("_reserved", "suffix"))
                .replace("SCAN_ADDRESSES", &addresses)
                .replace("SCAN_STRIDE", &stride)
                .replace("SCAN_IDENTITY", identity)
                .replace("SCAN_OP", &format!("{}u32", op as u32))
                .replace("SCAN_ORDER", if minimum { "ak <= bk" } else { "ak >= bk" })
                .replace("SCAN_TILE_ORDER", if minimum { "le_tile" } else { "ge_tile" }),
        );
    }
    println!("cargo:rerun-if-changed=src/burn-i32-reduction.rs.in");
    let integer_reduction = fs::read_to_string("src/burn-i32-reduction.rs.in").expect("read integer reduction source");
    use d3os_nvidia::compute::reduction::Reduction;
    for op in Reduction::ALL.into_iter().chain(Reduction::INTEGERS) {
        let minimum = matches!(op, Reduction::Min | Reduction::ArgMin | Reduction::IntMin | Reduction::IntArgMin);
        source.push_str(
            &if op.is_integer() { &integer_reduction } else { &reduction }
                .replace("REDUCE_ENTRY", op.symbol())
                .replace("REDUCE_PARAMETERS", &parameters)
                .replace("REDUCE_ADDRESSES", &addresses)
                .replace("REDUCE_STRIDE", &stride)
                .replace(
                    "REDUCE_IDENTITY",
                    if op.is_integer() {
                        if minimum { "2147483647i32" } else { "-2147483647i32 - 1i32" }
                    } else if minimum {
                        "bits_f32(0x7f800000u32)"
                    } else {
                        "bits_f32(unsigned(-8388608i32))"
                    },
                )
                .replace("REDUCE_COMPARE", if minimum { "lt_tile" } else { "gt_tile" })
                .replace("REDUCE_EXTREME", if minimum { "reduce_min" } else { "reduce_max" })
                .replace("REDUCE_OP", &format!("{}u32", (op as u32) % 7)),
        );
    }
    println!("cargo:rerun-if-changed=src/burn-argwhere.rs.in");
    let argwhere = fs::read_to_string("src/burn-argwhere.rs.in").expect("read argwhere template");
    let parameters = (0..8)
        .map(|a| format!("dim{a}: u32"))
        .chain((12..16).map(|a| format!("_reserved{a}: u32")))
        .collect::<Vec<_>>()
        .join(", ");
    let coordinates = (0..8).rev().map(|axis| {
        let advance = if axis == 0 { String::new() } else { format!("linear = linear / broadcast_scalar(dim{axis}, shape![128]);\n") };
        format!("let coordinate = linear % broadcast_scalar(dim{axis}, shape![128]);\n{advance}let coordinate = select(valid, coordinate, broadcast_scalar(unsigned(-1i32), shape![128]));\nunsafe {{ write_tile(output, row * broadcast_scalar(rank, shape![128]) + broadcast_scalar({axis}u32, shape![128]), coordinate, mask & broadcast_scalar(rank > {axis}u32, shape![128])) }};\n")
    }).collect::<String>();
    source.push_str(
        &argwhere
            .replace("ARGWHERE_PARAMETERS", &parameters)
            .replace("ARGWHERE_COORDINATES", &coordinates),
    );
    println!("cargo:rerun-if-changed=src/burn-sort.rs.in");
    let sort = fs::read_to_string("src/burn-sort.rs.in").expect("read sort template");
    let (helpers, entry) = sort.split_once("#[cutile::entry()]").expect("sort entry");
    let mut address_names = vec!["offset".to_owned()];
    address_names.extend((0..8).map(|a| format!("dim{a}")));
    address_names.extend((0..8).map(|a| format!("stride{a}")));
    let address_arguments = address_names.join(", ");
    let address_parameters = address_names.iter().map(|n| format!("{n}: u32")).collect::<Vec<_>>().join(", ");
    let addresses = (0..8).rev().map(|axis| {
        let advance = if axis == 0 { String::new() } else { format!("linear = linear / broadcast_scalar(dim{axis}, shape![128]);") };
        format!("let coordinate = linear % broadcast_scalar(dim{axis}, shape![128]);\n{advance}\nlet coordinate: Tile<i32, {{ [128] }}> = bitcast(coordinate);\naddress = address + coordinate * broadcast_scalar(signed(stride{axis}), shape![128]);\n")
    }).collect::<String>();
    source.push_str(
        &helpers
            .replace("SORT_ADDRESS_PARAMETERS", &address_parameters)
            .replace("SORT_ADDRESSES", &addresses),
    );
    let mut sort_parameters = ["count", "_rank", "_mode", "_axis", "offset", "extent", "length", "suffix"]
        .map(str::to_owned)
        .to_vec();
    sort_parameters.extend(address_names.into_iter().skip(1));
    sort_parameters.extend(["width", "search_steps"].map(str::to_owned));
    sort_parameters.extend((26..32).map(|a| format!("_reserved{a}")));
    let sort_parameters = sort_parameters.iter().map(|n| format!("{n}: u32")).collect::<Vec<_>>().join(", ");
    for mode in d3os_nvidia::compute::sort::Mode::ALL {
        source.push_str("#[cutile::entry()]");
        source.push_str(
            &entry
                .replace("SORT_ENTRY", mode.symbol())
                .replace("SORT_PARAMETERS", &sort_parameters)
                .replace("SORT_ADDRESS_ARGUMENTS", &address_arguments)
                .replace("SORT_INITIAL", &mode.initial().to_string())
                .replace("SORT_INTEGER", &mode.integer().to_string())
                .replace("SORT_DESCENDING", &mode.descending().to_string())
                .replace("SORT_TOPK", &mode.topk().to_string()),
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
    for (inner, rows) in [(8, 16), (16, 16), (64, 16), (16, 32)] {
        let name = if rows == 16 {
            format!("compensated_matrix_{inner}")
        } else {
            format!("compensated_matrix_{inner}_m{rows}")
        };
        source.push_str(
            &matrix
                .replace("compensated_matrix_template", &name)
                .replace("TILE_ROWS_U32", &format!("{rows}u32"))
                .replace("TILE_ROWS", &rows.to_string())
                .replace("TILE_INNER_U32", &format!("{inner}u32"))
                .replace("TILE_INNER", &inner.to_string())
                .replace("TILE_A_VALUES", &(rows * inner).to_string())
                .replace("TILE_B_VALUES", &(32 * inner).to_string())
                .replace("TILE_C_VALUES", &(32 * rows).to_string()),
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
        let vector = if format == 8 {
            "vector_project_q8".to_owned()
        } else {
            format!("vector_project_k::<{format}>")
        };
        source.push_str(&format!("#[cutile::entry()] unsafe fn vector_project_{name}(a:*const f32,b:*const f32,out:*mut f32,_m:u32,k:u32,n:u32,_format:u32){{unsafe{{ {vector}(a,b,out,k,n) }} }}\n"));
        source.push_str(&format!("#[cutile::entry()] unsafe fn project_m32_{name}(a:*const f32,b:*const f32,out:*mut f32,m:u32,k:u32,n:u32,_format:u32){{unsafe{{ compensated_matrix_16_m32::<true,false,false,{format}>(a,b,out,m,k,n,1u32) }} }}\n"));
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
