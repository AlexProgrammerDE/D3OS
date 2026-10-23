use std::{env, fs, path::PathBuf};
fn main() {
    let mut source = String::from("#[cutile::module] pub mod kernels { use cutile::core::*;\n");
    for file in [
        "common.rs",
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
    source.push('}');
    fs::write(PathBuf::from(env::var_os("OUT_DIR").expect("build directory")).join("programs.rs"), source).expect("write tile module");
}
