use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let source = root.join("os/application/doom");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut sources: Vec<_> = fs::read_dir(source.join("doomgeneric/doomgeneric"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "c"))
        .collect();
    // Match the native source list rather than compiling platform alternatives.
    let native = fs::read_to_string(source.join("Makefile.toml")).unwrap();
    sources.retain(|path| native.contains(&format!("/{}\"", path.file_name().unwrap().to_str().unwrap())));
    sources.push(source.join("src/engine-report.c"));
    sources.sort();
    let mut objects = Vec::new();
    for path in sources {
        println!("cargo:rerun-if-changed={}", path.display());
        let object = output.join(path.file_stem().unwrap()).with_extension("o");
        let status = Command::new("gcc")
            .args([
                "-c",
                "-std=gnu99",
                "-O3",
                "-march=x86-64-v3",
                "-D_DEFAULT_SOURCE",
                "-fpic",
                "-fno-stack-protector",
                "-ffreestanding",
                "-nostdlib",
            ])
            .arg(&path)
            .arg("-o")
            .arg(&object)
            .status()
            .expect("run DOOM C compiler");
        assert!(status.success(), "C compilation failed: {}", path.display());
        objects.push(object);
    }
    println!("cargo:rerun-if-changed={}", source.join("Makefile.toml").display());
    println!("cargo:rerun-if-changed={}", source.join("doomgeneric/doomgeneric").display());
    let archive = output.join("libdoom_engine.a");
    assert!(Command::new("ar").arg("rcs").arg(&archive).args(&objects).status().unwrap().success());
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=doom_engine");
    println!("cargo:rustc-link-lib=m");
    let start = native.find("[tasks.link]").expect("native link contract");
    let end = native[start..].find("condition").map_or(native.len(), |end| start + end);
    for value in native[start..end].split('"').filter(|value| value.starts_with("--wrap=")) {
        println!("cargo:rustc-link-arg=-Wl,{value}");
    }
}
