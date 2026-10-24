//! Compile Rust tile programs ahead of time for the native D3OS driver.
use cutile::compile_api::KernelCompiler;
use cutile_ir::{
    Module,
    bytecode::{BytecodeVersion, write_bytecode_version},
};
use d3os_nvidia::compute::{KERNEL_COUNT, Kernel};
use std::{
    path::{Path, PathBuf},
    process::Command,
};
include!(concat!(env!("OUT_DIR"), "/programs.rs"));
mod link;

fn threads(name: &str) -> u32 {
    match name {
        "delta_fused" | "delta_fused_batch" | "delta_fused_four_taps" => 256,
        "raster_fuzz" | "raster_fuzz_bytes" => 32,
        _ => 128,
    }
}

fn compile(output: &Path, family: &str, names: &[&str], toolkit: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut module = Module::new(family);
    for name in names {
        eprintln!("Compiling cuTile {name}");
        let artifact = KernelCompiler::new(kernels::__module_ast_self, "kernels", name)
            .target("sm_89")
            .options(cutile::tile_kernel::CompileOptions::default().num_worker_warps_per_cta((threads(name) / 32).max(4) as i32))
            .bytecode_version(BytecodeVersion::V13_4)
            .compile()?;
        link::append(&mut module, artifact.module(), name)?;
    }
    let ir = output.join(format!("{family}.tilebc"));
    let cubin = output.join(format!("{family}.cubin"));
    std::fs::write(output.join(format!("{family}.mlir")), module.to_mlir_text())?;
    std::fs::write(&ir, write_bytecode_version(&module, BytecodeVersion::V13_4)?)?;
    let status = Command::new(toolkit.join("bin/tileiras"))
        .args(["--gpu-name", "sm_89", "--opt-level", "3", "-o"])
        .arg(&cubin)
        .arg(&ir)
        .status()?;
    if !status.success() {
        return Err(format!("tileiras failed for {family}: {status}").into());
    }
    let bytes = std::fs::read(&cubin)?;
    let elf = goblin::elf::Elf::parse(&bytes)?;
    for (index, relocations) in &elf.shdr_relocs {
        let target = &elf.section_headers[elf.section_headers[*index].sh_info as usize];
        let name = elf.shdr_strtab.get_at(target.sh_name);
        // Tile launch metadata is host-only for these raw-pointer kernels.
        // The native loader requires every GPU-visible section to be linked.
        if !relocations.is_empty() && target.sh_flags & u64::from(goblin::elf::section_header::SHF_ALLOC) != 0 && name != Some(".nv.constant4") {
            return Err(format!("unsupported GPU relocation in {name:?}").into());
        }
    }
    let mut metadata = Vec::new();
    for name in names {
        let section = |prefix: &str| {
            elf.section_headers
                .iter()
                .find(|s| elf.shdr_strtab.get_at(s.sh_name) == Some(&format!("{prefix}{name}")))
        };
        let code = section(".text.").ok_or_else(|| format!("missing entry {name}"))?;
        let info = section(".nv.info.").ok_or_else(|| format!("missing launch metadata {name}"))?;
        let info = &bytes[usize::try_from(info.sh_offset)?..usize::try_from(info.sh_offset + info.sh_size)?];
        let required = d3os_nvidia::cubin::required_threads(info).map_err(|e| format!("invalid launch metadata {name}: {e:?}"))?;
        if required != Some([threads(name), 1, 1]) {
            return Err(format!("incorrect thread contract for {name}: {required:?}").into());
        }
        metadata.push(serde_json::json!({"entry":name,"threads":threads(name),"registers":code.sh_info>>24,"code_bytes":code.sh_size}));
    }
    std::fs::write(output.join(format!("{family}.json")), serde_json::to_vec_pretty(&metadata)?)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = PathBuf::from(std::env::args().nth(1).ok_or("usage: d3os-gpu-kernels OUTPUT_DIRECTORY [ENTRY ...]")?);
    let toolkit = PathBuf::from(std::env::var_os("CUDA_TOOLKIT_PATH").ok_or("CUDA_TOOLKIT_PATH must point to the host compiler toolkit")?);
    std::fs::create_dir_all(&output)?;
    let names: Vec<String> = std::env::args().skip(2).collect();
    if names.is_empty() {
        compile(&output, "vector-add", &["vector_add"], &toolkit)?;
        compile(&output, "tensor-matmul", &["tensor_matmul_f16", "tensor_matmul_bf16"], &toolkit)?;
        let mut inference: Vec<&str> = (0..KERNEL_COUNT).map(|id| Kernel::from_raw(id as u32).expect("kernel ID").symbol()).collect();
        inference.extend([
            "vector_project_tf32x3",
            "vector_project_bf16_weights",
            "vector_project_paired_bf16_weights",
            "vector_project_wide_tf32x3",
            "vector_project_wide_bf16_weights",
            "vector_project_unpaired_wide_bf16_weights",
            "gated_project_paired_bf16_weights",
            "delta_convolution_batch",
            "delta_recurrent_batch",
            "delta_grouped_recurrent_batch",
            "delta_fused_four_taps",
            "attention_split_batch",
            "attention_merge_splits",
            "tensor_matmul_tf32x3_k16",
            "vector_project_q8_0",
            "vector_project_q3k",
            "project_q3k",
            "dequantize_q3k",
            "embedding_q3k",
            "vector_project_q4k",
            "project_q4k",
            "dequantize_q4k",
            "embedding_q4k",
            "vector_project_q5k",
            "project_q5k",
            "dequantize_q5k",
            "embedding_q5k",
            "vector_project_q6k",
            "project_q6k",
            "dequantize_q6k",
            "embedding_q6k",
            "timestamp_probe",
            "local_stack_probe",
        ]);
        inference.extend(d3os_nvidia::compute::elementwise::Elementwise::ALL.iter().skip(1).map(|op| op.symbol()));
        inference.extend(d3os_nvidia::compute::reduction::Reduction::ALL.iter().skip(1).map(|op| op.symbol()));
        inference.extend(d3os_nvidia::compute::reduction::Reduction::INTEGERS.iter().map(|op| op.symbol()));
        inference.extend(d3os_nvidia::compute::random::Mode::ALL.iter().skip(1).map(|op| op.symbol()));
        inference.extend(d3os_nvidia::compute::words::WordOp::ALL.iter().skip(1).map(|op| op.symbol()));
        inference.extend(d3os_nvidia::compute::indexing::Scatter::ALL.iter().skip(1).map(|op| op.symbol()));
        inference.extend(["burn_avg_pool", "burn_adaptive_avg_pool", "burn_adaptive_avg_pool_backward"]);
        compile(&output, "inference", &inference, &toolkit)?;
    } else {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        compile(&output, "kernels", &names, &toolkit)?;
    }
    Ok(())
}
