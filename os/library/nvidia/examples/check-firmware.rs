use d3os_nvidia::{abi::*, firmware};
use std::{env, fs, path::PathBuf};

fn main() {
    let directory = PathBuf::from(env::args().nth(1).expect("firmware directory"));
    let booter = fs::read(directory.join("booter_load-570.144.bin")).unwrap();
    let parsed = firmware::signed_booter(&booter).expect("valid signed SEC2 booter");
    println!(
        "NVIDIA_BOOTER_VALID bytes={} code={:#x}+{:#x} data={:#x}+{:#x}",
        parsed.image.len(),
        parsed.code_offset,
        parsed.code_size,
        parsed.data_offset,
        parsed.data_size
    );
    let bios = fs::read(directory.join("vbios.rom")).unwrap();
    let fwsec = firmware::fwsec(&bios, (16376u64 << 20) - (2 << 20)).expect("valid VBIOS FWSEC");
    if let Ok(reference) = fs::read(directory.join("fwsec-reference.bin")) {
        assert_eq!(fwsec.image, reference);
        println!("NVIDIA_FWSEC_REFERENCE_MATCH");
    }
    println!(
        "NVIDIA_FWSEC_VALID bytes={} imem={} dmem={}",
        fwsec.image.len(),
        fwsec.descriptor.get(FalconDescriptor::IMEMLoadSize).unwrap(),
        fwsec.descriptor.get(FalconDescriptor::DMEMLoadSize).unwrap()
    );
}
