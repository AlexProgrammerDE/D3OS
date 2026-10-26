//! The actual Python stager and Rust parser must agree on the hash domain.
use d3os_nvidia::cubin::qualification::{MODULE_RECEIPT_BYTES, ModuleQualification, RuntimeInventory};
use sha2::{Digest, Sha256};
use std::{path::Path, process::Command};

#[test]
fn host_receipt_hashes_the_exact_records_exposed_to_guest_admission() {
    let stager = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools/nvidia/guest-compiler.py");
    let output = Command::new("python3")
        .args([
            "-B",
            "-c",
            r#"
import hashlib, importlib.util, struct, sys
spec = importlib.util.spec_from_file_location('guest_compiler', sys.argv[1])
stager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stager)
records = {
    'usr/share/nvidia/inference.tilebc': {'bytes': 96, 'sha256': '55' * 32},
    'usr/cuda/bin/tileiras': {'bytes': 128, 'sha256': '44' * 32},
}
canonical = stager.runtime_inventory(records)
qualification = {'source_sha256': '11' * 32, 'cubin_sha256': '22' * 32, 'sass_sha256': '33' * 32}
receipt = stager.module_receipt(qualification, 1, '44' * 32, hashlib.sha256(canonical).hexdigest())
sys.stdout.buffer.write(receipt + b'D3QRT001' + struct.pack('<I', len(records)) + canonical)
"#,
        ])
        .arg(stager)
        .output()
        .expect("Python is required for the host-staging integration test");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let receipt = ModuleQualification::parse(&output.stdout[..MODULE_RECEIPT_BYTES]).unwrap();
    let runtime = RuntimeInventory::parse(&output.stdout[MODULE_RECEIPT_BYTES..]).unwrap();
    let digest: [u8; 32] = Sha256::digest(runtime.canonical_bytes()).into();
    assert_eq!(*receipt.runtime_sha256(), digest);
    assert_eq!(runtime.records().len(), 2);
    assert_eq!(runtime.records()[0].sha256(), receipt.compiler_sha256());
    assert_ne!(digest, <[u8; 32]>::from(Sha256::digest(&output.stdout[MODULE_RECEIPT_BYTES..])));
}
