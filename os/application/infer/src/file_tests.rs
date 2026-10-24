use super::File;
use alloc::{format, string::String, vec};
use naming::shared_types::{OpenOptions, SeekOrigin};

pub(super) fn firmware() -> Result<(), String> {
    const SIZE: usize = (2 << 20) + 137;
    let file = File(naming::open("/fwcfg/fixture", OpenOptions::READONLY).map_err(|e| format!("firmware open: {e:?}"))?);
    let size = naming::seek(file.0, 0, SeekOrigin::End).map_err(|e| format!("firmware size: {e:?}"))?;
    if size != SIZE {
        return Err(format!("firmware size {size}, expected {SIZE}"));
    }
    for (offset, length) in [
        (0, 17),
        (1, 513),
        (4093, 8197),
        ((1 << 20) - 11, 4099),
        ((2 << 20) - 13, 150),
        (17, (1 << 20) + 37),
    ] {
        naming::seek(file.0, offset as isize, SeekOrigin::Start).map_err(|e| format!("firmware seek: {e:?}"))?;
        let mut data = vec![0xcc; length + 2];
        data[0] = 0xd3;
        data[length + 1] = 0xf0;
        let count = naming::read(file.0, &mut data[1..length + 1]).map_err(|e| format!("firmware read: {e:?}"))?;
        if count == 0
            || count > length
            || count > SIZE - offset
            || data[0] != 0xd3
            || data[length + 1] != 0xf0
            || data[1 + count..length + 1].iter().any(|&b| b != 0xcc)
            || data[1..count + 1]
                .iter()
                .enumerate()
                .any(|(i, &b)| b != (((offset + i) * 37 + (offset + i) / 251) % 256) as u8)
        {
            return Err(format!("firmware data or guard mismatch at {offset}"));
        }
    }
    naming::seek(file.0, 0, SeekOrigin::End).map_err(|e| format!("firmware end: {e:?}"))?;
    let mut guard = [0xd3; 19];
    if naming::read(file.0, &mut guard).map_err(|e| format!("firmware EOF: {e:?}"))? != 0 || guard != [0xd3; 19] {
        return Err("firmware EOF changed output".into());
    }
    let output = File(naming::open("/fwcfg/fixture", OpenOptions::WRITEONLY).map_err(|e| format!("firmware write open: {e:?}"))?);
    if naming::write(output.0, &[1, 2, 3]).is_ok() {
        return Err("firmware file accepted a write".into());
    }
    let second = File(naming::open("/fwcfg/second", OpenOptions::READONLY).map_err(|e| format!("second firmware file: {e:?}"))?);
    for _ in 0..3 {
        naming::seek(second.0, 11, SeekOrigin::Start).map_err(|e| format!("second firmware seek: {e:?}"))?;
        let mut data = [0; 73];
        if naming::read(second.0, &mut data).map_err(|e| format!("second firmware read: {e:?}"))? != data.len()
            || data.iter().enumerate().any(|(i, &b)| b != (((i + 11) * 19 + 7) % 256) as u8)
        {
            return Err("firmware selectors mixed file contents".into());
        }
        naming::seek(file.0, 4093, SeekOrigin::Start).map_err(|e| format!("first firmware reseek: {e:?}"))?;
        if naming::read(file.0, &mut data).map_err(|e| format!("first firmware reread: {e:?}"))? != data.len()
            || data.iter().enumerate().any(|(i, &b)| b != (((i + 4093) * 37 + (i + 4093) / 251) % 256) as u8)
        {
            return Err("firmware selector did not restore the first file".into());
        }
    }
    super::report(format_args!(
        "INFERENCE_OK fwcfg-self-test size={SIZE} reads=12 files=2 seek=ok guards=ok readonly=ok"
    ));
    Ok(())
}
