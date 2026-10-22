//! GPU inference operators. All pointers and dimensions are checked by D3OS.
use core::arch::asm;

/// # Safety
/// Output contains count writable words.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn buffer_zero(output: u64, count: u32) {
    let i = super::index();
    if i < count {
        unsafe {
            (output as *mut u32).add(i as usize).write(0);
        }
    }
}

/// # Safety
/// Input/output contain count elements. Exact in-place operation is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn inference_relu(input: u64, output: u64, count: u32) {
    let i = super::index();
    if i < count {
        let value = unsafe { (input as *const f32).add(i as usize).read() };
        unsafe { (output as *mut f32).add(i as usize).write(if value < 0.0 { 0.0 } else { value }) };
    }
}

/// # Safety
/// Input/output contain count values, bias contains cols values. Output may
/// exactly alias input, but cannot overlap bias.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bias_row(input: u64, bias: u64, output: u64, count: u32, cols: u32) {
    let i = super::index();
    if i < count {
        unsafe {
            (output as *mut f32)
                .add(i as usize)
                .write((input as *const f32).add(i as usize).read() + (bias as *const f32).add((i % cols) as usize).read())
        };
    }
}

/// # Safety
/// NCHW input/output contain count values, bias contains channels values.
/// Output may exactly alias input, but cannot overlap bias.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bias_channel(input: u64, bias: u64, output: u64, count: u32, spatial: u32, channels: u32) {
    let i = super::index();
    if i < count {
        unsafe {
            (output as *mut f32)
                .add(i as usize)
                .write((input as *const f32).add(i as usize).read() + (bias as *const f32).add((i / spatial % channels) as usize).read())
        };
    }
}

/// # Safety
/// Input is one CHW image, output is [channels*kh*kw, oh*ow]. Buffers do not
/// overlap. Dimensions, extents and all products are validated by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn im2col(
    input: u64, output: u64, channels: u32, height: u32, width: u32, kh: u32, kw: u32, oh: u32, ow: u32, sh: u32, sw: u32, ph: u32, pw: u32,
) {
    let i = super::index();
    let spatial = oh * ow;
    if i < channels * kh * kw * spatial {
        let column = i / spatial;
        let y = (i % spatial / ow) * sh + column / kw % kh;
        let x = (i % ow) * sw + column % kw;
        let value = match (y.checked_sub(ph), x.checked_sub(pw)) {
            (Some(y), Some(x)) if y < height && x < width => unsafe {
                (input as *const f32).add(((column / (kh * kw) * height + y) * width + x) as usize).read()
            },
            _ => 0.0,
        };
        unsafe { (output as *mut f32).add(i as usize).write(value) };
    }
}

/// # Safety
/// Disjoint input [planes,height,width] and output [planes,oh,ow].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn max_pool(
    input: u64, output: u64, planes: u32, height: u32, width: u32, kh: u32, kw: u32, sh: u32, sw: u32, oh: u32, ow: u32,
) {
    let i = super::index();
    if i < planes * oh * ow {
        let plane = i / (oh * ow);
        let y = (i / ow % oh) * sh;
        let x = (i % ow) * sw;
        let mut largest = f32::NEG_INFINITY;
        for ky in 0..kh {
            for kx in 0..kw {
                let value = unsafe { (input as *const f32).add(((plane * height + y + ky) * width + x + kx) as usize).read() };
                if value.is_nan() || value > largest {
                    largest = value;
                }
            }
        }
        unsafe { (output as *mut f32).add(i as usize).write(largest) };
    }
}

/// # Safety
/// Input/output are rows*cols floats. Exact in-place operation is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn inference_softmax(input: u64, output: u64, rows: u32, cols: u32) {
    let row = super::index();
    if row >= rows {
        return;
    }
    let mut largest = f32::NEG_INFINITY;
    for col in 0..cols {
        let value = unsafe { (input as *const f32).add((row * cols + col) as usize).read() };
        if !value.is_finite() {
            largest = f32::NAN;
            break;
        }
        if value > largest {
            largest = value;
        }
    }
    let mut sum = 0.0;
    for col in 0..cols {
        let value = unsafe { (input as *const f32).add((row * cols + col) as usize).read() } - largest;
        let exponent: f32;
        unsafe {
            asm!("ex2.approx.f32 {d}, {s};", d = out(reg32) exponent, s = in(reg32) (value * core::f32::consts::LOG2_E), options(pure, nomem, nostack));
        }
        unsafe { (output as *mut f32).add((row * cols + col) as usize).write(exponent) };
        sum += exponent;
    }
    for col in 0..cols {
        unsafe {
            let p = (output as *mut f32).add((row * cols + col) as usize);
            p.write(p.read() / sum);
        }
    }
}
