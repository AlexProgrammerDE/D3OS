//! Native indexed GPU rasterizer for the pinned doomgeneric draw interface.
use alloc::{collections::BTreeMap, vec, vec::Vec};
use core::{ffi::c_void, ptr, slice};
use d3os_compute::{
    Error, Result,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{KERNEL_COUNT, Kernel, ProfileEntry};
use spin::{Mutex, Once};

const WIDTH: usize = 320;
const HEIGHT: usize = 200;
const PIXELS: usize = WIDTH * HEIGHT;

unsafe extern "C" {
    static mut I_VideoBuffer: *mut u8;
    static mut dc_source: *const u8;
    static mut dc_colormap: *const u8;
    static mut dc_translation: *const u8;
    static mut dc_x: i32;
    static mut dc_yl: i32;
    static mut dc_yh: i32;
    static mut dc_iscale: i32;
    static mut dc_texturemid: i32;
    static mut centery: i32;
    static mut columnofs: [i32; WIDTH];
    static mut viewwindowy: i32;
    static mut viewheight: i32;
    static mut ds_y: i32;
    static mut ds_x1: i32;
    static mut ds_x2: i32;
    static mut ds_xfrac: i32;
    static mut ds_yfrac: i32;
    static mut ds_xstep: i32;
    static mut ds_ystep: i32;
    static mut ds_source: *const u8;
    static mut ds_colormap: *const u8;
    static mut fuzzpos: i32;
    static fuzzoffset: [i32; 50];
    static mut colormaps: *const u8;
    static gammatable: [[u8; 256]; 5];
    static mut usegamma: i32;

    fn __real_R_RenderPlayerView(player: *mut c_void);
    fn __real_R_DrawColumn();
    fn __real_R_DrawColumnLow();
    fn __real_R_DrawTranslatedColumn();
    fn __real_R_DrawTranslatedColumnLow();
    fn __real_R_DrawFuzzColumn();
    fn __real_R_DrawFuzzColumnLow();
    fn __real_R_DrawSpan();
    fn __real_R_DrawSpanLow();
    fn __real_I_SetPalette(palette: *mut u8);
    fn __real_I_FinishUpdate();
    fn __real_I_Quit();
}

static RENDERER: Once<Mutex<Renderer>> = Once::new();

pub(super) fn enabled() -> bool {
    RENDERER.get().is_some()
}

#[inline]
pub(super) fn cycles() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

pub(super) fn displayed(elapsed: u64) {
    if let Some(renderer) = RENDERER.get() {
        let mut renderer = renderer.lock();
        renderer.display_cycles = renderer.display_cycles.wrapping_add(elapsed);
    }
}

pub(super) fn report_profile() {
    if let Some(renderer) = RENDERER.get() {
        let renderer = renderer.lock();
        let frames = renderer.frames.max(1) as u64;
        log::info!(
            "NVIDIA_DOOM_PROFILE frames={} render_cycles_per_frame={} palette_cycles_per_frame={} display_cycles_per_frame={}",
            renderer.frames,
            renderer.render_cycles / frames,
            renderer.palette_cycles / frames,
            renderer.display_cycles / frames
        );
        if let Ok(after) = renderer.device.profile() {
            for (index, (before, after)) in renderer.profile_start.iter().zip(after).enumerate() {
                let calls = after.calls.wrapping_sub(before.calls);
                if calls != 0 {
                    let kernel = Kernel::from_raw(index as u32).expect("profile index is a built-in kernel");
                    log::info!(
                        "NVIDIA_DOOM_KERNEL_PROFILE symbol={} calls={calls} cycles_per_frame={}",
                        kernel.symbol(),
                        after.cycles.wrapping_sub(before.cycles) / frames
                    );
                }
            }
        }
        log::info!("NVIDIA_DOOM_PROFILE_DONE");
    }
}

struct Renderer {
    device: Nvidia,
    verify: bool,
    active: bool,
    fuzz: bool,
    commands: Vec<u32>,
    textures: Vec<u8>,
    snapshots: BTreeMap<(usize, usize), u32>,
    base: GpuBuffer<u32>,
    output: GpuBuffer<u32>,
    rgba: GpuBuffer<u32>,
    palette: GpuBuffer<u32>,
    pixels: Vec<u32>,
    frames: usize,
    primitives: usize,
    low_primitives: usize,
    fuzz_columns: usize,
    frame_start: i64,
    frame_start_cycles: u64,
    render_cycles: u64,
    palette_cycles: u64,
    display_cycles: u64,
    profile_start: [ProfileEntry; KERNEL_COUNT],
}

pub(super) fn init(verify: bool) -> Result<()> {
    let device = Nvidia::new()?;
    let renderer = Renderer {
        device,
        verify,
        active: false,
        fuzz: false,
        commands: Vec::new(),
        textures: Vec::new(),
        snapshots: BTreeMap::new(),
        base: device.allocate_words(PIXELS / 4)?,
        output: device.allocate_words(PIXELS / 4)?,
        rgba: device.allocate_words(PIXELS * 4)?,
        palette: device.allocate_words(256)?,
        pixels: if verify { vec![0; PIXELS / 4] } else { Vec::new() },
        frames: 0,
        primitives: 0,
        low_primitives: 0,
        fuzz_columns: 0,
        frame_start: 0,
        frame_start_cycles: 0,
        render_cycles: 0,
        palette_cycles: 0,
        display_cycles: 0,
        profile_start: device.profile()?,
    };
    RENDERER.call_once(|| Mutex::new(renderer));
    log::info!("NVIDIA_DOOM_READY backend=nvidia verify={verify} width={WIDTH} height={HEIGHT}");
    Ok(())
}

impl Renderer {
    unsafe fn screen() -> &'static mut [u8] {
        unsafe { slice::from_raw_parts_mut(I_VideoBuffer, PIXELS) }
    }

    unsafe fn screen_words() -> Result<&'static mut [u32]> {
        let pointer = unsafe { I_VideoBuffer.cast::<u32>() };
        if pointer.is_null() || !pointer.is_aligned() {
            return Err(Error::InvalidParameter);
        }
        // The pinned C engine allocates this byte surface with malloc. No C
        // drawing runs while the renderer lock owns this synchronous transfer.
        Ok(unsafe { slice::from_raw_parts_mut(pointer, PIXELS / 4) })
    }

    fn begin(&mut self) -> Result<()> {
        self.frame_start_cycles = cycles();
        self.active = true;
        self.primitives = 0;
        self.frame_start = time::systime().num_milliseconds();
        self.commands.clear();
        self.textures.clear();
        self.snapshots.clear();
        self.device.write_words(&self.base, 0, unsafe { Self::screen_words()? })
    }

    fn flush(&mut self) -> Result<()> {
        if self.commands.is_empty() {
            return Ok(());
        }
        let commands = self.device.upload_words(&self.commands)?;
        let textures = self.device.upload_bytes(&self.textures)?;
        let parameters = [WIDTH, HEIGHT, self.commands.len() / 12, self.textures.len()];
        if self.fuzz {
            self.device
                .dispatch_words(Kernel::FuzzBytes, &[(&self.base, 0), (&commands, 0), (&textures, 0)], &parameters)?;
        } else {
            let bins = d3os_compute::raster::command_bins(&self.commands, WIDTH, HEIGHT)?;
            let bin_count = bins.len();
            let bins = self.device.upload_words(&bins)?;
            self.device.dispatch_words(
                Kernel::RasterBinnedBytes,
                &[(&self.base, 0), (&commands, 0), (&textures, 0), (&self.output, 0), (&bins, 0)],
                &[parameters[0], parameters[1], parameters[2], parameters[3], bin_count],
            )?;
            core::mem::swap(&mut self.base, &mut self.output);
        }
        if self.verify {
            self.device.read_words_into(&self.base, 0, &mut self.pixels)?;
            if let Some(index) = self
                .pixels
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .zip(unsafe { Self::screen() }.iter())
                .position(|(gpu, &cpu)| gpu != cpu)
            {
                log::error!(
                    "NVIDIA_DOOM_MISMATCH frame={} pixel={index} gpu={} cpu={} fuzz={} commands={}",
                    self.frames,
                    self.pixels[index / 4].to_le_bytes()[index % 4],
                    unsafe { Self::screen()[index] },
                    self.fuzz,
                    self.commands.len() / 12
                );
                return Err(Error::DeviceFault);
            }
        }
        self.commands.clear();
        self.textures.clear();
        self.snapshots.clear();
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        self.flush()?;
        self.device.read_words_into(&self.base, 0, unsafe { Self::screen_words()? })?;
        self.active = false;
        self.frames += 1;
        self.render_cycles = self.render_cycles.wrapping_add(cycles().wrapping_sub(self.frame_start_cycles));
        if self.frames == 1 || self.frames.is_multiple_of(60) {
            log::info!(
                "NVIDIA_DOOM_FRAME_OK frame={} primitives={} verify={} render_ms={} low_primitives={} fuzz_columns={} backend=nvidia",
                self.frames,
                self.primitives,
                self.verify,
                time::systime().num_milliseconds() - self.frame_start,
                self.low_primitives,
                self.fuzz_columns
            );
        }
        Ok(())
    }

    unsafe fn snapshot(&mut self, pointer: *const u8, length: usize) -> Result<u32> {
        if pointer.is_null() || length == 0 || length > 4096 {
            return Err(Error::InvalidParameter);
        }
        let key = (pointer as usize, length);
        let bytes = unsafe { slice::from_raw_parts(pointer, length) };
        if let Some(&offset) = self.snapshots.get(&key) {
            // A zone allocation can be purged and reused even within a frame.
            if &self.textures[offset as usize..offset as usize + length] == bytes {
                return Ok(offset);
            }
        }
        let offset = u32::try_from(self.textures.len()).map_err(|_| Error::SizeOverflow)?;
        self.textures.extend_from_slice(bytes);
        self.snapshots.insert(key, offset);
        Ok(offset)
    }

    fn phase(&mut self, fuzz: bool) -> Result<()> {
        if !self.commands.is_empty() && self.fuzz != fuzz {
            self.flush()?;
        }
        self.fuzz = fuzz;
        Ok(())
    }

    unsafe fn column(&mut self, low: bool, translated: bool) -> Result<()> {
        self.phase(false)?;
        let (x, yl, yh, step, center, mid, window) = unsafe { (dc_x, dc_yl, dc_yh, dc_iscale, centery, dc_texturemid, viewwindowy) };
        if yh < yl {
            return Ok(());
        }
        let x = x
            .checked_mul(if low { 2 } else { 1 })
            .filter(|&x| (0..WIDTH as i32).contains(&x))
            .ok_or(Error::InvalidParameter)?;
        let x = unsafe { ptr::addr_of!(columnofs).cast::<i32>().add(x as usize).read() };
        let start = mid.wrapping_add(yl.wrapping_sub(center).wrapping_mul(step));
        let last = i64::from(start) + i64::from(yh - yl) * i64::from(step);
        let max_index = if translated {
            (i64::from(start) >> 16).max(last >> 16)
        } else if i64::from(start).div_euclid(128 << 16) != last.div_euclid(128 << 16) {
            127
        } else {
            ((i64::from(start) >> 16) & 127).max((last >> 16) & 127)
        };
        let length = usize::try_from(max_index + 1).map_err(|_| Error::InvalidParameter)?;
        let source = unsafe { self.snapshot(dc_source, length)? };
        let palette = unsafe { self.snapshot(dc_colormap, 256)? };
        let translation = if translated { unsafe { self.snapshot(dc_translation, 256)? } } else { 0 };
        self.commands.extend_from_slice(&[
            0,
            x as u32,
            (yl + window) as u32,
            (yh + window) as u32,
            start as u32,
            step as u32,
            source,
            length as u32,
            palette,
            translation,
            0,
            u32::from(low) | (u32::from(translated) << 1),
        ]);
        self.primitives += 1;
        self.low_primitives += usize::from(low);
        Ok(())
    }

    unsafe fn span(&mut self, low: bool) -> Result<()> {
        self.phase(false)?;
        let (x1, x2, y, xf, yf, xs, ys, window) = unsafe {
            (
                ds_x1,
                ds_x2,
                ds_y,
                ds_xfrac as u32,
                ds_yfrac as u32,
                ds_xstep as u32,
                ds_ystep as u32,
                viewwindowy,
            )
        };
        if x2 < x1 {
            return Ok(());
        }
        let x = x1
            .checked_mul(if low { 2 } else { 1 })
            .filter(|&x| (0..WIDTH as i32).contains(&x))
            .ok_or(Error::InvalidParameter)?;
        let offset = unsafe { ptr::addr_of!(columnofs).cast::<i32>().add(x as usize).read() };
        let end = offset + (x2 - x1 + 1) * if low { 2 } else { 1 } - 1;
        let source = unsafe { self.snapshot(ds_source, 4096)? };
        let palette = unsafe { self.snapshot(ds_colormap, 256)? };
        self.commands.extend_from_slice(&[
            1,
            offset as u32,
            (y + window) as u32,
            end as u32,
            ((xf << 10) & 0xffff0000) | ((yf >> 6) & 0xffff),
            ((xs << 10) & 0xffff0000) | ((ys >> 6) & 0xffff),
            source,
            4096,
            palette,
            0,
            0,
            u32::from(low),
        ]);
        self.primitives += 1;
        self.low_primitives += usize::from(low);
        // The original low-detail span routine changes these globals.
        if low {
            unsafe {
                ds_x1 <<= 1;
                ds_x2 <<= 1;
            }
        }
        Ok(())
    }

    unsafe fn fuzz(&mut self, low: bool) -> Result<()> {
        self.phase(true)?;
        unsafe {
            if dc_yl == 0 {
                dc_yl = 1;
            }
            if dc_yh == viewheight - 1 {
                dc_yh = viewheight - 2;
            }
            if dc_yh < dc_yl {
                return Ok(());
            }
            let x = dc_x
                .checked_mul(if low { 2 } else { 1 })
                .filter(|&x| (0..WIDTH as i32).contains(&x))
                .ok_or(Error::InvalidParameter)?;
            let x = ptr::addr_of!(columnofs).cast::<i32>().add(x as usize).read();
            let source = u32::try_from(self.textures.len()).map_err(|_| Error::SizeOverflow)?;
            if fuzzoffset.iter().any(|&offset| offset % WIDTH as i32 != 0) {
                return Err(Error::InvalidParameter);
            }
            self.textures.extend(fuzzoffset.iter().flat_map(|offset| offset.to_le_bytes()));
            let palette = self.snapshot(colormaps.add(6 * 256), 256)?;
            self.commands.extend_from_slice(&[
                2,
                x as u32,
                (dc_yl + viewwindowy) as u32,
                (dc_yh + viewwindowy) as u32,
                fuzzpos as u32,
                0,
                source,
                50,
                palette,
                0,
                0,
                u32::from(low),
            ]);
            if !self.verify {
                fuzzpos = (fuzzpos + dc_yh - dc_yl + 1) % 50;
            }
        }
        self.primitives += 1;
        self.low_primitives += usize::from(low);
        self.fuzz_columns += 1;
        Ok(())
    }

    unsafe fn set_palette(&mut self, palette: *const u8) -> Result<()> {
        let gamma = unsafe { usegamma.clamp(0, 4) as usize };
        let colors: Vec<u32> = unsafe { slice::from_raw_parts(palette, 768) }
            .chunks_exact(3)
            .map(|rgb| unsafe {
                (u32::from(gammatable[gamma][rgb[0] as usize]) << 16)
                    | (u32::from(gammatable[gamma][rgb[1] as usize]) << 8)
                    | u32::from(gammatable[gamma][rgb[2] as usize])
            })
            .collect();
        self.device.write_words(&self.palette, 0, &colors)
    }

    fn present(&mut self) -> Result<()> {
        let start = cycles();
        self.device.write_words(&self.base, 0, unsafe { Self::screen_words()? })?;
        self.device.dispatch_words(
            Kernel::PaletteBytes,
            &[(&self.base, 0), (&self.palette, 0), (&self.rgba, 0)],
            &[WIDTH, HEIGHT, 2],
        )?;
        let output = unsafe { slice::from_raw_parts_mut(super::DG_ScreenBuffer as *mut u32, PIXELS * 4) };
        self.device.read_words_into(&self.rgba, 0, output)?;
        self.palette_cycles = self.palette_cycles.wrapping_add(cycles().wrapping_sub(start));
        Ok(())
    }
}

fn checked(result: Result<()>) {
    if let Err(error) = result {
        log::error!("NVIDIA_DOOM_FAIL {error}");
        panic!("GPU rasterizer failed: {error}");
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_R_RenderPlayerView(player: *mut c_void) {
    if let Some(renderer) = RENDERER.get() {
        checked(renderer.lock().begin());
    }
    unsafe {
        __real_R_RenderPlayerView(player);
    }
    if let Some(renderer) = RENDERER.get() {
        checked(renderer.lock().end());
    }
}

unsafe fn draw(real: unsafe extern "C" fn(), kind: u32, low: bool, translated: bool) {
    if let Some(renderer) = RENDERER.get() {
        let mut renderer = renderer.lock();
        if renderer.active {
            checked(unsafe {
                match kind {
                    0 => renderer.column(low, translated),
                    1 => renderer.span(low),
                    _ => renderer.fuzz(low),
                }
            });
            let verify = renderer.verify;
            // The CPU reference must see the original span coordinates.
            if verify && kind == 1 && low {
                unsafe {
                    ds_x1 >>= 1;
                    ds_x2 >>= 1;
                }
            }
            drop(renderer);
            if verify {
                unsafe {
                    real();
                }
            }
            return;
        }
    }
    unsafe {
        real();
    }
}

macro_rules! column {
    ($wrapper:ident, $real:ident, $kind:expr, $low:expr, $translated:expr) => {
        #[unsafe(no_mangle)]
        unsafe extern "C" fn $wrapper() {
            unsafe {
                draw($real, $kind, $low, $translated);
            }
        }
    };
}
column!(__wrap_R_DrawColumn, __real_R_DrawColumn, 0, false, false);
column!(__wrap_R_DrawColumnLow, __real_R_DrawColumnLow, 0, true, false);
column!(__wrap_R_DrawTranslatedColumn, __real_R_DrawTranslatedColumn, 0, false, true);
column!(__wrap_R_DrawTranslatedColumnLow, __real_R_DrawTranslatedColumnLow, 0, true, true);
column!(__wrap_R_DrawSpan, __real_R_DrawSpan, 1, false, false);
column!(__wrap_R_DrawSpanLow, __real_R_DrawSpanLow, 1, true, false);
column!(__wrap_R_DrawFuzzColumn, __real_R_DrawFuzzColumn, 2, false, false);
column!(__wrap_R_DrawFuzzColumnLow, __real_R_DrawFuzzColumnLow, 2, true, false);

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_I_SetPalette(palette: *mut u8) {
    unsafe {
        __real_I_SetPalette(palette);
    }
    if let Some(renderer) = RENDERER.get() {
        checked(unsafe { renderer.lock().set_palette(palette) });
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_I_FinishUpdate() {
    if let Some(renderer) = RENDERER.get() {
        checked(renderer.lock().present());
        unsafe {
            super::DG_DrawFrame();
        }
    } else {
        unsafe {
            __real_I_FinishUpdate();
        }
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_I_Quit() -> ! {
    if let Some(renderer) = RENDERER.get() {
        log::info!("NVIDIA_DOOM_EXIT frames={}", renderer.lock().frames);
    }
    unsafe { __real_I_Quit() };
    unsafe { libc::stdlib::system::exit(0) }
}
