# Native GPU rendering for DOOM

DOOM runs through native Rust raster kernels on the passed-through RTX 4090 Laptop GPU.
The renderer uses the same compute service and coherent transfers as AlexNet and Qwen.
The D3OS console framebuffer provides the final display.
The driver does not load a host CUDA runtime or a Vulkan compatibility layer.

## Engine integration

The application links the pinned `doomgeneric` engine and provides its platform callbacks in `os/application/doom/src/doom.rs`.
The engine draws walls and sprites with textured columns, and floors and ceilings with textured spans.
Its renderer writes palette indices into `I_VideoBuffer`.
The video layer converts those indices to RGB pixels and doubles the image to 640 by 400.
`DG_DrawFrame` then copies the RGB image into the boot framebuffer.

The relevant engine routines are in the pinned [column and span renderer](https://github.com/ozkl/doomgeneric/blob/fc601639494e089702a1ada082eb51aaafc03722/doomgeneric/r_draw.c).
Palette conversion and scaling are in its [video layer](https://github.com/ozkl/doomgeneric/blob/fc601639494e089702a1ada082eb51aaafc03722/doomgeneric/i_video.c).
The Rust adapter replaces scene drawing, palette conversion, and scaling.
GNU linker wrappers redirect the engine entry points without changes to the pinned submodule.
The default `doom` command retains CPU rendering.
`doom --gpu` selects native GPU rendering, and `doom --verify-gpu` also runs the CPU reference.

## Compute path

The rasterizer builds ordered lists for each screen column and row.
A pixel merges its column and row lists in command order.
This removes a scan of unrelated commands from every pixel.
The GPU checks every list offset and command ID before access.
The scan kernel remains available for operator reference checks.
See [native NVIDIA performance](nvidia-performance.md) for uncapped measurements and the Linux comparison method.


The CPU retains game logic, BSP traversal, visibility, and draw-command construction.
The Rust application snapshots texture bytes and lighting tables into owned word buffers.
Each 12-word command contains a draw type, coordinates, fixed-point texture steps, resource offsets, and flags.
Columns cover walls and sprites, including translated sprites and low-detail duplication.
Spans cover floors and ceilings with the engine's packed fixed-point coordinates.

The raster kernel assigns one GPU thread to each pixel in the 320-by-200 indexed image.
That thread processes commands in order, so later primitives overwrite earlier primitives without races.
The kernel checks resource offsets before each texture or table read.
The syscall separately validates ownership, buffer extents, products, and output aliasing.

Invisibility uses a separate fuzz kernel because each write depends on neighboring framebuffer pixels.
One GPU thread preserves the engine's sequential feedback, including overlapping and double-width columns.
The adapter resolves each batch before the draw type changes between ordinary primitives and fuzz.

Snapshots include source lengths for sprite posts and translation tables for translated sprites.
The snapshot cache compares bytes on reuse because the zone allocator can recycle an engine pointer within a frame.
GPU results return to the indexed engine buffer before the view renderer returns.
The CPU then draws HUD patches, menus, automap, and screen wipes against that synchronized image.
The GPU converts the completed indexed image to RGB and scales it to 640 by 400.
The application copies those pixels into the console framebuffer.

## Frame transfer

The inference service now transfers data through a bounded 1 MiB staging allocation in coherent guest RAM.
Native GPU copy kernels handle uploads and readback, while completion fences separate CPU access from GPU access.
The device retains the staging allocation for its complete lifetime, including dispatch errors.
The GPU also clears application buffers before the driver returns their handles.

NVIDIA's [MMU definitions](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/kernel-open/nvidia-uvm/hwref/volta/gv100/dev_mmu.h) provide the MMU-v2 system-coherent aperture.
The staging path maps owned guest physical RAM through that aperture in the existing GPU address space.
Six hardware cases pass exact-bit comparisons, output guards, partial chunks, and both sides of staging boundaries under VFIO.
The tests include NaNs and signed zero, so a floating-point comparison cannot conceal a changed payload.
Both pretrained inference models retain their full numerical checks through this transfer path.
The renderer uses that path for texture upload and frame readback.

## Verification

The operator test compares textured columns, spans, translated colors, double-width drawing, ordered overdraw, fuzz feedback, and palette scaling with CPU results.
Invalid resource offsets and output guards exercise buffer boundaries.
These tests use the raw word API, so RGB bit patterns do not pass through floating-point validation.

The game check uses checksum-pinned Freedoom 0.13.0 assets.
It compares every indexed pixel after each GPU batch in reference mode.
The input sequence exercises movement, turning, firing, low detail, automap, menus, and invisibility.
The check requires 600 rendered views and both press and release events for the main controls.
It then quits the game and runs the native GPU operator suite in the same VM.
The GPU-only check uses the same sequence without CPU scene drawing.

The frame-rate log includes transfers, game logic, and console presentation.
The `render_ms` field measures view rendering and its transfers, before HUD drawing and final presentation.
This renderer does not provide a general 3D API, hardware scanout, audio support, or persistent save files.
It uses compute shaders for the original indexed renderer rather than the GPU's triangle rasterizer.
The binned kernel visits only the relevant column and row lists for each pixel.
Higher resolutions still need separate measurements of command construction, GPU work, and framebuffer copies.

The [VFIO guide](nvidia-vfio.howto.md#9-run-doom-on-the-gpu) contains asset, validation, and interactive commands.
