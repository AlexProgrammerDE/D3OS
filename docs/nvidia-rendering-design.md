# Native GPU rendering for DOOM

The current compute driver provides a feasible path to GPU rasterization for the existing DOOM application.
This path uses native Rust compute kernels and the existing D3OS console framebuffer.
It does not require a CUDA host runtime or a Vulkan compatibility layer.
This document describes the implementation path, not a completed renderer.

## Current renderer

The application links the pinned `doomgeneric` engine and provides its platform callbacks in `os/application/doom/src/doom.rs`.
The engine draws walls and sprites with textured columns, and floors and ceilings with textured spans.
Its renderer writes palette indices into `I_VideoBuffer`.
The video layer converts those indices to RGB pixels and doubles the image to 640 by 400.
`DG_DrawFrame` then copies the RGB image into the boot framebuffer.

The relevant engine routines are in the pinned [column and span renderer](https://github.com/ozkl/doomgeneric/blob/fc601639494e089702a1ada082eb51aaafc03722/doomgeneric/r_draw.c).
Palette conversion and scaling are in its [video layer](https://github.com/ozkl/doomgeneric/blob/fc601639494e089702a1ada082eb51aaafc03722/doomgeneric/i_video.c).
Moving only that conversion to the GPU leaves scene rasterization on the CPU.
A GPU renderer must replace the column and span drawing loops too.

## Compute path

The CPU retains game logic, BSP traversal, visibility, and draw-command construction.
The Rust application submits ordered column and span commands with texture and lighting-table handles.
A compute kernel resolves each output pixel from those commands and samples the corresponding texture.
Each output pixel has one GPU writer, which preserves command order without concurrent overdraw writes.
The first implementation can use the original 320-by-200 indexed image before RGB conversion and scaling.

The command format must include fixed-point texture coordinates and steps, clipping ranges, texture bounds, and lighting tables.
The syscall must validate every command and referenced resource before dispatch.
Queued primitives need owned texture snapshots or resources that remain valid until the frame completes.
An engine pointer alone cannot identify a persistent texture because the zone allocator can purge and reuse its storage.
Sprite posts also need explicit source lengths instead of an unconditional 128-byte column read.
Translated sprites need their translation table in addition to the lighting table.
Fuzz effects read neighboring pixels and require a separate ordering rule.
HUD patches, menus, screen wipes, and screenshot reads also need explicit synchronization with the rendered view.

The integration must keep the upstream submodule unchanged.
Native build adapters can redirect the relevant draw entry points to the Rust renderer.
The application must retain a selectable CPU renderer for comparison and machines without native GPU support.
That selection must remain explicit in diagnostics and performance results.
The GPU path must resolve pending drawing before a CPU routine reads or modifies the indexed image.

## Frame transfer

The current inference service transfers data with uncached CPU accesses through BAR1.
Those transfers already dominate model loading and are unsuitable for frequent frame readback.
The next driver requirement is a bounded staging allocation in guest RAM that the GPU can access.
The driver must preserve that allocation until all submitted GPU work completes.
CPU and GPU ownership must change only at explicit completion fences.

NVIDIA's [MMU definitions](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/kernel-open/nvidia-uvm/hwref/volta/gv100/dev_mmu.h) provide the MMU-v2 system-coherent aperture.
The candidate staging path maps owned guest physical RAM through that aperture in the existing GPU address space.
Hardware tests must establish cache visibility, correct addresses under VFIO, bounds, and completion before this path handles frames.
The same path can accelerate inference uploads and downloads.

## Acceptance checks

1. Compare GPU column and span output with the CPU renderer, including clipping and fixed-point wraparound.
2. Compare captured game frames with translated sprites, fuzz, HUD, menus, and wipes.
3. Measure frame time with transfers and presentation included.
4. Run a timed demo, then verify interactive movement, turning, and firing in the D3OS VM.
5. Verify cleanup after game exit and after a GPU dispatch error.

The renderer also needs a reproducible WAD fixture for automated tests.
No GPU DOOM execution or frame-rate claim follows from the current inference tests.
