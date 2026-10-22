# Prepare the NVIDIA debug machine

These commands prepare the inspected Fedora laptop for D3OS GPU development.
Run each stage separately and check its result before continuing.
The two reboot commands end your current session.
The Hybrid-mode and VFIO steps below were verified on this machine on October 22, 2026.

The hardware-free implementation already runs pretrained AlexNet inside D3OS on the CPU.
The NVIDIA hardware driver remains to be implemented after this setup.
Read [the inference guide](inference.howto.md) to run the existing tests.

## Hardware this guide targets

- Intel graphics: `0000:00:02.0`, driver `i915`.
- RTX 4090 Laptop GPU: `0000:01:00.0`, PCI ID `10de:2757`.
- NVIDIA audio: `0000:01:00.1`, PCI ID `10de:22bb`.
- The two NVIDIA functions currently occupy IOMMU group 15 together.
- The internal panel initially used NVIDIA in `AsusMuxDgpu` mode; it now uses Intel in Hybrid mode.

These addresses and IDs are specific to this machine.
Disconnect external displays attached to NVIDIA during setup.
Keep access to the firmware GPU setting and a Linux text console for recovery.

## 1. Move the panel to Intel

Inspect the current firmware settings:

```bash
asusctl armoury get dgpu_disable
asusctl armoury get gpu_mux_mode
```

Set Hybrid mode, which keeps NVIDIA available while routing the panel through integrated graphics:

```bash
sudo asusctl armoury set dgpu_disable 0
sudo asusctl armoury set gpu_mux_mode 1
```

Save your work, then reboot:

```bash
sudo systemctl reboot
```

The [ASUS GPU switching documentation](https://opengamingcollective.github.io/asusctl/faq/gpu-switching.html) defines this pair as Hybrid mode and requires a reboot.
Do not select Integrated mode: it disables the NVIDIA device.
The installed `supergfxctl` reports only `AsusMuxDgpu`, so this guide uses the supported `asusctl armoury` interface.
If the commands fail or the settings revert, select Hybrid or Optimus in firmware and repeat the checks below.

## 2. Verify the host display and device isolation

After reboot, run:

```bash
asusctl armoury get dgpu_disable
asusctl armoury get gpu_mux_mode
lspci -nnk -s 00:02.0
lspci -nnk -s 01:00

for connector in /sys/class/drm/card*-*; do
    [ -f "$connector/enabled" ] || continue
    [ "$(cat "$connector/enabled")" = enabled ] || continue
    card="${connector##*/}"
    card="${card%%-*}"
    printf '%s -> %s\n' "${connector##*/}" \
        "$(basename "$(readlink -f "/sys/class/drm/$card/device")")"
done

readlink -f /sys/bus/pci/devices/0000:01:00.0/iommu_group
ls /sys/bus/pci/devices/0000:01:00.0/iommu_group/devices/
cat /sys/bus/pci/devices/0000:01:00.0/reset_method
```

Continue only when:

- `dgpu_disable` is 0 and `gpu_mux_mode` is 1.
- Intel uses `i915`, and each enabled display connector maps to `0000:00:02.0`.
- Both NVIDIA functions remain visible with the IDs listed above.
- Their IOMMU group contains exactly `0000:01:00.0` and `0000:01:00.1`.

The group number can change after reboot.
The previous inspection reported `flr bus` as available GPU reset methods; actual recovery still needs testing.
If the display still maps to NVIDIA, stop here and resolve the MUX setting before reserving the GPU.

## 3. Reserve NVIDIA for VFIO on the next boot

Install the VM and inspection tools:

```bash
sudo dnf install qemu-system-x86 edk2-ovmf pciutils
modinfo vfio_pci
test -c /dev/kvm
```

The machine already has IOMMU groups, so no extra IOMMU boot options are needed for this inspected configuration.
[VFIO requires ownership of the complete group](https://www.kernel.org/doc/html/latest/driver-api/vfio.html), including the audio function.
NVIDIA will be unavailable to host applications while this reservation is active.

Create a new dracut configuration file.
If the file already exists, inspect it before replacing it:

```bash
sudo test ! -e /etc/dracut.conf.d/d3os-vfio.conf && \
sudo tee /etc/dracut.conf.d/d3os-vfio.conf >/dev/null <<'EOF'
force_drivers+=" vfio_pci "
add_drivers+=" vfio vfio_iommu_type1 i915 "
EOF
```

Rebuild the running kernel's initramfs and configure that kernel's boot entry:

```bash
sudo dracut --force --kver "$(uname -r)"
sudo grubby --update-kernel="/boot/vmlinuz-$(uname -r)" \
    --args='rd.driver.pre=vfio-pci vfio-pci.ids=10de:2757,10de:22bb'
sudo grubby --info="/boot/vmlinuz-$(uname -r)"
```

[Dracut's `force_drivers` option](https://man7.org/linux/man-pages/man5/dracut.conf.5.html) loads the VFIO driver early.
The PCI ID argument reserves both NVIDIA functions.
Only the running kernel's boot entry and initramfs are changed, leaving other installed kernels as a recovery option.
Do not continue if dracut or grubby reports an error.

Save your work and reboot into the same kernel version:

```bash
sudo systemctl reboot
```

## 4. Verify the handoff

Run these commands after reboot:

```bash
lspci -nnk -s 00:02.0
lspci -nnk -s 01:00
group_path=$(readlink -f /sys/bus/pci/devices/0000:01:00.0/iommu_group)
ls "$group_path/devices/"
ls -l /dev/kvm /dev/vfio/vfio "/dev/vfio/${group_path##*/}"
sudo journalctl -b -k --grep='DMAR|IOMMU|vfio'
```

Intel must still use `i915`.
Both NVIDIA functions must report `Kernel driver in use: vfio-pci`.
Repeat the enabled-connector loop from stage 2 to confirm the host panel still uses Intel.

For an unprivileged QEMU session, grant your login user access to this specific group:

```bash
group_path=$(readlink -f /sys/bus/pci/devices/0000:01:00.0/iommu_group)
sudo chown "$(id -un)" "/dev/vfio/${group_path##*/}"
test -r /dev/kvm && test -w /dev/kvm
test -r "/dev/vfio/${group_path##*/}" && test -w "/dev/vfio/${group_path##*/}"
```

The ownership change lasts until the device node is recreated, usually at reboot.
If the KVM permission checks fail, add your user to `kvm` with `sudo usermod -aG kvm "$(id -un)"`, then log out and back in.
Keep SELinux and the IOMMU enabled.

At this point the host is ready for the next development session.
Passing the GPU through does not yet make inference run on it.
Do not use the repository's older `qemu-pci.sh` template without adapting its example addresses and permissions.

## 5. Boot D3OS with the physical GPU

VFIO must pin the VM's RAM.
The launcher checks the current process's locked-memory limit before starting QEMU.
In the terminal that will start the VM, raise the shell's limit:

```bash
sudo prlimit --pid "$$" --memlock=17179869184:17179869184
ulimit -Hl
```

The expected limit is `16777216` KiB.
Child processes inherit that limit until the shell exits.
For an agent session, use its process ID instead of `$$`.

From the repository root, run:

```bash
cargo make --no-workspace --profile production image
python3 tools/nvidia/run-qemu.py --boot-check
```

The launcher checks that the IOMMU group contains exactly the GPU and audio function and that both use `vfio-pci`.
It starts QEMU without root privileges and keeps the virtual display for the D3OS console.
The boot check requires both NVIDIA functions in the guest PCI scan, a successful native register probe, and a successful CPU inference self-test.
Success prints `NVIDIA_PASSTHROUGH_BOOT_OK`.
Read `target/nvidia/vm/serial.log` and `target/nvidia/vm/qemu.log` for diagnostics.

Without `--boot-check`, the VM stays running until stopped.
Use `--memory-mib 12288` for a VM that will load the Qwen assets.
Use `--gdb-port 1234` to pause at reset for a local debugger connection.
The kernel reports `NVIDIA_PROBE_OK` when its native register probe succeeds.
That marker confirms register access, not initialized compute queues or GPU inference.

## Restore NVIDIA to the host

Run this from the kernel entry changed in stage 3, with the VM stopped:

```bash
sudo mv /etc/dracut.conf.d/d3os-vfio.conf \
    /etc/dracut.conf.d/d3os-vfio.conf.disabled
sudo grubby --update-kernel="/boot/vmlinuz-$(uname -r)" \
    --remove-args='rd.driver.pre=vfio-pci vfio-pci.ids=10de:2757,10de:22bb'
sudo dracut --force --kver "$(uname -r)"
sudo systemctl reboot
```

Hybrid mode can remain enabled for normal host use.
To restore the original panel routing afterward, set `dgpu_disable` to 0 and `gpu_mux_mode` to 0 with `asusctl`, then reboot again.
If the desktop cannot start, use a text console or boot an unchanged kernel to repair the changed entry.
