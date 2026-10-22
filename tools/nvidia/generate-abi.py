#!/usr/bin/env python3
"""Generate checked Rust record schemas from a pinned NVIDIA metadata reference."""

import argparse
import ctypes
import hashlib
import subprocess
import sys
from pathlib import Path

REVISION = "81d16543a19d61c82a94df30feba2af8396275af"
METADATA_SHA256 = "261256ac4322e046ff2db28235ae2ca8f68df081af1ce1509b407e814194aed6"
ROOT = Path(__file__).resolve().parents[2]
# Only declarations are used. No host GPU driver or CUDA library is loaded.
TYPES = {
    "GspFwWprMeta": "WprMeta",
    "MESSAGE_QUEUE_INIT_ARGUMENTS": "QueueArguments",
    "GSP_ARGUMENTS_CACHED": "GspArguments",
    "LibosMemoryRegionInitArgument": "LibosRegion",
    "msgqTxHeader": "QueueHeader",
    "GSP_MSG_QUEUE_ELEMENT": "QueueElement",
    "rpc_message_header_v": "RpcHeader",
    "GspSystemInfo": "SystemInfo",
    "PACKED_REGISTRY_TABLE": "RegistryTable",
    "PACKED_REGISTRY_ENTRY": "RegistryEntry",
    "RM_RISCV_UCODE_DESC": "RiscvDescriptor",
    "struct_nvfw_bin_hdr": "BinaryHeader",
    "struct_nvfw_hs_header_v2": "SignedHeader",
    "struct_nvfw_hs_load_header_v2": "LoadHeader",
    "struct_nvfw_hs_load_header_v2_app": "LoadApp",
    "BIT_HEADER_V1_00": "BitHeader",
    "BIT_DATA_FALCON_DATA_V2": "FalconData",
    "FALCON_UCODE_TABLE_HDR_V1": "FalconTableHeader",
    "FALCON_UCODE_TABLE_ENTRY_V1": "FalconTableEntry",
    "FALCON_UCODE_DESC_V3": "FalconDescriptor",
    "FALCON_APPLICATION_INTERFACE_HEADER_V1": "InterfaceHeader",
    "FALCON_APPLICATION_INTERFACE_ENTRY_V1": "InterfaceEntry",
    "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3": "DmemMapper",
    "FWSECLIC_READ_VBIOS_DESC": "ReadVbios",
    "FWSECLIC_FRTS_REGION_DESC": "FrtsRegion",
    "FWSECLIC_FRTS_CMD": "FrtsCommand",
}
TYPES.update(
    {
        "rpc_rc_triggered_v": "RcError",
        "NV2080_CTRL_PERF_BOOST_PARAMS": "PerfBoost",
        "NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS": "GrInfo",
        "struct_NV2080_CTRL_INTERNAL_GR_INFO": "GrInfoEntry",
        "rpc_gsp_rm_alloc_v": "RmAlloc",
        "rpc_gsp_rm_control_v": "RmControl",
        "rpc_set_page_directory_v": "SetPageDirectory",
        "struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05": "PageDirectory",
        "NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS": "VirtualMemoryAlloc",
        "NV0000_ALLOC_PARAMETERS": "RootAlloc",
        "NV0080_ALLOC_PARAMETERS": "DeviceAlloc",
        "NV2080_ALLOC_PARAMETERS": "SubdeviceAlloc",
        "NV_VASPACE_ALLOCATION_PARAMETERS": "VaspaceAlloc",
        "NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS": "DeviceInfoTable",
        "NV2080_CTRL_FIFO_DEVICE_ENTRY": "DeviceInfoEntry",
        "struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS": "ReservedPdes",
        "struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_level": "ReservedLevel",
        "NV_MEMORY_DESC_PARAMS": "MemoryDescriptor",
        "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS": "FifoAlloc",
        "NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS": "ContextInfo",
        "struct_NV2080_CTRL_INTERNAL_ENGINE_CONTEXT_BUFFER_INFO": "ContextBufferInfo",
        "NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS": "PromoteContext",
        "NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY": "PromoteEntry",
        "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS": "WorkToken",
        "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS": "ChannelGroupAlloc",
        "NV_CTXSHARE_ALLOCATION_PARAMETERS": "ContextShareAlloc",
        "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS": "GroupSchedule",
        "NVA06F_CTRL_BIND_PARAMS": "ChannelBind",
    }
)
RM_CONSTANTS = [
    "NVC6C0_SET_OBJECT",
    "NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A",
    "NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A",
    "NVC6C0_SEND_PCAS_A",
    "NVC6C0_SEND_SIGNALING_PCAS2_B",
    "NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
    "NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR",
    "NV2080_CTRL_CMD_PERF_BOOST",
    "NVC6C0_SET_SHADER_LOCAL_MEMORY_A",
    "NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A",
    "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO",
    "NV2080_CTRL_GR_INFO_INDEX_SM_VERSION",
    "NV01_MEMORY_VIRTUAL",
    "NV01_ROOT",
    "NV01_DEVICE_0",
    "NV20_SUBDEVICE_0",
    "FERMI_VASPACE_A",
    "AMPERE_CHANNEL_GPFIFO_A",
    "ADA_COMPUTE_A",
    "AMPERE_DMA_COPY_B",
    "NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE",
    "NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES",
    "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO",
    "NV2080_CTRL_CMD_GPU_PROMOTE_CTX",
    "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN",
    "NVA06F_CTRL_CMD_BIND",
    "NVA06F_CTRL_CMD_GPFIFO_SCHEDULE",
    "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE",
    "KEPLER_CHANNEL_GROUP_A",
    "FERMI_CONTEXT_SHARE_A",
    "NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING",
    "NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED",
    "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC",
]
CONSTANTS = [
    "GSP_FW_WPR_META_REVISION",
    "GSP_FW_WPR_META_MAGIC",
    "NV_VGPU_MSG_SIGNATURE_VALID",
    "NV_VGPU_MSG_RESULT_RPC_PENDING",
    "NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO",
    "NV_VGPU_MSG_FUNCTION_SET_REGISTRY",
    "NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_GET_WORK_SUBMIT_TOKEN",
    "NV_VGPU_MSG_FUNCTION_CTRL_GPU_PROMOTE_CTX",
    "NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY",
    "NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD",
    "NV_VGPU_MSG_EVENT_GSP_INIT_DONE",
    "NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER",
    "NV_VGPU_MSG_EVENT_OS_ERROR_LOG",
    "NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED",
    "NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC",
    "NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL",
    "NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER",
    "REGISTRY_TABLE_ENTRY_TYPE_DWORD",
    "FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER",
    "BIT_TOKEN_FALCON_DATA",
    "FALCON_UCODE_ENTRY_APPID_FWSEC_PROD",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=f"tinygrad source archive at {REVISION}",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    metadata = source / "tinygrad/runtime/autogen/nv.py"
    digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    if digest != METADATA_SHA256:
        parser.error("NVIDIA metadata does not match the pinned source revision")
    rm_digest = hashlib.sha256(
        (source / "tinygrad/runtime/autogen/nv_570.py").read_bytes()
    ).hexdigest()
    if rm_digest != "411e74f3a283d3f4bed0294fb6a54a34ddf6fe624a3db8ec58df97ab66909abf":
        parser.error("RM metadata does not match the pinned source revision")
    sys.path.insert(0, str(source))
    from tinygrad.runtime.autogen import nv
    from tinygrad.runtime.autogen import nv_570 as rm

    output = [
        f"// Generated by tools/nvidia/generate-abi.py from tinygrad {REVISION}.",
        f"// NVIDIA metadata SHA-256: {digest}",
        f"// RM 570 metadata SHA-256: {rm_digest}",
        "// See LICENSE.tinygrad for the reference metadata license.",
        "#![allow(non_upper_case_globals)]",
        "use crate::{Bits, Field};",
    ]
    integer_types = {
        ctypes.c_uint8: "u8",
        ctypes.c_uint16: "u16",
        ctypes.c_uint32: "u32",
        ctypes.c_uint64: "u64",
        ctypes.c_int8: "i8",
        ctypes.c_int16: "i16",
        ctypes.c_int32: "i32",
        ctypes.c_int64: "i64",
    }
    for name, rust_name in TYPES.items():
        record = getattr(nv, name, None) or getattr(rm, name)
        output.extend(
            [
                f"pub struct {rust_name};",
                f"impl {rust_name} {{",
                f"    pub const SIZE: usize = {ctypes.sizeof(record)};",
            ]
        )
        for field, typ, offset, *bits in record._real_fields_:
            if bits:
                raise ValueError(
                    f"bit field requires explicit encoding: {name}.{field}"
                )
            size = ctypes.sizeof(typ)
            if offset + size > ctypes.sizeof(record):
                raise ValueError(f"field outside record: {name}.{field}")
            rust_type = integer_types.get(typ, f"[u8; {size}]")
            identifier = "r#type" if field == "type" else field
            output.append(
                f"    pub const {identifier}: Field<{rust_type}> = Field::new({offset});"
            )
        output.append("}")
    output.extend(["pub struct QmdFields;", "impl QmdFields {"])
    prefix = "NVC6C0_QMDV03_00_"
    for name, value in vars(rm).items():
        if not name.startswith(prefix):
            continue
        if isinstance(value, tuple):
            high, low = value
            output.append(
                f"    pub const {name[len(prefix) :]}: Bits = Bits::new({high}, {low});"
            )
        elif callable(value):
            for index in range(8):
                high, low = value(index)
                output.append(
                    f"    pub const {name[len(prefix) :]}_{index}: Bits = Bits::new({high}, {low});"
                )
    output.append("}")
    for name in CONSTANTS + RM_CONSTANTS:
        value = getattr(rm if name in RM_CONSTANTS else nv, name)
        ty = "u64" if value > 0xFFFFFFFF else "u32"
        output.append(f"pub const {name}: {ty} = {value:#x};")
    path = ROOT / "os/library/nvidia/src/abi.rs"
    path.write_text("\n".join(output) + "\n")
    subprocess.run(["rustfmt", "--edition", "2024", str(path)], check=True)
    (path.parent.parent / "LICENSE.tinygrad").write_bytes(
        (source / "LICENSE").read_bytes()
    )
    print(f"Generated {len(TYPES)} record schemas: {path}")


if __name__ == "__main__":
    main()
