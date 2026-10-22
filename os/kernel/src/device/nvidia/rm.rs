//! Native GSP resource management and compute-channel construction.

use super::{
    Device, Error, Result,
    mmu::{Allocation, Memory},
};
use alloc::vec::Vec;
use d3os_nvidia::{Record, abi::*, context};
use log::info;

const PRIVILEGED_CLIENT: u32 = 0xc1e00004;
const USER_CLIENT: u32 = 0xc1000000;

struct ContextBuffer {
    id: u16,
    allocation: Allocation,
    physical: bool,
    virtual_address: bool,
}

pub(super) struct Compute {
    pub(super) memory: Memory,
    handle: u32,
    contexts: Vec<ContextBuffer>,
    pub channel: u32,
    pub token: u32,
    pub fifo: Allocation,
    pub(super) entries: usize,
    pub(super) sass_version: u64,
    pub(super) notifier: u64,
    pub(super) instance: u64,
    runlist: u32,
    pub(super) local_bytes_per_tpc: u64,
    pub(super) local_memory: Option<Allocation>,
    reserved_virtual: u64,
    pub(super) command_buffer: Option<Allocation>,
    pub(super) put: usize,
}

impl Device {
    fn rpc(&mut self, function: u32, payload: &[u8]) -> Result<Vec<u8>> {
        let mut gsp = self.gsp.take().ok_or(Error::Firmware)?;
        let result = gsp.call(self, function, payload);
        self.gsp = Some(gsp);
        result
    }
}

impl Compute {
    pub(super) fn new(device: &Device) -> Result<Self> {
        let mut memory = Memory::new(device)?;
        let fifo = memory.allocate(device, 4096, 4096)?;
        Ok(Self {
            memory,
            handle: 0xcf000000,
            contexts: Vec::new(),
            channel: 0,
            token: 0,
            fifo,
            entries: 32,
            sass_version: 0,
            notifier: 0,
            instance: 0,
            runlist: 0,
            local_bytes_per_tpc: 0,
            local_memory: None,
            reserved_virtual: 0,
            command_buffer: None,
            put: 0,
        })
    }

    fn alloc(&mut self, device: &mut Device, client: u32, parent: u32, class: u32, parameters: &[u8]) -> Result<u32> {
        let handle = self.handle;
        self.handle = self.handle.checked_add(1).ok_or(Error::Firmware)?;
        let mut args = Record::<{ RmAlloc::SIZE }>::default();
        args.set(RmAlloc::hClient, client)?;
        args.set(RmAlloc::hParent, parent)?;
        args.set(RmAlloc::hObject, handle)?;
        args.set(RmAlloc::hClass, class)?;
        args.set(RmAlloc::paramsSize, parameters.len() as u32)?;
        let mut payload = args.as_bytes().to_vec();
        payload.extend_from_slice(parameters);
        info!("NVIDIA RM allocate class={class:#x} parent={parent:#x} handle={handle:#x}");
        let response = device.rpc(NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, &payload)?;
        let response = Record::<{ RmAlloc::SIZE }>::from_bytes(&response)?;
        let status = response.get(RmAlloc::status)?;
        if status != 0 {
            return Err(Error::Rpc {
                function: class,
                result: status,
            });
        }
        Ok(if class == NV01_ROOT { client } else { handle })
    }

    fn control<const N: usize>(&mut self, device: &mut Device, client: u32, object: u32, command: u32, parameters: &Record<N>) -> Result<Record<N>> {
        let mut args = Record::<{ RmControl::SIZE }>::default();
        args.set(RmControl::hClient, client)?;
        args.set(RmControl::hObject, object)?;
        args.set(RmControl::cmd, command)?;
        args.set(RmControl::paramsSize, N as u32)?;
        let mut payload = args.as_bytes().to_vec();
        payload.extend_from_slice(parameters.as_bytes());
        info!("NVIDIA RM control command={command:#x} object={object:#x}");
        let response = device.rpc(NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, &payload)?;
        let header = Record::<{ RmControl::SIZE }>::from_bytes(&response)?;
        let status = header.get(RmControl::status)?;
        if status != 0 {
            return Err(Error::Rpc {
                function: command,
                result: status,
            });
        }
        Ok(Record::<N>::from_bytes(response.get(RmControl::SIZE..).ok_or(Error::Firmware)?)?)
    }

    fn descriptor(physical: u64, bytes: u64) -> Result<[u8; MemoryDescriptor::SIZE]> {
        let mut desc = Record::<{ MemoryDescriptor::SIZE }>::default();
        desc.set(MemoryDescriptor::base, physical)?;
        desc.set(MemoryDescriptor::size, bytes)?;
        desc.set(MemoryDescriptor::addressSpace, 2)?;
        Ok(desc.into_bytes())
    }

    fn fifo_parameters(&mut self, device: &Device, fifo: Allocation, entries: u32) -> Result<Record<{ FifoAlloc::SIZE }>> {
        let instance = self.memory.allocate_physical(device, 4096, 4096)?;
        self.instance = instance;
        let methods = self.memory.allocate_physical(device, 0x5000, 4096)?;
        let mut params = Record::<{ FifoAlloc::SIZE }>::default();
        params.set(FifoAlloc::gpFifoOffset, fifo.virtual_address)?;
        params.set(FifoAlloc::gpFifoEntries, entries)?;
        params.set(FifoAlloc::instanceMem, Self::descriptor(instance, 4096)?)?;
        params.set(FifoAlloc::ramfcMem, Self::descriptor(instance, 0x200)?)?;
        params.set(FifoAlloc::mthdbufMem, Self::descriptor(methods, 0x5000)?)?;
        params.set(FifoAlloc::userdMem, Self::descriptor(fifo.physical + u64::from(entries) * 8, 0x400)?)?;
        let mut offsets = [0u8; 64];
        offsets[..8].copy_from_slice(&(u64::from(entries) * 8).to_le_bytes());
        params.set(FifoAlloc::userdOffset, offsets)?;
        Ok(params)
    }

    fn promote(&mut self, device: &mut Device, client: u32, subdevice: u32, channel: u32, ids: &[u16], physical: bool, virtual_address: bool) -> Result<()> {
        let mut params = Record::<{ PromoteContext::SIZE }>::default();
        params.set(PromoteContext::engineType, 1)?;
        // hClient describes the optional hVirtMemory handle. Direct buffer
        // entries identify the channel through hChanClient and hObject.
        params.set(PromoteContext::hChanClient, client)?;
        params.set(PromoteContext::hObject, channel)?;
        params.set(PromoteContext::entryCount, ids.len() as u32)?;
        let mut entries = [0u8; 512];
        for (i, id) in ids.iter().enumerate() {
            let buffer = self.contexts.iter().find(|buffer| buffer.id == *id).ok_or(Error::Firmware)?;
            let pa = physical && buffer.physical;
            let va = virtual_address && buffer.virtual_address;
            let mut entry = Record::<{ PromoteEntry::SIZE }>::default();
            entry.set(PromoteEntry::bufferId, *id)?;
            entry.set(PromoteEntry::bInitialize, u8::from(pa))?;
            entry.set(PromoteEntry::bNonmapped, u8::from(pa && !va))?;
            entry.set(PromoteEntry::gpuPhysAddr, if pa { buffer.allocation.physical } else { 0 })?;
            entry.set(PromoteEntry::gpuVirtAddr, if va { buffer.allocation.virtual_address } else { 0 })?;
            entry.set(PromoteEntry::size, if pa { buffer.allocation.bytes } else { 0 })?;
            entry.set(PromoteEntry::physAttr, if pa { 4 } else { 0 })?;
            entries[i * PromoteEntry::SIZE..(i + 1) * PromoteEntry::SIZE].copy_from_slice(entry.as_bytes());
        }
        params.set(PromoteContext::promoteEntry, entries)?;
        self.control(device, client, subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, &params)?;
        Ok(())
    }

    fn copy_reserved(&mut self, device: &mut Device, client: u32, vaspace: u32, reserved: u64) -> Result<()> {
        let tables = self.memory.page_tables(device, reserved, 3)?;
        let mut levels = [0u8; 144];
        for (i, physical) in tables.into_iter().enumerate() {
            let mut level = Record::<{ ReservedLevel::SIZE }>::default();
            level.set(ReservedLevel::physAddress, physical)?;
            level.set(ReservedLevel::size, if i == 0 { 32 } else { 4096 })?;
            level.set(ReservedLevel::pageShift, [47, 38, 29][i])?;
            level.set(ReservedLevel::aperture, 1)?;
            levels[i * ReservedLevel::SIZE..(i + 1) * ReservedLevel::SIZE].copy_from_slice(level.as_bytes());
        }
        let mut params = Record::<{ ReservedPdes::SIZE }>::default();
        params.set(ReservedPdes::pageSize, 512 << 20)?;
        params.set(ReservedPdes::virtAddrLo, reserved)?;
        params.set(ReservedPdes::virtAddrHi, reserved + (512 << 20) - 1)?;
        params.set(ReservedPdes::numLevelsToCopy, 3)?;
        params.set(ReservedPdes::levels, levels)?;
        self.control(device, client, vaspace, NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, &params)?;

        Ok(())
    }

    pub(super) fn initialize(&mut self, device: &mut Device) -> Result<()> {
        device.registers.write(0x1704, 0)?;
        let client = PRIVILEGED_CLIENT;
        self.alloc(device, client, 0, NV01_ROOT, Record::<{ RootAlloc::SIZE }>::default().as_bytes())?;
        let mut params = Record::<{ DeviceAlloc::SIZE }>::default();
        params.set(DeviceAlloc::hClientShare, client)?;
        let dev = self.alloc(device, client, client, NV01_DEVICE_0, params.as_bytes())?;
        let subdev = self.alloc(device, client, dev, NV20_SUBDEVICE_0, Record::<{ SubdeviceAlloc::SIZE }>::default().as_bytes())?;
        let vaspace = self.alloc(device, client, dev, FERMI_VASPACE_A, Record::<{ VaspaceAlloc::SIZE }>::default().as_bytes())?;
        let table = self.control(
            device,
            client,
            subdev,
            NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE,
            &Record::<{ DeviceInfoTable::SIZE }>::default(),
        )?;
        let entries = table.get(DeviceInfoTable::entries)?;
        for entry in entries
            .chunks_exact(DeviceInfoEntry::SIZE)
            .take(table.get(DeviceInfoTable::numEntries)? as usize)
        {
            let entry = Record::<{ DeviceInfoEntry::SIZE }>::from_bytes(entry)?;
            let words = entry.get(DeviceInfoEntry::engineData)?;
            let engine = u32::from_le_bytes(words[8..12].try_into().map_err(|_| Error::Firmware)?);
            let runlist = u32::from_le_bytes(words[12..16].try_into().map_err(|_| Error::Firmware)?);
            info!("NVIDIA engine={engine} runlist={runlist}");
            if engine == 1 {
                info!("NVIDIA graphics engine data={words:02x?} pbdma={:02x?}", entry.get(DeviceInfoEntry::pbdmaIds)?);
                self.runlist = runlist;
            }
        }
        let reserved = self.memory.reserve_virtual(512 << 20, 512 << 20)?;
        self.reserved_virtual = reserved;
        self.copy_reserved(device, client, vaspace, reserved)?;

        let golden_fifo = self.fifo;
        let mut params = self.fifo_parameters(device, golden_fifo, 32)?;
        params.set(FifoAlloc::hVASpace, vaspace)?;
        params.set(FifoAlloc::engineType, 1)?;
        params.set(FifoAlloc::cid, 3)?;
        params.set(FifoAlloc::internalFlags, 0x1a)?;
        params.set(FifoAlloc::flags, 0x200320)?;
        params.set(FifoAlloc::userdMem, Self::descriptor(golden_fifo.physical + 32 * 8, 0x20)?)?;
        let channel = self.alloc(device, client, dev, AMPERE_CHANNEL_GPFIFO_A, params.as_bytes())?;
        let context_info = self.control(
            device,
            client,
            subdev,
            NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO,
            &Record::<{ ContextInfo::SIZE }>::default(),
        )?;
        let raw = context_info.get(ContextInfo::engineContextBuffersInfo)?;
        for (id, index, add, physical, virtual_address) in [
            (0, 0, 0x40000, true, true),
            (2, 16, 0, true, true),
            (3, 17, 0, false, true),
            (4, 18, 0, false, true),
            (5, 19, 0, false, true),
            (6, 20, 0, false, true),
            (9, 23, 0, true, true),
            (10, 24, 0, true, false),
            (11, 24, 0, true, true),
        ] {
            let buffer_info = Record::<{ ContextBufferInfo::SIZE }>::from_bytes(&raw[index * ContextBufferInfo::SIZE..])?;
            let alignment = if id == 5 {
                2 << 20
            } else {
                u64::from(buffer_info.get(ContextBufferInfo::alignment)?).max(4096)
            };
            let size = u64::from(buffer_info.get(ContextBufferInfo::size)?).checked_add(add).ok_or(Error::Alignment)?;
            let bytes = size.checked_add(alignment - 1).ok_or(Error::Alignment)? & !(alignment - 1);
            info!("NVIDIA context id={id} bytes={bytes:#x} alignment={alignment:#x}");
            let allocation = self.memory.allocate(device, bytes, alignment)?;
            self.contexts.push(ContextBuffer {
                id,
                allocation,
                physical,
                virtual_address,
            });
        }
        self.promote(device, client, subdev, channel, &[0, 2, 3, 4, 5, 6, 9, 10, 11], true, true)?;
        self.alloc(device, client, channel, ADA_COMPUTE_A, &[])?;
        self.alloc(device, client, channel, AMPERE_DMA_COPY_B, &[])?;
        info!("NVIDIA_GOLDEN_CONTEXT_OK");
        self.initialize_user(device)
    }

    fn initialize_user(&mut self, device: &mut Device) -> Result<()> {
        let client = USER_CLIENT;
        self.alloc(device, client, 0, NV01_ROOT, Record::<{ RootAlloc::SIZE }>::default().as_bytes())?;
        let mut params = Record::<{ DeviceAlloc::SIZE }>::default();
        params.set(DeviceAlloc::hClientShare, client)?;
        let dev = self.alloc(device, client, client, NV01_DEVICE_0, params.as_bytes())?;
        let subdev = self.alloc(device, client, dev, NV20_SUBDEVICE_0, Record::<{ SubdeviceAlloc::SIZE }>::default().as_bytes())?;
        let info = self.control(
            device,
            client,
            subdev,
            NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO,
            &Record::<{ GrInfo::SIZE }>::default(),
        )?;
        let raw = info.get(GrInfo::engineInfo)?;
        let sm =
            Record::<{ GrInfoEntry::SIZE }>::from_bytes(&raw[NV2080_CTRL_GR_INFO_INDEX_SM_VERSION as usize * GrInfoEntry::SIZE..])?.get(GrInfoEntry::data)?;
        self.sass_version = u64::from(((sm & 0xf00) >> 4) | (sm & 0xf));
        info!("NVIDIA shader SM version={sm:#x} SASS={:#x}", self.sass_version);
        let value = |index: usize| -> Result<u64> {
            Ok(u64::from(
                Record::<{ GrInfoEntry::SIZE }>::from_bytes(&raw[index * GrInfoEntry::SIZE..])?.get(GrInfoEntry::data)?,
            ))
        };
        let (warps, gpcs, tpcs, sms) = (value(13)?, value(20)?, value(23)?, value(32)?);
        self.local_bytes_per_tpc = (0x240 * 32 * warps * sms + 0x7fff) & !0x7fff;
        let local_bytes = (self.local_bytes_per_tpc * gpcs * tpcs + 0x1ffff) & !0x1ffff;
        info!("NVIDIA topology gpcs={gpcs} tpcs_per_gpc={tpcs} sm_per_tpc={sms} warps_per_sm={warps} local_bytes={local_bytes:#x}");
        self.local_memory = Some(self.memory.allocate(device, local_bytes, 0x20000)?);
        let mut boost = Record::<{ PerfBoost::SIZE }>::default();
        boost.set(PerfBoost::duration, u32::MAX)?;
        boost.set(PerfBoost::flags, 0x52)?;
        self.control(device, client, subdev, NV2080_CTRL_CMD_PERF_BOOST, &boost)?;

        let mut virtual_memory = Record::<{ VirtualMemoryAlloc::SIZE }>::default();
        virtual_memory.set(VirtualMemoryAlloc::limit, 0x1ffffffffffff)?;
        self.alloc(device, client, dev, NV01_MEMORY_VIRTUAL, virtual_memory.as_bytes())?;
        let mut params = Record::<{ VaspaceAlloc::SIZE }>::default();
        params.set(VaspaceAlloc::vaBase, 4096)?;
        params.set(VaspaceAlloc::vaSize, 0x1fffffb000000)?;
        params.set(VaspaceAlloc::flags, NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)?;
        let vaspace = self.alloc(device, client, dev, FERMI_VASPACE_A, params.as_bytes())?;
        let mut directory = Record::<{ PageDirectory::SIZE }>::default();
        directory.set(PageDirectory::physAddress, self.memory.root)?;
        directory.set(PageDirectory::numEntries, 4)?;
        directory.set(PageDirectory::flags, 8)?;
        directory.set(PageDirectory::hVASpace, vaspace)?;
        directory.set(PageDirectory::subDeviceId, 1)?;
        directory.set(PageDirectory::pasid, u32::MAX)?;
        let mut request = Record::<{ SetPageDirectory::SIZE }>::default();
        request.set(SetPageDirectory::hClient, client)?;
        request.set(SetPageDirectory::hDevice, dev)?;
        request.set(SetPageDirectory::pasid, u32::MAX)?;
        request.set(SetPageDirectory::params, directory.into_bytes())?;
        device.rpc(NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, request.as_bytes())?;
        let mut group_params = Record::<{ ChannelGroupAlloc::SIZE }>::default();
        group_params.set(ChannelGroupAlloc::engineType, 1)?;
        group_params.set(ChannelGroupAlloc::hVASpace, vaspace)?;
        let group = self.alloc(device, client, dev, KEPLER_CHANNEL_GROUP_A, group_params.as_bytes())?;
        let mut share_params = Record::<{ ContextShareAlloc::SIZE }>::default();
        share_params.set(ContextShareAlloc::hVASpace, vaspace)?;
        share_params.set(ContextShareAlloc::flags, 0)?;
        let share = self.alloc(device, client, group, FERMI_CONTEXT_SHARE_A, share_params.as_bytes())?;
        self.fifo = self.memory.allocate(device, 0x10000, 4096)?;
        let mut params = self.fifo_parameters(device, self.fifo, 4096)?;
        params.set(FifoAlloc::hContextShare, share)?;
        // Kernel-owned inference queue, VRAM notifier, and no ECC notifier.
        params.set(FifoAlloc::internalFlags, 2 | (3 << 2) | (1 << 4))?;
        params.set(FifoAlloc::hPhysChannelGroup, group)?;
        let notifier = self.memory.allocate(device, 4096, 4096)?;
        self.notifier = notifier.physical;
        params.set(FifoAlloc::hObjectError, notifier.physical as u32)?;
        params.set(FifoAlloc::errorNotifierMem, Self::descriptor(notifier.physical, 4096)?)?;
        let channel = self.alloc(device, client, group, AMPERE_CHANNEL_GPFIFO_A, params.as_bytes())?;
        let mut bind = Record::<{ ChannelBind::SIZE }>::default();
        bind.set(ChannelBind::engineType, 1)?;
        self.control(device, client, channel, NVA06F_CTRL_CMD_BIND, &bind)?;
        self.alloc(device, client, channel, ADA_COMPUTE_A, &[])?;
        self.initialize_context_image(device)?;
        self.copy_reserved(device, client, vaspace, self.reserved_virtual)?;
        let mut token_request = Record::<{ WorkToken::SIZE }>::default();
        token_request.set(WorkToken::workSubmitToken, u32::MAX)?;
        let response = self.control(device, client, channel, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &token_request)?;
        self.entries = 4096;
        self.token = response.get(WorkToken::workSubmitToken)? | (self.runlist << 16);
        self.channel = channel;
        self.dump_instance(device)?;
        let mut schedule = Record::<{ GroupSchedule::SIZE }>::default();
        schedule.set(GroupSchedule::bEnable, 1)?;
        self.control(device, client, group, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, &schedule)?;
        info!("NVIDIA_COMPUTE_CHANNEL_OK channel={channel:#x} token={:#x}", self.token);
        Ok(())
    }

    pub(super) fn dump_instance(&self, device: &Device) -> Result<()> {
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        for offset in [0xac, 0x210, 0x214, 0x218] {
            info!("NVIDIA instance offset={offset:#x} value={:?}", vram.read(self.instance as usize + offset));
        }
        Ok(())
    }

    fn initialize_context_image(&mut self, device: &Device) -> Result<()> {
        // Golden global buffers remain mapped in this GPU address space. Clone
        // the engine state, relocate its self pointer, then bind the private image.
        let template = self.contexts.iter().find(|buffer| buffer.id == 0).ok_or(Error::Firmware)?.allocation;
        let image = self.memory.allocate(device, template.bytes, 4096)?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        let header = context::relocate_header(
            &vram.read_bytes(template.physical as usize, context::HEADER_BYTES)?,
            image.bytes,
            image.virtual_address,
        )?;
        for offset in (0..image.bytes).step_by(4096) {
            let page = vram.read_bytes((template.physical + offset) as usize, 4096)?;
            vram.copy_from((image.physical + offset) as usize, &page)?;
        }
        vram.copy_from(image.physical as usize, &header)?;
        let mut instance = Record::<{ context::INSTANCE_BYTES }>::from_bytes(&vram.read_bytes(self.instance as usize, context::INSTANCE_BYTES)?)?;
        context::bind_virtual_image(&mut instance, image.virtual_address)?;
        vram.copy_from(self.instance as usize, instance.as_bytes())?;
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        info!("NVIDIA_CONTEXT_OK image={:#x} bytes={:#x} mode=virtual", image.virtual_address, image.bytes);
        Ok(())
    }
}
