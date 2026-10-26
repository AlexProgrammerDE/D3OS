"""Recognize one reviewed SM89 typed-math library-scratch schedule.

This certificate is deliberately narrower than a SASS dataflow proof. It binds
every instruction, predicate, register, immediate and control-flow destination
in the reviewed entry and its six helpers. Only whitespace, label spelling,
compiler helper numbering and helper placement may vary. Any other code change
requires a new review, including the address-bound argument below. Input SASS
must be freshly produced from the supplied cubin by the pinned nvdisasm binary.
This does not certify numerical accuracy, GPU execution or performance.
"""

import hashlib
import importlib.util
import json
import re
import struct
import subprocess
from pathlib import Path

ENTRY = "burn_float_arithmetic"
TEXT = f".text.{ENTRY}"
PROFILE = "sm89-typed-trig-library-scratch-v1"
REVISION = "2f47ad0036a7c18ec11d460f70b297a5824f9788"
SOURCE_HASHES = {
    "src/burn-float-arithmetic.rs.in": "c064901d1f484cc0fadad54f11a93406a24efd1dd23012ad70b20cbc8ad312fa",
    "src/common.rs": "8a005c8bbd8c3074b5cdee6b6fe57d9c4e42da485d59f9784ba278d9d12f487f",
    "src/burn-packed-common.rs": "e6fb080724e8b8d1c29b778b37ac34bbe01480325e71549c7e1874fbe6aec413",
    "src/main.rs": "28a6ad898c46ae396d8caa2cc94b8b8e39a8a2b945dbbfbd2bd7d47fe547f541",
    "src/link.rs": "205ac4376fb5ddcb0905c94379d6ad0f6c6cfc77f763585e190d1e5bb0defdd4",
    "build.rs": "0b1e68f6ccb6d60b3d6569c2957bae878c73bde32bb313fc69aa603fb77adf8c",
    "Cargo.lock": "13923daab0897a3f175cae8c314ec58476e959b32c6d2b296e7e5b6bc5ea0f43",
    "rust-toolchain.toml": "85db324345b7937c49ea2ac8b962d741f710719f4f751cab809985cbe6afbe3c",
}
TOOL_HASHES = {
    "tileiras": "8733d2ef6ae234ba5b415d185c957b8cc37577507a2850313a8c0e5c7c3ac54d",
    "nvdisasm": "fa719d636e2dcebdbce466e07ab69f144feabc2fdb3eda52380fbd4a5c189faf",
    "rustc": "859254978c0a0402c32f949f6de0d99aee73be8d15f45aac00ae1448aac51e74",
}
PROVENANCE = {
    "source_sha256": SOURCE_HASHES,
    "tool_sha256": TOOL_HASHES,
    "cuda_rust_revision": REVISION,
    "rust_toolchain": "1.98.1",
    "compile_options": "default;num_worker_warps_per_cta=4;sm_89;tile_ir=13.4;O3",
}
TRIG = f"${ENTRY}$__internal_trig_reduction_slowpathd"
CB2_BYTES = 784
CB2_SHA256 = "33c905a8f3e888c04b037b4305bb69a0bf27d98bdf3cf43301d5c96efecd612f"
# SHA256 of JSON instruction rows, using separators=(',', ':'). Destinations
# resolve to [function identity, byte offset within that function]. Hashes bind
# the entire schedule, including guards, address construction, calls, returns,
# arithmetic and padding, rather than just the local instruction inventory.
SCHEDULES = {
    "burn_float_arithmetic": (
        98304,
        "5082df769c85cbbdee3c84c2b98ff5bed8109c753736c282914f003afb149d4e",
    ),
    "$__cuda_sm20_dblrcp_rn_slowpath_v3": (
        720,
        "e54da592196875833f2553fa7233a42d0e398fcfde450c9f97797e9565a64aa9",
    ),
    "$__cuda_sm20_div_rn_f64_full": (
        1552,
        "4b709772bb3676156a202de0de8f7b228a0fccfd62b401e76dfea08593316245",
    ),
    "$__cuda_sm20_dsqrt_rn_f64_mediumpath_v1": (
        768,
        "31e440d92afef53977930b833e1ed9159d404224c128b447d0dc8218840ee683",
    ),
    "$__cuda_sm20_sqrt_rn_f32_slowpath": (
        368,
        "8cb8007dcf69687506955ffe8cacb2da5cf80eccb3e19428ad86292a45912754",
    ),
    "$__cuda_sm3x_div_rn_noftz_f32_slowpath": (
        1664,
        "0ba52ec2597a2058d1b661aa4aecffb0018953250eb2b4a3b5dabd6bfe949ade",
    ),
    "$burn_float_arithmetic$__internal_trig_reduction_slowpathd": (
        2736,
        "2c0d09db955b574ba8db2b9ab003b4424f6b05855705b4d77ae81bad8783e937",
    ),
}
TABLES = {
    8: (
        "__cudart_i2opi_d",
        144,
        "7ede62875c004cc8c2847b487cb377f75c153593f45bab10b272128e31690cde",
    ),
    16: (
        "__cudart_sin_cos_coeffs",
        128,
        "bf6a9587812432cf5649f28013165faa5a19d2ad0f66fd69cbf13096c2ea4263",
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def collect_provenance(crate, toolkit, disassembler):
    """Bind the production source/command and tools before and after compilation."""
    import tomllib

    dependencies = tomllib.loads((crate / "Cargo.toml").read_text())["dependencies"]
    revisions = {dependencies[name]["rev"] for name in ("cutile", "cutile-ir")}
    rustc = subprocess.run(
        ["rustup", "which", "--toolchain", "1.98.1", "rustc"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    return {
        "source_sha256": {name: file_digest(crate / name) for name in SOURCE_HASHES},
        "tool_sha256": {
            "tileiras": file_digest(toolkit / "bin/tileiras"),
            "nvdisasm": file_digest(disassembler),
            "rustc": file_digest(rustc),
        },
        "cuda_rust_revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "rust_toolchain": tomllib.loads((crate / "rust-toolchain.toml").read_text())[
            "toolchain"
        ]["channel"],
        "compile_options": PROVENANCE["compile_options"],
    }


def load_qualifier():
    spec = importlib.util.spec_from_file_location(
        "typed_math_cubin_qualification",
        Path(__file__).with_name("cubin-qualification.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Cubin:
    """Read the resource, symbol and REL records needed by this certificate."""

    def __init__(self, data):
        self.data = data
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
        offset, stride, count, names_index = (
            header[6],
            header[11],
            header[12],
            header[13],
        )
        raw = [
            struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride)
            for i in range(count)
        ]
        names = self.slice(raw[names_index][4], raw[names_index][5])
        self.sections = []
        for index, row in enumerate(raw):
            name, kind, flags, _, start, size, link, info, _, item_size = row
            self.sections.append(
                {
                    "index": index,
                    "name": self.string(names, name),
                    "kind": kind,
                    "flags": flags,
                    "start": start,
                    "size": size,
                    "link": link,
                    "info": info,
                    "item_size": item_size,
                }
            )
        self.named = {section["name"]: section for section in self.sections}
        if len(self.named) != len(self.sections):
            raise ValueError("duplicate ELF section names")
        symtabs = [section for section in self.sections if section["kind"] == 2]
        if len(symtabs) != 1:
            raise ValueError("expected exactly one ELF symbol table")
        table = symtabs[0]
        self.symbol_table = table["index"]
        if table["item_size"] != 24 or table["size"] % 24:
            raise ValueError("unsupported symbol table layout")
        names = self.contents(self.section(table["link"]))
        self.symbols = []
        for index in range(table["size"] // 24):
            name, info, _, section, value, size = struct.unpack_from(
                "<IBBHQQ", self.contents(table), index * 24
            )
            self.symbols.append(
                {
                    "index": index,
                    "name": self.string(names, name),
                    "kind": info & 15,
                    "section": section,
                    "value": value,
                    "size": size,
                }
            )

    def slice(self, start, size):
        if start < 0 or size < 0 or start + size > len(self.data):
            raise ValueError("truncated ELF payload")
        return self.data[start : start + size]

    @staticmethod
    def string(table, offset):
        end = table.find(b"\0", offset)
        if not 0 <= offset < len(table) or end < 0:
            raise ValueError("invalid ELF string")
        return table[offset:end].decode("ascii")

    def section(self, index):
        if not 0 <= index < len(self.sections):
            raise ValueError("invalid ELF section index")
        return self.sections[index]

    def contents(self, section):
        if section["kind"] == 8:
            raise ValueError("expected initialized ELF data")
        return self.slice(section["start"], section["size"])

    def records(self, name):
        data = self.contents(self.named[name])
        cursor = 0
        records = []
        while cursor < len(data):
            if cursor + 4 > len(data):
                raise ValueError("truncated NVIDIA info record")
            form, kind, size = struct.unpack_from("<BBH", data, cursor)
            length = size if form == 4 else 0
            if form not in (1, 2, 3, 4) or cursor + 4 + length > len(data):
                raise ValueError("unsupported NVIDIA info record")
            payload = data[cursor + 4 : cursor + 4 + length]
            records.append((kind, form, payload))
            cursor += 4 + length
        return records


def identity(name):
    return re.sub(r"^\$__internal_\d+_", "", name)


def schedule(cubin, sass, qualifier):
    """Resolve labels against ELF helpers, retaining every executable operand."""
    text = cubin.named[TEXT]
    functions = [
        s for s in cubin.symbols if s["kind"] == 2 and s["section"] == text["index"]
    ]
    entries = [s for s in functions if s["name"] == ENTRY]
    if (
        len(entries) != 1
        or entries[0]["value"] != 0
        or entries[0]["size"] != text["size"]
    ):
        raise ValueError("entry must own the complete executable section")
    helpers = sorted(
        (s for s in functions if s["name"] != ENTRY), key=lambda s: s["value"]
    )
    if not helpers:
        raise ValueError("missing reviewed helpers")
    entry_end = helpers[0]["value"]
    ranges = [(ENTRY, 0, entry_end)]
    cursor = entry_end
    for helper in helpers:
        if helper["value"] != cursor or not helper["size"] or helper["size"] % 16:
            raise ValueError("overlapping or incomplete helper ranges")
        ranges.append((identity(helper["name"]), cursor, cursor + helper["size"]))
        cursor += helper["size"]
    if cursor != text["size"] or len({name for name, _, _ in ranges}) != len(ranges):
        raise ValueError("unknown, duplicate or incomplete helper inventory")

    labels, rows, pending = {}, [], []
    active = False
    # The production qualifier has already rejected undecoded lines and gaps.
    for line in sass.splitlines():
        section = qualifier.SECTION.fullmatch(line)
        if section:
            active = section[1] == TEXT
        elif active:
            stripped = line.strip()
            if re.fullmatch(r"[\w.$@]+:", stripped):
                pending.append(stripped[:-1])
            instruction = qualifier.INSTRUCTION.fullmatch(line)
            if instruction:
                offset = int(instruction[1], 16)
                for label in pending:
                    if label in labels:
                        raise ValueError("duplicate executable label")
                    labels[label] = offset
                pending.clear()
                predicate = re.search(r"\*/\s*(@!?U?P(?:\d+|T))\s+", line)
                rows.append(
                    (
                        offset,
                        predicate[1] if predicate else "",
                        instruction[2],
                        instruction[3].strip(),
                    )
                )
    for symbol in functions:
        if labels.get(symbol["name"]) != symbol["value"]:
            raise ValueError("SASS function label disagrees with ELF linkage")

    def location(offset):
        for name, start, end in ranges:
            if start <= offset < end:
                return f"{name}:{offset - start:#x}"
        raise ValueError("control-flow target outside reviewed executable functions")

    def reference(match):
        label = match[1]
        if label not in labels:
            raise ValueError(f"undefined executable target: {label}")
        return "`(" + location(labels[label]) + ")"

    normalized = {name: [] for name, _, _ in ranges}
    calls = []
    for offset, predicate, opcode, operands in rows:
        owner = next(name for name, start, end in ranges if start <= offset < end)
        if opcode.startswith("CALL"):
            target = re.fullmatch(r"`\(([^)]+)\)", operands)
            if not target or not any(
                s["name"] == target[1] and s["value"] == labels.get(target[1])
                for s in helpers
            ):
                raise ValueError("CALL does not target a defined reviewed helper")
            calls.append({"offset": offset, "target": identity(target[1])})
        operands = re.sub(r"`\(([^)]+)\)", reference, operands)
        # nvdisasm prints the indirect-branch target inventory in the operand.
        operands = re.sub(
            r"(?<![\w.])(\.L_x_\d+)(?![\w.])", lambda m: reference(m), operands
        )
        normalized[owner].append([predicate, opcode, re.sub(r"\s+", "", operands)])
    fingerprints = {
        name: (
            len(values) * 16,
            digest(json.dumps(values, separators=(",", ":")).encode()),
        )
        for name, values in normalized.items()
    }
    return fingerprints, rows, calls, functions


def resources(cubin, entries, functions):
    text = cubin.named[TEXT]
    expected = {"entry": ENTRY, "code_bytes": 106112, "registers": 71, "threads": 128}
    if [entry for entry in entries if entry.get("entry") == ENTRY] != [expected]:
        raise ValueError("entry metadata exceeds or differs from reviewed resources")
    if text["size"] != expected["code_bytes"] or text["info"] >> 24 != 71:
        raise ValueError("ELF code/register resources differ from reviewed limits")
    shared = cubin.named.get(f".nv.shared.{ENTRY}")
    if shared and shared["size"]:
        raise ValueError("typed math must not allocate static shared memory")
    info = cubin.records(f".nv.info.{ENTRY}")
    for kind, expected_words in ((0x10, (128, 1, 1)), (0x1E, (0,))):
        if [(form, payload) for key, form, payload in info if key == kind] != [
            (4, struct.pack("<" + "I" * len(expected_words), *expected_words))
        ]:
            raise ValueError("unknown launch geometry or CRS stack requirement")
    selected = {symbol["index"]: symbol for symbol in functions}
    for kind in (0x11, 0x12):
        observed = {}
        for key, form, payload in cubin.records(".nv.info"):
            if key != kind:
                continue
            if form != 4 or len(payload) % 8:
                raise ValueError("unknown frame/minimum-stack record")
            for index, size in struct.iter_unpack("<II", payload):
                if index not in selected:
                    continue
                if index in observed:
                    raise ValueError("duplicate frame/minimum-stack record")
                observed[index] = size
        expected_sizes = {
            index: (96 if s["name"] == ENTRY else 0)
            for index, s in selected.items()
            if kind == 0x11 or s["name"] == ENTRY
        }
        if observed != expected_sizes:
            raise ValueError("unreviewed frame/minimum-stack requirement")


def tables(cubin):
    constants = cubin.named[f".nv.constant2.{ENTRY}"]
    if (
        constants["kind"] != 1
        or not constants["flags"] & 2
        or constants["size"] != CB2_BYTES
        or digest(cubin.contents(constants)) != CB2_SHA256
    ):
        raise ValueError("reviewed CB2 constant payload changed")
    constant = cubin.named[".nv.constant4"]
    if constant["size"] != 24 or not constant["flags"] & 2:
        raise ValueError("unknown CB4 table layout")
    relocations = {}
    for group in cubin.sections:
        if group["kind"] not in (4, 9):
            continue
        target = cubin.section(group["info"])
        if not target["flags"] & 2:
            continue
        if (
            target != constant
            or group["kind"] != 9
            or group["item_size"] != 16
            or group["size"] % 16
            or group["link"] != cubin.symbol_table
        ):
            raise ValueError("unknown allocated relocation/helper linkage")
        for offset, value in struct.iter_unpack("<QQ", cubin.contents(group)):
            index, kind = value >> 32, value & 0xFFFFFFFF
            if (
                offset in relocations
                or offset not in (0, 8, 16)
                or kind != 2
                or index >= len(cubin.symbols)
            ):
                raise ValueError("unknown table relocation")
            symbol = cubin.symbols[index]
            section = cubin.section(symbol["section"])
            if (
                not symbol["section"]
                or not section["flags"] & 2
                or symbol["kind"] != 1
                or symbol["value"] + symbol["size"] > section["size"]
            ):
                raise ValueError("undefined or out-of-range table object")
            if struct.unpack_from("<Q", cubin.contents(constant), offset)[0] != 0:
                raise ValueError("unexpected implicit table relocation addend")
            if offset == 0:
                if (
                    symbol["name"] != "___NV_TILE_LAUNCH_META_DATA___"
                    or section["kind"] != 8
                ):
                    raise ValueError("unknown launch metadata linkage")
            else:
                name, size, sha = TABLES[offset]
                if (
                    symbol["name"] != name
                    or symbol["size"] != size
                    or section["name"] != ".nv.global.init"
                ):
                    raise ValueError("unknown math table object")
                if (
                    digest(
                        cubin.contents(section)[
                            symbol["value"] : symbol["value"] + size
                        ]
                    )
                    != sha
                ):
                    raise ValueError("math table content changed")
            relocations[offset] = symbol["name"]
    if set(relocations) != {0, 8, 16}:
        raise ValueError("incomplete math-table linkage")
    return relocations


def classify(rows):
    """Inventory every local site, within the exact certified address schedule."""
    local = [row for row in rows if row[2].split(".")[0] in ("LDL", "STL")]
    if len(local) != 131:
        raise ValueError("unclassified local instruction")
    arrays = []
    for start in range(0, 126, 7):
        group = local[start : start + 7]
        covered = []
        for _, predicate, opcode, operands in group[:4]:
            match = re.fullmatch(r"\[R1(?:\+(0x[0-9a-f]+))?\], R\d+", operands)
            if predicate or not match or opcode not in ("STL", "STL.64"):
                raise ValueError("unclassified F32 scratch store")
            offset = int(match[1], 16) if match[1] else 0
            covered.extend(range(offset, offset + (8 if opcode == "STL.64" else 4)))
        if sorted(covered) not in (list(range(28)), list(range(28, 56))):
            raise ValueError("F32 scratch store escapes reviewed array")
        for (_, predicate, opcode, operands), read_offset in zip(
            group[4:], (24, 20, 16)
        ):
            expected_predicate = f"@P{(start // 7) % 2}" if read_offset == 16 else ""
            if (
                predicate != expected_predicate
                or opcode != "LDL"
                or not re.fullmatch(rf"R\d+, \[R\d+\+{read_offset:#x}\]", operands)
            ):
                raise ValueError("unclassified F32 scratch load")
        arrays.append(
            {
                "kind": "F32 reduction products",
                "frame_range": [min(covered), max(covered) + 1],
                "sites": [row[0] for row in group],
            }
        )
    # Address construction and the loop guards are bound by the helper hash.
    if [(p, op, operands) for _, p, op, operands in local[126:]] != [
        ("", "STL.64", "[R18], R66"),
        ("", "STL.64", "[R61+-0x78], R66"),
        ("@P0", "LDL.64", "R24, [R1+0x40]"),
        ("", "LDL.64", "R16, [R1+0x48]"),
        ("", "LDL.64", "R18, [R1+0x50]"),
    ]:
        raise ValueError("unclassified F64 scratch access")
    # Reviewed guards admit finite large F32 fields 143..254. The index is
    # ((exponent & 0xe0)-128)>>5. All three U32 reads stay in a 28-byte array.
    for exponent in range(143, 255):
        index = ((exponent & 0xE0) - 128) >> 5
        for offset in (24, 20, 16):
            address = offset - 4 * index
            if not 0 <= address <= 24:
                raise ValueError("F32 reviewed address-bound argument failed")
    # F64 finite large fields 1054..2046: the loop and final store write at
    # most five U64 words starting at R1+0x38, within [0x38,0x60).
    for exponent in range(1054, 2047):
        index = (exponent - 1024) >> 6
        begin, end = 15 - index, min(19 - index, 18)
        for word in [*range(end - begin), index + end - 15]:
            if not 0 <= word < 5:
                raise ValueError("F64 reviewed address-bound argument failed")
    arrays.append(
        {
            "kind": "F64 reduction products",
            "frame_range": [56, 96],
            "sites": [row[0] for row in local[126:]],
        }
    )
    return arrays


def qualify(cubin, sass, entries, provenance):
    """Fail closed unless fresh integrated code matches the reviewed certificate."""
    if provenance != PROVENANCE:
        raise ValueError("unreviewed typed-math compiler/source provenance")
    qualifier = load_qualifier()
    production = qualifier.qualify(cubin, sass)
    elf = Cubin(cubin)
    fingerprints, rows, calls, functions = schedule(elf, sass, qualifier)
    if fingerprints != SCHEDULES:
        raise ValueError(
            "typed-math instruction schedule changed; scratch requalification required"
        )
    resources(elf, entries, functions)
    linkage = tables(elf)
    arrays = classify(rows)
    return {
        "profile": PROFILE,
        "production_profile": production["metadata_unused_profile"],
        "entry": ENTRY,
        "frame_bytes": 96,
        "minimum_stack_bytes": 96,
        "registers": 71,
        "code_bytes": 106112,
        "threads": [128, 1, 1],
        "local_load_sites": 57,
        "local_store_sites": 74,
        "call_sites": calls,
        "classified_arrays": arrays,
        "unclassified_local_sites": [],
        "schedule_sha256": {name: sha for name, (_, sha) in fingerprints.items()},
        "table_linkage": linkage,
        "provenance": provenance,
        "constant2_sha256": CB2_SHA256,
        "constant2_bytes": CB2_BYTES,
        "cubin_sha256": digest(cubin),
        "sass_sha256": digest(sass.encode()),
    }
