"""Check the observed nvdisasm 13.4 SM89 direct-constant profile.

This is a conservative inventory check for trusted nvdisasm output, not a
general SASS decoder or dataflow proof. Every allocated executable ELF section,
including embedded helper functions, must have a complete 16-byte instruction
inventory. Literal CB0/CB2 banks may use the observed LDC register addressing:
their bank selection cannot reach CB4. One observed half-widening form reads
only a literal CB0 word. CB4 supports only the qualified integer table-pointer
forms below, entirely within bytes [8, 24).
"""

import hashlib
import json
import re
import struct

PROFILE = "sm89-direct-constant-v2"
INT8_MMA_PROFILE = "sm89-direct-constant-v3"
INT8_DOT_PROFILE = "sm89-direct-constant-v4"
INT8_DOT_CB2_PROFILE = "sm89-direct-constant-v5"
VERSION = "13.4.92"
ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
SECTION = re.compile(r"\s*\.section\s+([^,\s]+),[^\n]*")
INSTRUCTION = re.compile(
    r"\s*/\*([0-9a-f]+)\*/\s*(?:@!?U?P(?:\d+|T)\s+)?"
    r"([A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\s*([^;]*);\s*(?://.*)?"
)
CONSTANT = re.compile(r"\bc\[(0x[0-9a-f]+)\]\[([^\[\]]+)\]")
INDEXED = re.compile(r"(?:R\d+|UR\d+)(?:[+-]0x[0-9a-f]+)?")
REGISTER = r"(?:R\d+|RZ)"
# Only the observed signed dot32 form, with a literal zero accumulator.
# Its zero-accumulator i8 dot is bounded by 32*128*128, so SAT is inert.
# Optional matrix-operand reuse hints retain the same register-only operands.
INT8_MMA_OPCODE = "IMMA.16832.S8.S8.SAT"
INT8_MMA_REGISTER = r"R(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-4])"
INT8_MMA_OPERANDS = re.compile(
    rf"{INT8_MMA_REGISTER}\s*,\s*{INT8_MMA_REGISTER}(?:\.reuse)?\.ROW\s*,\s*"
    rf"{INT8_MMA_REGISTER}(?:\.reuse)?\.COL\s*,\s*RZ"
)
# Rust NVPTX emits this signed packed dot4 with register-only operands.
# A register accumulator covers the second dot4 in each eight-value group.
# The observed first multiplicand may carry a register reuse hint. No other
# modifier slot or constant-addressing form inherits that allowance.
INT8_DOT_OPCODE = "IDP.4A.S8.S8"
INT8_DOT_OPERANDS = re.compile(
    rf"{INT8_MMA_REGISTER}\s*,\s*{INT8_MMA_REGISTER}(?:\.reuse)?\s*,\s*"
    rf"{INT8_MMA_REGISTER}\s*,\s*(?:{INT8_MMA_REGISTER}|RZ)"
)
# The observed byte-sum form reads one aligned word from its own read-only
# CB2 section. Every literal offset is bounded against the actual ELF bytes.
INT8_DOT_CB2_OPERANDS = re.compile(
    rf"{INT8_MMA_REGISTER}\s*,\s*{INT8_MMA_REGISTER}\s*,\s*"
    rf"c\[0x2\]\[(0x[0-9a-f]+)\]\s*,\s*(?:{INT8_MMA_REGISTER}|RZ)"
)
# The observed Q4 upper-nibble extraction has register-only data and the
# literal 0x4 selector. Other signedness, selectors and modifiers stay rejected.
U32_EXTRACT_OPCODE = "SGXT.U32"
U32_EXTRACT_OPERANDS = re.compile(
    rf"{INT8_MMA_REGISTER}\s*,\s*{INT8_MMA_REGISTER}\s*,\s*0x4"
)
CB4 = r"c\[0x4\]\[0x[0-9a-f]+\]"
# nvdisasm prints a space between the brackets for this packed-half operand.
# Match the complete observed widening form, without extending CB4 syntax.
CB0_HALF_WIDEN = re.compile(r"R\d+,\s*-RZ,\s*c\[0x0\]\s*\[(0x[0-9a-f]+)\]\.H0_H0")
# No floating-point, vector, or unknown CB4 instruction inherits a guessed
# width. IMAD.WIDE reads eight bytes only in its final accumulator operand.
CB4_FORMS = {
    "IADD3": (4, rf"R\d+,\s*(?:(?:P\d+|PT),\s*)?{REGISTER},\s*{CB4},\s*{REGISTER}"),
    # Observed high-word table-pointer addition with an incoming carry.
    "IADD3.X": (4, rf"R\d+,\s*RZ,\s*{CB4},\s*RZ,\s*P\d+,\s*!PT"),
    "IMAD.X": (4, rf"R\d+,\s*{REGISTER},\s*{REGISTER},\s*{CB4},\s*(?:P\d+|PT)"),
    "IMAD.WIDE": (8, rf"R\d+,\s*{REGISTER},\s*{REGISTER},\s*{CB4}"),
    "IMAD.WIDE.U32": (8, rf"R\d+,\s*{REGISTER},\s*{REGISTER},\s*{CB4}"),
}
# The typed Power entry uses these packed-half comparisons for special cases.
# Admit only the observed register/immediate forms, with no constant-bank read.
HALF_PREDICATE = r"!?P(?:\d+|T)"
HALF_COMPARE_PREFIX = r"P\d+,\s*PT,\s*R\d+\.H0_H0,\s*"
HALF_COMPARE_FORMS = {
    "HSETP2.NEU.AND": HALF_COMPARE_PREFIX + rf"RZ\.H0_H0,\s*{HALF_PREDICATE}",
    "HSETP2.GEU.AND": HALF_COMPARE_PREFIX + rf"RZ\.H0_H0,\s*{HALF_PREDICATE}",
    "HSETP2.EQ.OR": HALF_COMPARE_PREFIX + rf"(?:1,\s*1|RZ\.H0_H0),\s*{HALF_PREDICATE}",
    "HSETP2.EQ.AND": HALF_COMPARE_PREFIX + rf"-1,\s*-1,\s*{HALF_PREDICATE}",
}
# Observed instruction families. Unknown/undecoded mnemonics fail closed even
# when no explicit constant operand is printed. CB4 has the narrower forms above.
OPCODES = frozenset(
    [
        "ATOMG",
        "ATOMS",
        "BAR",
        "BRA",
        "BREAK",
        "BRX",
        "BRXU",
        "BSSY",
        "BSYNC",
        "CALL",
        "CS2R",
        "DADD",
        "DEPBAR",
        "DFMA",
        "DMUL",
        "DSETP",
        "EXIT",
        "F2F",
        "F2FP",
        "F2I",
        "FADD",
        "FCHK",
        "FFMA",
        "FLO",
        "FMNMX",
        "FMUL",
        "FRND",
        "FSEL",
        "FSETP",
        "HADD2",
        "HFMA2",
        "HMMA",
        "HMUL2",
        "I2F",
        "I2FP",
        "IABS",
        "IADD3",
        "IMAD",
        "IMNMX",
        "ISETP",
        "LD",
        "LDC",
        "LDG",
        "LDGDEPBAR",
        "LDGSTS",
        "LDL",
        "LDS",
        "LDSM",
        "LEA",
        "LOP3",
        "MOV",
        "MUFU",
        "NOP",
        "P2R",
        "PLOP3",
        "POPC",
        "PRMT",
        "R2P",
        "R2UR",
        "RED",
        "REDUX",
        "RET",
        "S2R",
        "S2UR",
        "SEL",
        "SHF",
        "SHFL",
        "ST",
        "STG",
        "STL",
        "STS",
        "UFLO",
        "UIADD3",
        "UIMAD",
        "UISETP",
        "ULDC",
        "ULEA",
        "ULOP3",
        "UMOV",
        "UPRMT",
        "USEL",
        "USHF",
        "VOTEU",
        "WARPSYNC",
    ]
)
TEXT_DIRECTIVES = frozenset(
    (".sectioninfo", ".align", ".global", ".type", ".size", ".other", ".weak")
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def executable_sections(cubin):
    """Inventory actual allocated executable sections, without ELF dependencies."""
    if len(cubin) < ELF_HEADER.size:
        raise ValueError("truncated ELF64 header")
    header = ELF_HEADER.unpack_from(cubin)
    ident, kind, machine, version = header[:4]
    if (
        ident[:7] != b"\x7fELF\x02\x01\x01"
        or kind != 2
        or machine != 190
        or version != 1
    ):
        raise ValueError("expected little-endian ELF64 CUDA executable")
    if header[8] != ELF_HEADER.size or (header[7] >> 8) & 0xFF != 89:
        raise ValueError("unsupported ELF header or SM target (expected SM89)")
    offset, stride, count, names_index = header[6], header[11], header[12], header[13]
    if stride != SECTION_HEADER.size or not count or not 0 < names_index < count:
        raise ValueError("unsupported ELF section table")
    if offset < ELF_HEADER.size or offset + stride * count > len(cubin):
        raise ValueError("truncated ELF section table")
    headers = [
        SECTION_HEADER.unpack_from(cubin, offset + stride * i) for i in range(count)
    ]
    names_header = headers[names_index]
    if names_header[1] != 3:
        raise ValueError("section names must use a string table")
    names_offset, names_size = names_header[4:6]
    if names_offset + names_size > len(cubin):
        raise ValueError("truncated ELF section names")
    names = cubin[names_offset : names_offset + names_size]
    sections = {}
    ranges = []
    for section in headers:
        name_offset, section_type, flags, _, start, size = section[:6]
        if name_offset >= len(names):
            raise ValueError("invalid ELF section name offset")
        end = names.find(b"\0", name_offset)
        if end < 0:
            raise ValueError("unterminated ELF section name")
        try:
            name = names[name_offset:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("unsupported ELF section name") from error
        if section_type != 8 and start + size > len(cubin):
            raise ValueError(f"{name}: truncated section contents")
        if flags & 6 != 6:
            continue
        if not name.startswith(".text.") or section_type != 1:
            raise ValueError(f"{name}: unsupported executable section")
        if not size or size % 16 or start % 16:
            raise ValueError(
                f"{name}: code must contain whole aligned 16-byte instructions"
            )
        if name in sections or any(
            start < other_end and other_start < start + size
            for other_start, other_end in ranges
        ):
            raise ValueError(f"{name}: duplicate or overlapping executable section")
        sections[name] = size
        ranges.append((start, start + size))
    if not sections:
        raise ValueError("no allocated executable text sections")
    return sections


def dot_constant2_bytes(cubin, text_section):
    """Read one unambiguous, allocated, read-only per-entry constant bank.

    executable_sections() already validated the ELF header and table bounds.
    Global/local overlap is rejected because the native adapter would otherwise
    choose the last matching binding. This extension needs one exact binding.
    """
    header = ELF_HEADER.unpack_from(cubin)
    offset, stride, count, names_index = header[6], header[11], header[12], header[13]
    headers = [
        SECTION_HEADER.unpack_from(cubin, offset + stride * i) for i in range(count)
    ]
    strings = headers[names_index]
    names = cubin[strings[4] : strings[4] + strings[5]]
    expected = ".nv.constant2." + text_section.removeprefix(".text.")
    matches = []
    for section in headers:
        name_offset, kind, flags, _, start, size = section[:6]
        name = names[name_offset : names.find(b"\0", name_offset)].decode("ascii")
        if name not in (expected, ".nv.constant2"):
            continue
        if (
            name != expected
            or kind != 1
            or flags & 7 != 2
            or not 0 < size <= 65536
            or start < ELF_HEADER.size
            or start + size > len(cubin)
        ):
            raise ValueError("dot4 requires a bounded read-only per-entry CB2 section")
        matches.append(cubin[start : start + size])
    if len(matches) != 1:
        raise ValueError("dot4 requires one unambiguous per-entry CB2 binding")
    return matches[0]


def constant_accesses(opcode, operands):
    """Return qualified CB4 ranges; reject all unrecognized constant syntax."""
    if opcode == "HADD2.F32":
        widening = CB0_HALF_WIDEN.fullmatch(operands)
        if widening and int(widening[1], 16) % 4 == 0:
            return []
    accesses = list(CONSTANT.finditer(operands))
    remainder = CONSTANT.sub("", operands)
    if re.search(r"\bcx?\s*[\[(]", remainder, re.IGNORECASE):
        raise ValueError("indirect bank or unrecognized constant operand")
    if opcode.split(".")[0] in ("LDC", "ULDC") and not accesses:
        raise ValueError("constant load has no recognized constant operand")
    ranges = []
    for match in accesses:
        operand = next(part.strip() for part in operands.split(",") if match[0] in part)
        if operand not in (
            match[0],
            "-" + match[0],
            "+" + match[0],
            "|" + match[0] + "|",
            "-|" + match[0] + "|",
        ):
            raise ValueError(
                "unsupported constant operand modifiers or trailing syntax"
            )
        bank = int(match[1], 16)
        address = match[2]
        direct = re.fullmatch(r"0x[0-9a-f]+", address)
        if bank in (0, 2):
            if direct:
                continue
            # Fixed-bank indexing is observed on LDC/ULDC loads. No register
            # selects the bank, so these forms remain disjoint from CB4.
            if opcode in ("LDC", "LDC.U16", "ULDC", "ULDC.64") and INDEXED.fullmatch(
                address
            ):
                continue
            raise ValueError("unsupported fixed-bank constant addressing")
        if bank != 4 or not direct:
            raise ValueError("unsupported or indirect constant bank access")
        form = CB4_FORMS.get(opcode)
        if len(accesses) != 1 or form is None or not re.fullmatch(form[1], operands):
            raise ValueError(f"unsupported CB4 instruction form: {opcode}")
        width = form[0]
        address = int(address, 16)
        if address % 4 or address < 8 or address + width > 24:
            raise ValueError(
                "CB4 access is outside the qualified table-pointer range [8, 24)"
            )
        ranges.append((address, width))
    return ranges


def qualify(cubin: bytes, sass: str) -> dict:
    """Return deterministic qualification evidence or raise ValueError.

    The caller adds source_sha256/source_bytes to the returned record, stores
    the exact cubin/SASS inputs, and binds the actual disassembler provenance.
    The trusted disassembly must use the default nvdisasm 13.4 text format.
    """
    sections = executable_sections(cubin)
    if re.findall(r"^\s*\.target\s+(\S+)\s*$", sass, re.MULTILINE) != ["sm_89"]:
        raise ValueError("expected exactly one SM89 disassembly target")
    # Confirm the actual cubin also contains the producer/profile recorded by
    # nvdisasm. The stager separately verifies the executable version and digest.
    for marker in (f"V{VERSION}", "-O 3 -arch sm_89 "):
        if marker not in sass or marker.encode("ascii") not in cubin:
            raise ValueError("missing CUDA 13.4.92 SM89/O3 producer metadata")
    seen_sections = set()
    counts = {name: 0 for name in sections}
    active = None
    cb4_accesses = []
    has_int8_mma = False
    has_int8_dot = False
    cb2_dot_accesses = []
    u32_extracts = []
    dot_constants = {}
    for line_number, line in enumerate(sass.splitlines(), 1):
        boundary = SECTION.fullmatch(line)
        if boundary:
            name = boundary[1]
            if name.startswith(".text") and name not in sections:
                raise ValueError(
                    f"{name}: SASS text section absent from executable inventory"
                )
            active = name if name in sections else None
            if active:
                if active in seen_sections:
                    raise ValueError(f"{active}: duplicate SASS section")
                seen_sections.add(active)
            continue
        if active is None:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        instruction = INSTRUCTION.fullmatch(line)
        if instruction:
            offset = int(instruction[1], 16)
            opcode, operands = instruction[2], instruction[3].strip()
            # Increasing exact offsets reject gaps, duplicates, reordering and
            # out-of-section decode, including alignment padding and helpers.
            if offset != counts[active] * 16 or offset >= sections[active]:
                raise ValueError(
                    f"{active}:{offset:#x}: incomplete or duplicate instruction inventory"
                )
            half_compare = HALF_COMPARE_FORMS.get(opcode)
            if opcode.split(".")[0] == "IDP":
                if opcode != INT8_DOT_OPCODE:
                    raise ValueError(
                        f"{active}:{offset:#x}: unsupported signed INT8 dot4 form"
                    )
                if not INT8_DOT_OPERANDS.fullmatch(operands):
                    literal = INT8_DOT_CB2_OPERANDS.fullmatch(operands)
                    if literal is None:
                        raise ValueError(
                            f"{active}:{offset:#x}: unsupported signed INT8 dot4 form"
                        )
                    if active not in dot_constants:
                        dot_constants[active] = dot_constant2_bytes(cubin, active)
                    address = int(literal[1], 16)
                    if address % 4 or address + 4 > len(dot_constants[active]):
                        raise ValueError(
                            f"{active}:{offset:#x}: dot4 CB2 word outside declared bounds"
                        )
                    cb2_dot_accesses.append(
                        {
                            "section": active,
                            "offset": offset,
                            "constant_offset": address,
                            "bytes": 4,
                        }
                    )
                has_int8_dot = True
            elif opcode.split(".")[0] == "SGXT":
                if opcode != U32_EXTRACT_OPCODE or not U32_EXTRACT_OPERANDS.fullmatch(
                    operands
                ):
                    raise ValueError(
                        f"{active}:{offset:#x}: unsupported unsigned register extraction"
                    )
                u32_extracts.append({"section": active, "offset": offset, "literal": 4})
            elif opcode.split(".")[0] == "IMMA":
                if opcode != INT8_MMA_OPCODE or not INT8_MMA_OPERANDS.fullmatch(
                    operands
                ):
                    raise ValueError(
                        f"{active}:{offset:#x}: unsupported signed INT8 MMA form"
                    )
                has_int8_mma = True
            elif half_compare is not None:
                if not re.fullmatch(half_compare, operands):
                    raise ValueError(
                        f"{active}:{offset:#x}: unsupported packed-half comparison form"
                    )
            elif opcode.split(".")[0] not in OPCODES:
                raise ValueError(
                    f"{active}:{offset:#x}: unsupported/undecoded instruction {opcode}"
                )
            try:
                accesses = constant_accesses(opcode, operands)
            except ValueError as error:
                raise ValueError(f"{active}:{offset:#x}: {error}") from error
            for constant_offset, width in accesses:
                cb4_accesses.append(
                    {
                        "section": active,
                        "offset": offset,
                        "opcode": opcode,
                        "constant_offset": constant_offset,
                        "bytes": width,
                    }
                )
            counts[active] += 1
        elif (
            re.fullmatch(r"[\w.$@]+:", stripped)
            or stripped.split()[0] in TEXT_DIRECTIVES
        ):
            continue
        else:
            raise ValueError(
                f"{active}: line {line_number}: undecoded or unsupported text content"
            )
    if seen_sections != sections.keys() or any(
        counts[name] * 16 != size for name, size in sections.items()
    ):
        raise ValueError("SASS does not cover every executable section byte")
    sass_bytes = sass.encode("utf-8")
    record = {
        "schema_version": 1,
        "metadata_unused_profile": (
            INT8_DOT_CB2_PROFILE
            if cb2_dot_accesses or u32_extracts
            else INT8_DOT_PROFILE
            if has_int8_dot
            else INT8_MMA_PROFILE
            if has_int8_mma
            else PROFILE
        ),
        "target_sm": 89,
        "optimization_level": 3,
        "compiler_version": VERSION,
        "cubin_sha256": sha256(cubin),
        "cubin_bytes": len(cubin),
        "sass_sha256": sha256(sass_bytes),
        "sass_bytes": len(sass_bytes),
        "code_bytes": sum(sections.values()),
        "instruction_count": sum(counts.values()),
        "sections": [
            {"name": name, "bytes": size, "instruction_count": counts[name]}
            for name, size in sorted(sections.items())
        ],
        "cb4_accesses": cb4_accesses,
    }
    if cb2_dot_accesses:
        record["cb2_dot_accesses"] = cb2_dot_accesses
    if u32_extracts:
        record["u32_extracts"] = u32_extracts
    record["evidence_sha256"] = sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return record
