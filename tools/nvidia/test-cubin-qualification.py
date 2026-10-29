"""Host-only qualification/receipt tests with synthetic ELF and SASS fixtures."""

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qualification = load_module("cubin_qualification", "cubin-qualification.py")
guest = load_module("guest_compiler", "guest-compiler.py")
native = load_module("native_qemu", "run-qemu.py")
PRODUCER = b"Cuda compilation tools, release 13.4, V13.4.92\0-O 3 -arch sm_89 \0"


def elf(sections, *, constants=()):
    """Build small ELF64 inventories; instruction semantics come from test SASS."""
    names = bytearray(b"\0.shstrtab\0.note.nv.tkinfo\0")
    headers = [(0,) * 10]
    data = bytearray(64)
    name_offsets = {}
    for name, size in sections:
        name_offsets[name] = len(names)
        names.extend(name.encode("ascii") + b"\0")
        data.extend(b"\0" * (-len(data) % 16))
        headers.append((name_offsets[name], 1, 6, 0, len(data), size, 0, 0, 16, 0))
        data.extend(bytes(size))
    for name, payload, kind, flags in constants:
        name_offset = len(names)
        names.extend(name.encode("ascii") + b"\0")
        data.extend(b"\0" * (-len(data) % 16))
        headers.append(
            (name_offset, kind, flags, 0, len(data), len(payload), 0, 0, 4, 0)
        )
        data.extend(payload)
    note_offset = len(data)
    data.extend(PRODUCER)
    headers.append((11, 7, 0, 0, note_offset, len(PRODUCER), 0, 0, 1, 0))
    names_offset = len(data)
    data.extend(names)
    headers.append((1, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0))
    data.extend(b"\0" * (-len(data) % 8))
    table_offset = len(data)
    for header in headers:
        data.extend(struct.pack("<IIQQQQIIQQ", *header))
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        data,
        0,
        b"\x7fELF\x02\x01\x01" + bytes(9),
        2,
        190,
        1,
        0,
        0,
        table_offset,
        0x06005904,
        64,
        0,
        0,
        64,
        len(headers),
        len(headers) - 1,
    )
    return bytes(data)


def sass(sections):
    text = [
        ".target sm_89",
        '.string "Cuda compilation tools, release 13.4, V13.4.92"',
        '.string "-O 3 -arch sm_89 "',
    ]
    for name, instructions in sections:
        text.extend((f'.section {name},"ax",@progbits', ".align 128", f"{name}:"))
        text.extend(
            f"/*{offset * 16:04x}*/ {instruction};"
            for offset, instruction in enumerate(instructions)
        )
    return "\n".join(text) + "\n"


def qualify_instructions(*instructions):
    name = ".text.entry"
    return qualification.qualify(
        elf([(name, len(instructions) * 16)]), sass([(name, instructions)])
    )


class QualificationTests(unittest.TestCase):
    def test_packed_half_widening_reads_only_a_literal_cb0_word(self):
        for spacing in ("", " ", "\t"):
            instruction = f"HADD2.F32 R9, -RZ, c[0x0]{spacing}[0x198].H0_H0"
            result = qualify_instructions(instruction)
            self.assertEqual(result["instruction_count"], 1)
            self.assertEqual(result["cb4_accesses"], [])
        observed = "HADD2.F32 R9, -RZ, c[0x0] [0x198].H0_H0"
        for bad in (
            observed.replace("0x0]", "0x4]"),
            observed.replace("0x0]", "0x2]"),
            observed.replace("0x0]", "R0]"),
            observed.replace("c[0x0]", "cx[UR0]"),
            observed.replace("0x198", "UR1"),
            observed.replace("0x198", "R2+0x198"),
            observed.replace("0x198", "0x199"),
            observed.replace("0x198", "0x19a"),
            observed.replace("H0_H0", "H1_H1"),
            observed.replace("H0_H0", "H0_H1"),
            observed.replace("H0_H0", "H0_H0.F32"),
            observed.replace("-RZ", "RZ"),
            observed.replace("-RZ", "R4"),
            observed.replace("HADD2.F32", "HADD2.F32.FTZ"),
            observed.replace("HADD2.F32", "HFMA2"),
            observed + ", c[0x4][0x8]",
        ):
            with self.subTest(instruction=bad), self.assertRaises(ValueError):
                qualify_instructions(bad)

    def test_power_half_comparisons_retain_register_only_admission(self):
        observed = (
            "HSETP2.NEU.AND P0, PT, R35.H0_H0, RZ.H0_H0, PT",
            "HSETP2.EQ.OR P0, PT, R2.H0_H0, 1, 1, !P0",
            "HSETP2.EQ.OR P0, PT, R2.H0_H0, RZ.H0_H0, !P0",
            "@!P1 HSETP2.GEU.AND P3, PT, R35.H0_H0, RZ.H0_H0, PT",
            "HSETP2.EQ.AND P0, PT, R2.H0_H0, -1, -1, !P0",
        )
        result = qualify_instructions(*observed)
        self.assertEqual(result["instruction_count"], len(observed))
        self.assertEqual(result["cb4_accesses"], [])
        for instruction in (
            observed[0].replace("NEU.AND", "UNKNOWN.AND"),
            observed[0].replace("H0_H0", "H1_H1", 1),
            observed[0].replace("RZ.H0_H0", "c[0x4][0x8]"),
            observed[0].replace("RZ.H0_H0", "c[0x2][0x8]"),
            observed[0].replace("RZ.H0_H0", "R4.H0_H0"),
            observed[1].replace("1, 1", "2, 2"),
            observed[4].replace("-1, -1", "-2, -2"),
            observed[4].replace("!P0", "R0"),
        ):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                qualify_instructions(instruction)

    def test_direct_table_pointer_widths_and_evidence(self):
        result = qualify_instructions(
            "IADD3 R2, P0, R4, c[0x4][0x10], RZ",
            "IMAD.X R3, RZ, RZ, c[0x4][0x14], P0",
            "IMAD.WIDE R2, R2, R3, c[0x4][0x8]",
            "IMAD.WIDE.U32 R2, R2, R3, c[0x4][0x10]",
            "IADD3 R2, R4, c[0x4][0xc], RZ",
        )
        self.assertEqual(result["instruction_count"], 5)
        self.assertEqual(result["code_bytes"], 80)
        self.assertEqual(
            [access["bytes"] for access in result["cb4_accesses"]], [4, 4, 8, 8, 4]
        )
        self.assertEqual(
            [access["constant_offset"] for access in result["cb4_accesses"]],
            [16, 20, 8, 16, 12],
        )
        evidence = result.pop("evidence_sha256")
        self.assertEqual(
            evidence,
            hashlib.sha256(
                json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )

    def test_carry_table_pointer_add_reads_one_bounded_word(self):
        instruction = "IADD3.X R3, RZ, c[0x4][0x14], RZ, P0, !PT"
        result = qualify_instructions(instruction)
        self.assertEqual(
            [(a["constant_offset"], a["bytes"]) for a in result["cb4_accesses"]],
            [(20, 4)],
        )
        for bad in (
            instruction.replace("0x14", "0x4"),
            instruction.replace("0x14", "0x18"),
            instruction.replace("0x14", "0x15"),
            instruction.replace("0x14", "R0"),
            instruction.replace("P0, !PT", "R0, !PT"),
            instruction.replace("P0, !PT", "P0, PT"),
            instruction.replace("RZ, c", "R4, c"),
            instruction.replace("IADD3.X", "IADD3.X.SAT"),
        ):
            with self.subTest(instruction=bad), self.assertRaises(ValueError):
                qualify_instructions(bad)

    def test_fixed_safe_banks_and_supported_indexing(self):
        for bank in (0, 2):
            for opcode, register in (
                ("LDC", "R2"),
                ("LDC.U16", "R2"),
                ("ULDC", "UR2"),
                ("ULDC.64", "UR2"),
            ):
                for address in ("0x0", "R0", "R9+0x10", "UR4", "UR4-0x10"):
                    with self.subTest(bank=bank, opcode=opcode, address=address):
                        result = qualify_instructions(
                            f"{opcode} {register}, c[0x{bank:x}][{address}]"
                        )
                        self.assertEqual(result["cb4_accesses"], [])

    def test_unsupported_constant_forms_reject(self):
        instructions = (
            "IADD3 R2, R4, c[0x4][0x0], RZ",
            "IADD3 R2, R4, c[0x4][0x4], RZ",
            "IADD3 R2, R4, c[0x4][0x7], RZ",
            "IADD3 R2, R4, c[0x4][0x18], RZ",
            "IMAD.WIDE R2, R2, R3, c[0x4][0x4]",
            "IMAD.WIDE R2, R2, R3, c[0x4][0x14]",
            "IMAD.WIDE R2, c[0x4][0x8], R3, R4",
            "FFMA R2, R4, c[0x4][0x10], RZ",
            "ULDC.64 UR2, c[0x4][0x8]",
            "LDC.64 R2, c[0x4][0x8]",
            "LDC R2, c[0x4][R0]",
            "ULDC UR2, c[0x4][UR0+0x10]",
            "LDC R2, c[R0][0x8]",
            "LDC R2, cx[UR0][0x8]",
            "LDC R2, c[0x1][0x8]",
            "LDC R2, c[0x6][0x8]",
            "LDC R2, c[0x2][R0+R1]",
            "LDC R2, c[0x2][R0+]",
            "LDC R2, c[0x2][R0+0x10+0x4]",
            "LDC R2, c[0x2][0x8][R0]",
            "LDC R2, c[0x2][0x8]+R0",
            "LDC R2, c[0x2][0x8",
            "LDC R2, C[0x2][0x8]",
            "LDC R2, [R0]",
            "MOV R2, c[0x2][R0]",
            "IADD3 R2, R4, -c[0x4][0x10], RZ",
        )
        for instruction in instructions:
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                qualify_instructions(instruction)

    def test_whole_sections_and_embedded_helpers(self):
        sections = [
            (".text.entry", ["EXIT", "NOP"]),
            (".text.helper", ["IADD3 R2, R4, c[0x4][0x4], RZ"]),
        ]
        cubin = elf([(name, len(instructions) * 16) for name, instructions in sections])
        with self.assertRaises(ValueError):
            qualification.qualify(cubin, sass(sections))
        with self.assertRaises(ValueError):
            qualification.qualify(cubin, sass(sections[:1]))
        sections[1][1][0] = "RET.REL.NODEC R0"
        result = qualification.qualify(cubin, sass(sections))
        self.assertEqual(
            [section["instruction_count"] for section in result["sections"]], [2, 1]
        )

    def test_instruction_inventory_rejects_partial_or_unknown_decode(self):
        cubin = elf([(".text.entry", 32)])
        original = sass([(".text.entry", ["EXIT", "NOP"])])
        mutations = (
            original.replace("/*0010*/ NOP;", ""),
            original.replace("/*0010*/", "/*0000*/"),
            original.replace("/*0010*/", "/*0020*/"),
            original.replace("/*0010*/", "/*0011*/"),
            original + "/*0020*/ NOP;\n",
            original.replace("/*0010*/ NOP;", "/*0010*/ .byte 0x00"),
            original.replace("/*0010*/ NOP;", "/*0010*/ .inst 0x00"),
            original.replace("/*0010*/ NOP;", "/*0010*/ UNKNOWN R0;"),
            original + ".zero 16\n",
            original + '.section .text.hidden,"ax",@progbits\n',
            original + '.section .text.entry,"ax",@progbits\n',
            original.replace(
                "/*0010*/ NOP;", '.section .nv.info,"",@progbits\n/*0010*/ NOP;'
            ),
            original.replace("sm_89", "sm_90", 1),
            original.replace("V13.4.92", "V13.4.93"),
            original.replace("-O 3", "-O 2"),
        )
        for mutation in mutations:
            with self.subTest(sass=mutation), self.assertRaises(ValueError):
                qualification.qualify(cubin, mutation)

    def test_malformed_elf_inventory(self):
        original = elf([(".text.entry", 32), (".text.helper", 16)])
        table = struct.unpack_from("<Q", original, 40)[0]
        mutations = [original[:size] for size in (0, 63, len(original) - 1)]
        for position, fmt, value in (
            (4, "B", 1),
            (5, "B", 2),
            (18, "H", 62),
            (48, "I", 0x06005A04),
            (58, "H", 32),
            (60, "H", 0),
            (62, "H", 0xFFFF),
            (table + 64 + 4, "I", 8),
            (table + 64 + 32, "Q", 31),
            (table + 64 + 24, "Q", len(original) + 16),
            (table + 128, "I", 0xFFFFFF),
            (table + 128 + 24, "Q", 64),
            (table + 128, "I", struct.unpack_from("<I", original, table + 64)[0]),
        ):
            data = bytearray(original)
            struct.pack_into("<" + fmt, data, position, value)
            mutations.append(bytes(data))
        for mutation in mutations:
            with (
                self.subTest(sha=hashlib.sha256(mutation).hexdigest()),
                self.assertRaises(ValueError),
            ):
                qualification.executable_sections(mutation)


class Int8DotQualificationTests(unittest.TestCase):
    def test_signed_register_dot_with_zero_or_register_accumulation(self):
        for instruction in (
            "IDP.4A.S8.S8 R10, R10, R13, RZ",
            "@P0 IDP.4A.S8.S8 R10, R21, R20, R10",
            "@!P6 IDP.4A.S8.S8 R0, R254, R1, R253",
        ):
            result = qualify_instructions(instruction)
            self.assertEqual(
                result["metadata_unused_profile"], qualification.INT8_DOT_PROFILE
            )
            self.assertEqual(result["cb4_accesses"], [])
            self.assertEqual(result["instruction_count"], 1)
        mixed = qualify_instructions(
            "IMMA.16832.S8.S8.SAT R8, R8.ROW, R22.COL, RZ",
            "IDP.4A.S8.S8 R10, R21, R20, R10",
        )
        self.assertEqual(
            mixed["metadata_unused_profile"], qualification.INT8_DOT_PROFILE
        )

    def test_literal_cb2_dot_reads_only_declared_aligned_words(self):
        for address in (0, 4, 65532):
            instruction = f"IDP.4A.S8.S8 R1, R2, c[0x2][{address:#x}], RZ"
            result = qualification.qualify(
                elf(
                    [(".text.entry", 16)],
                    constants=[(".nv.constant2.entry", b"\x01" * (address + 4), 1, 2)],
                ),
                sass([(".text.entry", [instruction])]),
            )
            self.assertEqual(
                result["metadata_unused_profile"], qualification.INT8_DOT_CB2_PROFILE
            )
            self.assertEqual(result["cb4_accesses"], [])
            self.assertEqual(
                result["cb2_dot_accesses"],
                [
                    {
                        "section": ".text.entry",
                        "offset": 0,
                        "constant_offset": address,
                        "bytes": 4,
                    }
                ],
            )
        for address in (1, 2, 5, 8, 65536, 1 << 40):
            with self.subTest(address=address), self.assertRaises(ValueError):
                qualification.qualify(
                    elf(
                        [(".text.entry", 16)],
                        constants=[(".nv.constant2.entry", bytes(8), 1, 2)],
                    ),
                    sass(
                        [
                            (
                                ".text.entry",
                                [f"IDP.4A.S8.S8 R1, R2, c[0x2][{address:#x}], R3"],
                            )
                        ]
                    ),
                )

    def test_literal_cb2_dot_rejects_ambiguous_missing_and_mutable_banks(self):
        local = (".nv.constant2.entry", bytes(4), 1, 2)
        variants = [
            [],
            [(".nv.constant2.other", bytes(4), 1, 2)],
            [(".nv.constant2", bytes(4), 1, 2)],
            [local, local],
            [local, (".nv.constant2", bytes(4), 1, 2)],
        ]
        variants.extend(
            [
                [(local[0], bytes(size), kind, flags)]
                for size, kind, flags in [
                    (0, 1, 2),
                    (65540, 1, 2),
                    (4, 8, 2),
                    (4, 3, 2),
                    (4, 1, 0),
                    (4, 1, 3),
                    (4, 1, 6),
                ]
            ]
        )
        for constants in variants:
            with (
                self.subTest(metadata=[(n, len(b), k, f) for n, b, k, f in constants]),
                self.assertRaises(ValueError),
            ):
                qualification.qualify(
                    elf([(".text.entry", 16)], constants=constants),
                    sass([(".text.entry", ["IDP.4A.S8.S8 R1, R2, c[0x2][0x0], RZ"])]),
                )

    def test_literal_cb2_dot_retains_bank_operand_and_register_restrictions(self):
        binary = elf(
            [(".text.entry", 16)], constants=[(".nv.constant2.entry", bytes(4), 1, 2)]
        )
        invalid = [
            f"IDP.4A.S8.S8 R1, R2, {value}, RZ"
            for value in [
                "c[0x0][0x0]",
                "c[0x4][0x8]",
                "c[0x2][R0]",
                "c[0x2][0x0+R0]",
                "cx[UR0]",
                "-c[0x2][0x0]",
                "c[0x2][0x0].H0",
            ]
        ]
        invalid.extend(
            [
                "IDP.4A.S8.S8 c[0x2][0x0], R1, R2, RZ",
                "IDP.4A.S8.S8 R0, c[0x2][0x0], R2, RZ",
                "IDP.4A.S8.S8 R0, R1, R2, c[0x2][0x0]",
                "IDP.4A.U8.S8 R0, R1, c[0x2][0x0], RZ",
                "IDP.4A.S8.S8 R255, R1, c[0x2][0x0], RZ",
            ]
        )
        for instruction in invalid:
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                qualification.qualify(binary, sass([(".text.entry", [instruction])]))

    def test_observed_unsigned_extraction_keeps_its_exact_operand_contract(self):
        accepted = qualify_instructions("SGXT.U32 R1, R254, 0x4")
        self.assertEqual(
            accepted["metadata_unused_profile"], qualification.INT8_DOT_CB2_PROFILE
        )
        self.assertEqual(
            accepted["u32_extracts"],
            [{"section": ".text.entry", "offset": 0, "literal": 4}],
        )
        invalid = [
            f"{op} R1, R2, 0x4"
            for op in ["SGXT", "SGXT.S32", "SGXT.U16", "SGXT.U32.SAT"]
        ]
        invalid.extend(
            f"SGXT.U32 R1, R2, {count}"
            for count in ["4", "0x3", "0x5", "0x20", "R0", "UR0", "c[0x2][0x0]"]
        )
        invalid.extend(
            [
                "SGXT.U32 R255, R1, 0x4",
                "SGXT.U32 R0, RZ, 0x4",
                "SGXT.U32 R1, c[0x0][0x160], 0x4",
                "SGXT.U32 R1, R2.reuse, 0x4",
                "SGXT.U32 R1, R2, 0x4, R3",
            ]
        )
        for instruction in invalid:
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                qualify_instructions(instruction)

    def test_other_dot_shapes_types_and_modifiers_reject(self):
        for opcode in (
            "IDP",
            "IDP.4A",
            "IDP.2A.S8.S8",
            "IDP.4A.U8.S8",
            "IDP.4A.S8.U8",
            "IDP.4A.S16.S16",
            "IDP.4A.S8.S8.SAT",
            "IDP.4A.S8.S8.REUSE",
            "IDP.4A.S8.S8.UNKNOWN",
        ):
            with self.subTest(opcode=opcode), self.assertRaises(ValueError):
                qualify_instructions(f"{opcode} R10, R21, R20, R10")

    def test_non_register_operands_and_unobserved_register_modifiers_reject(self):
        operands = ["R10", "R21", "R20", "R10"]
        for slot in range(4):
            for value in (
                "R255",
                "R999",
                "R01",
                "UR0",
                "URZ",
                "0",
                "0x0",
                "-R0",
                "R0.reuse",
                "R0.ROW",
                "R0.H0",
                "|R0|",
                "[R0]",
                "R0+0x4",
                "c[0x0][0x160]",
                "c[0x2][0x8]",
                "c[0x4][0x8]",
                "cx[UR0]",
            ):
                changed = operands.copy()
                changed[slot] = value
                with (
                    self.subTest(slot=slot, value=value),
                    self.assertRaises(ValueError),
                ):
                    qualify_instructions("IDP.4A.S8.S8 " + ", ".join(changed))
        for slot in range(3):
            changed = operands.copy()
            changed[slot] = "RZ"
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                qualify_instructions("IDP.4A.S8.S8 " + ", ".join(changed))


class Int8MmaQualificationTests(unittest.TestCase):
    """Synthetic inventories with the register form observed in retained SASS."""

    observed = "IMMA.16832.S8.S8.SAT R8, R8.ROW, R22.COL, RZ"

    def test_observed_signed_zero_accumulator_form(self):
        for instruction in (
            self.observed,
            "@P0 " + self.observed,
            "@!P6 " + self.observed,
            "@!UP0\tIMMA.16832.S8.S8.SAT\tR0,\tR16.ROW, R24.COL,RZ",
            "IMMA.16832.S8.S8.SAT R252 , R248.ROW , R250.COL , RZ",
            "IMMA.16832.S8.S8.SAT R44, R36.reuse.ROW, R44.COL, RZ",
            "IMMA.16832.S8.S8.SAT R44, R44.ROW, R86.reuse.COL, RZ",
            "IMMA.16832.S8.S8.SAT R44, R36.reuse.ROW, R86.reuse.COL, RZ",
        ):
            with self.subTest(instruction=instruction):
                result = qualify_instructions(instruction)
                self.assertEqual(
                    result["metadata_unused_profile"], "sm89-direct-constant-v3"
                )
                self.assertEqual(result["instruction_count"], 1)
                self.assertEqual(result["cb4_accesses"], [])

    def test_other_shapes_types_or_modifiers_reject(self):
        for opcode in (
            "IMMA",
            "IMMA.16832.S8.S8",
            "IMMA.16816.S8.S8.SAT",
            "IMMA.16864.S8.S8.SAT",
            "IMMA.16832.U8.S8.SAT",
            "IMMA.16832.S8.U8.SAT",
            "IMMA.16832.S4.S4.SAT",
            "IMMA.16832.S8.S8.SATFINITE",
            "IMMA.16832.S8.S8.SAT.REUSE",
        ):
            with self.subTest(opcode=opcode), self.assertRaises(ValueError):
                qualify_instructions(
                    self.observed.replace("IMMA.16832.S8.S8.SAT", opcode)
                )

    def test_non_register_operands_layouts_and_accumulators_reject(self):
        operands = ["R8", "R8.ROW", "R22.COL", "RZ"]
        mutations = (
            (0, "RZ"),
            (0, "UR8"),
            (0, "R255"),
            (0, "R999"),
            (0, "R08"),
            (0, "R8.reuse"),
            (0, "R8.ROW"),
            (0, "-R8"),
            (1, "R8"),
            (1, "R8.COL"),
            (1, "RZ.ROW"),
            (1, "UR8.ROW"),
            (1, "R8.ROW.reuse"),
            (1, "R8.reuse.reuse.ROW"),
            (1, "R8.REUSE.ROW"),
            (1, "|R8|.ROW"),
            (2, "R22"),
            (2, "R22.ROW"),
            (2, "R22.COL.H0"),
            (2, "R255.COL"),
            (2, "R22.COL.reuse"),
            (2, "R22.reuse.reuse.COL"),
            (2, "R22.REUSE.COL"),
            (2, "R255.reuse.COL"),
            (2, "RZ.reuse.COL"),
            (2, "UR22.reuse.COL"),
            (2, "-R22.reuse.COL"),
            (3, "R0"),
            (3, "R8"),
            (3, "URZ"),
            (3, "-RZ"),
            (3, "RZ.reuse"),
        )
        # Constants, immediates and indirect addresses cannot occupy any slot.
        for slot, replacement in (
            *mutations,
            *(
                (slot, value)
                for slot in range(4)
                for value in (
                    "0",
                    "0x0",
                    "c[0x0][0x160]",
                    "c[0x2][0x8]",
                    "c[0x4][0x8]",
                    "c[R0][0x8]",
                    "cx[UR0]",
                    "[R8]",
                    "R8+0x4",
                )
            ),
        ):
            changed = operands.copy()
            changed[slot] = replacement
            with (
                self.subTest(slot=slot, operand=replacement),
                self.assertRaises(ValueError),
            ):
                qualify_instructions("IMMA.16832.S8.S8.SAT " + ", ".join(changed))
        for suffix in (", RZ", ", c[0x4][0x8]", " garbage"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                qualify_instructions(self.observed + suffix)

    def test_mma_does_not_relax_inventory_or_producer_checks(self):
        cubin = elf([(".text.entry", 32)])
        original = sass([(".text.entry", [self.observed, "EXIT"])])
        for mutation in (
            original.replace("/*0010*/ EXIT;", ""),
            original.replace("/*0010*/", "/*0000*/"),
            original.replace("/*0000*/", "/*0010*/", 1),
            original + "/*0020*/ NOP;\n",
            original.replace("sm_89", "sm_90", 1),
            original.replace("V13.4.92", "V13.4.93"),
        ):
            with self.subTest(sass=mutation), self.assertRaises(ValueError):
                qualification.qualify(cubin, mutation)

    def test_profile_depends_only_on_validated_executable_mma(self):
        baseline = qualify_instructions("EXIT")
        synthetic_cubin = elf([(".text.entry", 16)])
        annotated = sass([(".text.entry", ["EXIT"])]) + "// " + self.observed + "\n"
        annotated += '.section .nv.info,"",@progbits\n' + self.observed + ";\n"
        self.assertEqual(
            qualification.qualify(synthetic_cubin, annotated)[
                "metadata_unused_profile"
            ],
            "sm89-direct-constant-v2",
        )
        for sections in (
            [(".text.entry", [self.observed]), (".text.helper", ["EXIT"])],
            [(".text.entry", ["EXIT"]), (".text.helper", [self.observed])],
        ):
            with self.subTest(sections=sections):
                result = qualification.qualify(
                    elf(
                        [
                            (name, len(instructions) * 16)
                            for name, instructions in sections
                        ]
                    ),
                    sass(sections),
                )
                self.assertEqual(
                    result["metadata_unused_profile"], "sm89-direct-constant-v3"
                )
                evidence = result.pop("evidence_sha256")
                self.assertEqual(
                    evidence,
                    hashlib.sha256(
                        json.dumps(
                            result, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                )
        self.assertEqual(qualify_instructions("EXIT"), baseline)


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.kernels = self.root / "kernels"
        self.toolkit = self.root / "toolkit"
        self.directory = self.root / "stage"
        (self.kernels / "host-audit").mkdir(parents=True)
        self.manifest = {"kernels": {}}
        self.cubin = elf([(".text.entry", 16)])
        self.sass = sass([(".text.entry", ["EXIT"])])
        for family in guest.FAMILIES:
            suffix = "ptx" if family == "tensor-matmul" else "tilebc"
            source = self.kernels / f"{family}.{suffix}"
            source.write_bytes(family.encode())
            cubin = self.kernels / f"host-audit/{family}.cubin"
            cubin.write_bytes(self.cubin)
            (self.kernels / f"host-audit/{family}.sass").write_text(self.sass)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            self.manifest["kernels"][family] = {
                "input": source.name,
                "bytes": source.stat().st_size,
                "sha256": source_hash,
                "host_audit": {
                    "cubin": f"host-audit/{family}.cubin",
                    "sass": f"host-audit/{family}.sass",
                    "architecture": "sm_89",
                    "sass_verified": True,
                    "bytes": len(self.cubin),
                    "sha256": hashlib.sha256(self.cubin).hexdigest(),
                },
                "legacy_qualification": qualification.qualify(self.cubin, self.sass)
                | {"source_sha256": source_hash, "source_bytes": source.stat().st_size},
            }
        self.write_manifest()
        for path in (
            "bin/tileiras",
            "bin/ptxas",
            "lib/libnvvm.so.4",
            "nvvm/libdevice/libdevice.10.bc",
        ):
            destination = self.toolkit / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.encode())
        self.library = self.root / "libc.so.6"
        self.library.write_bytes(b"test runtime library")
        version = patch.object(
            guest.subprocess, "check_output", return_value="Cuda V13.4.92\n"
        )
        runtime = patch.object(
            guest,
            "runtime_assets",
            return_value=[(self.library, Path("lib64/libc.so.6"))],
        )
        version.start()
        runtime.start()
        self.addCleanup(version.stop)
        self.addCleanup(runtime.stop)

    def write_manifest(self):
        (self.kernels / "manifest.json").write_text(json.dumps(self.manifest))

    def prepare(self, **options):
        return guest.prepare(self.kernels, self.toolkit, self.directory, **options)

    def test_optional_qwen_module_is_qualified_and_sealed_like_core_modules(self):
        record = self.manifest["kernels"]["inference"]
        record = json.loads(json.dumps(record))
        record["input"] = "qwen-vector.ptx"
        (self.kernels / record["input"]).write_bytes(
            (self.kernels / "inference.tilebc").read_bytes()
        )
        self.manifest["kernels"]["qwen-vector"] = record
        self.write_manifest()
        assets = {
            str(destination): source
            for source, destination in self.prepare(precompiled=True)
        }
        staged = json.loads(
            (self.directory / "guest-compiler-manifest.json").read_text()
        )
        self.assertEqual(
            guest.family_names(staged["legacy_qualifications"]),
            (*guest.FAMILIES, "qwen-vector"),
        )
        receipt = assets["usr/share/nvidia/qwen-vector.qualification"].read_bytes()
        self.assertEqual(struct.unpack_from("<I", receipt, 24)[0], 2)
        self.assertEqual(receipt[-32:].hex(), staged["runtime_inventory_sha256"])
        self.assertIn("usr/share/nvidia/qwen-vector.ptx", staged["assets"])
        self.assertIn("usr/share/nvidia/qwen-vector.cubin", staged["assets"])
        (self.kernels / record["input"]).write_bytes(b"changed source")
        with self.assertRaises(RuntimeError):
            self.prepare(precompiled=True)

    def test_optional_families_cannot_remove_core_or_add_arbitrary_modules(self):
        for names in [("inference", "qwen-vector"), (*guest.FAMILIES, "unknown")]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                guest.family_names(names)

    def test_precompiled_modules_are_covered_by_the_same_runtime_inventory(self):
        assets = {
            str(destination): source
            for source, destination in self.prepare(precompiled=True)
        }
        manifest = json.loads(
            (self.directory / "guest-compiler-manifest.json").read_text()
        )
        self.assertEqual(manifest["module_mode"], "qualified_cache")
        self.assertEqual(len(manifest["assets"]), 11)
        for family in guest.FAMILIES:
            cached = f"usr/share/nvidia/{family}.cubin"
            identity = manifest["assets"][cached]
            self.assertEqual(identity["bytes"], len(self.cubin))
            self.assertEqual(
                identity["sha256"],
                hashlib.sha256(assets[cached].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                identity["sha256"],
                manifest["legacy_qualifications"][family]["cubin_sha256"],
            )
        runtime = assets["usr/share/nvidia/runtime.qualification"].read_bytes()
        self.assertEqual(struct.unpack_from("<I", runtime, 8)[0], 11)
        self.assertEqual(runtime[12:], guest.runtime_inventory(manifest["assets"]))
        for family in guest.FAMILIES:
            receipt = assets[f"usr/share/nvidia/{family}.qualification"].read_bytes()
            self.assertEqual(receipt[-32:], hashlib.sha256(runtime[12:]).digest())

    def test_precompiled_staging_rejects_changed_audited_code(self):
        cubin = self.kernels / "host-audit/inference.cubin"
        changed = bytearray(cubin.read_bytes())
        changed[-1] ^= 1
        cubin.write_bytes(changed)
        with self.assertRaises(RuntimeError):
            self.prepare(precompiled=True)
        self.assertFalse(self.directory.exists())

    def test_precompiled_mode_cannot_be_misattributed_to_linux(self):
        with self.assertRaises(ValueError):
            self.prepare(precompiled=True, execution_os="linux")

    def test_receipts_bind_actual_assets_and_exact_host_output(self):
        assets = self.prepare()
        staged = {
            destination.as_posix(): source.read_bytes()
            for source, destination in assets
        }
        runtime = staged["usr/share/nvidia/runtime.qualification"]
        self.assertEqual(runtime[:8], b"D3QRT001")
        count = struct.unpack_from("<I", runtime, 8)[0]
        self.assertEqual(count, 8)
        offset = 12
        paths = []
        for _ in range(count):
            length = struct.unpack_from("<I", runtime, offset)[0]
            offset += 4
            path = runtime[offset : offset + length].decode()
            offset += length
            size = struct.unpack_from("<Q", runtime, offset)[0]
            offset += 8
            checksum = runtime[offset : offset + 32]
            offset += 32
            self.assertEqual(size, len(staged[path]))
            self.assertEqual(checksum, hashlib.sha256(staged[path]).digest())
            paths.append(path)
        self.assertEqual(offset, len(runtime))
        self.assertEqual(paths, sorted(paths))
        runtime_hash = hashlib.sha256(runtime[12:]).digest()
        for family in guest.FAMILIES:
            module = staged[f"usr/share/nvidia/{family}.qualification"]
            self.assertEqual(len(module), 192)
            self.assertEqual(module[:8], b"D3QMOD01")
            kind = 2 if family == "tensor-matmul" else 1
            self.assertEqual(
                struct.unpack_from("<6I", module, 8), (1, 1, 89, 3, kind, 0)
            )
            compiler = "ptxas" if kind == 2 else "tileiras"
            expected = [
                hashlib.sha256(family.encode()).digest(),
                hashlib.sha256(self.cubin).digest(),
                hashlib.sha256(self.sass.encode()).digest(),
                hashlib.sha256(staged[f"usr/cuda/bin/{compiler}"]).digest(),
                runtime_hash,
            ]
            self.assertEqual([module[i : i + 32] for i in range(32, 192, 32)], expected)

    def test_evidence_mutations_reject_before_staging(self):
        original = copy.deepcopy(self.manifest)
        for section, field, value in (
            (None, "sha256", "00" * 32),
            (None, "bytes", 0),
            (None, "input", "other.tilebc"),
            ("host_audit", "sha256", "00" * 32),
            ("host_audit", "bytes", 0),
            ("host_audit", "cubin", "../outside.cubin"),
            ("host_audit", "sass_verified", False),
            ("legacy_qualification", "source_sha256", "00" * 32),
            ("legacy_qualification", "sass_sha256", "00" * 32),
            ("legacy_qualification", "metadata_unused_profile", "unqualified"),
            ("legacy_qualification", "instruction_count", 2),
        ):
            self.manifest = copy.deepcopy(original)
            record = self.manifest["kernels"][guest.FAMILIES[0]]
            target = record if section is None else record[section]
            target[field] = value
            self.write_manifest()
            with (
                self.subTest(section=section, field=field),
                self.assertRaises(RuntimeError),
            ):
                self.prepare()
            self.assertFalse(self.directory.exists())

    def test_actual_evidence_mutations_reject(self):
        family = guest.FAMILIES[0]
        for relative in (
            f"{family}.tilebc",
            f"host-audit/{family}.cubin",
            f"host-audit/{family}.sass",
        ):
            path = self.kernels / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(path=relative), self.assertRaises(RuntimeError):
                self.prepare()
            path.write_bytes(original)

    def test_absent_audit_and_duplicate_or_unknown_family_reject(self):
        original = copy.deepcopy(self.manifest)
        for field in ("host_audit", "legacy_qualification"):
            self.manifest = copy.deepcopy(original)
            del self.manifest["kernels"][guest.FAMILIES[0]][field]
            self.write_manifest()
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.prepare()
        self.manifest = copy.deepcopy(original)
        self.manifest["kernels"]["unknown"] = {}
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            self.prepare()
        raw = json.dumps(original)
        (self.kernels / "manifest.json").write_text(raw[:-1] + ', "kernels": {}}')
        with self.assertRaises(RuntimeError):
            self.prepare()

    def test_compiler_version_rejects_before_receipt(self):
        with (
            patch.object(
                guest.subprocess, "check_output", return_value="Cuda V13.4.93\n"
            ),
            self.assertRaises(RuntimeError),
        ):
            self.prepare()

    def test_inventory_is_sorted_bounded_and_unambiguous(self):
        record = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
        first = guest.runtime_inventory({"usr/b": record, "lib64/a": record})
        second = guest.runtime_inventory({"lib64/a": record, "usr/b": record})
        self.assertEqual(first, second)
        self.assertNotEqual(
            first, guest.runtime_inventory({"usr/bb": record, "lib64/a": record})
        )
        for path in (
            "",
            ".",
            "/usr/a",
            "usr//a",
            "usr/./a",
            "usr/../a",
            "usr/a/",
            "usr\\a",
            "usr/\x00a",
            "a" * 513,
            "usr/a.qualification",
            "usr/manifest.json",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                guest.runtime_inventory({path: record})
        for count in (0, 129):
            with self.subTest(count=count), self.assertRaises(ValueError):
                guest.runtime_inventory({f"usr/{i}": record for i in range(count)})
        self.assertTrue(
            guest.runtime_inventory({f"usr/{i}": record for i in range(128)})
        )

    def test_zero_or_invalid_inventory_identities_reject(self):
        checksum = hashlib.sha256(b"abc").hexdigest()
        for size in (0, -1, 1 << 64, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                guest.runtime_inventory({"usr/a": {"bytes": size, "sha256": checksum}})
        for checksum in ("00" * 32, "00" * 31, "invalid"):
            with self.subTest(checksum=checksum), self.assertRaises(ValueError):
                guest.runtime_inventory({"usr/a": {"bytes": 3, "sha256": checksum}})

    def test_zero_module_identity_rejects_in_every_field(self):
        checksum = hashlib.sha256(b"abc").hexdigest()
        record = {
            "source_sha256": checksum,
            "cubin_sha256": checksum,
            "sass_sha256": checksum,
        }
        for field in record:
            with self.subTest(field=field), self.assertRaises(ValueError):
                guest.module_receipt(record | {field: "00" * 32}, 1, checksum, checksum)
        for compiler, runtime in (("00" * 32, checksum), (checksum, "00" * 32)):
            with (
                self.subTest(compiler=compiler, runtime=runtime),
                self.assertRaises(ValueError),
            ):
                guest.module_receipt(record, 1, compiler, runtime)


class ModuleAdmissionTests(unittest.TestCase):
    def fixture(self, cached, *, legacy=False):
        manifest = {
            "module_mode": "qualified_cache" if cached else "native_compile",
            "assets": {},
            "legacy_qualifications": {},
        }
        lines = []
        for family in guest.FAMILIES:
            source = hashlib.sha256(("source:" + family).encode()).hexdigest()
            cubin = hashlib.sha256(("cubin:" + family).encode()).hexdigest()
            manifest["assets"][f"usr/share/nvidia/{family}.tilebc"] = {
                "bytes": 96,
                "sha256": source,
            }
            manifest["legacy_qualifications"][family] = {
                "source_sha256": source,
                "cubin_sha256": cubin,
                "cubin_bytes": 128,
            }
            if cached:
                manifest["assets"][f"usr/share/nvidia/{family}.cubin"] = {
                    "bytes": 128,
                    "sha256": cubin,
                }
            phases = (
                "compile_ms=12 image_load_ms=0 execute_ms=0 output_ms=12"
                if cached
                else "compile_ms=12 image_load_ms=2 execute_ms=8 output_ms=2"
            )
            mode = (
                "qualified_cache compiler_executed=false"
                if cached
                else "native_compile compiler_executed=true"
            )
            suffix = "" if legacy else " mode=" + mode
            lines.append(
                f"[1.000] NVIDIA_COMPILE_MODULE os=d3os family={family} input=tile_ir input_sha256={source} target=sm_89 {phases} bytes=128 sha256={cubin} exit=0{suffix}"
            )
        return manifest, lines

    def check(self, manifest, lines, cached):
        return native.validate_module_admission(
            "\n".join(lines), manifest, guest.FAMILIES, precompiled=cached
        )

    def test_native_legacy_and_explicit_cache_modes_bind_all_three_families(self):
        for cached, legacy in [(False, False), (False, True), (True, False)]:
            with self.subTest(cached=cached, legacy=legacy):
                manifest, lines = self.fixture(cached, legacy=legacy)
                lines = ["\x1b[36m" + line + "\x1b[39m" for line in lines]
                self.assertEqual(
                    set(self.check(manifest, lines, cached)), set(guest.FAMILIES)
                )

    def test_missing_duplicate_unknown_and_prefixed_records_reject(self):
        manifest, lines = self.fixture(True)
        mutations = [
            lines[:-1],
            lines + [lines[0]],
            [lines[0], lines[0], lines[2]],
            [lines[0].replace("vector-add", "unknown"), *lines[1:]],
            [
                line.replace("NVIDIA_COMPILE_MODULE", "NOT_NVIDIA_COMPILE_MODULE")
                for line in lines
            ],
        ]
        for mutated in mutations:
            with self.subTest(lines=mutated), self.assertRaises(RuntimeError):
                self.check(manifest, mutated, True)

    def test_output_source_timing_mode_and_contradictory_status_mutations_reject(self):
        for cached in [False, True]:
            manifest, lines = self.fixture(cached)
            source = manifest["legacy_qualifications"]["vector-add"]["source_sha256"]
            cubin = manifest["legacy_qualifications"]["vector-add"]["cubin_sha256"]
            replacements = [
                (source, "00" * 32),
                (cubin, "00" * 32),
                ("bytes=128", "bytes=127"),
                ("target=sm_89", "target=sm_90"),
                ("compile_ms=12", "compile_ms=13"),
                ("exit=0", "exit=1"),
            ]
            replacements += (
                [
                    ("compiler_executed=false", "compiler_executed=true"),
                    (
                        "image_load_ms=0 execute_ms=0 output_ms=12",
                        "image_load_ms=0 execute_ms=1 output_ms=11",
                    ),
                ]
                if cached
                else [
                    (
                        "mode=native_compile compiler_executed=true",
                        "mode=qualified_cache compiler_executed=false",
                    )
                ]
            )
            for old, new in replacements:
                with (
                    self.subTest(cached=cached, replacement=(old, new)),
                    self.assertRaises(RuntimeError),
                ):
                    self.check(
                        manifest, [lines[0].replace(old, new), *lines[1:]], cached
                    )
            for suffix in [" status=error", " exit=1", " mode=native_compile"]:
                with (
                    self.subTest(cached=cached, suffix=suffix),
                    self.assertRaises(RuntimeError),
                ):
                    self.check(manifest, [lines[0] + suffix, *lines[1:]], cached)

    def test_cache_requires_bound_cubin_and_explicit_zero_execution(self):
        manifest, lines = self.fixture(True)
        for field, value in [("sha256", "00" * 32), ("bytes", 127)]:
            changed = copy.deepcopy(manifest)
            changed["assets"]["usr/share/nvidia/vector-add.cubin"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.check(changed, lines, True)
        changed = copy.deepcopy(manifest)
        del changed["assets"]["usr/share/nvidia/vector-add.cubin"]
        with self.assertRaises(RuntimeError):
            self.check(changed, lines, True)
        with self.assertRaises(RuntimeError):
            self.check(manifest, [line.split(" mode=", 1)[0] for line in lines], True)
        with self.assertRaises(RuntimeError):
            self.check(manifest, lines, False)


if __name__ == "__main__":
    unittest.main()
