"""CPU artifact tests. Set TYPED_MATH_SCRATCH_ARTIFACTS to a fresh entry export.

Optionally set TYPED_MATH_SCRATCH_BASELINE_SASS to the old inference-family
disassembly to exercise the complete production audit with both entry versions.
Set TYPED_MATH_SCRATCH_PREVIOUS_ARTIFACTS to the corrected Power24 v2 export
to check that it cannot enter under the corrected profile.
Set TYPED_MATH_SCRATCH_REJECTED_ARTIFACTS to the first Erf25 export with the
incomplete unary predicate to check that it cannot enter the final profile.
Set TYPED_MATH_SCRATCH_PRIOR_ERF_REPORT to the accepted Erf25 v1 report to
check that renewal preserves its certificate and rejects stale provenance.
"""

import copy
import importlib.util
import json
import os
import re
import struct
import unittest
from collections import Counter
from pathlib import Path


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load("scratch_test_subject", "typed-math-scratch.py")
prepare = load("prepare_test_subject", "prepare-kernels.py")


class ScratchArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        location = os.environ.get("TYPED_MATH_SCRATCH_ARTIFACTS")
        if not location:
            raise unittest.SkipTest(
                "requires a freshly disassembled typed-math entry export"
            )
        directory = Path(location)
        cls.cubin = (directory / "kernels.cubin").read_bytes()
        cls.sass = (directory / "kernels.sass").read_text()
        cls.entries = json.loads((directory / "kernels.json").read_text())
        cls.elf = scratch.Cubin(cls.cubin)

    def qualify(self, *, cubin=None, sass=None, entries=None, provenance=None):
        return scratch.qualify(
            self.cubin if cubin is None else cubin,
            self.sass if sass is None else sass,
            self.entries if entries is None else entries,
            scratch.PROVENANCE if provenance is None else provenance,
        )

    def change_instruction(self, offset, transform):
        pattern = (
            rf"(^[ \t]*/\*{offset:04x}\*/[ \t]*"
            r"(?:@!?U?P(?:\d+|T)[ \t]+)?[A-Z][^\n]+)$"
        )

        def replace(match):
            changed = transform(match[0])
            self.assertNotEqual(changed, match[0])
            return changed

        mutated, count = re.subn(pattern, replace, self.sass, flags=re.MULTILINE)
        self.assertEqual(count, 1)
        return mutated

    def mutate_record(self, section_name, kind, transform):
        data = bytearray(self.cubin)
        section = self.elf.named[section_name]
        cursor, end = section["start"], section["start"] + section["size"]
        while cursor < end:
            form, key, size = struct.unpack_from("<BBH", data, cursor)
            length = size if form == 4 else 0
            if key == kind and transform(data, cursor + 4, length) is not False:
                return bytes(data)
            cursor += 4 + length
        self.fail("requested NVIDIA record does not exist")

    def test_reviewed_entry_classifies_every_local_site(self):
        result = self.qualify()
        arrays = result["classified_arrays"]
        self.assertEqual(len(arrays), 28)
        self.assertEqual(sum(len(array["sites"]) for array in arrays), 194)
        self.assertEqual(
            {tuple(array["frame_range"]) for array in arrays},
            {(0, 28), (28, 56), (56, 96)},
        )
        self.assertEqual(result["unclassified_local_sites"], [])
        self.assertEqual(len(result["call_sites"]), 135)
        self.assertEqual(result["registers"], 88)
        self.assertEqual(result["code_bytes"], 224512)
        self.assertEqual(result["frame_bytes"], 96)
        self.assertEqual(result["minimum_stack_bytes"], 96)
        self.assertEqual(len(result["helper_frame_bytes"]), 7)
        self.assertEqual(set(result["helper_frame_bytes"].values()), {0})
        self.assertEqual(result["local_load_sites"], 84)
        self.assertEqual(result["local_store_sites"], 110)
        self.assertEqual(sum(a["frame_range"] == [28, 56] for a in arrays), 18)
        self.assertEqual(sum(a["frame_range"] == [0, 28] for a in arrays), 9)

    def test_registered_erf_dispatch_and_arithmetic_are_bound(self):
        _, rows, calls, _ = scratch.schedule(
            self.elf, self.sass, scratch.load_qualifier()
        )
        self.assertEqual(
            Counter(call["target"] for call in calls),
            {
                "$__cuda_sm20_dblrcp_rn_slowpath_v3": 6,
                "$__cuda_sm20_div_rn_f64_full": 24,
                "$__cuda_sm20_dsqrt_rn_f64_mediumpath_v1": 3,
                "$__cuda_sm20_sqrt_rn_f32_slowpath": 9,
                "$__cuda_sm3x_div_rn_noftz_f32_slowpath": 81,
                f"${scratch.ENTRY}$__internal_accurate_pow": 3,
                scratch.TRIG: 9,
            },
        )
        dispatches = [
            row
            for row in rows
            if row[2] == "ISETP.NE.U32.AND" and ", 0x21, PT" in row[3]
        ]
        self.assertEqual(len(dispatches), 8)
        # Exercise each emitted Erf region. These mutations check schedule
        # binding only; instruction counts do not establish numerical accuracy.
        for dispatch in dispatches:
            arithmetic = next(
                row
                for row in rows
                if row[0] > dispatch[0] and row[2] in ("DFMA", "FFMA")
            )
            changes = (
                (dispatch[0], lambda row: row.replace("0x21,", "0x20,")),
                (
                    arithmetic[0],
                    lambda row: row.replace("DFMA", "DADD").replace("FFMA", "FADD"),
                ),
            )
            for offset, transform in changes:
                with self.subTest(offset=offset), self.assertRaises(ValueError):
                    self.qualify(sass=self.change_instruction(offset, transform))

    def test_sdk_generated_unary_erf_predicate_is_bound(self):
        changes = (
            (0x3C0, lambda row: row.replace("0x21,", "0x22,")),
            (0x3C0, lambda row: row.replace("ISETP.EQ.U32.OR", "ISETP.EQ.U32.AND")),
            (0x410, lambda row: row.replace("P4 ;", "!P4 ;")),
        )
        for offset, transform in changes:
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(offset, transform))

    def test_observed_table_carry_has_no_broader_cb4_admission(self):
        q = scratch.load_qualifier()
        _, rows, _, _ = scratch.schedule(self.elf, self.sass, q)
        carries = [row for row in rows if row[2] == "IADD3.X" and "c[0x4]" in row[3]]
        self.assertEqual(len(carries), 2)
        changes = (
            lambda row: row.replace("0x14]", "0x18]"),
            lambda row: row.replace("0x14]", "0x4]"),
            lambda row: row.replace("0x14]", "R1]"),
            lambda row: row.replace("R45, RZ,", "R45, R2,"),
            lambda row: row.replace("], RZ,", "], R2,"),
            lambda row: row.replace("!PT", "PT"),
            lambda row: row.replace("IADD3.X", "IADD3.X.U32"),
            lambda row: row.replace("IADD3.X", "FADD"),
        )
        for carry in carries:
            for index, transform in enumerate(changes):
                changed = self.change_instruction(carry[0], transform)
                # Reject independently of the scratch schedule fingerprint.
                with (
                    self.subTest(offset=carry[0], form=index),
                    self.assertRaises(ValueError),
                ):
                    q.qualify(self.cubin, changed)
                with (
                    self.subTest(offset=carry[0], scratch_form=index),
                    self.assertRaises(ValueError),
                ):
                    self.qualify(sass=changed)

    def test_unknown_local_forms_bases_offsets_and_slots_fail(self):
        changes = {
            "unknown local opcode": (
                0x82B0,
                lambda row: row.replace("LDL ", "LDL.128 "),
            ),
            "unknown store base": (0x8140, lambda row: row.replace("[R1+", "[R2+")),
            "out-of-array store": (0x8140, lambda row: row.replace("+0x1c]", "+0x60]")),
            "unknown load base": (0x82B0, lambda row: row.replace("[R13+", "[R17+")),
            "out-of-array load": (0x82B0, lambda row: row.replace("+0x18]", "+0x1c]")),
            "new save slot": (
                0x82E0,
                lambda row: re.sub(r"@P0 SHF.*;", "STL [R1+0x60], R0 ;", row),
            ),
            "F32 bound": (0x8210, lambda row: row.replace("0x5,", "0x4,")),
            "F32 pointer scale": (0x8280, lambda row: row.replace("-0x4,", "-0x8,")),
            "F64 pointer base": (0x362A0, lambda row: row.replace("0x38,", "0x40,")),
            "F64 loop bound": (0x362F0, lambda row: row.replace("0x12,", "0x13,")),
            "F64 pointer stride": (0x364E0, lambda row: row.replace("0x8,", "0x10,")),
            "F64 final offset": (0x36600, lambda row: row.replace("-0x78]", "-0x70]")),
            "missing predicate": (0x36610, lambda row: row.replace("@P0", "   ")),
            "return address": (0x3700, lambda row: row.replace("0x3730", "0x3740")),
        }
        for name, (offset, transform) in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(offset, transform))

    def test_unknown_and_wrong_helper_targets_fail(self):
        for target in (
            "missing_helper",
            "$__internal_0_$__cuda_sm20_dblrcp_rn_slowpath_v3",
            ".L_x_765",
        ):
            changed = self.change_instruction(
                0x20E0,
                lambda row, target=target: re.sub(r"`\([^)]+\)", f"`({target})", row),
            )
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.qualify(sass=changed)
        changed = self.change_instruction(
            0x36270, lambda row: row.replace(".L_x_765", ".L_x_766")
        )
        with self.assertRaises(ValueError):
            self.qualify(sass=changed)

    def test_each_reduction_group_rejects_changed_addressing(self):
        arrays = self.qualify()["classified_arrays"]
        # Exercise all twenty-seven inlined reductions, including the new Tan
        # copies and the BF16/F16 paths with different register allocation.
        for array in arrays[:-1]:
            offset = array["sites"][4]
            changed = self.change_instruction(
                offset, lambda row: row.replace("+0x18]", "+0x1c]")
            )
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.qualify(sass=changed)

    def test_every_local_site_rejects_application_spills_without_hash_checks(self):
        _, rows, _, _ = scratch.schedule(self.elf, self.sass, scratch.load_qualifier())
        scratch.classify(rows)
        for index, row in enumerate(rows):
            if row[2].split(".")[0] not in ("LDL", "STL"):
                continue
            changed = rows.copy()
            changed[index] = (row[0], "", "STL", "[R1+0x60], R0")
            with self.subTest(offset=row[0]), self.assertRaises(ValueError):
                scratch.classify(changed)
        with self.assertRaises(ValueError):
            scratch.classify([*rows, (rows[-1][0] + 16, "", "STL", "[R1], R0")])

    def test_every_actual_call_rejects_a_different_resolved_helper(self):
        calls = self.qualify()["call_sites"]
        helpers = [
            symbol["name"]
            for symbol in self.elf.symbols
            if symbol["kind"] == 2 and symbol["name"] != scratch.ENTRY
        ]
        power = f"${scratch.ENTRY}$__internal_accurate_pow"
        self.assertEqual(sum(call["target"] == power for call in calls), 3)
        for call in calls:
            target = next(
                helper
                for helper in helpers
                if scratch.identity(helper) != call["target"]
            )
            changed = self.change_instruction(
                call["offset"],
                lambda row, target=target: re.sub(r"`\([^)]+\)", f"`({target})", row),
            )
            with self.subTest(offset=call["offset"]), self.assertRaises(ValueError):
                self.qualify(sass=changed)

    def test_every_reduction_pointer_and_f64_call_guard_is_bound(self):
        _, rows, calls, _ = scratch.schedule(
            self.elf, self.sass, scratch.load_qualifier()
        )
        for array in self.qualify()["classified_arrays"][:-1]:
            read = next(row for row in rows if row[0] == array["sites"][4])
            base = re.search(r"\[(R\d+)\+", read[3])[1]
            pointer = next(
                row
                for row in reversed(rows)
                if row[0] < read[0]
                and row[2] == "IMAD"
                and re.fullmatch(rf"{base}, R\d+, -0x4, R(?:1|50)", row[3])
            )
            changed = self.change_instruction(
                pointer[0], lambda row: row.replace("-0x4,", "-0x8,")
            )
            with self.subTest(pointer=pointer[0]), self.assertRaises(ValueError):
                self.qualify(sass=changed)
        trig_calls = [call for call in calls if call["target"] == scratch.TRIG]
        self.assertEqual(len(trig_calls), 9)
        for call in trig_calls:
            guard = next(
                row
                for row in reversed(rows)
                if row[0] < call["offset"]
                and row[2] == "DSETP.GE.AND"
                and "2.14748364800000000000e+09" in row[3]
            )
            changed = self.change_instruction(
                guard[0], lambda row: row.replace("DSETP.GE.AND", "DSETP.LE.AND")
            )
            with self.subTest(guard=guard[0]), self.assertRaises(ValueError):
                self.qualify(sass=changed)

    def test_power_computation_and_revised_trig_inputs_are_bound(self):
        # Power has no local accesses. Bind its arithmetic and the trig helper's
        # changed input-register setup as strictly as the reduction arrays.
        changes = {
            0x35A00: lambda row: row.replace("DMUL", "DADD"),
            0x36210: lambda row: row.replace("0x14,", "0x15,"),
            0x36220: lambda row: row.replace("R2 ;", "R3 ;"),
            0x36250: lambda row: row.replace("RZ ;", "0x1 ;"),
            0x36240: lambda row: row.replace("0x7ff,", "0x7fe,"),
            0x362B0: lambda row: row.replace("SRZ", "SR_CLOCKLO"),
        }
        for offset, transform in changes.items():
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(offset, transform))

    def test_every_helper_rejects_a_nonzero_frame(self):
        helpers = [
            symbol
            for symbol in self.elf.symbols
            if symbol["kind"] == 2 and symbol["name"] != scratch.ENTRY
        ]
        for helper in helpers:

            def add_frame(data, offset, length, helper_index=helper["index"]):
                for position in range(offset, offset + length, 8):
                    if struct.unpack_from("<I", data, position)[0] == helper_index:
                        struct.pack_into("<I", data, position + 4, 16)
                        return
                return False

            with self.subTest(helper=helper["name"]), self.assertRaises(ValueError):
                self.qualify(cubin=self.mutate_record(".nv.info", 0x11, add_frame))

    def test_f64_power_integer_cutoff_and_identities_are_bound(self):
        changes = {
            "parity limit": (0x43C0, lambda row: row.replace("0x433,", "0x434,")),
            "cutoff shift clamp": (0x4690, lambda row: row.replace("0x409,", "0x40a,")),
            "integer product": (
                0x47D0,
                lambda row: row.replace("IMAD.WIDE.U32", "IMAD.HI.U32"),
            ),
            "product low comparison": (
                0x4810,
                lambda row: row.replace("GE.U32", "GT.U32"),
            ),
            "product high comparison": (
                0x4820,
                lambda row: row.replace("GE.U32", "GT.U32"),
            ),
            "large exponent cutoff": (
                0x4860,
                lambda row: row.replace("0x4090cbff", "0x4090cbfe"),
            ),
            "nonzero repair": (0x47E0, lambda row: row.replace("0x1,", "0x2,")),
            "exponent one identity": (0x4B00, lambda row: row.replace("R8,", "R9,")),
            "negative one infinite identity": (
                0x4D50,
                lambda row: row.replace("P3 ;", "!P3 ;"),
            ),
            "base one identity": (0x4D80, lambda row: row.replace("P1 ;", "!P1 ;")),
            "exponent zero identity": (
                0x4DA0,
                lambda row: row.replace("P0 ;", "!P0 ;"),
            ),
        }
        for name, (offset, transform) in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(offset, transform))

    def test_nonlocal_computation_and_guards_are_bound(self):
        changes = {
            "frame base": (0x15C0, lambda row: row.replace("0x1c,", "0x20,")),
            "finite guard": (0x80B0, lambda row: row.replace("NEU", "EQU")),
            "large guard": (0x8020, lambda row: row.replace("105615", "65535")),
            "reduction product": (
                0x8120,
                lambda row: row.replace("0x3c439041", "0x3c439040"),
            ),
            "F64 helper computation": (
                0x36450,
                lambda row: row.replace("R65,", "R66,"),
            ),
            "F64 table pointer scale": (
                0x36340,
                lambda row: row.replace("0x8", "0x10"),
            ),
        }
        for name, (offset, transform) in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(offset, transform))

    def test_malformed_decoded_inventory_fails(self):
        offset = 0x8140
        pattern = rf"^\s*/\*{offset:04x}\*/[^\n]+$"
        row = re.search(pattern, self.sass, re.MULTILINE)[0]
        changes = (
            self.sass.replace(row, ""),
            self.sass.replace(row, row + "\n" + row),
            self.change_instruction(offset, lambda row: row.replace(";", "")),
            self.change_instruction(
                offset,
                lambda row: re.sub(r"\*/\s+\w+", "*/ UNKNOWN", row),
            ),
        )
        for index, changed in enumerate(changes):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                self.qualify(sass=changed)

    def test_malformed_helper_ranges_and_labels_fail(self):
        q = scratch.load_qualifier()
        for mutation in ("gap", "overlap", "unknown", "duplicate"):
            elf = copy.deepcopy(self.elf)
            helpers = sorted(
                (
                    symbol
                    for symbol in elf.symbols
                    if symbol["kind"] == 2 and symbol["name"] != scratch.ENTRY
                ),
                key=lambda symbol: symbol["value"],
            )
            helper = helpers[0]
            if mutation == "gap":
                helper["value"] += 16
            elif mutation == "overlap":
                helper["size"] += 16
            elif mutation == "unknown":
                helper["name"] = "unreviewed_helper"
            else:
                helper["name"] = helpers[1]["name"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                scratch.schedule(elf, self.sass, q)
        label = re.search(r"^\.L_x_\d+:$", self.sass, re.MULTILINE)[0]
        with self.assertRaises(ValueError):
            self.qualify(sass=self.sass.replace(label, label + "\n" + label, 1))

    def test_label_and_compiler_helper_numbering_may_change(self):
        renamed = re.sub(
            r"\.L_x_(\d+)", lambda m: f".L_x_{int(m[1]) + 9000}", self.sass
        )
        # Symbol spellings must agree in both ELF and SASS. Equal-length names
        # exercise renumbering without rewriting the ELF string-table layout.
        cubin = self.cubin
        for index in range(5):
            old, new = f"$__internal_{index}_", f"$__internal_{index + 5}_"
            renamed = renamed.replace(old, new)
            cubin = cubin.replace(old.encode(), new.encode())
        self.assertEqual(
            self.qualify(cubin=cubin, sass=renamed)["schedule_sha256"],
            self.qualify()["schedule_sha256"],
        )

    def test_helper_placement_normalization(self):
        # Unit-test only the normalized decoded schedule. This deliberately
        # does not present edited SASS as a fresh disassembly of edited code.
        q = scratch.load_qualifier()
        original = scratch.schedule(self.elf, self.sass, q)[0]
        elf = copy.deepcopy(self.elf)
        helpers = sorted(
            (
                symbol
                for symbol in elf.symbols
                if symbol["kind"] == 2
                and symbol["section"] == elf.named[scratch.TEXT]["index"]
                and symbol["name"] != scratch.ENTRY
            ),
            key=lambda s: s["value"],
        )
        first = helpers[0]["value"]
        moves = []
        cursor = first
        for helper in reversed(helpers):
            moves.append(
                (
                    helper["value"],
                    helper["value"] + helper["size"],
                    cursor - helper["value"],
                )
            )
            helper["value"] = cursor
            cursor += helper["size"]

        def move(match):
            offset = int(match[1], 16)
            for begin, end, delta in moves:
                if begin <= offset < end:
                    return f"/*{offset + delta:04x}*/"
            return match[0]

        changed = re.sub(r"/\*([0-9a-f]+)\*/", move, self.sass)
        self.assertEqual(scratch.schedule(elf, changed, q)[0], original)

    def test_frames_stack_geometry_and_register_caps_fail(self):
        for kind in (0x11, 0x12):

            def larger_frame(data, offset, length):
                for position in range(offset, offset + length, 8):
                    if struct.unpack_from("<I", data, position + 4)[0] == 96:
                        struct.pack_into("<I", data, position + 4, 112)
                        return
                return False

            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.qualify(cubin=self.mutate_record(".nv.info", kind, larger_frame))
        for kind, value in ((0x10, 256), (0x1E, 16)):
            changed = self.mutate_record(
                f".nv.info.{scratch.ENTRY}",
                kind,
                lambda data, offset, _, value=value: struct.pack_into(
                    "<I", data, offset, value
                ),
            )
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.qualify(cubin=changed)
        data = bytearray(self.cubin)
        table = struct.unpack_from("<Q", data, 40)[0]
        struct.pack_into(
            "<I",
            data,
            table + self.elf.named[scratch.TEXT]["index"] * 64 + 44,
            89 << 24,
        )
        with self.assertRaises(ValueError):
            self.qualify(cubin=bytes(data))
        for key, value in (("registers", 89), ("code_bytes", 224640), ("threads", 256)):
            entries = copy.deepcopy(self.entries)
            entries[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.qualify(entries=entries)

    def test_helper_frame_and_undefined_linkage_fail(self):
        def helper_frame(data, offset, length):
            struct.pack_into("<I", data, offset + 4, 16)

        with self.assertRaises(ValueError):
            self.qualify(cubin=self.mutate_record(".nv.info", 0x11, helper_frame))
        data = bytearray(self.cubin)
        symbol_table = self.elf.section(self.elf.symbol_table)
        helper = next(s for s in self.elf.symbols if s["name"] == scratch.TRIG)
        struct.pack_into(
            "<H", data, symbol_table["start"] + helper["index"] * 24 + 6, 0
        )
        with self.assertRaises(ValueError):
            self.qualify(cubin=bytes(data))

    def test_table_content_relocations_and_provenance_fail(self):
        data = bytearray(self.cubin)
        data[self.elf.named[".nv.global.init"]["start"]] ^= 1
        with self.assertRaises(ValueError):
            self.qualify(cubin=bytes(data))
        data = bytearray(self.cubin)
        data[self.elf.named[f".nv.constant2.{scratch.ENTRY}"]["start"] + 16] ^= 1
        # Data-only changes leave executable disassembly unchanged. The CB2
        # payload binding must reject this independently of the SASS hash.
        with self.assertRaises(ValueError):
            self.qualify(cubin=bytes(data))
        data = bytearray(self.cubin)
        struct.pack_into("<Q", data, self.elf.named[".nv.constant4"]["start"] + 8, 8)
        with self.assertRaises(ValueError):
            self.qualify(cubin=bytes(data))
        for key in scratch.PROVENANCE:
            provenance = copy.deepcopy(scratch.PROVENANCE)
            provenance[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.qualify(provenance=provenance)
        changed = self.change_instruction(
            0x36370, lambda row: row.replace("c[0x4][0x8]", "c[0x4][R1]")
        )
        with self.assertRaises(ValueError):
            self.qualify(sass=changed)

    def test_unknown_cb4_forms_and_relocation_layout_fail(self):
        forms = (
            lambda row: row.replace("IMAD.WIDE", "IMAD.WIDE.F32"),
            lambda row: row.replace("c[0x4][0x8]", "c[0x4][0x0]"),
            lambda row: row.replace("c[0x4][0x8]", "c[0x4][0x14]"),
        )
        for index, transform in enumerate(forms):
            with self.subTest(form=index), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(0x36370, transform))
        relocation = self.elf.named[".rel.nv.constant4"]
        for mutation in ("duplicate slot", "unknown slot", "unknown kind"):
            data = bytearray(self.cubin)
            offset = relocation["start"] + 16
            if mutation == "unknown kind":
                value = struct.unpack_from("<Q", data, offset + 8)[0]
                struct.pack_into("<Q", data, offset + 8, (value & ~0xFFFFFFFF) | 1)
            else:
                struct.pack_into(
                    "<Q", data, offset, 0 if mutation == "duplicate slot" else 24
                )
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.qualify(cubin=bytes(data))

    def test_each_source_and_tool_pin_rejects_stale_provenance(self):
        for group in ("source_sha256", "tool_sha256"):
            for name in scratch.PROVENANCE[group]:
                provenance = copy.deepcopy(scratch.PROVENANCE)
                provenance[group][name] = "0" * 64
                with (
                    self.subTest(group=group, name=name),
                    self.assertRaises(ValueError),
                ):
                    self.qualify(provenance=provenance)

    def test_protocol_renewal_preserves_erf_certificate_and_rejects_old_provenance(
        self,
    ):
        location = os.environ.get("TYPED_MATH_SCRATCH_PRIOR_ERF_REPORT")
        if not location:
            self.skipTest("requires the accepted Erf25 v1 report")
        previous = json.loads(Path(location).read_text())["qualification"]
        with self.assertRaises(ValueError):
            self.qualify(provenance=previous["provenance"])
        # Protocol registration may change provenance and artifact spelling,
        # while every schedule, resource, constant and local bound stays exact.
        renewed = json.loads(json.dumps(self.qualify()))
        for field in ("profile", "provenance", "cubin_sha256", "sass_sha256"):
            previous.pop(field)
            renewed.pop(field)
        self.assertEqual(renewed, previous)

    def test_historical_scratch_artifact_requires_its_own_profile(self):
        location = os.environ.get("TYPED_MATH_SCRATCH_HISTORICAL_ARTIFACTS")
        if not location:
            self.skipTest("requires the retained historical typed-math checkpoint")
        directory = Path(location)
        with self.assertRaises(ValueError):
            self.qualify(
                cubin=(directory / "kernels.cubin").read_bytes(),
                sass=(directory / "kernels.sass").read_text(),
                entries=json.loads((directory / "kernels.json").read_text()),
            )

    def test_previous_power_artifact_requires_its_own_profile(self):
        location = os.environ.get("TYPED_MATH_SCRATCH_PREVIOUS_ARTIFACTS")
        if not location:
            self.skipTest("requires the corrected Power24 v2 export")
        directory = Path(location)
        with self.assertRaises(ValueError):
            self.qualify(
                cubin=(directory / "kernels.cubin").read_bytes(),
                sass=(directory / "kernels.sass").read_text(),
                entries=json.loads((directory / "kernels.json").read_text()),
            )

    def test_incomplete_unary_erf_artifact_cannot_enter_final_profile(self):
        location = os.environ.get("TYPED_MATH_SCRATCH_REJECTED_ARTIFACTS")
        if not location:
            self.skipTest("requires the first Erf25 export before the unary fix")
        directory = Path(location)
        # Even claiming the current source provenance cannot admit old code.
        with self.assertRaises(ValueError):
            self.qualify(
                cubin=(directory / "kernels.cubin").read_bytes(),
                sass=(directory / "kernels.sass").read_text(),
                entries=json.loads((directory / "kernels.json").read_text()),
            )

    def test_complete_production_policy_preserves_zero_local_gates(self):
        location = os.environ.get("TYPED_MATH_SCRATCH_BASELINE_SASS")
        if not location:
            self.skipTest("requires the old complete inference-family disassembly")
        baseline = Path(location).read_text()
        bodies = prepare.audit("inference", baseline)
        self.assertNotRegex(bodies[scratch.ENTRY], r"\b(?:LDL|STL)\b")
        old = bodies[scratch.ENTRY]
        new = re.search(
            r"^\.text\.burn_float_arithmetic:$(.*?)(?=^\.text\.|\Z)",
            self.sass,
            re.MULTILINE | re.DOTALL,
        )[1]
        integrated = baseline.replace(old, new)
        with self.assertRaises(RuntimeError):
            prepare.audit("inference", integrated)
        # The context must describe the complete supplied module, not merely
        # the single-entry cubin whose body was inserted here.
        with self.assertRaises(RuntimeError):
            prepare.audit(
                "inference",
                integrated,
                cubin=self.cubin,
                entries=self.entries,
                provenance=scratch.PROVENANCE,
            )
        for name in ("burn_adam", "burn_remainder", "burn_integer_arithmetic"):
            changed = baseline.replace(bodies[name], "\n/*0000*/ LDL R0, [R1];\n")
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                prepare.audit("inference", changed)
        # Old sixteen-operation code still has no local-memory exception.
        changed = baseline.replace(old, old + "\n/*0000*/ STL [R1], R0;\n")
        with self.assertRaises(RuntimeError):
            prepare.audit(
                "inference",
                changed,
                cubin=self.cubin,
                entries=self.entries,
                provenance=scratch.PROVENANCE,
            )

    def test_combined_family_inventory_preserves_entry_and_peer_checks(self):
        location = os.environ.get("TYPED_MATH_SCRATCH_BASELINE_SASS")
        if not location:
            self.skipTest("requires the retained complete inference-family export")
        baseline = Path(location)
        peer_elf = scratch.Cubin(baseline.with_suffix(".cubin").read_bytes())
        peers = [
            section
            for section in peer_elf.sections
            if section["flags"] & 4
            and section["size"]
            and section["name"] != scratch.TEXT
        ]
        # This is a structural fixture made from actual executable bytes. It
        # exercises complete-family admission without producing a linked export.
        data = bytearray(self.cubin)
        header = list(struct.unpack_from("<16sHHIQQQIHHHHHH", data))
        sections = [
            list(struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * 64))
            for i in range(header[12])
        ]
        names = bytearray(self.elf.contents(self.elf.section(header[13])))
        blocks = {
            name: block
            for block, name in re.findall(
                r"(^\s*\.section\s+([^,\s]+),[^\n]*\n.*?)(?=^\s*\.section\s+|\Z)",
                baseline.read_text(),
                re.MULTILINE | re.DOTALL,
            )
        }
        combined_sass = self.sass
        for peer in peers:
            name_offset = len(names)
            names.extend(peer["name"].encode() + b"\0")
            data.extend(bytes(-len(data) % 128))
            sections.append(
                [
                    name_offset,
                    1,
                    6,
                    0,
                    len(data),
                    peer["size"],
                    0,
                    0,
                    128,
                    0,
                ]
            )
            data.extend(peer_elf.contents(peer))
            combined_sass += "\n" + blocks[peer["name"]]
        sections[header[13]][4:6] = len(data), len(names)
        data.extend(names)
        data.extend(bytes(-len(data) % 8))
        header[6], header[12] = len(data), len(sections)
        for section in sections:
            data.extend(struct.pack("<IIQQQQIIQQ", *section))
        struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, *header)
        entries = json.loads((baseline.parent.parent / "inference.json").read_text())
        entries = [entry for entry in entries if entry["entry"] != scratch.ENTRY]
        entries.extend(self.entries)
        self.assertEqual(len(peers) + 1, 248)
        self.assertEqual(len(entries), 248)
        combined_cubin = bytes(data)
        result = self.qualify(cubin=combined_cubin, sass=combined_sass, entries=entries)
        self.assertEqual(result["schedule_sha256"], self.qualify()["schedule_sha256"])
        production = scratch.load_qualifier().qualify(combined_cubin, combined_sass)
        self.assertEqual(len(production["sections"]), len(peers) + 1)
        bodies = prepare.audit(
            "inference",
            combined_sass,
            cubin=combined_cubin,
            entries=entries,
            provenance=scratch.PROVENANCE,
        )
        for name in ("burn_adam", "burn_remainder", "burn_integer_arithmetic"):
            changed = combined_sass.replace(bodies[name], "\n/*0000*/ STL [R1], R0;\n")
            with self.subTest(entry=name), self.assertRaises(RuntimeError):
                prepare.audit(
                    "inference",
                    changed,
                    cubin=combined_cubin,
                    entries=entries,
                    provenance=scratch.PROVENANCE,
                )
        with self.assertRaises(ValueError):
            self.qualify(cubin=combined_cubin, sass=self.sass, entries=entries)


if __name__ == "__main__":
    unittest.main()
