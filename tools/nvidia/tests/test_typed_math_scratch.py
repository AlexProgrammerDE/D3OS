"""CPU artifact tests. Set TYPED_MATH_SCRATCH_ARTIFACTS to a fresh entry export.

Optionally set TYPED_MATH_SCRATCH_BASELINE_SASS to the old inference-family
disassembly to exercise the complete production audit with both entry versions.
"""

import copy
import importlib.util
import json
import os
import re
import struct
import unittest
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
        pattern = rf"(^\s*/\*{offset:04x}\*/[^\n]+)$"
        mutated, count = re.subn(
            pattern, lambda match: transform(match[0]), self.sass, flags=re.MULTILINE
        )
        self.assertEqual(count, 1)
        self.assertNotEqual(mutated, self.sass)
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
        self.assertEqual(len(result["call_sites"]), 72)
        self.assertEqual(result["registers"], 79)
        self.assertEqual(result["code_bytes"], 152832)
        self.assertEqual(result["frame_bytes"], 96)
        self.assertEqual(result["minimum_stack_bytes"], 96)
        self.assertEqual(result["local_load_sites"], 84)
        self.assertEqual(result["local_store_sites"], 110)
        self.assertEqual(sum(a["frame_range"] == [28, 56] for a in arrays), 18)
        self.assertEqual(sum(a["frame_range"] == [0, 28] for a in arrays), 9)

    def test_unknown_local_forms_bases_offsets_and_slots_fail(self):
        changes = {
            "unknown local opcode": (
                0x5D90,
                lambda row: row.replace("LDL ", "LDL.128 "),
            ),
            "unknown store base": (0x5C10, lambda row: row.replace("[R1+", "[R2+")),
            "out-of-array store": (0x5C10, lambda row: row.replace("+0x1c]", "+0x60]")),
            "unknown load base": (0x5D90, lambda row: row.replace("[R35+", "[R36+")),
            "out-of-array load": (0x5D90, lambda row: row.replace("+0x18]", "+0x1c]")),
            "new save slot": (
                0x5DC0,
                lambda row: re.sub(r"@P0 SHF.*;", "STL [R1+0x60], R0 ;", row),
            ),
            "F32 bound": (0x5D40, lambda row: row.replace("0x5,", "0x4,")),
            "F32 pointer scale": (0x5D60, lambda row: row.replace("-0x4,", "-0x8,")),
            "F64 pointer base": (0x24AF0, lambda row: row.replace("0x38,", "0x40,")),
            "F64 loop bound": (0x24B40, lambda row: row.replace("0x12,", "0x13,")),
            "F64 pointer stride": (0x24D30, lambda row: row.replace("0x8,", "0x10,")),
            "F64 final offset": (0x24E50, lambda row: row.replace("-0x78]", "-0x70]")),
            "missing predicate": (0x24E60, lambda row: row.replace("@P0", "   ")),
            "return address": (0x3650, lambda row: row.replace("0x3670", "0x3680")),
        }
        for name, (offset, transform) in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(offset, transform))

    def test_unknown_and_wrong_helper_targets_fail(self):
        for target in (
            "missing_helper",
            "$__internal_0_$__cuda_sm20_dblrcp_rn_slowpath_v3",
            ".L_x_571",
        ):
            changed = self.change_instruction(
                0x2000,
                lambda row, target=target: re.sub(r"`\([^)]+\)", f"`({target})", row),
            )
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.qualify(sass=changed)
        changed = self.change_instruction(
            0x24AC0, lambda row: row.replace(".L_x_571", ".L_x_572")
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

    def test_nonlocal_computation_and_guards_are_bound(self):
        changes = {
            "frame base": (0x14F0, lambda row: row.replace("0x1c,", "0x20,")),
            "finite guard": (0x5B90, lambda row: row.replace("NEU", "EQU")),
            "large guard": (0x5B00, lambda row: row.replace("105615", "65535")),
            "reduction product": (
                0x5BF0,
                lambda row: row.replace("0x3c439041", "0x3c439040"),
            ),
            "F64 helper computation": (
                0x24CD0,
                lambda row: row.replace("R57,", "R56,"),
            ),
            "F64 table pointer scale": (
                0x24B90,
                lambda row: row.replace("0x8", "0x10"),
            ),
        }
        for name, (offset, transform) in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(sass=self.change_instruction(offset, transform))

    def test_malformed_decoded_inventory_fails(self):
        offset = 0x5C10
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
            80 << 24,
        )
        with self.assertRaises(ValueError):
            self.qualify(cubin=bytes(data))
        for key, value in (("registers", 80), ("code_bytes", 152960), ("threads", 256)):
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
            0x24BC0, lambda row: row.replace("c[0x4][0x8]", "c[0x4][R1]")
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
                self.qualify(sass=self.change_instruction(0x24BC0, transform))
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

    def test_historical_scratch_artifact_requires_its_own_profile(self):
        location = os.environ.get("TYPED_MATH_SCRATCH_HISTORICAL_ARTIFACTS")
        if not location:
            self.skipTest("requires the retained nineteen-operation checkpoint")
        directory = Path(location)
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


if __name__ == "__main__":
    unittest.main()
