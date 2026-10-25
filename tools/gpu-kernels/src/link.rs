//! Combine independently specialized cuTile entries into one verified module.
use cutile_ir::{
    Module,
    ir::{Attribute, Block, OpId, Region, RegionId, Value, ValueProducer},
};
use std::collections::{HashMap, HashSet};

/// Apply the fixed CTA contract to every architecture understood by the pinned
/// Tile IR assembler. The guest chooses machine code; the IR retains its shape.
pub fn portable_hints(module: &mut Module) -> Result<(), Box<dyn std::error::Error>> {
    for function in module.functions.clone() {
        let hints = module
            .op_mut(function)
            .attributes
            .iter_mut()
            .find_map(|(_, value)| match value {
                Attribute::OptimizationHints(hints) => Some(hints),
                _ => None,
            })
            .ok_or("entry has no launch hints")?;
        let baseline = hints
            .entries
            .iter()
            .find(|(target, _)| target == "sm_89")
            .ok_or("entry has no baseline CTA contract")?
            .1
            .clone();
        for target in [
            "sm_80", "sm_86", "sm_87", "sm_88", "sm_89", "sm_90", "sm_100", "sm_103", "sm_107", "sm_110", "sm_120", "sm_121",
        ] {
            if !hints.entries.iter().any(|(existing, _)| existing == target) {
                hints.entries.push((target.into(), baseline.clone()));
            }
        }
    }
    Ok(())
}

struct Import<'a> {
    source: &'a Module,
    target: &'a mut Module,
    arguments: HashMap<Value, Value>,
    results: HashMap<(OpId, u32), Value>,
}
impl Import<'_> {
    fn value(&self, value: Value) -> Value {
        match self.source.value_data(value).producer {
            ValueProducer::BlockArg { .. } => self.arguments[&value],
            ValueProducer::OpResult { op, result_index } => self.results[&(op, result_index)],
        }
    }
    fn region(&mut self, region: RegionId) -> RegionId {
        let id = self.target.alloc_region(Region { blocks: Vec::new() });
        for old_id in &self.source.region(region).blocks {
            let block = self.source.block(*old_id);
            let new_id = self.target.alloc_block(Block {
                args: Vec::new(),
                ops: Vec::new(),
            });
            self.target.region_mut(id).blocks.push(new_id);
            for (index, (value, ty)) in block.args.iter().enumerate() {
                let new = self.target.alloc_value(
                    ty.clone(),
                    ValueProducer::BlockArg {
                        block: new_id,
                        arg_index: index as u32,
                    },
                );
                self.arguments.insert(*value, new);
                self.target.block_mut(new_id).args.push((new, ty.clone()));
            }
            for op in &block.ops {
                let new = self.operation(*op);
                self.target.block_mut(new_id).ops.push(new);
            }
        }
        id
    }
    fn operation(&mut self, op: OpId) -> OpId {
        let old = self.source.op(op);
        let mut new = old.clone();
        new.operands = old.operands.iter().map(|&value| self.value(value)).collect();
        new.regions.clear();
        let id = self.target.alloc_op(new);
        for (index, ty) in old.result_types.iter().enumerate() {
            let value = self.target.alloc_value(
                ty.clone(),
                ValueProducer::OpResult {
                    op: id,
                    result_index: index as u32,
                },
            );
            self.results.insert((op, index as u32), value);
        }
        for region in &old.regions {
            let new = self.region(*region);
            self.target.op_mut(id).regions.push(new);
        }
        id
    }
}

/// Link independent entries transactionally, verifying the complete module once.
pub fn combine(family: &str, entries: &[(&str, Module)]) -> Result<Module, Box<dyn std::error::Error>> {
    let mut linked = Module::new(family);
    let mut exports = HashSet::new();
    for (export, source) in entries {
        if source.functions.len() != 1 || !source.globals.is_empty() {
            return Err("expected one self-contained cuTile entry without globals".into());
        }
        source.verify_dominance()?;
        source.verify_bytecode_indices()?;
        if !exports.insert(export) {
            return Err("duplicate exported entry".into());
        }
        let mut importer = Import {
            source,
            target: &mut linked,
            arguments: HashMap::new(),
            results: HashMap::new(),
        };
        let id = importer.operation(source.functions[0]);
        let name = importer
            .target
            .op_mut(id)
            .attributes
            .iter_mut()
            .find(|(key, _)| key == "sym_name")
            .ok_or("entry has no export name")?;
        name.1 = Attribute::String((*export).to_owned());
        importer.target.functions.push(id);
    }
    linked.verify_dominance()?;
    linked.verify_bytecode_indices()?;
    Ok(linked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::compile_api::KernelCompiler;
    use cutile_ir::bytecode::{BytecodeVersion, decoder::decode_bytecode, write_bytecode_version};

    fn program(name: &str) -> Module {
        KernelCompiler::new(crate::kernels::__module_ast_self, "kernels", name)
            .target("sm_89")
            .options(cutile::tile_kernel::CompileOptions::default().num_worker_warps_per_cta(4))
            .bytecode_version(BytecodeVersion::V13_4)
            .compile()
            .unwrap()
            .module()
            .clone()
    }

    #[test]
    fn importing_nested_regions_remaps_captures_and_multiple_results() {
        let vector = program("vector_add");
        let softmax = program("inference_softmax");
        let linked = combine("test", &[("vector", vector.clone()), ("softmax", softmax.clone())]).unwrap();
        assert_eq!(linked.functions.len(), 2);
        assert_eq!(linked.num_values(), vector.num_values() + softmax.num_values());
        linked.verify_dominance().unwrap();
        linked.verify_bytecode_indices().unwrap();
        let bytes = write_bytecode_version(&linked, BytecodeVersion::V13_4).unwrap();
        decode_bytecode(&bytes).unwrap();
        assert_eq!(vector.functions.len(), 1);
        assert_eq!(softmax.functions.len(), 1);
    }

    #[test]
    fn portable_hints_preserve_geometry_without_duplicates() {
        let mut module = program("vector_add");
        portable_hints(&mut module).unwrap();
        portable_hints(&mut module).unwrap();
        let entry = module.op(module.functions[0]);
        let hints = entry
            .attributes
            .iter()
            .find_map(|(_, value)| match value {
                Attribute::OptimizationHints(hints) => Some(hints),
                _ => None,
            })
            .unwrap();
        assert_eq!(hints.entries.len(), 12);
        assert!(hints.entries.iter().all(|(_, value)| *value == hints.entries[0].1));
        let bytes = write_bytecode_version(&module, BytecodeVersion::V13_4).unwrap();
        decode_bytecode(&bytes).unwrap();
    }

    #[test]
    fn invalid_or_duplicate_entries_leave_the_sources_unchanged() {
        let source = program("vector_add");
        let before = write_bytecode_version(&source, BytecodeVersion::V13_4).unwrap();
        assert!(combine("test", &[("vector", source.clone()), ("vector", source.clone())]).is_err());
        let mut invalid = source.clone();
        invalid.functions.push(invalid.functions[0]);
        assert!(combine("test", &[("vector", source.clone()), ("invalid", invalid)]).is_err());
        assert_eq!(before, write_bytecode_version(&source, BytecodeVersion::V13_4).unwrap());
    }
}
