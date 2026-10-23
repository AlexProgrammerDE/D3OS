//! Combine independently specialized cuTile entries into one verified module.
use cutile_ir::{
    Module,
    ir::{Attribute, Block, OpId, Region, RegionId, Value, ValueProducer},
};
use std::collections::HashMap;

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

pub fn append(target: &mut Module, source: &Module, export: &str) -> Result<(), Box<dyn std::error::Error>> {
    if source.functions.len() != 1 || !source.globals.is_empty() {
        return Err("expected one self-contained cuTile entry without globals".into());
    }
    source.verify_dominance()?;
    source.verify_bytecode_indices()?;
    if target.functions.iter().any(|&id| {
        target
            .op(id)
            .attributes
            .iter()
            .any(|(key, value)| key == "sym_name" && *value == Attribute::String(export.to_owned()))
    }) {
        return Err("duplicate exported entry".into());
    }
    let mut linked = target.clone();
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
    name.1 = Attribute::String(export.to_owned());
    importer.target.functions.push(id);
    importer.target.verify_dominance()?;
    importer.target.verify_bytecode_indices()?;
    *target = linked;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::compile_api::KernelCompiler;
    use cutile_ir::bytecode::{BytecodeVersion, decoder::decode_bytecode, write_bytecode_version};

    fn program(name: &str) -> Module {
        KernelCompiler::new(crate::kernels::__module_ast_self, "kernels", name)
            .target("sm_89")
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
        let mut linked = Module::new("test");
        append(&mut linked, &vector, "vector").unwrap();
        let watermark = linked.num_values();
        append(&mut linked, &softmax, "softmax").unwrap();
        assert_eq!(linked.functions.len(), 2);
        assert_eq!(linked.num_values(), watermark + softmax.num_values());
        linked.verify_dominance().unwrap();
        linked.verify_bytecode_indices().unwrap();
        let bytes = write_bytecode_version(&linked, BytecodeVersion::V13_4).unwrap();
        decode_bytecode(&bytes).unwrap();
        assert_eq!(vector.functions.len(), 1);
        assert_eq!(softmax.functions.len(), 1);
    }

    #[test]
    fn invalid_or_duplicate_entries_leave_the_destination_unchanged() {
        let source = program("vector_add");
        let mut target = Module::new("test");
        append(&mut target, &source, "vector").unwrap();
        let before = write_bytecode_version(&target, BytecodeVersion::V13_4).unwrap();
        assert!(append(&mut target, &source, "vector").is_err());
        let mut invalid = source.clone();
        invalid.functions.push(invalid.functions[0]);
        assert!(append(&mut target, &invalid, "invalid").is_err());
        assert_eq!(before, write_bytecode_version(&target, BytecodeVersion::V13_4).unwrap());
    }
}
