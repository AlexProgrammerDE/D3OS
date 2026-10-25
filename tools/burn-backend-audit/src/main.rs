//! Inventory pinned Burn traits without mistaking declarations for runtime support.
use proc_macro2::Span;
use quote::ToTokens;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use syn::{
    visit::{self, Visit},
    *,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TRAITS: &[&str] = &[
    "BackendTypes",
    "Backend",
    "AutodiffBackend",
    "DeviceOps",
    "Device",
    "TensorMetadata",
    "FloatTensorOps",
    "IntTensorOps",
    "BoolTensorOps",
    "QTensorOps",
    "ModuleOps",
    "ActivationOps",
    "TransactionOps",
    "DistributedOps",
];

#[derive(Clone, Serialize)]
struct Source {
    package: String,
    file: String,
    line: usize,
}

#[derive(Clone, Default, Serialize)]
struct Facts {
    calls: BTreeSet<String>,
    method_calls: BTreeSet<String>,
    backend_calls: BTreeSet<String>,
    declines: BTreeSet<String>,
    guards: BTreeSet<String>,
    dtype_variants: BTreeSet<String>,
    macros: BTreeSet<String>,
    expanded_macros: BTreeSet<String>,
    macro_parse_errors: BTreeSet<String>,
}

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}
fn last(path: &syn::Path) -> String {
    path.segments.last().unwrap().ident.to_string()
}
fn attributes(attrs: &[Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("cfg") || a.path().is_ident("cfg_attr"))
        .map(tokens)
        .collect()
}
fn test_only(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && matches!(&a.meta, Meta::List(l) if l.tokens.to_string() == "test"))
}
fn bounded(value: &impl ToTokens) -> String {
    tokens(value).chars().take(800).collect()
}

struct QuantizationMacro {
    operation: Expr,
    inputs: syn::punctuated::Punctuated<Expr, Token![,]>,
}

impl syn::parse::Parse for QuantizationMacro {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let tag: Ident = input.parse()?;
        if tag != "float_op" {
            return Err(syn::Error::new(tag.span(), "expected float_op"));
        }
        let operation = input.parse()?;
        input.parse::<Token![,]>()?;
        let inputs = syn::punctuated::Punctuated::parse_terminated(input)?;
        Ok(Self { operation, inputs })
    }
}

impl<'ast> Visit<'ast> for Facts {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = &*node.func {
            let name = last(&path.path);
            self.calls.insert(tokens(path));
            if matches!(
                path.path.segments.first().unwrap().ident.to_string().as_str(),
                "Self" | "B" | "Native" | "CubeBackend"
            ) {
                self.backend_calls.insert(name.clone());
            }
            if name == "unsupported" || name.starts_with("unsupported_") || name == "graph_unsupported" || name == "profile_unsupported" {
                self.declines.insert(tokens(path));
            }
        }
        visit::visit_expr_call(self, node);
    }
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.method_calls.insert(node.method.to_string());
        visit::visit_expr_method_call(self, node);
    }
    fn visit_macro(&mut self, node: &'ast Macro) {
        let name = last(&node.path);
        self.macros.insert(name.clone());
        if matches!(name.as_str(), "todo" | "unimplemented" | "panic") {
            self.declines.insert(format!("{name}!"));
        }
        if matches!(name.as_str(), "assert" | "assert_eq" | "assert_ne") {
            self.guards.insert(bounded(node));
        }
        if matches!(name.as_str(), "dequant_op_quant" | "dequant_op_flow") {
            match syn::parse2::<QuantizationMacro>(node.tokens.clone()) {
                Ok(expanded) => {
                    self.expanded_macros.insert(name);
                    self.backend_calls.extend(["dequantize".into(), "quantize_dynamic".into()]);
                    self.visit_expr(&expanded.operation);
                    for input in &expanded.inputs {
                        self.visit_expr(input);
                    }
                }
                Err(error) => {
                    self.macro_parse_errors.insert(error.to_string());
                }
            }
        }
        visit::visit_macro(self, node);
    }
    fn visit_expr_if(&mut self, node: &'ast ExprIf) {
        self.guards.insert(bounded(&node.cond));
        visit::visit_expr_if(self, node);
    }
    fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
        self.guards.insert(format!(
            "match {} => [{}]",
            bounded(&node.expr),
            node.arms.iter().map(|a| bounded(&a.pat)).collect::<Vec<_>>().join(", ")
        ));
        visit::visit_expr_match(self, node);
    }
    fn visit_path(&mut self, node: &'ast syn::Path) {
        let first = node.segments.first().unwrap().ident.to_string();
        if matches!(
            first.as_str(),
            "DType" | "FloatDType" | "IntDType" | "BoolDType" | "BoolStore" | "QuantValue" | "ScaleDtype" | "QuantStore"
        ) && node.segments.len() > 1
        {
            self.dtype_variants.insert(tokens(node));
        }
        visit::visit_path(self, node);
    }
}

#[derive(Clone, Serialize)]
struct Function {
    source: Source,
    conditions: Vec<String>,
    facts: Facts,
}
#[derive(Clone)]
struct Definition {
    signature: String,
    source: Source,
    conditions: Vec<String>,
    default: Option<Function>,
}
#[derive(Default)]
struct Index {
    traits: BTreeMap<String, Definition>,
    implementations: BTreeMap<(String, String), Vec<Function>>,
    types: BTreeMap<String, BTreeMap<String, String>>,
    files: BTreeMap<String, String>,
    unresolved_impl_macros: Vec<String>,
    trait_declarations: BTreeMap<String, Source>,
    inherent_helpers: BTreeMap<String, Vec<Function>>,
}

fn function(package: &str, file: &Path, span: Span, conditions: Vec<String>, body: &Block) -> Function {
    let mut facts = Facts::default();
    facts.visit_block(body);
    Function {
        source: Source {
            package: package.into(),
            file: file.to_string_lossy().into(),
            line: span.start().line,
        },
        conditions,
        facts,
    }
}

fn trait_aliases(tree: &UseTree, aliases: &mut BTreeMap<String, String>) {
    match tree {
        UseTree::Path(path) => trait_aliases(&path.tree, aliases),
        UseTree::Group(group) => {
            for item in &group.items {
                trait_aliases(item, aliases);
            }
        }
        UseTree::Rename(rename) if TRAITS.contains(&rename.ident.to_string().as_str()) => {
            aliases.insert(rename.rename.to_string(), rename.ident.to_string());
        }
        _ => {}
    }
}

impl Index {
    fn scan(&mut self, package: &str, root: &Path) -> Result<()> {
        let mut files = Vec::new();
        collect_files(root, &mut files)?;
        files.sort();
        for path in files {
            let source = fs::read_to_string(&path)?;
            let relative = path.strip_prefix(root)?;
            self.files
                .insert(format!("{package}/{}", relative.display()), format!("{:x}", Sha256::digest(source.as_bytes())));
            let ast = syn::parse_file(&source).map_err(|e| format!("{}: {e}", path.display()))?;
            self.items(package, relative, &ast.items, &attributes(&ast.attrs), &BTreeMap::new())?;
        }
        Ok(())
    }
    fn items(&mut self, package: &str, file: &Path, items: &[Item], conditions: &[String], inherited_aliases: &BTreeMap<String, String>) -> Result<()> {
        let mut aliases = inherited_aliases.clone();
        for item in items {
            if let Item::Use(item) = item {
                trait_aliases(&item.tree, &mut aliases);
            }
        }
        for item in items {
            match item {
                Item::Mod(m) if !test_only(&m.attrs) => {
                    if let Some((_, items)) = &m.content {
                        let inherited = [conditions.to_vec(), attributes(&m.attrs)].concat();
                        self.items(package, file, items, &inherited, &aliases)?;
                    }
                }
                Item::Trait(t) if TRAITS.contains(&t.ident.to_string().as_str()) && !test_only(&t.attrs) => {
                    self.trait_declarations.insert(
                        t.ident.to_string(),
                        Source {
                            package: package.into(),
                            file: file.to_string_lossy().into(),
                            line: t.ident.span().start().line,
                        },
                    );
                    for item in &t.items {
                        if let TraitItem::Fn(f) = item {
                            let key = format!("{}::{}", t.ident, f.sig.ident);
                            let conditions = [conditions.to_vec(), attributes(&t.attrs), attributes(&f.attrs)].concat();
                            let source = Source {
                                package: package.into(),
                                file: file.to_string_lossy().into(),
                                line: f.sig.ident.span().start().line,
                            };
                            let default = f.default.as_ref().map(|b| function(package, file, f.sig.ident.span(), conditions.clone(), b));
                            if self
                                .traits
                                .insert(
                                    key.clone(),
                                    Definition {
                                        signature: tokens(&f.sig),
                                        source,
                                        conditions,
                                        default,
                                    },
                                )
                                .is_some()
                            {
                                return Err(format!("duplicate trait declaration {key}").into());
                            }
                        }
                    }
                }
                Item::Impl(i) if i.trait_.is_none() && !test_only(&i.attrs) => {
                    let Type::Path(owner) = &*i.self_ty else { continue };
                    let owner = last(&owner.path);
                    if package == "d3os-burn-native" && ["NativeTensor", "NativeDevice", "BackendTensor", "Layout"].contains(&owner.as_str()) {
                        for item in &i.items {
                            if let ImplItem::Fn(f) = item {
                                let conditions = [conditions.to_vec(), attributes(&i.attrs), attributes(&f.attrs)].concat();
                                self.inherent_helpers.entry(format!("{owner}::{}", f.sig.ident)).or_default().push(function(
                                    package,
                                    file,
                                    f.sig.ident.span(),
                                    conditions,
                                    &f.block,
                                ));
                            }
                        }
                    }
                }
                Item::Impl(i) if !test_only(&i.attrs) => {
                    let Some((_, trait_path, _)) = &i.trait_ else { continue };
                    let name = last(trait_path);
                    let trait_name = aliases.get(&name).cloned().unwrap_or(name);
                    if !TRAITS.contains(&trait_name.as_str()) {
                        continue;
                    }
                    let Type::Path(owner) = &*i.self_ty else { continue };
                    let owner = last(&owner.path);
                    let base = [conditions.to_vec(), attributes(&i.attrs)].concat();
                    for item in &i.items {
                        match item {
                            ImplItem::Fn(f) if !test_only(&f.attrs) => {
                                let implementation = function(package, file, f.sig.ident.span(), [base.clone(), attributes(&f.attrs)].concat(), &f.block);
                                self.implementations
                                    .entry((owner.clone(), format!("{trait_name}::{}", f.sig.ident)))
                                    .or_default()
                                    .push(implementation);
                            }
                            ImplItem::Type(t) => {
                                self.types
                                    .entry(owner.clone())
                                    .or_default()
                                    .insert(format!("{trait_name}::{}", t.ident), tokens(&t.ty));
                            }
                            ImplItem::Macro(m) => {
                                self.unresolved_impl_macros
                                    .push(format!("{package}/{}: {}", file.display(), tokens(&m.mac.path)));
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn resolve(&self, owner: &str, key: &str) -> Resolution {
        if let Some(functions) = self.implementations.get(&(owner.into(), key.into())) {
            Resolution {
                kind: "override",
                functions: functions.clone(),
            }
        } else if let Some(default) = &self.traits[key].default {
            Resolution {
                kind: "inherited_default",
                functions: vec![default.clone()],
            }
        } else {
            Resolution {
                kind: "missing_required",
                functions: vec![],
            }
        }
    }
}

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.extension().is_some_and(|s| s == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

#[derive(Clone, Serialize)]
struct Resolution {
    kind: &'static str,
    functions: Vec<Function>,
}
#[derive(Serialize)]
struct Row {
    id: String,
    signature: String,
    declaration: Source,
    conditions: Vec<String>,
    native: Resolution,
    linux_cube: Resolution,
    linux_fusion: Resolution,
    autodiff_wrapper: Resolution,
    native_reachable_decline_declarations: BTreeSet<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_quantized_metadata: Option<Resolution>,
    runtime_conformance: &'static str,
}

// Backward reachability is cycle-safe. A declaration in a cycle is not a
// failure unless that cycle can reach a known decline.
fn reachable_declines(edges: &[Vec<usize>], blocked: &BTreeSet<usize>) -> Vec<BTreeSet<usize>> {
    let mut result = vec![BTreeSet::new(); edges.len()];
    for &node in blocked {
        result[node].insert(node);
    }
    loop {
        let previous = result.clone();
        for (node, next) in edges.iter().enumerate() {
            for &target in next {
                result[node].extend(previous[target].iter().copied());
            }
        }
        if result == previous {
            return result;
        }
    }
}

fn package_roots(repo: &Path) -> Result<BTreeMap<String, PathBuf>> {
    const PACKAGES: &[&str] = &["burn-backend", "burn-cubecl", "burn-fusion", "burn-autodiff", "cubecl-common", "cubecl"];
    let result = Command::new("cargo")
        .args(["+1.98.1", "metadata", "--locked", "--format-version", "1", "--manifest-path"])
        .arg(repo.join("tools/mnist-benchmark/Cargo.toml"))
        .output()?;
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).into_owned().into());
    }
    let value: serde_json::Value = serde_json::from_slice(&result.stdout)?;
    let mut roots = BTreeMap::new();
    for p in value["packages"].as_array().ok_or("cargo metadata lacks packages")? {
        let name = p["name"].as_str().ok_or("package lacks name")?;
        if PACKAGES.contains(&name) {
            let expected = if name.starts_with("cubecl") { "0.11.3" } else { "0.22.1" };
            if p["version"] != expected {
                return Err(format!("unexpected {name} version {}", p["version"]).into());
            }
            roots.insert(
                name.into(),
                PathBuf::from(p["manifest_path"].as_str().ok_or("missing manifest")?)
                    .parent()
                    .unwrap()
                    .join("src"),
            );
        }
    }
    for package in PACKAGES {
        if !roots.contains_key(*package) {
            return Err(format!("missing pinned package {package}").into());
        }
    }
    Ok(roots)
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: d3os-burn-backend-audit REPOSITORY OUTPUT.json".into());
    }
    let repo = PathBuf::from(&args[0]).canonicalize()?;
    let mut index = Index::default();
    let mut roots = package_roots(&repo)?;
    roots.insert("d3os-burn-native".into(), repo.join("os/library/burn-native/src"));
    for (package, root) in &roots {
        index.scan(package, root)?;
    }
    for trait_name in TRAITS {
        if !index.trait_declarations.contains_key(*trait_name) {
            return Err(format!("missing trait {trait_name}").into());
        }
    }
    for implementations in index.implementations.values() {
        if implementations.iter().any(|f| !f.facts.macro_parse_errors.is_empty()) {
            return Err("unparsed known quantization macro".into());
        }
    }
    if index
        .traits
        .values()
        .filter_map(|d| d.default.as_ref())
        .any(|f| !f.facts.macro_parse_errors.is_empty())
    {
        return Err("unparsed known quantization default macro".into());
    }
    let keys: Vec<_> = index.traits.keys().cloned().collect();
    let positions: BTreeMap<_, _> = keys.iter().enumerate().map(|(i, k)| (k.rsplit("::").next().unwrap(), i)).collect();
    if positions.len() != keys.len() {
        return Err("ambiguous backend method names need trait-qualified resolution".into());
    }
    let native: Vec<_> = keys
        .iter()
        .map(|k| {
            index.resolve(
                if k.starts_with("DeviceOps::") || k.starts_with("Device::") {
                    "NativeDevice"
                } else if k.starts_with("TensorMetadata::") {
                    "BackendTensor"
                } else if k.starts_with("AutodiffBackend::") {
                    "Autodiff"
                } else {
                    "Native"
                },
                k,
            )
        })
        .collect();
    let edges: Vec<Vec<_>> = native
        .iter()
        .map(|r| {
            r.functions
                .iter()
                .flat_map(|f| &f.facts.backend_calls)
                .filter_map(|name| positions.get(name.as_str()).copied())
                .collect()
        })
        .collect();
    let blocked: BTreeSet<_> = native
        .iter()
        .enumerate()
        .filter(|(_, r)| r.functions.iter().any(|f| !f.facts.declines.is_empty()))
        .map(|(i, _)| i)
        .collect();
    let reachable = reachable_declines(&edges, &blocked);
    let mut rows = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        let definition = &index.traits[key];
        let device = key.starts_with("DeviceOps::") || key.starts_with("Device::");
        let metadata = key.starts_with("TensorMetadata::");
        let autodiff = key.starts_with("AutodiffBackend::");
        rows.push(Row {
            id: key.clone(),
            signature: definition.signature.clone(),
            declaration: definition.source.clone(),
            conditions: definition.conditions.clone(),
            native: native[i].clone(),
            linux_cube: index.resolve(
                if device {
                    "Device"
                } else if metadata {
                    "CubeTensor"
                } else if autodiff {
                    "Autodiff"
                } else {
                    "CubeBackend"
                },
                key,
            ),
            linux_fusion: index.resolve(
                if device {
                    "Device"
                } else if metadata {
                    "FusionTensor"
                } else if autodiff {
                    "Autodiff"
                } else {
                    "Fusion"
                },
                key,
            ),
            autodiff_wrapper: index.resolve(
                if device {
                    "NativeDevice"
                } else if metadata {
                    "AutodiffTensor"
                } else {
                    "Autodiff"
                },
                key,
            ),
            native_reachable_decline_declarations: reachable[i].iter().map(|&j| keys[j].clone()).collect(),
            native_quantized_metadata: metadata.then(|| index.resolve("BackendQuantizedTensor", key)),
            runtime_conformance: "unverified_by_static_inventory",
        });
    }
    let report = serde_json::json!({
        "schema":1,"burn_version":"0.22.1","full_backend_parity":false,
        "scope":"Complete selected Burn trait method inventory; static evidence is not a runtime support verdict",
        "limitations":["cfg conditions are recorded, not evaluated","method calls lack type resolution","free helper functions and other macros are not expanded; inherent native helper bodies are recorded for review","reachability includes syntactic associated backend calls and the pinned dequant_op_quant/dequant_op_flow macros only; guarded or unreachable paths may be included","dtype variants and guards require semantic review","runtime device capability flags are separate evidence","fusion and autodiff wrappers do not prove inner kernels exist"],
        "traits":index.trait_declarations,"methods":rows,"native_inherent_helpers":index.inherent_helpers,"associated_types":index.types,"unresolved_impl_macros":index.unresolved_impl_macros,"source_sha256":index.files,
        "summary":{"methods":rows.len(),"native_overrides":rows.iter().filter(|r|r.native.kind=="override").count(),"native_inherited":rows.iter().filter(|r|r.native.kind=="inherited_default").count(),"native_missing_required":rows.iter().filter(|r|r.native.kind=="missing_required").count(),"potential_decline_paths":rows.iter().filter(|r|!r.native_reachable_decline_declarations.is_empty()).count()}
    });
    let output = PathBuf::from(&args[1]);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("BURN_BACKEND_INVENTORY_OK {}", report["summary"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cycles_only_propagate_declines_they_can_reach() {
        let graph = vec![vec![1], vec![0, 2], vec![], vec![4], vec![3]];
        let actual = reachable_declines(&graph, &BTreeSet::from([2]));
        assert_eq!(
            actual,
            vec![BTreeSet::from([2]), BTreeSet::from([2]), BTreeSet::from([2]), BTreeSet::new(), BTreeSet::new()]
        );
    }
}
