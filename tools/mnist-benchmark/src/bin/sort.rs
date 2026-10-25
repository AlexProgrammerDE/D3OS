//! CUDA-device API comparison; pinned Burn sorts via its host fallback.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend, execution) = (Device::cuda(0), "cuda", "host_fallback");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, execution) = (Device::flex(), "flex_cpu", "cpu");
    let include_empty = !std::env::args().any(|arg| arg == "--nonempty");
    let (domain, domain_name) = if std::env::args().any(|arg| arg == "--finite") {
        (d3os_ai::validation::sort::FloatDomain::Finite, "finite")
    } else {
        (d3os_ai::validation::sort::FloatDomain::Ieee, "ieee")
    };
    #[cfg(feature = "cuda")]
    let (policy, policy_name) = (d3os_ai::validation::sort::TopKOrdering::NanFirst, "nan_first");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (policy, policy_name) = (d3os_ai::validation::sort::TopKOrdering::TotalOrder, "total_order");
    d3os_ai::validation::sort::run(&device, include_empty, policy, domain).expect("public ordering conformance");
    device.sync().expect("sort synchronization");
    println!(
        "BURN_SORT_CONFORMANCE_OK os=linux backend={backend} sorting={execution} topk_order={policy_name} float_domain={domain_name} dtypes=f32,i32 sort=true argsort=true topk=true gradients=6 empty={include_empty} shared_views=true"
    );
}
