//! Matched ND indexing, gradients, scalar outputs, and exact device indices.
use d3os_ai::burn::tensor::Device;

fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let duplicates = !std::env::args().any(|a| a == "--unique");
    let scalar = !std::env::args().any(|a| a == "--no-scalar");
    let empty = !std::env::args().any(|a| a == "--nonempty");
    if std::env::args().any(|a| a == "--scalar-grad") {
        d3os_ai::validation::nd_indexing::scalar_gradients(&device).expect("scalar ND gradient conformance");
        device.sync().expect("scalar ND synchronization");
        println!("BURN_ND_SCALAR_GRAD_OK os=linux backend={backend} scatter_gradient_cases=5 gather_gradient_cases=1");
        return;
    }
    d3os_ai::validation::nd_indexing::run(&device, duplicates, empty, scalar).expect("ND indexing conformance");
    device.sync().expect("ND indexing synchronization");
    println!(
        "BURN_ND_INDEXING_OK os=linux backend={backend} dtypes=f32,i32 scatter_modes=5 widths=1,2,3 index_ranks=1,2,3 scatter_gradient_cases=15 gather_gradient_cases=3 scalar={scalar} exact_i32=true duplicates={duplicates} empty_slices={empty} shared_views=true"
    );
}
