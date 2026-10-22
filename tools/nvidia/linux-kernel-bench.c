// Linux-only reference. D3OS does not link or use these NVIDIA libraries.
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef uint64_t DevicePointer;
typedef void *Handle;
#define CUDA_FUNCTION(name, args) static int (*name) args
CUDA_FUNCTION(cuInit, (unsigned));
CUDA_FUNCTION(cuDeviceGet, (int *, int));
CUDA_FUNCTION(cuCtxCreate_v2, (Handle *, unsigned, int));
CUDA_FUNCTION(cuCtxDestroy_v2, (Handle));
CUDA_FUNCTION(cuMemAlloc_v2, (DevicePointer *, size_t));
CUDA_FUNCTION(cuMemFree_v2, (DevicePointer));
CUDA_FUNCTION(cuMemcpyHtoD_v2, (DevicePointer, const void *, size_t));
CUDA_FUNCTION(cuMemcpyDtoH_v2, (void *, DevicePointer, size_t));
CUDA_FUNCTION(cuModuleLoad, (Handle *, const char *));
CUDA_FUNCTION(cuModuleGetFunction, (Handle *, Handle, const char *));
CUDA_FUNCTION(cuLaunchKernel, (Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, Handle, void **, void **));
CUDA_FUNCTION(cuCtxSynchronize, (void));
CUDA_FUNCTION(cuEventCreate, (Handle *, unsigned));
CUDA_FUNCTION(cuEventRecord, (Handle, Handle));
CUDA_FUNCTION(cuEventSynchronize, (Handle));
CUDA_FUNCTION(cuEventElapsedTime, (float *, Handle, Handle));
CUDA_FUNCTION(cuEventDestroy_v2, (Handle));
CUDA_FUNCTION(cublasCreate_v2, (Handle *));
CUDA_FUNCTION(cublasDestroy_v2, (Handle));
CUDA_FUNCTION(cublasSgemm_v2, (Handle, int, int, int, int, int, const float *, DevicePointer, int, DevicePointer, int, const float *, DevicePointer, int));

static void check(int status, const char *operation) {
    if (status) {
        fprintf(stderr, "%s failed: %d\n", operation, status);
        exit(1);
    }
}
#define CHECK(call) check((call), #call)
#define LOAD(library, name) do { *(void **)(&name) = dlsym(library, #name); if (!name) { fprintf(stderr, "%s\n", dlerror()); exit(1); } } while (0)

static void *library(const char *name) {
    void *result = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (!result) { fprintf(stderr, "%s\n", dlerror()); exit(1); }
    return result;
}

static void benchmark(Handle function, Handle blas, unsigned rows, unsigned inner, unsigned cols) {
    const size_t sizes[3] = {(size_t)rows * inner, (size_t)inner * cols, (size_t)rows * cols};
    DevicePointer buffers[3];
    for (unsigned b = 0; b < 3; ++b) {
        float *values = malloc(sizes[b] * sizeof(float));
        if (!values) exit(1);
        for (size_t i = 0; i < sizes[b]; ++i) values[i] = b == 2 ? 0.f : ((int)(i % 31) - 15) / 32.f;
        CHECK(cuMemAlloc_v2(&buffers[b], sizes[b] * sizeof(float)));
        CHECK(cuMemcpyHtoD_v2(buffers[b], values, sizes[b] * sizeof(float)));
        free(values);
    }
    void *arguments[] = {&buffers[0], &buffers[1], &buffers[2], &rows, &inner, &cols};
    const unsigned tile = rows == 1 && cols <= 2048 ? 1 : 8;
    const unsigned blocks = (((rows + 15) / 16) * ((cols + tile - 1) / tile) + 3) / 4;
    Handle start, stop;
    CHECK(cuEventCreate(&start, 0));
    CHECK(cuEventCreate(&stop, 0));
    float *results[2];
    const unsigned repeats = 30;
    for (unsigned implementation = 0; implementation < 2; ++implementation) {
        for (unsigned iteration = 0; iteration <= repeats; ++iteration) {
            if (iteration == 1) CHECK(cuEventRecord(start, NULL));
            if (implementation == 0) {
                CHECK(cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0, NULL, arguments, NULL));
            } else {
                const float alpha = 1.f, beta = 0.f;
                // Row-major Y[M,N] = X[M,K] W[N,K]^T is column-major
                // Y[N,M] = W[K,N]^T X[K,M]. The default math mode is FP32.
                CHECK(cublasSgemm_v2(blas, 1, 0, (int)cols, (int)rows, (int)inner,
                    &alpha, buffers[1], (int)inner, buffers[0], (int)inner, &beta, buffers[2], (int)cols));
            }
        }
        CHECK(cuEventRecord(stop, NULL));
        CHECK(cuEventSynchronize(stop));
        float milliseconds;
        CHECK(cuEventElapsedTime(&milliseconds, start, stop));
        printf("LINUX_KERNEL_BENCH implementation=%s rows=%u inner=%u cols=%u repeats=%u mean_us=%.3f effective_gb_s=%.3f\n",
            implementation ? "cublas-fp32" : "d3os-kernel", rows, inner, cols, repeats,
            milliseconds * 1000 / repeats, (double)sizes[1] * sizeof(float) * repeats / (milliseconds * 1e6));
        fflush(stdout);
        results[implementation] = malloc(sizes[2] * sizeof(float));
        if (!results[implementation]) exit(1);
        CHECK(cuMemcpyDtoH_v2(results[implementation], buffers[2], sizes[2] * sizeof(float)));
    }
    for (size_t i = 0; i < sizes[2]; ++i) {
        const float a = results[0][i], b = results[1][i];
        if (!isfinite(a) || !isfinite(b) || fabsf(a - b) > 0.0002f + 0.0002f * fabsf(b)) {
            fprintf(stderr, "projection mismatch at %zu: %g versus %g\n", i, a, b);
            exit(1);
        }
    }
    for (unsigned i = 0; i < 2; ++i) free(results[i]);
    for (unsigned i = 0; i < 3; ++i) CHECK(cuMemFree_v2(buffers[i]));
    CHECK(cuEventDestroy_v2(start));
    CHECK(cuEventDestroy_v2(stop));
}

int main(void) {
    void *cuda = library("libcuda.so.1");
    LOAD(cuda, cuInit); LOAD(cuda, cuDeviceGet); LOAD(cuda, cuCtxCreate_v2); LOAD(cuda, cuCtxDestroy_v2);
    LOAD(cuda, cuMemAlloc_v2); LOAD(cuda, cuMemFree_v2);
    LOAD(cuda, cuMemcpyHtoD_v2); LOAD(cuda, cuMemcpyDtoH_v2);
    LOAD(cuda, cuModuleLoad); LOAD(cuda, cuModuleGetFunction);
    LOAD(cuda, cuLaunchKernel); LOAD(cuda, cuCtxSynchronize);
    LOAD(cuda, cuEventCreate); LOAD(cuda, cuEventRecord);
    LOAD(cuda, cuEventSynchronize); LOAD(cuda, cuEventElapsedTime); LOAD(cuda, cuEventDestroy_v2);
    void *cublas = library("libcublas.so.12");
    LOAD(cublas, cublasCreate_v2); LOAD(cublas, cublasSgemm_v2); LOAD(cublas, cublasDestroy_v2);
    CHECK(cuInit(0));
    int device;
    Handle context, module, function, blas;
    CHECK(cuDeviceGet(&device, 0));
    CHECK(cuCtxCreate_v2(&context, 0, device));
    CHECK(cuModuleLoad(&module, "/inference.cubin"));
    CHECK(cuModuleGetFunction(&function, module, "tensor_project_tf32x3"));
    CHECK(cublasCreate_v2(&blas));
    benchmark(function, blas, 1, 9216, 4096);
    benchmark(function, blas, 1, 1024, 3584);
    benchmark(function, blas, 1, 3584, 1024);
    benchmark(function, blas, 1, 1024, 248320);
    benchmark(function, blas, 55 * 55, 363, 64);
    CHECK(cublasDestroy_v2(blas));
    CHECK(cuCtxDestroy_v2(context));
    dlclose(cublas);
    dlclose(cuda);
    puts("LINUX_KERNEL_BENCH_OK");
    return 0;
}
