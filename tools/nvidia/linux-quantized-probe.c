// Linux-only diagnostic for the exact kernels used by the native D3OS backend.
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t DevicePointer;
typedef void *Handle;
enum { GUARD_WORDS = 64 };
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
CUDA_FUNCTION(cuModuleUnload, (Handle));
CUDA_FUNCTION(cuModuleGetFunction, (Handle *, Handle, const char *));
CUDA_FUNCTION(cuLaunchKernel, (Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, Handle, void **, void **));
CUDA_FUNCTION(cuCtxSynchronize, (void));
CUDA_FUNCTION(cuProfilerStart, (void));
CUDA_FUNCTION(cuProfilerStop, (void));

static void check(int status, const char *operation) {
    if (status) { fprintf(stderr, "%s failed: %d\n", operation, status); exit(1); }
}
#define CHECK(call) check((call), #call)
#define LOAD(name) do { *(void **)(&name) = dlsym(cuda, #name); if (!name) { fprintf(stderr, "%s\n", dlerror()); exit(1); } } while (0)

static unsigned dimension(const char *text) {
    char *end;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || !*text || *text == '-' || *end || value == 0 || value > UINT_MAX) exit(1);
    return (unsigned)value;
}

static void read_file(const char *path, void *destination, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file || fread(destination, 1, bytes, file) != bytes || fgetc(file) != EOF || fclose(file)) {
        fprintf(stderr, "cannot read exact probe input: %s\n", path); exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 7 && !(argc == 8 && strcmp(argv[7], "profile") == 0)) {
        fprintf(stderr, "usage: probe WEIGHTS INPUT REFERENCE INNER OUTPUTS FORMAT [profile]\n"); return 1;
    }
    int instrumented = argc == 8;
    unsigned rows = 1, inner = dimension(argv[4]), outputs = dimension(argv[5]), format = dimension(argv[6]);
    const char *entry;
    unsigned block, bytes;
    switch (format) {
        case 8: entry = "vector_project_q8_0"; block = 32; bytes = 34; break;
        case 11: entry = "vector_project_q3k"; block = 256; bytes = 110; break;
        case 12: entry = "vector_project_q4k"; block = 256; bytes = 144; break;
        case 13: entry = "vector_project_q5k"; block = 256; bytes = 176; break;
        case 14: entry = "vector_project_q6k"; block = 256; bytes = 210; break;
        default: return 1;
    }
    if (inner % block || (uint64_t)inner * outputs > UINT_MAX) return 1;
    size_t weight_bytes = (size_t)inner * outputs / block * bytes;
    size_t words[3] = {inner, (weight_bytes + 3) / 4, outputs};
    uint32_t *host[3];
    const uint32_t guard = 0xd305abcd;
    for (unsigned b = 0; b < 3; ++b) {
        host[b] = malloc((words[b] + 2 * GUARD_WORDS) * sizeof(uint32_t));
        if (!host[b]) return 1;
        for (size_t i = 0; i < words[b] + 2 * GUARD_WORDS; ++i) host[b][i] = guard;
    }
    read_file(argv[1], host[1] + GUARD_WORDS, weight_bytes);
    read_file(argv[2], host[0] + GUARD_WORDS, words[0] * sizeof(float));
    float *reference = malloc((size_t)outputs * sizeof(float));
    if (!reference) return 1;
    read_file(argv[3], reference, (size_t)outputs * sizeof(float));
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!cuda) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    LOAD(cuInit); LOAD(cuDeviceGet); LOAD(cuCtxCreate_v2); LOAD(cuCtxDestroy_v2);
    LOAD(cuMemAlloc_v2); LOAD(cuMemFree_v2); LOAD(cuMemcpyHtoD_v2); LOAD(cuMemcpyDtoH_v2);
    LOAD(cuModuleLoad); LOAD(cuModuleUnload); LOAD(cuModuleGetFunction); LOAD(cuLaunchKernel);
    LOAD(cuCtxSynchronize); LOAD(cuProfilerStart); LOAD(cuProfilerStop);
    CHECK(cuInit(0));
    int device;
    Handle context, module, function;
    CHECK(cuDeviceGet(&device, 0)); CHECK(cuCtxCreate_v2(&context, 0, device));
    CHECK(cuModuleLoad(&module, "/inference.cubin")); CHECK(cuModuleGetFunction(&function, module, entry));
    DevicePointer buffers[3], pointers[3];
    for (unsigned b = 0; b < 3; ++b) {
        CHECK(cuMemAlloc_v2(&buffers[b], (words[b] + 2 * GUARD_WORDS) * sizeof(uint32_t)));
        CHECK(cuMemcpyHtoD_v2(buffers[b], host[b], (words[b] + 2 * GUARD_WORDS) * sizeof(uint32_t)));
        pointers[b] = buffers[b] + GUARD_WORDS * sizeof(uint32_t);
    }
    void *arguments[] = {&pointers[0], &pointers[1], &pointers[2], &rows, &inner, &outputs, &format};
    unsigned blocks = (outputs + 3) / 4;
    for (unsigned i = 0; i < 256; ++i) CHECK(cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0, NULL, arguments, NULL));
    CHECK(cuCtxSynchronize());
    if (instrumented) CHECK(cuProfilerStart());
    CHECK(cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0, NULL, arguments, NULL));
    CHECK(cuCtxSynchronize());
    if (instrumented) CHECK(cuProfilerStop());
    CHECK(cuMemcpyDtoH_v2(host[2], buffers[2], (words[2] + 2 * GUARD_WORDS) * sizeof(uint32_t)));
    for (unsigned b = 0; b < 3; ++b) {
        uint32_t edges[2 * GUARD_WORDS];
        CHECK(cuMemcpyDtoH_v2(&edges[0], buffers[b], GUARD_WORDS * sizeof(uint32_t)));
        CHECK(cuMemcpyDtoH_v2(&edges[GUARD_WORDS], buffers[b] + (words[b] + GUARD_WORDS) * sizeof(uint32_t), GUARD_WORDS * sizeof(uint32_t)));
        for (unsigned i = 0; i < 2 * GUARD_WORDS; ++i) if (edges[i] != guard) return 1;
    }
    double max_absolute = 0;
    for (unsigned i = 0; i < outputs; ++i) {
        float actual;
        memcpy(&actual, &host[2][i + GUARD_WORDS], sizeof(actual));
        double error = fabs((double)actual - reference[i]);
        if (!isfinite(actual) || !isfinite(reference[i]) || error > 0.0002 + 0.0002 * fabs(reference[i])) {
            fprintf(stderr, "projection mismatch at %u: %.9g versus %.9g\n", i, actual, reference[i]); return 1;
        }
        if (error > max_absolute) max_absolute = error;
    }
    printf("LINUX_QUANTIZED_PROBE_OK os=linux backend=native-cutile instrumented=%s entry=%s inner=%u outputs=%u format=%u values=%u max_absolute=%.9g guards=pass\n",
        instrumented ? "true" : "false", entry, inner, outputs, format, outputs, max_absolute);
    for (unsigned b = 0; b < 3; ++b) { CHECK(cuMemFree_v2(buffers[b])); free(host[b]); }
    free(reference); CHECK(cuModuleUnload(module)); CHECK(cuCtxDestroy_v2(context)); dlclose(cuda);
    return 0;
}
