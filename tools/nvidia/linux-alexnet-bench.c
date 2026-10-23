// Isolated Linux reference. D3OS does not load these NVIDIA libraries.
#include <cuda.h>
#include <cudnn.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DECLARE(name) static __typeof__(name) *p_##name
DECLARE(cuInit);
DECLARE(cuDeviceGet);
DECLARE(cuCtxCreate_v4);
DECLARE(cuCtxDestroy_v2);
DECLARE(cuMemAlloc_v2);
DECLARE(cuMemcpyHtoD_v2);
DECLARE(cuMemcpyDtoH_v2);
DECLARE(cuMemcpyDtoD_v2);
DECLARE(cuCtxSynchronize);
DECLARE(cudnnCreate);
DECLARE(cudnnDestroy);
DECLARE(cudnnGetVersion);
DECLARE(cudnnGetLastErrorString);
DECLARE(cudnnCreateTensorDescriptor);
DECLARE(cudnnSetTensor4dDescriptor);
DECLARE(cudnnDestroyTensorDescriptor);
DECLARE(cudnnCreateFilterDescriptor);
DECLARE(cudnnSetFilter4dDescriptor);
DECLARE(cudnnDestroyFilterDescriptor);
DECLARE(cudnnCreateConvolutionDescriptor);
DECLARE(cudnnSetConvolution2dDescriptor);
DECLARE(cudnnDestroyConvolutionDescriptor);
DECLARE(cudnnSetConvolutionMathType);
DECLARE(cudnnGetConvolution2dForwardOutputDim);
DECLARE(cudnnGetConvolutionForwardAlgorithm_v7);
DECLARE(cudnnConvolutionForward);
DECLARE(cudnnAddTensor);
DECLARE(cudnnCreateActivationDescriptor);
DECLARE(cudnnSetActivationDescriptor);
DECLARE(cudnnActivationForward);
DECLARE(cudnnDestroyActivationDescriptor);
DECLARE(cudnnCreatePoolingDescriptor);
DECLARE(cudnnSetPooling2dDescriptor);
DECLARE(cudnnPoolingForward);
DECLARE(cudnnDestroyPoolingDescriptor);
static int (*p_cublasCreate_v2)(void **);
static int (*p_cublasDestroy_v2)(void *);
static int (*p_cublasSetMathMode)(void *, int);
static int (*p_cublasSgemm_v2)(void *, int, int, int, int, int, const float *,
                               CUdeviceptr, int, CUdeviceptr, int,
                               const float *, CUdeviceptr, int);

static void require(int condition, const char *operation) {
  if (!condition) {
    fprintf(stderr, "LINUX_ALEXNET_FAIL %s\n", operation);
    exit(EXIT_FAILURE);
  }
}
static void check(int status, const char *operation) {
  if (status) {
    fprintf(stderr, "LINUX_ALEXNET_FAIL %s status=%d\n", operation, status);
    if (p_cudnnGetLastErrorString) {
      char message[2048] = {0};
      p_cudnnGetLastErrorString(message, sizeof(message));
      fprintf(stderr, "%s\n", message);
    }
    exit(EXIT_FAILURE);
  }
}
#define CHECK(call) check((int)(call), #call)
#define LOAD(library, name)                                                    \
  do {                                                                         \
    *(void **)(&p_##name) = dlsym(library, #name);                             \
    require(p_##name != NULL, #name);                                          \
  } while (0)
static void *library(const char *name) {
  void *result = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (!result) {
    fprintf(stderr, "%s\n", dlerror());
    exit(EXIT_FAILURE);
  }
  return result;
}
static double now_ms(void) {
  struct timespec t;
  require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "clock_gettime");
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

typedef struct {
  const uint8_t *bytes;
  size_t size, position;
} Cursor;
typedef struct {
  char name[64];
  unsigned rank, shape[4];
  size_t elements;
  const uint8_t *data;
} Record;
typedef struct {
  uint8_t *data;
  Record records[32];
  unsigned count;
} Archive;
static const uint8_t *take(Cursor *c, size_t n) {
  require(n <= c->size - c->position, "archive extent");
  const uint8_t *result = c->bytes + c->position;
  c->position += n;
  return result;
}
static uint64_t integer(Cursor *c, unsigned width) {
  const uint8_t *b = take(c, width);
  uint64_t n = 0;
  for (unsigned i = 0; i < width; ++i)
    n |= (uint64_t)b[i] << (8 * i);
  return n;
}
static Archive read_archive(const char *path) {
  FILE *file = fopen(path, "rb");
  require(file != NULL, "open archive");
  require(fseek(file, 0, SEEK_END) == 0, "size archive");
  long size = ftell(file);
  require(size > 0 && fseek(file, 0, SEEK_SET) == 0, "seek archive");
  Archive result = {0};
  result.data = malloc((size_t)size);
  require(result.data != NULL, "archive memory");
  require(fread(result.data, 1, (size_t)size, file) == (size_t)size &&
              fclose(file) == 0,
          "read archive");
  Cursor c = {result.data, (size_t)size, 0};
  require(memcmp(take(&c, 8), "D3TENSR\0", 8) == 0 && integer(&c, 4) == 1,
          "FP32 archive format");
  result.count = (unsigned)integer(&c, 4);
  require(result.count > 0 && result.count <= 32, "archive count");
  for (unsigned i = 0; i < result.count; ++i) {
    Record *r = &result.records[i];
    unsigned length = (unsigned)integer(&c, 2);
    r->rank = (unsigned)integer(&c, 2);
    r->elements = (size_t)integer(&c, 8);
    require(length > 0 && length < sizeof(r->name) && r->rank > 0 &&
                r->rank <= 4,
            "record metadata");
    memcpy(r->name, take(&c, length), length);
    r->name[length] = 0;
    size_t elements = 1;
    for (unsigned j = 0; j < r->rank; ++j) {
      uint64_t dim = integer(&c, 8);
      require(dim > 0 && dim <= 1u << 20 && elements <= SIZE_MAX / dim,
              "record dimension");
      r->shape[j] = (unsigned)dim;
      elements *= (size_t)dim;
    }
    require(elements == r->elements && elements <= SIZE_MAX / sizeof(float),
            "record length");
    r->data = take(&c, elements * sizeof(float));
    for (unsigned j = 0; j < i; ++j)
      require(strcmp(r->name, result.records[j].name) != 0, "duplicate tensor");
  }
  require(c.position == c.size, "archive end");
  return result;
}
static Record *record(Archive *a, const char *name, unsigned rank,
                      const unsigned *shape) {
  for (unsigned i = 0; i < a->count; ++i)
    if (strcmp(a->records[i].name, name) == 0) {
      Record *r = &a->records[i];
      require(r->rank == rank &&
                  memcmp(r->shape, shape, rank * sizeof(*shape)) == 0,
              name);
      return r;
    }
  require(0, name);
  return NULL;
}
static float value(const Record *r, size_t i) {
  float f;
  memcpy(&f, r->data + i * sizeof(float), sizeof(f));
  return f;
}
static CUdeviceptr allocate(size_t bytes) {
  CUdeviceptr p;
  CHECK(p_cuMemAlloc_v2(&p, bytes));
  return p;
}
static CUdeviceptr upload(const Record *r) {
  for (size_t i = 0; i < r->elements; ++i)
    require(isfinite(value(r, i)), "finite model data");
  CUdeviceptr p = allocate(r->elements * sizeof(float));
  CHECK(p_cuMemcpyHtoD_v2(p, r->data, r->elements * sizeof(float)));
  return p;
}
static cudnnTensorDescriptor_t tensor(unsigned c, unsigned h, unsigned w) {
  cudnnTensorDescriptor_t d;
  CHECK(p_cudnnCreateTensorDescriptor(&d));
  CHECK(p_cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                                     (int)c, (int)h, (int)w));
  return d;
}
static void verify(CUdeviceptr output, const Record *r) {
  float *actual = malloc(r->elements * sizeof(float));
  require(actual != NULL, "verification buffer");
  CHECK(p_cuMemcpyDtoH_v2(actual, output, r->elements * sizeof(float)));
  float max_error = 0;
  for (size_t i = 0; i < r->elements; ++i) {
    float expected = value(r, i), error = fabsf(actual[i] - expected);
    if (!isfinite(actual[i]) || !isfinite(expected) ||
        error > 2e-4f + 2e-4f * fabsf(expected)) {
      fprintf(stderr,
              "LINUX_ALEXNET_LAYER_FAIL name=%s index=%zu actual=%.9g "
              "expected=%.9g\n",
              r->name, i, actual[i], expected);
      exit(EXIT_FAILURE);
    }
    if (error > max_error)
      max_error = error;
  }
  printf("LINUX_ALEXNET_LAYER_OK name=%s elements=%zu max_abs_error=%.9g\n",
         r->name, r->elements, max_error);
  free(actual);
}

typedef struct {
  const char *name;
  unsigned inputs, outputs, h, w;
  CUdeviceptr weights, bias, output, pooled;
  cudnnTensorDescriptor_t input_desc, output_desc, bias_desc, pool_desc;
  cudnnFilterDescriptor_t filter;
  cudnnConvolutionDescriptor_t convolution;
  cudnnConvolutionFwdAlgo_t algorithm;
  size_t workspace;
  Record *reference;
} Layer;
static CUdeviceptr forward(Layer *layers, CUdeviceptr input, cudnnHandle_t dnn,
                           void *blas, cudnnActivationDescriptor_t relu,
                           cudnnPoolingDescriptor_t pool, CUdeviceptr workspace,
                           int validate) {
  const float one = 1, zero = 0;
  for (unsigned i = 0; i < 8; ++i) {
    Layer *l = &layers[i];
    if (i < 5) {
      CHECK(p_cudnnConvolutionForward(
          dnn, &one, l->input_desc, (void *)(uintptr_t)input, l->filter,
          (void *)(uintptr_t)l->weights, l->convolution, l->algorithm,
          (void *)(uintptr_t)workspace, l->workspace, &zero, l->output_desc,
          (void *)(uintptr_t)l->output));
      CHECK(p_cudnnAddTensor(dnn, &one, l->bias_desc,
                             (void *)(uintptr_t)l->bias, &one, l->output_desc,
                             (void *)(uintptr_t)l->output));
    } else {
      CHECK(p_cuMemcpyDtoD_v2(l->output, l->bias, l->outputs * sizeof(float)));
      CHECK(p_cublasSgemm_v2(blas, 1, 0, (int)l->outputs, 1, (int)l->inputs,
                             &one, l->weights, (int)l->inputs, input,
                             (int)l->inputs, &one, l->output, (int)l->outputs));
    }
    if (i != 7)
      CHECK(p_cudnnActivationForward(
          dnn, relu, &one, l->output_desc, (void *)(uintptr_t)l->output, &zero,
          l->output_desc, (void *)(uintptr_t)l->output));
    if (validate)
      verify(l->output, l->reference);
    input = l->output;
    if (l->pooled) {
      CHECK(p_cudnnPoolingForward(dnn, pool, &one, l->output_desc,
                                  (void *)(uintptr_t)input, &zero, l->pool_desc,
                                  (void *)(uintptr_t)l->pooled));
      input = l->pooled;
    }
  }
  CHECK(p_cuCtxSynchronize());
  return input;
}

int main(int argc, char **argv) {
  require(argc == 3, "usage: linux-alexnet-bench MODEL FIXTURE");
  double start = now_ms();
  Archive weights = read_archive(argv[1]), fixture = read_archive(argv[2]);
  double file_ms = now_ms() - start;
  require(weights.count == 16 && fixture.count == 9,
          "complete AlexNet archives");
  void *cuda = library("libcuda.so.1"), *cudnn = library("libcudnn.so.9"),
       *cublas = library("libcublas.so.13");
  LOAD(cuda, cuInit);
  LOAD(cuda, cuDeviceGet);
  LOAD(cuda, cuCtxCreate_v4);
  LOAD(cuda, cuCtxDestroy_v2);
  LOAD(cuda, cuMemAlloc_v2);
  LOAD(cuda, cuMemcpyHtoD_v2);
  LOAD(cuda, cuMemcpyDtoH_v2);
  LOAD(cuda, cuMemcpyDtoD_v2);
  LOAD(cuda, cuCtxSynchronize);
  LOAD(cudnn, cudnnCreate);
  LOAD(cudnn, cudnnDestroy);
  LOAD(cudnn, cudnnGetVersion);
  LOAD(cudnn, cudnnGetLastErrorString);
  LOAD(cudnn, cudnnCreateTensorDescriptor);
  LOAD(cudnn, cudnnSetTensor4dDescriptor);
  LOAD(cudnn, cudnnDestroyTensorDescriptor);
  LOAD(cudnn, cudnnCreateFilterDescriptor);
  LOAD(cudnn, cudnnSetFilter4dDescriptor);
  LOAD(cudnn, cudnnDestroyFilterDescriptor);
  LOAD(cudnn, cudnnCreateConvolutionDescriptor);
  LOAD(cudnn, cudnnSetConvolution2dDescriptor);
  LOAD(cudnn, cudnnDestroyConvolutionDescriptor);
  LOAD(cudnn, cudnnSetConvolutionMathType);
  LOAD(cudnn, cudnnGetConvolution2dForwardOutputDim);
  LOAD(cudnn, cudnnGetConvolutionForwardAlgorithm_v7);
  LOAD(cudnn, cudnnConvolutionForward);
  LOAD(cudnn, cudnnAddTensor);
  LOAD(cudnn, cudnnCreateActivationDescriptor);
  LOAD(cudnn, cudnnSetActivationDescriptor);
  LOAD(cudnn, cudnnActivationForward);
  LOAD(cudnn, cudnnDestroyActivationDescriptor);
  LOAD(cudnn, cudnnCreatePoolingDescriptor);
  LOAD(cudnn, cudnnSetPooling2dDescriptor);
  LOAD(cudnn, cudnnPoolingForward);
  LOAD(cudnn, cudnnDestroyPoolingDescriptor);
  LOAD(cublas, cublasCreate_v2);
  LOAD(cublas, cublasDestroy_v2);
  LOAD(cublas, cublasSetMathMode);
  LOAD(cublas, cublasSgemm_v2);
  CHECK(p_cuInit(0));
  CUdevice device;
  CHECK(p_cuDeviceGet(&device, 0));
  CUcontext context;
  CHECK(p_cuCtxCreate_v4(&context, NULL, 0, device));
  cudnnHandle_t dnn;
  void *blas;
  CHECK(p_cudnnCreate(&dnn));
  CHECK(p_cublasCreate_v2(&blas));
  CHECK(p_cublasSetMathMode(blas, 0));
  cudnnActivationDescriptor_t relu;
  CHECK(p_cudnnCreateActivationDescriptor(&relu));
  CHECK(p_cudnnSetActivationDescriptor(relu, CUDNN_ACTIVATION_RELU,
                                       CUDNN_PROPAGATE_NAN, 0));
  cudnnPoolingDescriptor_t pool;
  CHECK(p_cudnnCreatePoolingDescriptor(&pool));
  CHECK(p_cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_MAX,
                                      CUDNN_PROPAGATE_NAN, 3, 3, 0, 0, 2, 2));
  const unsigned output_channels[] = {64, 192, 384, 256, 256, 4096, 4096, 1000};
  const char *names[] = {"features.0",   "features.3",  "features.6",
                         "features.8",   "features.10", "classifier.1",
                         "classifier.4", "classifier.6"};
  Layer layers[8] = {0};
  unsigned channels = 3, height = 224, width = 224;
  size_t workspace_size = 0;
  start = now_ms();
  for (unsigned i = 0; i < 8; ++i) {
    Layer *l = &layers[i];
    l->name = names[i];
    l->inputs = i < 5 ? channels : channels * height * width;
    l->outputs = output_channels[i];
    unsigned kernel = i == 0 ? 11 : i == 1 ? 5 : 3;
    unsigned shape[] = {l->outputs, l->inputs, kernel, kernel};
    char name[64];
    snprintf(name, sizeof(name), "%s.weight", l->name);
    l->weights = upload(record(&weights, name, i < 5 ? 4 : 2, shape));
    snprintf(name, sizeof(name), "%s.bias", l->name);
    l->bias = upload(record(&weights, name, 1, &l->outputs));
    l->input_desc = tensor(channels, height, width);
    if (i < 5) {
      CHECK(p_cudnnCreateFilterDescriptor(&l->filter));
      CHECK(p_cudnnSetFilter4dDescriptor(
          l->filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, (int)l->outputs,
          (int)channels, (int)kernel, (int)kernel));
      CHECK(p_cudnnCreateConvolutionDescriptor(&l->convolution));
      int padding = i < 2 ? 2 : 1, stride = i == 0 ? 4 : 1;
      CHECK(p_cudnnSetConvolution2dDescriptor(
          l->convolution, padding, padding, stride, stride, 1, 1,
          CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
      CHECK(p_cudnnSetConvolutionMathType(l->convolution, CUDNN_FMA_MATH));
      int n, c, h, w;
      CHECK(p_cudnnGetConvolution2dForwardOutputDim(
          l->convolution, l->input_desc, l->filter, &n, &c, &h, &w));
      require(n == 1 && c == (int)l->outputs && h > 0 && w > 0,
              "convolution shape");
      l->h = (unsigned)h;
      l->w = (unsigned)w;
    } else {
      l->h = 1;
      l->w = 1;
    }
    l->output_desc = tensor(l->outputs, l->h, l->w);
    l->bias_desc = tensor(l->outputs, 1, 1);
    l->output = allocate((size_t)l->outputs * l->h * l->w * sizeof(float));
    if (i < 5) {
      cudnnConvolutionFwdAlgoPerf_t algorithms[8];
      int count;
      CHECK(p_cudnnGetConvolutionForwardAlgorithm_v7(
          dnn, l->input_desc, l->filter, l->convolution, l->output_desc, 8,
          &count, algorithms));
      int selected = 0;
      for (int j = 0; j < count; ++j)
        if (algorithms[j].status == CUDNN_STATUS_SUCCESS &&
            algorithms[j].memory <= 256u * 1024 * 1024) {
          l->algorithm = algorithms[j].algo;
          l->workspace = algorithms[j].memory;
          selected = 1;
          break;
        }
      require(selected, "FP32 convolution algorithm");
      if (l->workspace > workspace_size)
        workspace_size = l->workspace;
      printf("LINUX_ALEXNET_ALGORITHM layer=%s algorithm=%d workspace=%zu "
             "math=fma\n",
             l->name, (int)l->algorithm, l->workspace);
    }
    unsigned reference_shape[] = {1, l->outputs, l->h, l->w};
    l->reference = record(&fixture, i == 7 ? "logits" : l->name, i < 5 ? 4 : 2,
                          reference_shape);
    channels = l->outputs;
    height = l->h;
    width = l->w;
    if (i == 0 || i == 1 || i == 4) {
      height = (height - 3) / 2 + 1;
      width = (width - 3) / 2 + 1;
      l->pool_desc = tensor(channels, height, width);
      l->pooled = allocate((size_t)channels * height * width * sizeof(float));
    }
  }
  CUdeviceptr workspace = workspace_size ? allocate(workspace_size) : 0;
  double load_ms = now_ms() - start;
  const unsigned image_shape[] = {1, 3, 224, 224};
  CUdeviceptr input = upload(record(&fixture, "input", 4, image_shape));
  for (unsigned run = 0; run < 3; ++run) {
    forward(layers, input, dnn, blas, relu, pool, workspace, 0);
    start = now_ms();
    for (unsigned i = 0; i < 100; ++i)
      forward(layers, input, dnn, blas, relu, pool, workspace, 0);
    double elapsed = now_ms() - start;
    printf("LINUX_ALEXNET_BENCH run=%u repeats=100 file_ms=%.3f load_ms=%.3f "
           "forward_total_ms=%.6f forward_mean_ms=%.6f weights=f32 "
           "arithmetic=f32 cudnn=%zu cudnn_math=fma cublas_math=default\n",
           run, file_ms, load_ms, elapsed, elapsed / 100, p_cudnnGetVersion());
  }
  CUdeviceptr logits =
      forward(layers, input, dnn, blas, relu, pool, workspace, 1);
  float scores[1000];
  CHECK(p_cuMemcpyDtoH_v2(scores, logits, sizeof(scores)));
  unsigned best = 0;
  for (unsigned i = 1; i < 1000; ++i)
    if (scores[i] > scores[best])
      best = i;
  require(best == 258, "dog class");
  for (unsigned i = 0; i < 8; ++i) {
    Layer *l = &layers[i];
    CHECK(p_cudnnDestroyTensorDescriptor(l->input_desc));
    CHECK(p_cudnnDestroyTensorDescriptor(l->output_desc));
    CHECK(p_cudnnDestroyTensorDescriptor(l->bias_desc));
    if (i < 5) {
      CHECK(p_cudnnDestroyFilterDescriptor(l->filter));
      CHECK(p_cudnnDestroyConvolutionDescriptor(l->convolution));
    }
    if (l->pooled)
      CHECK(p_cudnnDestroyTensorDescriptor(l->pool_desc));
  }
  CHECK(p_cudnnDestroyActivationDescriptor(relu));
  CHECK(p_cudnnDestroyPoolingDescriptor(pool));
  CHECK(p_cublasDestroy_v2(blas));
  CHECK(p_cudnnDestroy(dnn));
  CHECK(p_cuCtxDestroy_v2(context));
  free(weights.data);
  free(fixture.data);
  dlclose(cublas);
  dlclose(cudnn);
  dlclose(cuda);
  puts("LINUX_ALEXNET_OK reference_layers=8 class=258");
  return EXIT_SUCCESS;
}
