// Linux reference diagnostics only. Never linked into D3OS.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { unsigned int x, y, z; } dimensions;
typedef int (*launch_fn)(const void *, dimensions, dimensions, void **, size_t, void *);
typedef int (*launch_ex_fn)(const void *, const void *, void **);
typedef int (*name_fn)(const char **, const void *);

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t runtime_once = PTHREAD_ONCE_INIT;
static void *runtime;
static const void *seen[512];
static size_t count;

static void find_runtime(void) {
  // llama.cpp loads its CUDA backend locally, outside RTLD_NEXT's scope.
  runtime = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
  if (runtime == NULL) {
    fprintf(stderr, "LINUX_CUDA_TRACE_FAIL runtime=%s\n", dlerror());
    abort();
  }
}

static void *symbol(const char *name) {
  pthread_once(&runtime_once, find_runtime);
  void *result = dlsym(runtime, name);
  if (result == NULL) {
    fprintf(stderr, "LINUX_CUDA_TRACE_FAIL symbol=%s error=%s\n", name, dlerror());
    abort();
  }
  return result;
}

static void record(const void *function, const char *api) {
  pthread_mutex_lock(&lock);
  for (size_t i = 0; i < count; ++i) {
    if (seen[i] == function) {
      pthread_mutex_unlock(&lock);
      return;
    }
  }
  if (count == sizeof(seen) / sizeof(seen[0])) {
    fprintf(stderr, "LINUX_CUDA_TRACE_FAIL capacity=%zu\n", count);
    abort();
  }
  const char *name = NULL;
  int status = ((name_fn)symbol("cudaFuncGetName"))(&name, function);
  if (status != 0 || name == NULL) {
    fprintf(stderr, "LINUX_CUDA_TRACE_FAIL function_name_status=%d\n", status);
    abort();
  }
  seen[count++] = function;
  fprintf(stderr, "LINUX_CUDA_KERNEL api=%s name=%s\n", api, name);
  pthread_mutex_unlock(&lock);
}

int cudaLaunchKernel(const void *function, dimensions grid, dimensions block,
                     void **arguments, size_t shared, void *stream) {
  record(function, "cudaLaunchKernel");
  return ((launch_fn)symbol("cudaLaunchKernel"))(function, grid, block, arguments, shared, stream);
}

int cudaLaunchKernelExC(const void *config, const void *function, void **arguments) {
  record(function, "cudaLaunchKernelExC");
  return ((launch_ex_fn)symbol("cudaLaunchKernelExC"))(config, function, arguments);
}
