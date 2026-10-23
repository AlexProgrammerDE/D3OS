// Linux-only quantized-model baseline. D3OS does not link llama.cpp or CUDA.
#define _GNU_SOURCE
#include "llama.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BATCH 32
#define CAPACITY 2048
#define DECODE 128

static void require(bool condition, const char *operation) {
  if (!condition) {
    fprintf(stderr, "LINUX_LARGE_QWEN_FAIL %s errno=%d\n", operation, errno);
    exit(EXIT_FAILURE);
  }
}

static double now_ms(void) {
  struct timespec time;
  require(clock_gettime(CLOCK_MONOTONIC, &time) == 0, "clock_gettime");
  return time.tv_sec * 1000.0 + time.tv_nsec / 1e6;
}

static void evaluate(struct llama_context *context, struct llama_batch *batch,
                     const llama_token *tokens, int count, int position) {
  batch->n_tokens = count;
  for (int i = 0; i < count; ++i) {
    batch->token[i] = tokens[i];
    batch->pos[i] = position + i;
    batch->n_seq_id[i] = 1;
    batch->seq_id[i][0] = 0;
    batch->logits[i] = i + 1 == count;
  }
  require(llama_decode(context, *batch) == 0, "llama_decode");
  llama_synchronize(context);
}

static void memory(ggml_backend_dev_t device, const char *phase) {
  size_t free_bytes, total_bytes;
  ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
  printf("LINUX_LARGE_QWEN_MEMORY phase=%s free_bytes=%zu total_bytes=%zu "
         "used_bytes=%zu\n",
         phase, free_bytes, total_bytes, total_bytes - free_bytes);
}

int main(int argc, char **argv) {
  require(argc == 4, "usage: linux-large-qwen-bench MODEL LAYERS HIDDEN");
  const int layers = atoi(argv[2]), hidden = atoi(argv[3]);
  require((layers == 32 && hidden == 4096) || (layers == 64 && hidden == 5120),
          "model target");
  ggml_backend_load_all_from_path("/bin");
  llama_backend_init();
  ggml_backend_dev_t device =
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
  require(device != NULL && strcmp(ggml_backend_reg_name(
                                       ggml_backend_dev_backend_reg(device)),
                                   "CUDA") == 0,
          "CUDA GPU backend required");
  ggml_backend_dev_t devices[] = {device, NULL};
  memory(device, "before_load");
  struct llama_model_params model_params = llama_model_default_params();
  model_params.devices = devices;
  model_params.n_gpu_layers = -1;
  model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
  // Include embeddings and small tensors, which the default loader can keep
  // in CPU memory even when every transformer layer is offloaded.
  const struct llama_model_tensor_buft_override overrides[] = {
      {".*", ggml_backend_dev_buffer_type(device)}, {NULL, NULL}};
  model_params.tensor_buft_overrides = overrides;
  double start = now_ms();
  struct llama_model *model = llama_model_load_from_file(argv[1], model_params);
  require(model != NULL, "load quantized model");
  double load_ms = now_ms() - start;
  const struct llama_vocab *vocab = llama_model_get_vocab(model);
  require(llama_model_n_layer(model) == layers &&
              llama_model_n_embd(model) == hidden &&
              llama_vocab_n_tokens(vocab) == 248320,
          "model dimensions");
  printf("LINUX_LARGE_QWEN_MODEL layers=%d hidden=%d parameters=%" PRIu64
         " load_ms=%.3f\n",
         layers, hidden, llama_model_n_params(model), load_ms);
  memory(device, "after_load");

  struct llama_sampler *sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  require(sampler != NULL, "sampler chain");
  llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  struct llama_sampler_seq_config sampler_config = {.seq_id = 0,
                                                    .sampler = sampler};
  struct llama_context_params params = llama_context_default_params();
  params.n_ctx = CAPACITY;
  params.n_batch = BATCH;
  params.n_ubatch = BATCH;
  params.n_seq_max = 1;
  params.n_outputs_max = 1;
  params.n_outputs_max_per_seq = 1;
  params.n_threads = 4;
  params.n_threads_batch = 4;
  params.type_k = GGML_TYPE_F32;
  params.type_v = GGML_TYPE_F32;
  params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
  params.offload_kqv = true;
  params.op_offload = true;
  params.no_perf = false;
  params.samplers = &sampler_config;
  params.n_samplers = 1;
  start = now_ms();
  struct llama_context *context = llama_init_from_model(model, params);
  require(context != NULL, "create GPU context");
  printf("LINUX_LARGE_QWEN_CONTEXT state_ms=%.3f cache=f32 batch=%d n_ctx=%u\n",
         now_ms() - start, BATCH, llama_n_ctx(context));
  memory(device, "after_context");
  struct llama_batch batch = llama_batch_init(BATCH, 0, 1);
  // Explicit text-only, non-thinking chat. Keep these bytes in the evidence
  // so the native benchmark can use exactly the same prompt tokens later.
  const char *chat =
      "<|im_start|>user\nWhich country is largest by land area? "
      "Write a long, detailed essay about its geography, climate, "
      "population, and history.<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  llama_token prompt[256];
  int count =
      llama_tokenize(vocab, chat, (int)strlen(chat), prompt, 256, false, true);
  require(count > 0 && count <= 256, "tokenize prompt");
  printf("LINUX_LARGE_QWEN_PROMPT ids=[");
  for (int i = 0; i < count; ++i) {
    printf("%s%d", i ? "," : "", prompt[i]);
  }
  puts("]");
  for (int run = 0; run < 3; ++run) {
    llama_memory_clear(llama_get_memory(context), true);
    llama_sampler_reset(sampler);
    llama_synchronize(context);
    llama_perf_context_reset(context);
    start = now_ms();
    for (int position = 0; position < count; position += BATCH) {
      int rows = count - position < BATCH ? count - position : BATCH;
      evaluate(context, &batch, prompt + position, rows, position);
    }
    double prefill_ms = now_ms() - start;
    llama_token tokens[DECODE];
    uint64_t checksum = 0;
    start = now_ms();
    for (int i = 0; i < DECODE; ++i) {
      tokens[i] = llama_get_sampled_token_ith(context, -1);
      require(tokens[i] != LLAMA_TOKEN_NULL, "GPU greedy sampling");
      llama_sampler_accept(sampler, tokens[i]);
      checksum = checksum * 31 + (uint32_t)tokens[i];
      evaluate(context, &batch, &tokens[i], 1, count + i);
    }
    double decode_ms = now_ms() - start;
    struct llama_perf_context_data perf = llama_perf_context(context);
    require(perf.n_p_eval == count && perf.n_eval == DECODE &&
                isfinite(decode_ms) && decode_ms > 0,
            "exact evaluated work");
    printf("LINUX_LARGE_QWEN_BENCH run=%d warm=%s prompt_tokens=%d "
           "prefill_ms=%.3f decode_tokens=%d "
           "decode_ms=%.3f decode_tps=%.3f checksum=%" PRIu64
           " cache=f32 n_ctx=%u sampler=backend_greedy "
           "eos=continue_for_benchmark\n",
           run, run ? "true" : "false", count, prefill_ms, DECODE, decode_ms,
           DECODE * 1000.0 / decode_ms, checksum, llama_n_ctx(context));
    printf("LINUX_LARGE_QWEN_TOKENS run=%d ids=[", run);
    for (int i = 0; i < DECODE; ++i) {
      printf("%s%d", i ? "," : "", tokens[i]);
    }
    puts("]");
    if (run == 0) {
      printf("LINUX_LARGE_QWEN_ANSWER ");
      for (int i = 0; i < DECODE && !llama_vocab_is_eog(vocab, tokens[i]);
           ++i) {
        char piece[4096];
        int size = llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece),
                                        0, false);
        require(size >= 0, "decode token text");
        fwrite(piece, 1, (size_t)size, stdout);
      }
      puts("\nLINUX_LARGE_QWEN_ANSWER_END");
    }
  }
  memory(device, "after_decode");
  llama_batch_free(batch);
  llama_free(context);
  llama_sampler_free(sampler);
  llama_model_free(model);
  llama_backend_free();
  puts("LINUX_LARGE_QWEN_OK");
  return EXIT_SUCCESS;
}
