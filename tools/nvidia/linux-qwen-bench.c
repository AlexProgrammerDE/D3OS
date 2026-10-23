// Linux-only llama.cpp reference. D3OS does not link these libraries.
#define _GNU_SOURCE
#include "llama.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define DECODE 64
#define BATCH 32

static void require(bool condition, const char *operation) {
    if (!condition) {
        fprintf(stderr, "LINUX_MATCHED_QWEN_FAIL %s errno=%d\n", operation, errno);
        exit(EXIT_FAILURE);
    }
}

static double now_ms(void) {
    struct timespec time;
    require(clock_gettime(CLOCK_MONOTONIC, &time) == 0, "clock_gettime");
    return time.tv_sec * 1000.0 + time.tv_nsec / 1e6;
}

static uint32_t read_u32(FILE *file) {
    uint8_t bytes[4];
    require(fread(bytes, 1, sizeof(bytes), file) == sizeof(bytes), "read token contract");
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static void print_tokens(const char *phase, int run, const llama_token *tokens, int count) {
    printf("LINUX_MATCHED_QWEN_TOKENS phase=%s run=%d ids=[", phase, run);
    for (int i = 0; i < count; ++i) {
        printf("%s%d", i ? "," : "", tokens[i]);
    }
    puts("]");
}

static FILE *model_in_ram(const char *path) {
    FILE *source = fopen(path, "rb");
    require(source != NULL, "open model");
    int fd = memfd_create("matched-qwen", MFD_CLOEXEC);
    require(fd >= 0, "memfd_create");
    FILE *target = fdopen(fd, "w+b");
    require(target != NULL, "fdopen model");
    size_t capacity = 1024 * 1024;
    void *buffer = malloc(capacity);
    require(buffer != NULL, "model read buffer");
    size_t count;
    while ((count = fread(buffer, 1, capacity, source)) != 0) {
        require(fwrite(buffer, 1, count, target) == count, "write RAM model");
    }
    require(!ferror(source) && fclose(source) == 0 && fflush(target) == 0, "finish model read");
    rewind(target);
    free(buffer);
    return target;
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

int main(int argc, char **argv) {
    require(argc == 4, "usage: linux-qwen-bench MODEL TOKENS auto|off");
    bool flash_auto = strcmp(argv[3], "auto") == 0;
    require(flash_auto || strcmp(argv[3], "off") == 0, "Flash Attention setting");
    FILE *contract = fopen(argv[2], "rb");
    require(contract != NULL, "open token contract");
    char magic[8];
    require(fread(magic, 1, 8, contract) == 8 && memcmp(magic, "QWBENCH\0", 8) == 0, "contract magic");
    require(read_u32(contract) == 2, "contract version");
    uint32_t count = read_u32(contract);
    require(count > 0 && count <= BATCH && read_u32(contract) == DECODE, "contract lengths");
    uint32_t capacity = read_u32(contract);
    require(capacity == 256 && count + DECODE <= capacity, "matched context capacity");
    llama_token prompt[BATCH], expected[DECODE];
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t token = read_u32(contract);
        require(token < 248320, "prompt token range");
        prompt[i] = (llama_token)token;
    }
    for (int i = 0; i < DECODE; ++i) {
        uint32_t token = read_u32(contract);
        require(token < 248320, "decode token range");
        expected[i] = (llama_token)token;
    }
    require(fgetc(contract) == EOF && !ferror(contract) && fclose(contract) == 0, "contract end");
    print_tokens("prompt", -1, prompt, (int)count);
    print_tokens("decode_input", -1, expected, DECODE);

    ggml_backend_load_all_from_path("/bin");
    llama_backend_init();
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    require(device != NULL, "GPU backend required");
    require(strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") == 0, "CUDA backend required");
    printf("LINUX_MATCHED_QWEN_BACKEND name=%s description=%s\n", ggml_backend_dev_name(device), ggml_backend_dev_description(device));
    ggml_backend_dev_t devices[] = {device, NULL};
    double start = now_ms();
    FILE *ram = model_in_ram(argv[1]);
    double file_ms = now_ms() - start;
    struct llama_model_params model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.n_gpu_layers = -1;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    start = now_ms();
    struct llama_model *model = llama_model_load_from_file_ptr(ram, model_params);
    require(model != NULL, "load model from RAM");
    double load_ms = now_ms() - start;
    require(llama_model_n_layer(model) == 24 && llama_model_n_embd(model) == 1024 &&
            llama_vocab_n_tokens(llama_model_get_vocab(model)) == 248320, "model dimensions");
    require(fclose(ram) == 0, "close RAM model");

    // Compare independent sessions and a reusable context with cleared state.
    // Both feed the exact D3OS token trace, even if a different FP32 reduction
    // order changes greedy choices. Report those differences explicitly.
    for (int reuse = 0; reuse <= 1; ++reuse) {
        struct llama_context *context = NULL;
        struct llama_sampler *sampler = NULL;
        struct llama_sampler_seq_config sampler_config;
        for (int run = 0; run < 3; ++run) {
            start = now_ms();
            if (context == NULL) {
                sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
                require(sampler != NULL, "sampler chain");
                llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
                sampler_config = (struct llama_sampler_seq_config){.seq_id = 0, .sampler = sampler};
                struct llama_context_params params = llama_context_default_params();
                params.n_ctx = capacity;
                params.n_batch = BATCH;
                params.n_ubatch = BATCH;
                params.n_seq_max = 1;
                params.n_outputs_max = 1;
                params.n_outputs_max_per_seq = 1;
                params.n_threads = 4;
                params.n_threads_batch = 4;
                params.type_k = GGML_TYPE_F32;
                params.type_v = GGML_TYPE_F32;
                params.flash_attn_type = flash_auto ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
                params.offload_kqv = true;
                params.op_offload = true;
                params.no_perf = false;
                params.samplers = &sampler_config;
                params.n_samplers = 1;
                context = llama_init_from_model(model, params);
                require(context != NULL, "create context");
            } else {
                llama_memory_clear(llama_get_memory(context), true);
                llama_sampler_reset(sampler);
                llama_synchronize(context);
            }
            double state_ms = now_ms() - start;
            struct llama_batch batch = llama_batch_init(BATCH, 0, 1);
            llama_perf_context_reset(context);
            start = now_ms();
            evaluate(context, &batch, prompt, (int)count, 0);
            double prefill_ms = now_ms() - start;
            llama_token sampled[DECODE];
            uint64_t input_checksum = 0, sampled_checksum = 0;
            int mismatches = 0;
            start = now_ms();
            for (int i = 0; i < DECODE; ++i) {
                sampled[i] = llama_get_sampled_token_ith(context, -1);
                require(sampled[i] != LLAMA_TOKEN_NULL, "backend greedy sampling required");
                llama_sampler_accept(sampler, sampled[i]);
                mismatches += sampled[i] != expected[i];
                input_checksum = input_checksum * 31 + (uint32_t)expected[i];
                sampled_checksum = sampled_checksum * 31 + (uint32_t)sampled[i];
                evaluate(context, &batch, &expected[i], 1, (int)count + i);
            }
            double decode_ms = now_ms() - start;
            struct llama_perf_context_data perf = llama_perf_context(context);
            require(perf.n_p_eval == (int)count && perf.n_eval == DECODE, "exact evaluated token counts");
            printf("LINUX_MATCHED_QWEN_BENCH mode=%s run=%d warm=%s file_ms=%.3f load_ms=%.3f state_ms=%.3f "
                   "prompt_tokens=%u prefill_ms=%.3f prefill_tps=%.3f decode_tokens=%d decode_ms=%.3f decode_tps=%.3f "
                   "input_checksum=%" PRIu64 " sampled_checksum=%" PRIu64 " selection_mismatches=%d "
                   "cache=f32 batch=%d ubatch=%d n_ctx=%u sampler=backend_greedy flash_attention_requested=%s\n",
                   reuse ? "reset_context" : "new_context", run, run ? "true" : "false", file_ms, load_ms, state_ms,
                   count, prefill_ms, count * 1000.0 / prefill_ms, DECODE, decode_ms, DECODE * 1000.0 / decode_ms,
                   input_checksum, sampled_checksum, mismatches, BATCH, BATCH, llama_n_ctx(context), argv[3]);
            print_tokens("sampled", run, sampled, DECODE);
            llama_batch_free(batch);
            if (!reuse || run == 2) {
                llama_free(context);
                llama_sampler_free(sampler);
                context = NULL;
            }
        }
    }
    llama_model_free(model);
    llama_backend_free();
    puts("LINUX_MATCHED_QWEN_OK");
    return EXIT_SUCCESS;
}
