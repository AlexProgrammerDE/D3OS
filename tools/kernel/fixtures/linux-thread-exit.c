#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

enum { THREADS = 8, ROUNDS = 8 };

struct context {
    atomic_int cleanups;
    atomic_int simd_failed;
    uint64_t id;
};

static int check_simd_context(uint64_t id) {
    _Alignas(16) uint64_t pattern[4] = {
        0x1122334455667788ULL ^ id,
        0x8877665544332211ULL ^ id,
        0x55aa55aa55aa55aaULL ^ id,
        0xaa55aa55aa55aa55ULL ^ id,
    };
    int failed;
    // Linux syscalls preserve vector registers. Keep distinct values live
    // while yielding to other workers, including across CPU migration.
    __asm__ volatile (
        "movdqu 0(%[pattern]), %%xmm0\n"
        "movdqu 16(%[pattern]), %%xmm1\n"
        "movl $128, %%r8d\n"
        "1: movl $24, %%eax\n"
        "syscall\n"
        "movdqa %%xmm0, %%xmm2\n"
        "pcmpeqb 0(%[pattern]), %%xmm2\n"
        "pmovmskb %%xmm2, %%eax\n"
        "cmpl $65535, %%eax\n"
        "jne 2f\n"
        "movdqa %%xmm1, %%xmm2\n"
        "pcmpeqb 16(%[pattern]), %%xmm2\n"
        "pmovmskb %%xmm2, %%eax\n"
        "cmpl $65535, %%eax\n"
        "jne 2f\n"
        "decl %%r8d\n"
        "jnz 1b\n"
        "xorl %%eax, %%eax\n"
        "jmp 3f\n"
        "2: movl $1, %%eax\n"
        "3:\n"
        : "=&a" (failed)
        : [pattern] "r" (pattern)
        : "rcx", "r11", "r8", "xmm0", "xmm1", "xmm2", "cc", "memory"
    );
    return failed;
}

static void cleanup(void *argument) {
    struct context *context = argument;
    atomic_fetch_add(&context->cleanups, 1);
}

static void *worker(void *argument) {
    struct context *context = argument;
    atomic_store(&context->simd_failed, check_simd_context(context->id));
    pthread_cleanup_push(cleanup, argument);
    pthread_exit(argument);
    pthread_cleanup_pop(0);
    return NULL;
}

static int fail(int code) {
    fprintf(stderr, "LINUX_THREAD_EXIT_FAIL code=%d\n", code);
    return code;
}

int main(void) {
    for (int round = 0; round < ROUNDS; ++round) {
        pthread_t threads[THREADS];
        struct context contexts[THREADS];
        for (int index = 0; index < THREADS; ++index) {
            atomic_init(&contexts[index].cleanups, 0);
            atomic_init(&contexts[index].simd_failed, 0);
            contexts[index].id = (uint64_t)(round * THREADS + index + 1);
            if (pthread_create(&threads[index], NULL, worker, &contexts[index])) {
                return fail(1);
            }
        }
        for (int index = 0; index < THREADS; ++index) {
            void *result = NULL;
            if (pthread_join(threads[index], &result)) {
                return fail(2);
            }
            if (result != &contexts[index] || atomic_load(&contexts[index].cleanups) != 1) {
                return fail(3);
            }
            if (atomic_load(&contexts[index].simd_failed)) {
                return fail(4);
            }
        }
    }
    fprintf(stderr, "LINUX_THREAD_EXIT_OK threads=%d rounds=%d\n", THREADS, ROUNDS);
    return 0;
}
