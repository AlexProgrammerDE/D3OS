#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

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

static int check_reservation(void) {
    const size_t page = 4096;
    const size_t size = 1UL << 30;
    unsigned char *memory = mmap(NULL, size, PROT_NONE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        return fail(5);
    }
    for (size_t offset = 0; offset < size; offset += size / 8) {
        if (mprotect(memory + offset, page, PROT_READ | PROT_WRITE)) {
            return fail(6);
        }
        for (size_t index = 0; index < page; ++index) {
            if (memory[offset + index] != 0) {
                return fail(7);
            }
            memory[offset + index] = (unsigned char)(index + offset / (size / 8));
        }
        if (mprotect(memory + offset, page, PROT_NONE) ||
            mprotect(memory + offset, page, PROT_READ)) {
            return fail(8);
        }
        for (size_t index = 0; index < page; ++index) {
            if (memory[offset + index] != (unsigned char)(index + offset / (size / 8))) {
                return fail(9);
            }
        }
    }
    int fd = open("/tmp/reservation", O_CREAT | O_TRUNC | O_RDWR, 0600);
    const unsigned char expected[] = {31, 41, 59, 26};
    if (fd < 0 || write(fd, expected, sizeof(expected)) != (ssize_t)sizeof(expected)) {
        return fail(10);
    }
    unsigned char *file = mmap(memory + page, page, PROT_NONE,
                               MAP_PRIVATE | MAP_FIXED, fd, 0);
    if (file != memory + page) {
        return fail(11);
    }
    errno = 0;
    if (write(fd, file, 1) != -1 || errno != EFAULT) {
        return fail(12);
    }
    errno = 0;
    if (write(fd, memory + page * 2, 1) != -1 || errno != EFAULT) {
        return fail(13);
    }
    if (lseek(fd, 0, SEEK_SET) != 0) {
        return fail(14);
    }
    errno = 0;
    if (read(fd, memory, 1) != -1 || errno != EFAULT) {
        return fail(15);
    }
    if (mprotect(file, page, PROT_READ)) {
        return fail(16);
    }
    for (size_t index = 0; index < sizeof(expected); ++index) {
        if (file[index] != expected[index]) {
            return fail(17);
        }
    }
    close(fd);
    unlink("/tmp/reservation");

    // Replace both an untouched reservation and existing physical backing.
    unsigned char *replacement = mmap(memory, page * 3, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (replacement != memory) {
        return fail(18);
    }
    for (size_t index = 0; index < page * 3; ++index) {
        if (replacement[index] != 0) {
            return fail(19);
        }
    }
    if (mprotect(memory + size / 8, page, PROT_NONE) ||
        munmap(memory + page, page) || munmap(memory, size)) {
        return fail(20);
    }
    errno = 0;
    if (mprotect(memory, page, PROT_READ) != -1 || errno != ENOMEM) {
        return fail(21);
    }
    fprintf(stderr, "LINUX_RESERVATION_OK reserved_bytes=%zu touched_pages=8 file_prot_none=preserved replacement=zeroed\n", size);
    return 0;
}

int main(void) {
    int reservation = check_reservation();
    if (reservation) {
        return reservation;
    }
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
