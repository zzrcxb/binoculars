#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

#define ALWAYS_INLINE __attribute__((always_inline))

// assembly related
#define _maccess(P)                                                            \
    do {                                                                       \
        typeof(*(P)) _NO_USE;                                                  \
        __asm__ __volatile__("mov (%1), %0\n" : "=r"(_NO_USE) : "r"(P));       \
    } while (0)

#define _mwrite(P, V)                                                          \
    do {                                                                       \
        __asm__ __volatile__("mov %1, %0\n" : "=m"(*(P)) : "r"(V) : "memory"); \
    } while (0)

static inline void ALWAYS_INLINE _clflush(void const *p) {
    __asm__ __volatile__("clflush 0(%0)\n" : : "c"(p) : "rax");
}

static inline void ALWAYS_INLINE _clflush_v(void volatile *p) {
    __asm__ __volatile__("clflush 0(%0)\n" : : "c"(p) : "rax");
}

static inline void ALWAYS_INLINE _lfence(void) {
    __asm__ __volatile__("lfence\n");
}

static inline void ALWAYS_INLINE _mfence(void) {
    __asm__ __volatile__("mfence\n");
}

static inline ALWAYS_INLINE u64 _rdtscp(u32 *aux) {
    u64 rax, rdx;
    __asm__ __volatile__("rdtscp\n" : "=a"(rax), "=d"(rdx), "=c"(*aux) : :);
    return (rdx << 32) + rax;
}

#define SEL_NOSPEC(MASK, T, F)                                                 \
    (((MASK) & (typeof(MASK))(T)) | (~(MASK) & (typeof(MASK))(F)))

// timing related
#define _TIMEIT(TYPE, P)                                                       \
    do {                                                                       \
        t_start = _rdtscp(&_sig);                                              \
        _maccess((TYPE *)P);                                                   \
        t_end = _rdtscp(&_sig);                                                \
    } while (0);

static uint32_t _time_maccess(void *p, size_t size) {
    unsigned int _sig;
    uint64_t t_start, t_end;

    switch (size) {
    case 1:
        _TIMEIT(uint8_t, p);
        break;
    case 2:
        _TIMEIT(uint16_t, p);
        break;
    case 4:
        _TIMEIT(uint32_t, p);
        break;
    case 8:
        _TIMEIT(uint64_t, p);
        break;
    default:
        assert(false && "Unsupported size");
    }
    return t_end - t_start;
}

static unsigned _get_cache_hit_threshold(void) {
    unsigned const size = 4096;
    unsigned const count = 1000;
    size_t i;
    unsigned threshold;
    uint64_t hit_sum = 0, miss_sum = 0;

    char *data = malloc(size);
    if (!data)
        return 0;
    memset(data, 0, size);

    // warmup
    for (i = 0; i < count; i++) {
        _maccess(data);
    }

    // measure hits
    for (i = 0; i < count; i++) {
        hit_sum += _time_maccess(data, 1);
    }

    for (i = 0; i < count; i++) {
        _clflush(data);
        _mfence();
        miss_sum += _time_maccess(data, 1);
    }

    free(data);
    threshold = (miss_sum + 3 * hit_sum) / (4 * count);
    if (threshold > 150) {
        fprintf(stderr,
                "[WARN] Cache hit threshold is abnormally high at %u cycles, "
                "please check your system settings.\n",
                threshold);
    }
    return threshold;
}
