#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>

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
#define UNUSED __attribute__((unused))

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

static inline ALWAYS_INLINE u64 _rdtsc_google_begin(void) {
    u64 t;
    asm volatile("lfence\n\t"
                 "rdtsc\n\t"
                 "shl $32, %%rdx\n\t"
                 "or %%rdx, %0\n\t"
                 "lfence"
                 : "=a"(t)
                 :
                 // "memory" avoids reordering. rdx = TSC >> 32.
                 // "cc" = flags modified by SHL.
                 : "rdx", "memory", "cc");
    return t;
}

static inline ALWAYS_INLINE u64 _rdtscp_google_end(void) {
    u64 t;
    asm volatile("rdtscp\n\t"
                 "shl $32, %%rdx\n\t"
                 "or %%rdx, %0\n\t"
                 "lfence"
                 : "=a"(t)
                 :
                 // "memory" avoids reordering. rcx = TSC_AUX. rdx = TSC >> 32.
                 // "cc" = flags modified by SHL.
                 : "rcx", "rdx", "memory", "cc");
    return t;
}

#define _timer_start _rdtsc_google_begin
#define _timer_end   _rdtscp_google_end

// page table related functions
// assuming 4KB page and 4-level PT
#define PAGE_SHIFT (12u)
#define PAGE_SIZE (1u << PAGE_SHIFT)
#define INDEX_WIDTH (9u)
#define INDEX_COUNT (1u << INDEX_WIDTH)
#define INDEX_MASK (INDEX_COUNT - 1)

static inline ALWAYS_INLINE u64 addr_crafter(u64 pl4, u64 pl3, u64 pl2, u64 pl1) {
    u64 page = (pl4 << 27) + (pl3 << 18) + (pl2 << 9) + pl1;
    return page << 12;
}

// get index for a given pointer and a level of page table
// PGD (level 4) -> PUD (level 3) -> PMD (level 2) -> PTE (level 1)
static inline ALWAYS_INLINE u16 get_PL_index(void *ptr, u8 level) {
    u64 page = ((uintptr_t)ptr >> PAGE_SHIFT);
    return (page >> ((level - 1) * 9)) & 0x1ff;
}

// mmap related
static u8 *mmap_private(void *addr, u64 size) {
    u8 *ptr = mmap(addr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED)
        return NULL;
    return ptr;
}

static u8 *mmap_private_init(void *addr, u64 size, u8 init) {
    u8 *ptr = mmap_private(addr, size);
    if (ptr) {
        memset(ptr, init, size);
    }
    return ptr;
}

static u8 *mmap_shared(void *addr, u64 size) {
    u8 *ptr = mmap(addr, size, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED)
        return NULL;
    return ptr;
}

static u8 *mmap_shared_init(void *addr, u64 size, u8 init) {
    u8 *ptr = mmap_shared(addr, size);
    if (ptr) {
        memset(ptr, init, size);
    }
    return ptr;
}

// affinity
static bool set_affinity_priority(uint core, int prio) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    bool ret1 = sched_setaffinity(getpid(), sizeof(set), &set) != -1;
    bool ret2 = setpriority(PRIO_PROCESS, 0, prio) != -1;
    return ret1 && ret2;
}

// error message related
#define _error(...)                                                            \
    do {                                                                       \
        fprintf(stderr, __VA_ARGS__);                                          \
    } while (0)
