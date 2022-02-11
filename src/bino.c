#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fcntl.h>
#include <signal.h>
#include <sched.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>

#include "ptedit_header.h"
#include "utils.h"

static int CORE1, CORE2;

typedef struct pctrl_t {
    volatile bool start, ready;
    volatile u64 ts, *lats, idx;
} pctrl_t;

#define VICTIM_STORE_OFFSET (0x528u)
#define STORE_PROBE_BASE_ADDR (0x4b2592c00000ull)
static int __attribute__((noinline)) store_offset() {
    int ret = 0;
    const u32 MEASURES = 100;

    volatile pctrl_t *ctrl = (pctrl_t *)mmap_shared(NULL, sizeof(pctrl_t));
    ctrl->ready = false;
    ctrl->start = false;

    int pid = fork();
    if (pid == 0) {
        set_affinity_priority(CORE2, 0);
        u8 *page = mmap_private(NULL, PAGE_SIZE);
        if (!page) return 1;
        ctrl->ready = true;
        while (!ctrl->start);

        while (true) {
            _mwrite(&page[VICTIM_STORE_OFFSET], 0xff);
        }
    } else if (pid < 0) {
        _error("Failed to fork!\n");
    }

    set_affinity_priority(CORE1, 0);
    u8 *base_page = mmap_private((void *)STORE_PROBE_BASE_ADDR, PAGE_SIZE * INDEX_COUNT);
    u64 *results = malloc(sizeof(u64) * INDEX_COUNT);
    if (!base_page || !results) {
        _error("Failed to allocate probe pages or results\n");
        ret = 1;
        goto err;
    }
    memset(base_page, 0, PAGE_SIZE * INDEX_COUNT);
    memset(results, 0, sizeof(u64) * INDEX_COUNT);

    while (!ctrl->ready);
    ctrl->start = true;
    for (unsigned cnt = 0; cnt < INDEX_COUNT; cnt++) {
        u32 offset = cnt % INDEX_COUNT;
        u8 *addr = base_page + offset * PAGE_SIZE;
        for (u32 c = 0; c < MEASURES; c++) {
            _maccess(addr);
            ptedit_invalidate_tlb(addr);

            u64 t_start = _rdtsc_google_begin();
            _maccess(addr);
            u64 t_end = _rdtscp_google_end();
            results[offset] += (t_end - t_start);
        }
    }

    for (u32 offset = 0; offset < INDEX_COUNT; offset++) {
        printf("%u\t%lu\n", offset, results[offset] / MEASURES);
    }

err:
    kill(pid, SIGKILL);
    return ret;
}


#define VICTIM_LOAD_ADDR (0x5d21ca821000ull)
static int __attribute__((noinline)) vpn_latency() {
    int ret = 0;
    const u32 MEASURES = 100;

    volatile pctrl_t *ctrl = (pctrl_t *)mmap_shared(NULL, sizeof(pctrl_t));
    ctrl->ready = false;
    ctrl->start = false;
    ctrl->lats = (u64 *)mmap_shared(NULL, sizeof(u64) * INDEX_COUNT);
    if (!ctrl->lats) {
        return 1;
    }
    memset((void *)ctrl->lats, 0, sizeof(u64) * INDEX_COUNT);

    int pid = fork();
    if (pid == 0) {
        set_affinity_priority(CORE2, 0);
        u8 *page = mmap_private((void *)VICTIM_LOAD_ADDR, PAGE_SIZE);
        if (!page) return 1;
        memset(page, 0, PAGE_SIZE);
        ctrl->ready = true;

        while (true) {
            while (!ctrl->start);
            _maccess(page);
            ptedit_invalidate_tlb(page);
            u64 t_start = _rdtsc_google_begin();
            _maccess(page);
            u64 t_end = _rdtscp_google_end();
            ctrl->lats[ctrl->idx] += (t_end - t_start);
            ctrl->start = false;
        }
    } else if (pid < 0) {
        _error("Failed to fork!\n");
    }

    set_affinity_priority(CORE1, 0);
    u8 *page = mmap_private(NULL, PAGE_SIZE);
    if (!page) {
        _error("Failed to allocate the probe page\n");
        ret = 1;
        goto err;
    }
    memset(page, 0, PAGE_SIZE);

    while (!ctrl->ready);
    for (unsigned cnt = 0; cnt < INDEX_COUNT * MEASURES; cnt++) {
        u32 idx = (cnt % INDEX_COUNT);
        u32 offset = (cnt % INDEX_COUNT) << 3;
        ctrl->idx = idx;
        ctrl->start = true;
        while (ctrl->start) {
            _mwrite(&page[offset], 0xff);
            _mwrite(&page[offset], 0xff);
            _mwrite(&page[offset], 0xff);
        }
    }

    for (u32 offset = 0; offset < INDEX_COUNT; offset++) {
        printf("%u\t%lu\n", offset, ctrl->lats[offset] / MEASURES);
    }

err:
    kill(pid, SIGKILL);
    return ret;
}

static int __attribute__((noinline)) vpn_contention() {
    int ret = 0;
    const u32 MEASURES = 100;

    volatile pctrl_t *ctrl = (pctrl_t *)mmap_shared(NULL, sizeof(pctrl_t));
    ctrl->ready = false;
    ctrl->start = false;

    int pid = fork();
    if (pid == 0) {
        set_affinity_priority(CORE2, 0);
        u8 *page = mmap_private((void *)VICTIM_LOAD_ADDR, PAGE_SIZE);
        if (!page) return 1;
        memset(page, 0, PAGE_SIZE);
        ctrl->ready = true;

        while (!ctrl->start);
        while (true) {
            ptedit_invalidate_tlb(page);
            _maccess(page);
        }
    } else if (pid < 0) {
        _error("Failed to fork!\n");
    }

    set_affinity_priority(CORE1, 0);
    u8 *page = mmap_private(NULL, PAGE_SIZE);
    u64 *results = malloc(sizeof(u64) * INDEX_COUNT);
    if (!page || !results) {
        _error("Failed to allocate the probe page\n");
        ret = 1;
        goto err;
    }
    memset(page, 0, PAGE_SIZE);
    memset(results, 0, sizeof(u64) * INDEX_COUNT);

    while (!ctrl->ready);
    ctrl->start = true;
    for (unsigned cnt = 0; cnt < INDEX_COUNT * MEASURES; cnt++) {
        u32 idx = (cnt % INDEX_COUNT);
        u32 offset = (cnt % INDEX_COUNT) << 3;
        u64 t_start = _rdtsc_google_begin();
        for (u32 i = 0; i < 10000; i++) {
            _mwrite(&page[offset], 0xff);
        }
        u64 t_end = _rdtscp_google_end();
        results[idx] += (t_end - t_start);
    }

    for (u32 offset = 0; offset < INDEX_COUNT; offset++) {
        printf("%u\t%lu\n", offset, results[offset] / MEASURES);
    }

err:
    kill(pid, SIGKILL);
    return ret;
}


static int __attribute__((noinline)) contention_effect(bool alias) {
    int ret = 0;
    const u32 MEASURES = 100;
    u32 offset = alias ? 0x21ul << 3 : 0x528;

    volatile pctrl_t *ctrl = (pctrl_t *)mmap_shared(NULL, sizeof(pctrl_t));
    ctrl->ready = false;
    ctrl->start = false;

    int pid = fork();
    if (pid == 0) {
        set_affinity_priority(CORE2, 0);
        u8 *page = mmap_private(NULL, PAGE_SIZE);
        if (!page) return 1;
        memset(page, 0, PAGE_SIZE);
        ctrl->ready = true;

        while (!ctrl->start);
        while (true) {
            _mwrite(&page[offset], 0xff);
        }
    } else if (pid < 0) {
        _error("Failed to fork!\n");
    }

    set_affinity_priority(CORE1, 0);
    u8 *page = mmap_private((void *)VICTIM_LOAD_ADDR, PAGE_SIZE);
    u64 *results = malloc(sizeof(u64) * MEASURES);
    if (!page || !results) {
        _error("Failed to allocate the probe page\n");
        ret = 1;
        goto err;
    }
    memset(page, 0, PAGE_SIZE);
    memset(results, 0, sizeof(u64) * MEASURES);

    while (!ctrl->ready);
    ctrl->start = true;
    for (unsigned cnt = 0; cnt < MEASURES; cnt++) {
        _maccess(page);
        ptedit_invalidate_tlb(page);
        u64 t_start = _rdtsc_google_begin();
        _maccess(page);
        u64 t_end = _rdtscp_google_end();
        results[cnt] = t_end - t_start;
    }

    for (u32 i = 0; i < MEASURES; i++) {
        printf("%lu\n", results[i]);
    }

err:
    kill(pid, SIGKILL);
    return ret;
}


static void segv_handler(int signum) {
    sigset_t sigs;
    sigemptyset(&sigs);
    sigaddset(&sigs, signum);
    sigprocmask(SIG_UNBLOCK, &sigs, NULL);

    ptedit_cleanup();
    fprintf(stderr, "SEGFAULT!\n");
    exit(128 + SIGSEGV);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        return -1;
    }

    if (ptedit_init()) {
        return 1;
    }

    if (signal(SIGSEGV, segv_handler) == SIG_ERR) {
        fprintf(stderr, "Failed to register segfault handler\n");
        return 2;
    }

    CORE1 = atoi(argv[2]);
    CORE2 = atoi(argv[3]);

    int ret;
    if (strcmp(argv[1], "store_offset") == 0) {
        ret = store_offset();
    } else if (strcmp(argv[1], "vpn_latency") == 0) {
        ret = vpn_latency();
    } else if (strcmp(argv[1], "vpn_contention") == 0) {
        ret = vpn_contention();
    } else if (strcmp(argv[1], "contention_effect") == 0 && argc == 5) {
        ret = contention_effect(atoi(argv[4]));
    } else {
        fprintf(stderr, "Unknown option %s or insufficient arguments\n", argv[1]);
        ret = 2;
    }

    ptedit_cleanup();
    return ret;
}
