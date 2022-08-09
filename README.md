# <img src="https://verishare.org/05a3cb8261bf3af193c147c178c5560e78953f24/preview/" alt="Binoculars Logo" width="40"/> Binoculars Microarchitectural Side Channel Attack
Binoculars is the first *stateless-indirect* side-channel attack that exploits
*contention* between page walker loads and normal data stores.
There are two important characteristics that make Binoculars special:
- **Easy to observe, high signal-to-noise ratio**.
An attacker can create *significant* delays in thread execution time
(up to 16,000~20,000 cycles!) stemming from a single dynamic instruction.
This magnitude of delay dwarfs the one created
by any other microarchitectural side channel by at least two orders of magnitude;
- **Leaks a wide range virtual address bits in victim memory operations.**
The execution delays are address dependent.
With our attack primitives,
Binoculars can leak low-order bits (e.g., bits 11-3) and
high-order bits (e.g., virtual page number).
Using these two primitives,
we demonstrated two end-to-end attacks on real machines,
which include extracting the nonce $k$ in ECDSA with *a single victim run*
and fully breaking kernel ASLR (KASLR).

## What's Happening?
To perform a memory access, the hardware needs to first translate
its (virtual) address to the physical address.
The mapping from virtual addresses to physical addresses are stored
in a radix tree called *page table* (usually has four levels).
A page table lookup is called a *page walk* and
it is usually done by a hardware unit called the *page walker*.

During a page walk, the page walker performs a pointer-chasing from the root of the page table,
by issuing **multiple** *page walker loads* that read from each level of page table.
Similar to normal memory loads, these page walker loads go through cache hierarchy
and are **subject to address-dependent contention**.

We found that a page walker load from one hyperthread (HT) can experience
extreme contention (e.g., up to 20,000 cycles) caused by normal data stores from the sibling hyperthread.
The extreme contention happens if the page walker load and the stores are "4K-aliasing"
(or more precisely, they share the same VA bits 11-3).





## Research Paper
The Binoculars paper will appear in *USENIX Security'22* with the title
*"Binoculars: Contention-Based Side-Channel Attacks Exploiting the Page Walker"*.
You can find a preprint copy [here](https://verishare.org/1ef5fb8c8ab6746690ad440adc543addd9b47cd7/preview/)
and the BibTeX citation below:

```bibtex
@inproceedings{binoculars-attack,
    author = {Zirui Neil Zhao and Adam Morrison and Christopher W. Fletcher and Josep Torrellas},
    title = {Binoculars: {Contention-Based} {Side-Channel} Attacks Exploiting the Page Walker},
    booktitle = {31st USENIX Security Symposium (USENIX Security 22)},
    year = {2022},
}
```

## Affected Processors
We believe the root cause of the Binoculars
attack is related to Intel’s optimization of page walker loads.
Therefore, Binoculars is likely exclusive to Intel processors.
We verified Binoculars exist on the following Intel processors:
- Haswell-EP (Xeon E3-1246 v3)
- Skylake-X (i7-7820X)
- Cascade Lake-X (Xeon W-2245)

If you find Binoculars also exist on other processors (Intel or non-Intel),
you are welcome to report them by sending us a pull request
(please include the processor model, OS distribution, and kernel version).

## Mitigations


## Why is it Called Binoculars?
This side-channel is just like viewing through a binoculars from either end.
From one end, you can make things seem "closer" (extracting store offsets);
from the other end, you can make things seem "farther" (extracting virtual page numbers).
