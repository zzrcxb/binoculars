# <img src="https://verishare.org/05a3cb8261bf3af193c147c178c5560e78953f24/preview/" alt="Binoculars Logo" width="40"/> Binoculars Microarchitectural Side Channel Attack
Binoculars is the first *stateless-indirect* side-channel attack
(stateless means that it does not rely on persistent state changes like cache footprint,
indirect means that it is not a *direct* result of victim instructions).
The Binoculars attack is cross-hyperthread and it exploits *temproary* resource contention (and sometimes starvation)
between page walker loads from one thread and normal data stores from the sibling thread.
There are two important characteristics that make Binoculars special:
- **Easy to observe, high signal-to-noise ratio**.
An attacker can create *significant* delays in thread execution time
(up to 16,000~20,000 cycles!), all stemming from a single dynamic instruction.
This magnitude of delay dwarfs the one created
by any other microarchitectural side channel by at least two orders of magnitude;
- **Leaks a wide range virtual address bits in victim memory operations.**
The contention is address dependent.
We built two attack primitives that can monitor victim memory access patterns.
The first primitive, Store->Load channel, leaks low-order address bits (e.g., bits 11-3) of victim stores.
And the second primitive, Load->Store channel, leaks high-order virtual address bits (e.g., virtual page number) of all victim TLB-missing accesses.

Using these two primitives,
we demonstrated two end-to-end attacks on real machines,
which include extracting the nonce $k$ in ECDSA with *a single victim run*
and fully breaking kernel ASLR (KASLR).

## Try the Binoculars Attack
Please refer to this [document](EXP.md)
if you want to try the Binoculars attack.

## What's Happening?
To perform a memory access, the hardware needs to first translate
the virtual address to the physical address by looking up the TLB.
If the query results in a TLB miss,
a hardware unit, named *page walker*,
will initiate a *page walk* to read the virtual-to-physical mapping from the page table in the memory.

On x86-64, during a page walk, the page walker performs a pointer-chasing from the root of the radix page table,
by issuing **multiple** *page walker loads* that read from each level of page table.
Similar to normal memory loads, page walker loads also go through cache hierarchy
and are **subject to address-dependent contention**.

We found that a page walker load from a *reader* hyperthread (HT) can experience
extreme contention (e.g., being delayed for up to 20,000 cycles)
because of normal data stores from the sibling *writer* hyperthread.
The extreme contention happens if the page walker load and the stores are "4K-aliasing"
(or more precisely, they share the same VA bits 11-3).
By changing whether the attacker plays the role of the reader or the writer thread,
we can leak a wide range of virtual address bits of victim accesses.

We believe this surprising behavior is a result of an optimization in Intel processors.
The optimization is to issues page walker loads as "stuffed loads",
which bypass the instruction scheduler to avoid scheduling latency.
But an unexpected effect of this optimization is that the scheduler can no longer
detect and mediate the conflicts between the page walekr loads and other memory accesses.
As a result, the page walker will experience resource starvation,
until a watchdog alarm is triggered after \~20,000 cycles.
After that, the page walk is aborted and restarted, presumably with a higher priority.

## Research Paper
The Binoculars paper appears in *USENIX Security '22* with the title
*"Binoculars: Contention-Based Side-Channel Attacks Exploiting the Page Walker"*.
You can find a copy of the paper [here](https://www.usenix.org/system/files/sec22-zhao-zirui.pdf)
and the BibTeX citation below:

```bibtex
@inproceedings{binoculars-attack,
    author = {Zirui Neil Zhao and Adam Morrison and Christopher W. Fletcher and Josep Torrellas},
    title = {Binoculars: Contention-Based Side-Channel Attacks Exploiting the Page Walker},
    booktitle = {31st USENIX Security Symposium (USENIX Security 22)},
    year = {2022},
}
```

## Affected Processors
We believe the root cause of the Binoculars
attack is related to an Intel’s optimization of page walker loads.
Therefore, the Binoculars attack is likely exclusive to Intel processors.
We verified that the attack works on the following Intel processors:
- Haswell-EP (Xeon E3-1246 v3)
- Skylake-X (i7-7820X)
- Cascade Lake-X (Xeon W-2245)

If you find the Binoculars attack also works on other processors (Intel or non-Intel),
you are welcome to report them by sending us a pull request
(please include the processor model, OS distribution, and kernel version).

## Software Mitigations
Since the Binoculars is a cross-hyperthread side channel,
the easiest mitigation is to disable the hyperthreading
or only schedule mutually-trusted programs on the same physical core.
Another option is to rewrite sensitive vulnerable programs with data-oblivious programming practices.
Both mitigations may come with non-negligible performance loss.

## Why is it Called Binoculars?
Because this side channel is asymmetric. Depending on which end the attacker is on (i.e., the reader or the writer side),
the channels leaks different virtual address bits (high-order bits or low-order bits).
This behavior reminds us of viewing a (real-world) bincoulars from either end,
which can either bring objects closer or farther.
Hence the name, Binoculars.
