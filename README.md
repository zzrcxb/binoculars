# Binoculars
Binoculars is a new stateless-indirect side-channel attack that exploits
contention between page walker loads and normal data stores.
Binoculars has strong contention effects (low noise)
and leaks a wide range of virtual address bits.
It also requires no shared memory.

## Why is it Called Binoculars?
This side-channel is just like viewing through a binoculars from either end.
From one end, you can make things seem "closer" (extracting store offsets);
from the other end, you can make things seem "farther" (extracting virtual page numbers).

## Software Dependences
### Kernel Module
These PoCs depend on [PTEditor](https://github.com/misc0110/PTEditor),
which is a kernel module that helps page-table manipulation in user space.
Therefore, a kernel module build environment is required.

### Python
Our scripts require `Python >= 3.6` and Python libraries that are listed in `requirements.txt`.
You can install those libraries by executing
```bash
pip3 install -r requirements.txt --user
```

### Intel Pin
Our ECDSA PoC requires Intel Pin to trace its memory accesses.
We use `pin-3.21` but a lower version might also work.
Please make sure that the global varaible `$PIN_ROOT`
points to the root of the `Pin` directory.

### Other Dependences
Binoculars also requires `CMake`, `gcc`, and `ninja-build` or `GNU make`.
Your system likely has them already.

## Build
### Binoculars
Under the project's root directory,
execute:
```bash
mkdir build && cd build && cmake ..
```
CMake will detect the family of the CPU's microarchitecture
and whether KPTI is enabled.
Please check the CMake output to confirm the detection results are correct.
We currently support `Haswell`, `Skylake`, and `Cascade Lake-X`.

After that, under the build directory, execute command:
`make` or `ninja` depending on your build system.

The `build` directory should contain the follow binaries after building:
- `bino`: demonstrates the contention effects and our attack primitives;
- `kaslr`: PoC that breaks kernel ASLR;
- `mont-bino`: PoC that breaks ECDSA;
- `mont-blank`, `mont-memjam-dep`, `mont-memjam-para`:
PoCs that try to break ECDSA with other approaches for comparison purpose
(not part of Binoculars, detailed in the paper, you can ignore them for now).

### PTEditor
Under the `PTEditor` directory, execute
```bash
make
```
to build the kernel module.
Then load the module by executing:
```bash
sudo insmod module/pteditor.ko
```

### Memory Tracing Pintool
Under the `pin` directory, execute
```bash
make
```

## Demonstration of Binoculars
The binary `bino` demonstrates the contention effects
of Binoculars and our attack primitives,
including:
- Contention effects
- Store->Load Channel
- Load->Store Channel
- Load->Store Channel Port-Contention Variant

### Using scripts
The Jupyter notebook `demo.ipynb` contains the code to produce for
Figure 2, 4, 6, and 8 in the paper.

### Using the binary
You can also run the binary directly.
The binary `bino` takes the following arguments:
```bash
./bino <action> <core 1> <core 2>
```
where `<core 1>` and `<core 2>` are CPU IDs
of two logical cores that share the same physical core;
the action can be:
- `contention_effect`: Contention effects
- `store_offset`: Store->Load Channel
- `vpn_latency`: Load->Store Channel
- `vpn_contention`: Load->Store Channel Port-Contention Variant

For the `contention_effect` action,
it further takes a fifth argument `0 or 1`
to control whether there is 4K-aliasing.

## PoCs
### Break ECDSA
#### Find Target Offsets
Under the root of Binoculars, execute:
```bash
<path to Pin binary> -t pin/obj-intel64/memtrace.so -t config/tracer.tinfo -- ossl/openssl dgst -ecdsa-with-SHA1 -sign ossl/private.pem ossl/data > /dev/null
```
will trace stores to addresses that might be suitable for monitoring.
After the tracing, it should generate a file named `data.trace`,
which contains the store's PC offset within the text segment,
and the PL4, PL3, PL2, and PL1 indexes, and offset of the address
that the store writes to, like:
```
<PC offset> W PL4_index PL3_index PL2_index PL1_index offset
```

Execute command:
```bash
cut -d" " data.trace -f7 | sort | uniq -c | sort -n
```
to sort offsets by their occurences in an ascending order.
You can try an offset that is frequent
and use it to replace the value of `CENTRAL_ZERO` in `src/mont.h`.
Note that because we not only monitor the offset at `CENTRAL_ZERO`,
but also offsets around it,
it is better to pick an offset that is "surrounded" by neighboring offsets.
For example, if the output is:
```
...
 9690 0x8c8
 9690 0x8d8
10830 0x8b0
10830 0x8c0
10830 0x8d0
10830 0x8e0
```
offset `0x8d0` is a better choice than `0x8e0`.


### Measure End-to-End Accuracy
Inside Jupyter notebook `poc.ipynb`,
the second code block demonstrates extracting the nonce `k`
with oracle boundaries.
It uses 30 runs for training and 10 different runs to measure the accuracy.
If one offset does not perform well, please try different offsets.

## Break Kernel ASLR
The third and fourth code blocks in `poc.ipynb`
demonstrates the kernel ASLR break.
Please refer to theirs comments for more details.

## Expected Results
Expected results are under `expected/<uarch-name>`.

Hardware config:
```
Haswell-EP: Xeon E3-1246 v3
Skylake-X: i7-7820X
Cascade Lake-X: Xeon W-2245
```

System config:
```
OS: Ubuntu 20.04.3 LTS (Focal Fossa)
Kernel: 5.4.0-89-generic
Boot options: default speculative execution attack mitigations
Compiler: GCC 9.3.0
```
