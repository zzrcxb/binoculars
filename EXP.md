# <img src="https://verishare.org/05a3cb8261bf3af193c147c178c5560e78953f24/preview/" alt="Binoculars Logo" width="40"/> Running the Experiments

You can clone the repo by running:
```bash
git clone --recursive https://github.com/zzrcxb/binoculars.git
```

Before running any experiment, here are the required software and hardware dependences:

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
points to the root of the Intel Pin directory.

### Other Dependences
Binoculars also requires `CMake`, `gcc`, and `ninja-build` or `GNU make`.
Your system likely has them already.

## Hardware Dependence
We tested our PoCs on three different families of microarchitectures: Haswell, Skylake, and Cascade Lake.
Therefore, it is strongly recommended to try the PoCs on a processor that belongs to one of these three families.
You can find the microarchitecture family of your processor by executing:
```bash
gcc -march=native -Q --help=target | grep march | head -n 1 | tr -d '[:space:]' | cut -d"=" -f2 | cut -d"-" -f1
```
If its output matches `haswell.*`, `skylake.*`, or `cascadelake.*`, then your processor is supported by default.


## Build
### Binoculars
Under the project's root directory,
execute:
```bash
mkdir build && cd build && cmake ..
```

After that, under the build directory, execute command:
`make` or `ninja` depending on your build system.

The `build` directory should contain the follow binaries after building:
- `bino`: demonstrates the contention effects and our attack primitives;
- `kaslr`: PoC that breaks kernel ASLR;
- `mont-bino`: PoC that breaks ECDSA;

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

### Pin-Based Memory Tracing Tool
Please make sure that the global variable `$PIN_ROOT` points to the root of the Intel Pin directory.
Then, under the `pin` directory, execute
```bash
make
```
to build the tracing tool.

### Our System Configurations

Hardware config:
```
Haswell-EP: Xeon E3-1246 v3
Skylake-X: i7-7820X
Cascade Lake-X: Xeon W-2245
```

System config:
```
OS: Ubuntu 20.04.3 Server LTS (Focal Fossa)
Kernel: 5.4.0 (booted with default speculative execution attack mitigations)
Compiler: GCC 9.3.0
```


## Demos
The binary `bino` demonstrates the contention behavior exploited by the Binoculars attack and our attack primitives,
including:
- Contention Behaviors
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
- `store_offset`: Store->Load channel
- `vpn_latency`: Load->Store channel
- `vpn_contention`: Load->Store channel port-contention variant

For the `contention_effect` action,
it further takes a fifth argument with options of `0` or `1`
to control whether there is 4K-aliasing between the page walker loads and normal data stores.

## PoCs
### Break ECDSA

In the ECDSA break, we monitor victim stores to one of the four big-number variables.
We detect those stores with the Store->Load channel by monitoring the page offsets (i.e., bits 11-0) that are associated with the variable.
Because the offset depends on the OS and memory allocator details,
if you are using a system that is different from our configuration,
the hardcoded offset in our code may not work (well) on your system.
In that case, you can find suitable offsets to monitor following the Step 1.
Otherwise, you can jump to the Step 2.


#### **Step 1: Find Target Offsets**
Under the root of Binoculars, execute:
```bash
<path to Pin binary> -t pin/obj-intel64/memtrace.so -t config/tracer.tinfo -- ossl/openssl dgst -ecdsa-with-SHA1 -sign ossl/private.pem ossl/data > /dev/null
```
will trace stores to addresses that might be suitable for monitoring.
After the tracing, it should generate a file named `data.trace`,
which contains the store's PC offset within the text segment,
the PL4, PL3, PL2, and PL1 indexes, and offset of the address
that the store writes to, like:
```
<PC offset> W PL4_index PL3_index PL2_index PL1_index offset
```

Execute command:
```bash
cut -d" " data.trace -f7 | sort | uniq -c | sort -n
```
to sort offsets by their occurences in an ascending order.
These offsets are our candidates.
You can candidate offsets from the more occurences end.
To try an offset, you need to replace the value of `CENTRAL_ZERO` in `src/mont.c` with the offset value in hexdecimal.

Note that because these big number varaibles spans multiple double words,
each variable has multiple corresponding offsets.
Monitoring the "central" offset that is "surrounded" by neighboring offsets usually is the best choice.
For example, if the output is:
```
...
 9690 0x8b8
 9690 0x8c8
 9690 0x8d8
10830 0x8b0
10830 0x8c0
10830 0x8d0
10830 0x8e0
```
it means that there are stores to offsets: `0x8b0`, `0x8b8`, `0x8c0`, `0x8c8`, `0x8d0`, `0x8d8`, and `0x8e0`.
Then, `0x8c8` is a better choice than `0x8e0`, because `0x8c8` is the more towards the center.


#### **Step 2 Measure Nonce `k` Prediction Accuracy with Oracle Boundaries**
Inside Jupyter notebook `poc.ipynb`,
the second code block demonstrates extracting the nonce `k`
with *oracle boundaries*.
It uses 30 runs for training and 10 different runs to measure the accuracy.
If one offset does not perform well, please try different offsets following the instruction in Step 1.

### Realistic Nonce `k` Prediction
The Jupyter notebook `mont.ipynb` contains a more realistic nonce `k` recovery demo without using oracle boundaries.
It follows the training methodology described in the paper.
Please refer to the comments in each cell for more details.

## Break Kernel ASLR
The third and fourth code blocks in `poc.ipynb`
demonstrate the kernel ASLR break.
Please refer to the comments for more details.
