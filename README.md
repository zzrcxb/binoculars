# Binoculars

## Why binoculars?
This side-channel is just like viewing through a binoculars from either end.
From one end, you can make things seem "closer" (extracting store offsets);
from the other side, you can make things seem "farther" (extracting virtual page numbers).

## Requirements
These PoCs depend on [PTEditor](https://github.com/misc0110/PTEditor),
which is a kernel module that helps page-table manipulation in user space.
Therefore, a kernel module build environment is required.

For the `runner.py` script to work, you will need
`Python >= 3.6`, and then execute
```pip3 install -r requirements.txt --user```
to install required libraries.

## Build
Execute `make pteditor` to build PTEditor and load the kernel module.

Execute `make` to build PoCs.

## PoCs
`store_offset`: PoC that recovers 11-3 bits of a store's address,
which is `0x528` in our PoC.

`load_page_throughput`: PoC that recovers 47-12 bits of a load's address (VPN),
by measuring sender's throughput.
Each level of page-table entry has an index of `0x87`, `0x65`, `0x43`, and `0x21`.

`load_page_contention`: PoC that recovers 47-12 bits of a load's address,
by observing contention at the receiver end.
Each level of page-table entry has an index of `0x87`, `0x65`, `0x43`, and `0x21`.

## Execute
### Use `runner.py` (Recommended)
`runner.py` will execute one of the PoCs for several times and output averaged
(denoised) results. Please refer to its help information for more details.

Some examples:

Run PoC `store_offset`
(the script will try to detect logical cores that share the same physical core
and pin processes to those cores):

```./runner.py store_offset```

Run PoC `store_offset` on cores 1 and 3:

```./runner.py -c1,3 store_offset```

Run PoC `load_page_throughput` for 30 times (instead of the default 10 times):

```./runner.py -i 30 load_page_throughput```

After it's finished, it will output a PDF that has the name `<PoC_name>.pdf`,
and also output raw data as TSV to `stdout`
(if you want to save the raw data: `./runner.py ... | tee raw_data.tsv`,
the three columns correspond to `offset`, `avg. value`, `standard deviation`).

### Directly run `bin/bino`
The output binray `bin/bino` takes an argument,
which indicates the PoC you are trying to invoke.
Possible options are: `store_offset`, `load_page_throughput`, and `load_page_contention`.

To force processes run on the same physical core,
run it with:
```taskset -c <core1_id>,<core2_id> bin/bino <option>```.
Logical cores `core1` and `core2` should share the same physical core.
The easiest way to find such two cores is reading from `/proc/cpuinfo`,
and find two entries that have the same `core id` and `physical id` but
different `processor` fields.

## Understand Outputs
If you used "runner.py" to generate plots in the previous step,
you should see a PDF that has the name `<PoC_name>.pdf`.
The x-axis represents a store's offet or a page-table entry's index.
The y-axis represents an averaged measured values (e.g., latency/throughput, averaged over N runs).
The solid blue line represents the averaged measured values for each offset/index value.
The light blue shade represents the standard deviation for each offset/index value.
The horizontal orange dashed line represents the average value of *all* data points.
The horizontal grey dashed lines represent `[avg. - 2 * stddev, avg. + 2 * stddev]`.
The data points with red crosses represent the values that the attacker would recover.

## Expected Results
Expected results are under `expected/<uarch-name>`.

Hardware config:
```
Haswell-EP: Xeon E3-1246 v3
Skylake-X: i7-7820X
```

System config:
```
OS: Ubuntu 20.04.3 LTS (Focal Fossa)
Kernel: 5.4.0-89-generic
Boot options: default with all Spectre mitigations
Compiler: GCC 9.3.0
```
