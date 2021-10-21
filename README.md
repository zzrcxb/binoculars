# Binoculars

## Why binoculars?
This side-channel is just like viewing through a binoculars from either end.
From one end, you can make things seem "closer" (extracting store offsets);
from the other side, you can make things seem "farther" (extracting load page).

## Requirements
These PoCs depend on [PTEditor](https://github.com/misc0110/PTEditor),
which is a kernel module that helps page-table manipulation in user space.
Therefore, a kernel module build environment is required.

For the `runner.py` script to work, you will need
`Python >= 3.6`, and then execute
`pip3 install -r requirements.txt --user`
to install required libraries.

## Build
Execute `make pteditor` to build PTEditor and load the kernel module.

Execute `make` to build PoCs.

## Execute
### PoCs
`store_offset`: PoC that recovers 11-3 bits of a store's address.
`load_page_throughput`: PoC that recovers 47-12 bits of a load's address,
by measuring sender's throughput.
`load_page_contention`: PoC that recovers 47-12 bits of a load's address,
by observing contention at the receiver end.

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
Possible options are: `store_offset`, `load_page_throughput`, `load_page_contention`.

To force processes run on the same physical core,
run it with:
```taskset -c <core1_id>,<core2_id> bin/bino <option>```
logical cores `core1` and `core2` should share the same physical core.
The easiest way to find such two cores is reading from `/proc/cpuinfo`,
and find two entries that have the same `core id` and `physical id` but
different `processor` field.
