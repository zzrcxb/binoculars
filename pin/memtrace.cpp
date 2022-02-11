#include "pin.H" // NOLINT(redefinition_different_typedef)
#include <fstream>
#include <iostream>
#include <map>
#include <stdio.h>
#include <vector>

using namespace std;

KNOB<string> KnobDataOutfile(KNOB_MODE_WRITEONCE, "pintool", "d", "data.trace",
                            "specify output file name");

KNOB<string> KnobTracerfile(KNOB_MODE_WRITEONCE, "pintool", "t", "tracer.tinfo",
                            "specify tracer file name");

#define NAME_LEN 32

struct TracerPoint {
public:
    bool read_enable, write_enable, after, single;
    char name[NAME_LEN];
};

FILE *DATA_OUT;
char TRACE_NAME[NAME_LEN] = "\0";
bool READ_ENABLE = false, WRITE_ENABLE = false;

ADDRINT IMAGE_BASE = 0;
map<ADDRINT, TracerPoint> TRACE_ACTIONS;

void dataReadTracer(void *ip, void *_addr) {
    if (READ_ENABLE) {
        ADDRINT ip_pic = (ADDRINT)ip - IMAGE_BASE;
        ADDRINT addr = (ADDRINT)_addr;
        unsigned long page = addr >> 12;
        fprintf(DATA_OUT, "%#lx R %#05lx %#05lx %#05lx %#05lx %#05lx\n", ip_pic,
                (page >> 27) & 0x1ff, (page >> 18) & 0x1ff, (page >> 9) & 0x1ff,
                page & 0x1ff, (unsigned long)(addr & 0xfff));
    }
}

void dataWriteTracer(void *ip, void *_addr) {
    if (WRITE_ENABLE) {
        ADDRINT ip_pic = (ADDRINT)ip - IMAGE_BASE;
        ADDRINT addr = (ADDRINT)_addr;
        unsigned long page = addr >> 12;
        fprintf(DATA_OUT, "%#lx W %#05lx %#05lx %#05lx %#05lx %#05lx\n", ip_pic,
                (page >> 27) & 0x1ff, (page >> 18) & 0x1ff, (page >> 9) & 0x1ff,
                page & 0x1ff, (unsigned long)(addr & 0xfff));
    }
}

void hitTracer(TracerPoint *tp) {
    READ_ENABLE = tp->read_enable;
    WRITE_ENABLE = tp->write_enable;
    strcpy_s(TRACE_NAME, NAME_LEN, tp->name);
    TRACE_NAME[NAME_LEN - 1] = '\0';
}

void Instruction(INS ins, void *v) {
    ADDRINT offset = INS_Address(ins) - IMAGE_BASE;
    if (TRACE_ACTIONS.find(offset) != TRACE_ACTIONS.end()) {
        TracerPoint &tp = TRACE_ACTIONS[offset];
        if (tp.single) {
            uint32_t numMemOps = INS_MemoryOperandCount(ins);
            for (uint32_t memOp = 0; memOp < numMemOps; memOp++) {
                if (tp.read_enable && INS_MemoryOperandIsRead(ins, memOp)) {
                    INS_InsertPredicatedCall(
                        ins, IPOINT_BEFORE, (AFUNPTR)dataReadTracer,
                        IARG_INST_PTR, IARG_MEMORYOP_EA, memOp, IARG_END);
                }

                if (tp.write_enable && INS_MemoryOperandIsWritten(ins, memOp)) {
                    INS_InsertPredicatedCall(
                        ins, IPOINT_BEFORE, (AFUNPTR)dataWriteTracer,
                        IARG_INST_PTR, IARG_MEMORYOP_EA, memOp, IARG_END);
                }
            }
        } else {
            auto *action = &TRACE_ACTIONS[offset];
            auto loc = action->after ? IPOINT_AFTER : IPOINT_BEFORE;
            INS_InsertCall(ins, loc, (AFUNPTR)hitTracer, IARG_PTR, action, IARG_END);
        }
    }
}

void ImageLoad(IMG img, void *v) {
    if (IMG_IsMainExecutable(img)) {
        IMAGE_BASE = IMG_LoadOffset(img);
        fprintf(stderr, "[PIN] image \"%s\" is loaded at %#0lx\n",
                IMG_Name(img).c_str(), (unsigned long)IMAGE_BASE);
    }
}

void Fini(int32_t code, void *v) {
    fclose(DATA_OUT);
}

/**
 * @brief load tracer actions
 * a tracer action file usually has an extension ".tinfo".
 * from high-level, its format is like the following:
 * <hex offset in the executable> <action> <flags> <tracer name>
 *      <action> can be ENABLE, DISABLE, or SINGLE
 *      <flags> is a string that can contain "R" or/and "W".
 *          Each flag corresponds to recording reads, writes, and instructions.
 *          (e.g., flag "RW" records all data accesses)
 *      <tracer name> is a string **WITHOUT** any space
 * @return int; 0 for success, not-zero for failure
 */
int LoadTracerPoints() {
    ifstream infile(KnobTracerfile.Value().c_str());
    if (infile.is_open()) {
        string type, flags, name;
        ADDRINT offset;
        while (infile >> hex >> offset >> type >> flags >> name) {
            TracerPoint tp;
            memset(&tp, 0, sizeof(TracerPoint));
            bool enable = (type == "ENABLE" || type == "SINGLE");
            for (auto &c : flags) {
                switch (c) {
                    case 'R': {
                        tp.read_enable = enable;
                        break;
                    }
                    case 'W': {
                        tp.write_enable = enable;
                        break;
                    }
                    default: {
                        fprintf(stderr, "Invalid flag '%c'\n", c);
                        infile.close();
                        return 1;
                    }
                }
            }
            tp.single = type == "SINGLE";
            strcpy_s(tp.name, NAME_LEN, name.c_str());
            tp.name[NAME_LEN - 1] = '\0';
            tp.after = enable;
            TRACE_ACTIONS[offset] = tp;
        }
        infile.close();
        return 0;
    } else {
        fprintf(stderr, "[PIN] Trace point file \"%s\" not found.",
                KnobTracerfile.Value().c_str());
        return 1;
    }
}


int32_t Usage() {
    PIN_ERROR("This Pintool prints R/W traces for given instructions\n" +
              KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

int main(int argc, char *argv[]) {
    int ret = 0;

    if (PIN_Init(argc, argv))
        return Usage();

    if (LoadTracerPoints()) {
        fprintf(stderr, "Failed to parser \"%s\"\n",
                KnobTracerfile.Value().c_str());
        return -1;
    }

    DATA_OUT = fopen(KnobDataOutfile.Value().c_str(), "w");
    if (!DATA_OUT) {
        ret = 1;
        goto err;
    }

    IMG_AddInstrumentFunction(ImageLoad, 0);

    // Register Instruction to be called to instrument instructions
    INS_AddInstrumentFunction(Instruction, 0);

    // Register Fini to be called when the application exits
    PIN_AddFiniFunction(Fini, 0);

    // Start the program, never returns
    PIN_StartProgram();

err:
    fclose(DATA_OUT);
    remove(KnobDataOutfile.Value().c_str());
    return ret;
}
