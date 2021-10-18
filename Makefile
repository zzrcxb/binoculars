CC := gcc
SRC := src
OBJ := obj
BIN := bin
INC := include

PTEDITOR := PTEditor
CFLAGS := -O1 -g -Werror -I$(INC) -I$(PTEDITOR)
LDFLAGS := -O1

SRCS := $(shell find $(SRC) -type f -name "*.c")
BINS := $(patsubst $(SRC)/%, $(BIN)/%, $(SRCS:.c=))
OBJS := $(patsubst $(SRC)/%, $(OBJ)/%, $(SRCS:.c=.o))
DEPS := $(patsubst $(SRC)/%, $(OBJ)/%, $(SRCS:.c=.d))

.PHONY: all clean pte_editor pte_editor_install

all: $(BINS)

$(BIN)/%: $(OBJ)/%.o
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $<

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -MMD -o $@ -c $<

clean:
	rm -rf $(BIN) $(OBJ)

pte_editor:
	@echo "Building PTEditor"
	make -C $(PTEDITOR)/

pte_editor_install:
	sudo insmod $(PTEDITOR)/module/pteditor.ko

-include $(DEPS)
