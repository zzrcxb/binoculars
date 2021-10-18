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

.PHONY: all clean pteditor pteditor_build pteditor_install pteditor_uninstall

all: $(BINS)

$(BIN)/%: $(OBJ)/%.o
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $<

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -MMD -o $@ -c $<

clean:
	rm -rf $(BIN) $(OBJ)

pteditor: pteditor_build pteditor_install

pteditor_build:
	@echo "Building PTEditor"
	make -C $(PTEDITOR)/

pteditor_install:
	sudo insmod $(PTEDITOR)/module/pteditor.ko

pteditor_uninstall:
	sudo rmmod $(PTEDITOR)/module/pteditor.ko

-include $(DEPS)
