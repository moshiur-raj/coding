CC = gcc
CFLAGS = -Wall -Werror
LIBS = -lm
SRCDIR = .
OBJDIR = $(SRCDIR)/build
EXDIR = .
OBJS = $(OBJDIR)/main.o $(OBJDIR)/core.o $(OBJDIR)/rpn_generator.o $(OBJDIR)/shared.o $(OBJDIR)/stacks.o

all : calculator

build :
	@echo "creating build directory"
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o : $(SRCDIR)/%.c
	@echo ">> Compiling $^..."
	$(CC) -c $(CFLAGS) $^ -o $@

calculator : build $(OBJS)
	@echo ">>> Starting to build the executable"
	$(CC) $(OBJS) $(CFLAGS) $(LIBS) -o $(EXDIR)/calculator

test : calculator
	$(EXDIR)/calculator <<< +1 +2 +

run : calculator
	$(EXDIR)/calculator

clean :
	@echo ">>> Removing builds"
	rm -rf $(OBJDIR) $(EXDIR)/calculator
