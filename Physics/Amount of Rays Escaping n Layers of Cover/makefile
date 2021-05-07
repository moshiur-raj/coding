CC = gcc
CFLAGS = -Wall -Werror -O2 -march=native
LIBS = -lpthread
SRCDIR = .
OBJDIR = ./build
EXDIR = .
OBJS = $(OBJDIR)/main.o $(OBJDIR)/calc.o $(OBJDIR)/pcg_basic.o

all : calculate

$(OBJDIR)/%.o : $(SRCDIR)/%.c
	@echo ">> Compiling $^"
	$(CC) -c $(CFLAGS) $^ -o $@

calculate : $(OBJS)
	@echo ">>> Building the executable"
	$(CC) $(OBJS) $(CFLAGS) $(LIBS) $(OPTS) -o $(EXDIR)/calculate

test : calculate
	$(EXDIR)/calculate 1 2 1 0 1 1000000 5

run : calculate $(SRCDIR)/main.py
	python main.py

clean :
	@echo ">>> Removing builds"
	rm $(OBJS) calculate
