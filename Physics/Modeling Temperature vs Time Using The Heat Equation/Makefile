CC=gcc
CFLAGS=-Wall -O3 -march=native
LIBS=-lm

all : calculate

calculate : ./main.c ./initialization.c ./core.c
	$(CC) ./main.x ./initialization.c ./core.c $(CFLAGS) $(LIBS) -o ./calculate

run : calculate ./main.py
	python main.py && mpv ./animation.mkv

play : mpc ./animation.mkv
