CC = gcc
CFLAGS = -I. -O3 -march=native -mtune=native -std=c17 -flto -fpic -Wall -lpthread -lm -lgsl

libising.so: ./*.c ./*.h
	$(CC) ./*.c $(CFLAGS) -shared -o ./libising.so

run: libising.so main.py
	python ./main.py

clean:
	rm -f ./libising.so
