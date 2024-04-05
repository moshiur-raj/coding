CC = gcc
CFLAGS = -I. -O2 -std=c17 -flto -fPIC -Wall -lpthread -lm -lgsl

libising.so: ./*.c ./*.h
	$(CC) ./*.c $(CFLAGS) -shared -o ./libising.so

run: libising.so main.py
	python ./main.py

clean:
	rm -f ./libising.so
