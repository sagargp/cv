CC=g++-4.7
CCFLAGS=-g
CFLAGS=-fPIC -O3 -std=c++11
LIBS=`pkg-config opencv --cflags --libs`

all: test

test: test.cc
	${CC} ${CFLAGS} ${CCFLAGS} ${INCLUDES} -o test test.cc ${LIBS}

clean:
	rm -f test
