CC = g++
INCLUDES = $(shell pkg-config --cflags opencv4)
CFLAGS = -std=c++11 -Wall -pedantic -g -fopenmp ${INCLUDES}
LDFLAGS = $(shell pkg-config --libs opencv4)

SRC = $(wildcard *.cpp)
HDR = $(wildcard *.hpp)
OBJ = $(SRC:.cpp=.o)

underwater: ${OBJ}
	${CC} -o $@ ${CFLAGS} ${LDFLAGS} $^

%.o: %.cpp
	${CC} -c ${CFLAGS} $<

${OBJ}: ${HDR}

clean:
	rm -f underwater *.o
