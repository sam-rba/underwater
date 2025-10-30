CC = g++
INCLUDES = -I/usr/include/opencv4
CFLAGS = -std=c++11 -Wall -pedantic -g -fopenmp ${INCLUDES}
LDFLAGS = -lopencv_imgcodecs -lopencv_xphoto -lopencv_imgproc -lopencv_core

SRC = $(wildcard *.cpp)
HDR = $(wildcard *.hpp)
OBJ = $(SRC:.cpp=.o)

underwater: ${OBJ}
	${CC} -o $@ ${CFLAGS} ${LDFLAGS} $^

%.o: %.cpp
	${CC} -c ${CFLAGS} $<

clean:
	rm -f underwater *.o
