CC = g++
INCLUDES = -I/usr/include/opencv4
CFLAGS = -std=c++11 -Wall -pedantic -g ${INCLUDES}
LDFLAGS = -lopencv_flann -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_photo -lopencv_video -lopencv_dnn -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

SRC = underwater.cpp
HDR = $(wildcard *.hpp)
OBJ = $(SRC:.cpp=.o)

underwater: ${OBJ}
	${CC} -o $@ ${CFLAGS} ${LDFLAGS} $^

%.o: %.cpp
	${CC} -c ${CFLAGS} $<

clean:
	rm -f rretinex *.o
