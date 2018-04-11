main: main.o NN.o
	g++ main.o NN.o -o main -larmadillo

NN.o: src/NN.cpp src/NN.h
	g++ -std=c++11 -Wall -Wextra -O2 src/NN.cpp -c

main.o: src/main.cpp
	g++ -std=c++11 -Wall -Wextra -O2 src/main.cpp -c

clean:
	rm main.o
	rm NN.o
