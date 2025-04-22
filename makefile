NVCC=$(shell which nvcc)
CUDAVERSION=$(shell $(NVCC) -V | grep -Po "release\s\K\w+\.\w")
CC=$(NVCC) --std=c++17
CUDA_INCLUDEPATH="/usr/local/cuda-$(CUDAVERSION)/targets/x86_64-linux/include"
INCLUDE=-I ./cuda-samples -I /usr/include/opencv4 -I$(CUDA_INCLUDEPATH)
LIBS=$(shell pkg-config --libs opencv4) -lnppist -lnppig -lnppidei  -lnppitc -lnppicc -lnppial -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage

.PHONY: default
default: mosaic


src/main.o:	src/main.cu
	$(CC) -c $(INCLUDE) -o $@ $< $(LIBS)

src/optionsparser.o:	src/optionsparser.cpp
	$(CC) -c $(INCLUDE) -o $@ $< $(LIBS)

src/astroio.o:	src/astroio.cu
	$(CC) -c $(INCLUDE) -o $@ $< $(LIBS)

mosaic:	 src/main.o src/optionsparser.o src/astroio.o
	$(CC) $(INCLUDE) -o $@  src/main.o src/optionsparser.o src/astroio.o $(LIBS) 


clean: 
	rm mosaic  src/*.o
