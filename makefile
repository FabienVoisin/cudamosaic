CC=nvcc --std=c++17
INCLUDE=-I /mnt/sdd/cuda-samples -I /usr/include/opencv4 -I /usr/local/cuda-12.6/targets/x86_64-linux/include/
LIBS=$(shell pkg-config --libs opencv4) -lnppist -lnppig -lnppidei  -lnppitc -lnppicc -lnppial -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage

.PHONY: default
default: mosaic


src/main.o:	src/main.cu
	$(CC) -c $(INCLUDE) -o $@ $< $(LIBS)


mosaic:	 src/main.o
	$(CC) $(INCLUDE) -o $@  src/main.o $(LIBS) 


clean: 
	rm mosaic  src/*.o
