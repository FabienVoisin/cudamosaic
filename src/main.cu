#include "struct.cuh"
#include "astroio.h"
#include <cstdlib>
#include <vector>
#include <string>
#include <filesystem>
#include "optionsparser.h"
/*Extern variables */
extern std::string directorypath;
extern std::string outputfilename;

/*global variable : NPP buffers */
Npp8u *maxbuffer;
Npp8u *sumbuffer;
int squaresize;
const Npp8u threshold=40;

void setupmaxbuffer(astrojpg_rgb_<Npp8u> &image1){
    size_t  maxbufferhostsize;
    NppiSize osizeROI={(int)image1.nppgreyimage.width(),(int)image1.nppgreyimage.height()};
    nppiMaxIndxGetBufferHostSize_8u_C1R(osizeROI, &maxbufferhostsize);
    cudaMalloc((void**)&maxbuffer,maxbufferhostsize);
}

void setupsumbuffer(int squaresize){
    size_t  sumbufferhostsize;
    NppiSize osumROI={squaresize,squaresize};
    nppiSumGetBufferHostSize_32f_C1R(osumROI, &sumbufferhostsize);
    cudaMalloc((void**)&sumbuffer,sumbufferhostsize);
}

void initfirstimage(astrojpg_rgb_<Npp8u> &image1){
    setupmaxbuffer(image1);
    setupsumbuffer(squaresize);
    image1.getgreyimage(); 
    image1.getsignalimage(image1.nppgreyimage,threshold);
    image1.getmaxpixel(image1.nppgreyimage,image1.maxpixelposition,maxbuffer);
    cudaDeviceSynchronize();
    image1.createROIdata(squaresize);

}

void mosaicimages(std::vector<std::string> &files, astrojpg_rgb_<Npp32f> &imagetotal, astrojpg_rgb_<Npp8u> &image1){
    
    int differencex,differencey;
    imagetotal.getgreyimage();
    imagetotal.getsignalimage(imagetotal.nppgreyimage,threshold);
    imagetotal.Correlationimage(image1.maskimage,sumbuffer);
    imagetotal.getmaxpixel(imagetotal.correlationimage,imagetotal.maxcorrposition,maxbuffer);

    for (std::string file : files ){
        std::cout<<"filename="<<file<<std::endl;
        astrojpg_rgb_<Npp8u> iterimage(file);
        
        iterimage.getgreyimage();
        
        iterimage.getsignalimage(iterimage.nppgreyimage,threshold);
        iterimage.Correlationimage(image1.maskimage,sumbuffer);
        iterimage.getmaxpixel(iterimage.correlationimage,iterimage.maxcorrposition,maxbuffer);
        
        differencex=iterimage.maxcorrposition.x-imagetotal.maxcorrposition.x;
        differencey=iterimage.maxcorrposition.y-imagetotal.maxcorrposition.y;
        cv::Point_<int> offsetposition={differencex,differencey};
        imagetotal.stackimage(iterimage);
        std::cout<<imagetotal.nppinputimage.width()<<","<<imagetotal.nppinputimage.height()<<std::endl;
        cudaDeviceSynchronize();
        std::cout<<"filename="<<file<<std::endl;
    }
}

void normaliseimage(astrojpg_rgb_<Npp32f> &imagetotal){
    imagetotal.normaliseimage<Npp32f,3>(imagetotal.nppinputimage);
    imagetotal.normaliseimage<Npp32f,1>(imagetotal.nppgreyimage);
}

void outputmosaic(astrojpg_rgb_<Npp32f> &imagetotal){
    saveastro<Npp32f,3>(imagetotal.nppinputimage,outputfilename);    
    saveastro<Npp32f,3>(imagetotal.exposuremap,"finalresultexp.jpg");
}

int main(int argc, char **argv){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaMemPool_t memPool;
    size_t free, total;
    cudaMemGetInfo(&free,&total);
    std::cout<<"free="<<free<<",total="<<total<<std::endl;
    std::vector<std::string> files;
    
    optionparser(argc, argv);
    if (directorypath.empty()){
        std::cout<<"Input directory has not been mentioned"<<std::endl;
        exit(1);
    }
    listfiles(directorypath,files);
    std::stable_sort(files.begin(), files.end());
    squaresize=111;
    astrojpg_rgb_<Npp8u> image1(files[0]);
    initfirstimage(image1);

    astrojpg_rgb_<Npp32f> imagetotal(files[1]);
    mosaicimages(files,imagetotal,image1);
    normaliseimage(imagetotal);
    outputmosaic(imagetotal);

    
    return 0;
}