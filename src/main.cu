#include "struct.cuh"
#include "gpufuncs.cuh"
#include "astroio.h"

int main(){
    astrojpg_8u_rgb image1("Orion/orion_1.jpg");
    image1.getgreyimage(); //get the grey image 
    Npp8u* maxbuffer;
    Npp8u* sumbuffer;
    size_t  maxbufferhostsize;
    
    NppiSize osizeROI={(int)image1.nppgreyimage.width(),(int)image1.nppgreyimage.height()};
    
    nppiMaxIndxGetBufferHostSize_8u_C1R(osizeROI, &maxbufferhostsize);

    cudaMalloc((void**)&maxbuffer,maxbufferhostsize);

    image1.getmaxpixel(image1.nppgreyimage,maxbuffer);
    std::cout<<"Maximum pixel: x="<<image1.maxpixelposition.x<<",y="<<image1.maxpixelposition.y<<std::endl;

    const Npp8u threshold=10;
    image1.getsignalimage(threshold);

    saveastro<Npp8u,1>(image1.nppgreyimage,"greyimage.jpg");
    saveastro<Npp8u,1>(image1.signalimage,"signalimage.jpg");
    /* We then need to create the data that will be use for the convolution*/
    int squaresize=111; //square size 
    npp::ImageNPP_32s_C1 maskimage(squaresize,squaresize);

    size_t  sumbufferhostsize;
    NppiSize omaskROI={squaresize,squaresize};
    nppiSumGetBufferHostSize_32f_C1R(omaskROI, &sumbufferhostsize);
    cudaMalloc((void**)&sumbuffer,sumbufferhostsize);
    std::cout<<"Am I here"<<std::endl;
    cudaDeviceSynchronize();
    CreateROIdata(image1.signalimage,image1.maxpixelposition,squaresize,maskimage);
    saveastro<Npp32s,1>(maskimage,"mask.jpg");
    /*We need to load a new  image */
    std::cout<<"Am I here 2"<<std::endl;
    astrojpg_8u_rgb image2("Orion/orion_48.jpg");
    image2.getgreyimage();
    image2.getsignalimage(threshold);
    /*Create three new images, the correlation image, the exposure map and the new combinbed image*/
    npp::ImageNPP_8u_C1 correlation(image2.nppgreyimage.size());
    Correlationimage(image2.signalimage,maskimage,sumbuffer,correlation);
    std::cout<<"FINAL COUNTDOWN"<<std::endl;
    saveastro<Npp8u,1>(correlation,"correlationexample.jpg");

    
    return 0;
}