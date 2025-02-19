#include "struct.cuh"
#include "astroio.h"




int main(){
    astrojpg_rgb_<Npp8u> image1("Orion/orion_1.jpg");
    image1.getgreyimage(); //get the grey image 
    Npp8u* maxbuffer;
    Npp8u* sumbuffer;
    size_t  maxbufferhostsize;
    
    NppiSize osizeROI={(int)image1.nppgreyimage.width(),(int)image1.nppgreyimage.height()};
    
    nppiMaxIndxGetBufferHostSize_8u_C1R(osizeROI, &maxbufferhostsize);

    cudaMalloc((void**)&maxbuffer,maxbufferhostsize);

    image1.getmaxpixel(image1.nppgreyimage,image1.maxpixelposition,maxbuffer);
    std::cout<<"Maximum pixel: x="<<image1.maxpixelposition.x<<",y="<<image1.maxpixelposition.y<<std::endl;

    const Npp8u threshold=40;
    image1.getsignalimage(image1.nppgreyimage,threshold);

    saveastro<Npp8u,1>(image1.nppgreyimage,"greyimage.jpg");
    saveastro<Npp8u,1>(image1.signalimage,"signalimage.jpg");
    /* We then need to create the data that will be use for the convolution*/
    int squaresize=111; //square size 

    size_t  sumbufferhostsize;
    NppiSize omaskROI={squaresize,squaresize};
    nppiSumGetBufferHostSize_32f_C1R(omaskROI, &sumbufferhostsize);
    cudaMalloc((void**)&sumbuffer,sumbufferhostsize);
    std::cout<<"Am I here"<<std::endl;
    cudaDeviceSynchronize();
    image1.createROIdata(squaresize);
    saveastro<Npp32s,1>(image1.maskimage,"mask.jpg");
    //image1.Correlationimage(image1.maskimage,sumbuffer);
    //saveastro<Npp8u,1>(image1.correlationimage,"autocorrelation1.jpg");
    //image1.getmaxpixel(image1.correlationimage,image1.maxcorrposition,maxbuffer);
    //std::cout<<"Maximum corr pixel: x="<<image1.maxcorrposition.x<<",y="<<image1.maxcorrposition.y<<std::endl;

    /*We need to load a new  image */
    std::cout<<"Am I here 2"<<std::endl;
    astrojpg_rgb_<Npp8u> image2("Orion/orion_1.jpg");
    image2.getgreyimage();
    image2.getsignalimage(image2.nppgreyimage,threshold);
    /*Create three new images, the correlation image, the exposure map and the new combinbed image*/
    //npp::ImageNPP_8u_C1 correlation(image2.nppgreyimage.size());
    image2.Correlationimage(image1.maskimage,sumbuffer);
    std::cout<<"FINAL COUNTDOWN"<<std::endl;
    saveastro<Npp8u,1>(image2.correlationimage,"correlationexample.jpg");
    image2.getmaxpixel(image2.correlationimage,image2.maxcorrposition,maxbuffer);
    std::cout<<"Maximum corr pixel: x="<<image2.maxcorrposition.x<<",y="<<image2.maxcorrposition.y<<std::endl;
    astrojpg_rgb_<Npp8u> image3("Orion/orion_4.jpg");
    image3.getgreyimage();
    image3.getsignalimage(image3.nppgreyimage,threshold);
    image3.Correlationimage(image1.maskimage,sumbuffer);
    saveastro<Npp8u,1>(image3.correlationimage,"correlationexample2.jpg");
    image3.getmaxpixel(image3.correlationimage,image3.maxcorrposition,maxbuffer);
    std::cout<<"Maximum corr pixel: x="<<image3.maxcorrposition.x<<",y="<<image3.maxcorrposition.y<<std::endl;

    /*Create a new function for mosaicing the stuff*/
    unsigned int differencex,differencey;
    
    astrojpg_rgb_<Npp32f> imagetotal("Orion/orion_1.jpg");
    imagetotal.getgreyimage();
    imagetotal.getsignalimage(imagetotal.nppgreyimage,threshold);
    imagetotal.Correlationimage(image1.maskimage,sumbuffer);
    imagetotal.getmaxpixel(imagetotal.correlationimage,imagetotal.maxcorrposition,maxbuffer);
    differencex=image3.maxcorrposition.x-imagetotal.maxcorrposition.x;
    differencey=image3.maxcorrposition.y-imagetotal.maxcorrposition.y;
    std::cout<<"diff="<<imagetotal.maxcorrposition.x<<","<<imagetotal.maxcorrposition.y<<std::endl;
    cv::Point_<int> offsetposition={differencex,differencey};
    imagetotal.stackimage(image3,offsetposition);
    saveastro<Npp32f,3>(imagetotal.nppinputimage,"finalresult.jpg");
    saveastro<Npp32f,1>(imagetotal.exposuremap,"finalresultexp.jpg");
    return 0;
}