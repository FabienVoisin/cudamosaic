#include "struct.cuh"

int main(){
    astrojpg_8u_rgb image1("Orion/orion_1.jpg");
    image1.getgreyimage(); //get the grey image 
    Npp8u* maxbuffer;
    size_t  maxbufferhostsize;
    
    NppiSize osizeROI={(int)image1.nppgreyimage.width(),(int)image1.nppgreyimage.height()};
    
    nppiMaxIndxGetBufferHostSize_8u_C1R(osizeROI, &maxbufferhostsize);

    cudaMalloc((void**)&maxbuffer,maxbufferhostsize);

    image1.getmaxpixel(image1.nppgreyimage,maxbuffer);
    std::cout<<"Maximum pixel: x="<<image1.maxpixelposition.x<<",y="<<image1.maxpixelposition.y;

    const Npp8u threshold=10;
    image1.getsignalimage(threshold);
    saveastro<Npp8u,1>(image1.signalimage,"signalimage.jpg");

    
    return 0;
}