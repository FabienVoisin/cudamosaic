#include "struct.cuh"
#include "astroio.h"
#include <cstdlib>
#include <vector>
#include <string>
#include <filesystem>

void listfiles(std::string directorypath,std::vector<std::string> &list_of_files){
    for (const auto& entry : std::filesystem::directory_iterator(directorypath)) {
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        //std::cout<<outfilename_str<<std::endl;
        list_of_files.push_back(outfilename_str);
    }
}

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaMemPool_t memPool;
    size_t free, total;
    cudaMemGetInfo(&free,&total);
    std::cout<<"free="<<free<<",total="<<total<<std::endl;
    astrojpg_rgb_<Npp8u> image1("Orion/orion_20.jpg");
    std::cout<<image1.nppinputimage.width()<<","<<image1.nppinputimage.height();
    std::vector<std::string> files;
    listfiles("/mnt/sdd/cudamosaic/Orion/",files);
    std::stable_sort(files.begin(), files.end());
    std::cout<<files[1]<<std::endl;
    //for (auto filename: files) std::cout<<filename<<std::endl;
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
    std::cout<<"Am I here 3"<<std::endl;
    image2.getgreyimage();
    
    image2.getsignalimage(image2.nppgreyimage,threshold);
    /*Create three new images, the correlation image, the exposure map and the new combinbed image*/
    //npp::ImageNPP_8u_C1 correlation(image2.nppgreyimage.size());
    image2.Correlationimage(image1.maskimage,sumbuffer);
    std::cout<<"Am I here 4"<<std::endl;
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
    int differencex,differencey;
    
    astrojpg_rgb_<Npp32f> imagetotal("Orion/orion_1.jpg");
    imagetotal.getgreyimage();
    imagetotal.getsignalimage(imagetotal.nppgreyimage,threshold);
    imagetotal.Correlationimage(image1.maskimage,sumbuffer);
    imagetotal.getmaxpixel(imagetotal.correlationimage,imagetotal.maxcorrposition,maxbuffer);
    cudaDeviceSynchronize();
    std::cout<<"am i here now"<<std::endl;
    cudaMemGetInfo(&free,&total);
    std::cout<<"free="<<free<<",total="<<total<<std::endl;
    for (int i = 1 ; i<10 ;i++ ){
        cudaMemGetInfo(&free,&total);
    std::cout<<"free="<<free<<",total="<<total<<std::endl;
        std::cout<<"filename="<<files[i]<<std::endl;
        astrojpg_rgb_<Npp8u> iterimage(files[i]);
        
        iterimage.getgreyimage();
        
        iterimage.getsignalimage(iterimage.nppgreyimage,threshold);
        
        iterimage.Correlationimage(image1.maskimage,sumbuffer);
        
        iterimage.getmaxpixel(iterimage.correlationimage,iterimage.maxcorrposition,maxbuffer);
        
        differencex=iterimage.maxcorrposition.x-imagetotal.maxcorrposition.x;
        differencey=iterimage.maxcorrposition.y-imagetotal.maxcorrposition.y;
        cv::Point_<int> offsetposition={differencex,differencey};
        std::cout<<"totalpos="<<imagetotal.maxcorrposition.x<<","<<imagetotal.maxcorrposition.y<<std::endl;
        std::cout<<"pos="<<iterimage.maxcorrposition.x<<","<<iterimage.maxcorrposition.y<<std::endl;
        std::cout<<"diff="<<differencex<<","<<differencey<<std::endl;
        std::cout<<"Am I now"<<std::endl;
        imagetotal.stackimage(iterimage,offsetposition);
        cudaDeviceSynchronize();
    }
    
    //std::cout<<"diff="<<imagetotal.maxcorrposition.x<<","<<imagetotal.maxcorrposition.y<<std::endl;
    //cv::Point_<int> offsetposition={differencex,differencey};
    //imagetotal.stackimage(image3,offsetposition);
    saveastro<Npp32f,3>(imagetotal.nppinputimage,"finalresult.jpg");
    saveastro<Npp32f,1>(imagetotal.exposuremap,"finalresultexp.jpg");
    return 0;
}