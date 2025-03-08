#include "struct.cuh"
#include "astroio.h"
#include <cstdlib>
#include <vector>
#include <string>
#include <filesystem>
#include "optionsparser.h"
extern std::string directorypath;
extern std::string outputfilename;
void listfiles(std::string directorypath,std::vector<std::string> &list_of_files){
    for (const auto& entry : std::filesystem::directory_iterator(directorypath)) {
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        //std::cout<<outfilename_str<<std::endl;
        list_of_files.push_back(outfilename_str);
    }
}

int main(int argc, char ** argv){
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
    astrojpg_rgb_<Npp8u> image1(files[0]);
    std::cout<<image1.nppinputimage.width()<<","<<image1.nppinputimage.height();
    
    
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
    
    std::cout<<"Am I here 4"<<std::endl;
    std::cout<<"FINAL COUNTDOWN"<<std::endl;
    
    /*Create a new function for mosaicing the stuff*/
    int differencex,differencey;
    
    astrojpg_rgb_<Npp32f> imagetotal(files[1]);
    imagetotal.getgreyimage();
    imagetotal.getsignalimage(imagetotal.nppgreyimage,threshold);
    imagetotal.Correlationimage(image1.maskimage,sumbuffer);
    imagetotal.getmaxpixel(imagetotal.correlationimage,imagetotal.maxcorrposition,maxbuffer);
    cudaDeviceSynchronize();
    std::cout<<"am i here now"<<std::endl;
    cudaMemGetInfo(&free,&total);
    std::cout<<"free="<<free<<",total="<<total<<std::endl;
    int i=0;
    for (std::string file : files ){
        std::cout<<"filename="<<file<<std::endl;
        astrojpg_rgb_<Npp8u> iterimage(file);
        
        iterimage.getgreyimage();
        
        iterimage.getsignalimage(iterimage.nppgreyimage,threshold);
        //std::string itersignal="itersignal"+std::to_string(i)+".jpg";
        //saveastro<Npp8u,1>(iterimage.signalimage,itersignal);
        iterimage.Correlationimage(image1.maskimage,sumbuffer);
        //std::string itercorr="itercorr"+std::to_string(i)+".jpg";
        //saveastro<Npp8u,1>(iterimage.correlationimage,itercorr);
        iterimage.getmaxpixel(iterimage.correlationimage,iterimage.maxcorrposition,maxbuffer);
        
        differencex=iterimage.maxcorrposition.x-imagetotal.maxcorrposition.x;
        differencey=iterimage.maxcorrposition.y-imagetotal.maxcorrposition.y;
        cv::Point_<int> offsetposition={differencex,differencey};
        std::cout<<"totalwidth="<<imagetotal.nppinputimage.width()<<","<<imagetotal.nppinputimage.height()<<std::endl;
    
        std::cout<<"diff="<<differencex<<","<<differencey<<std::endl;
        std::cout<<"Am I now"<<std::endl;
        imagetotal.stackimage(iterimage);
        std::cout<<imagetotal.nppinputimage.width()<<","<<imagetotal.nppinputimage.height()<<std::endl;
        cudaDeviceSynchronize();
        //std::string iterstack="finalresult_"+std::to_string(i)+".jpg";
        //saveastro<Npp32f,3>(imagetotal.nppinputimage,iterstack);
        //std::string iterexp="finalexp_"+std::to_string(i)+".jpg";
        //saveastro<Npp32f,1>(imagetotal.exposuremap,iterexp);
        i++;
    }
    
    //std::cout<<"diff="<<imagetotal.maxcorrposition.x<<","<<imagetotal.maxcorrposition.y<<std::endl;
    //cv::Point_<int> offsetposition={differencex,differencey};
    //imagetotal.stackimage(image3,offsetposition);
    saveastro<Npp32f,3>(imagetotal.nppinputimage,outputfilename);
    saveastro<Npp32f,1>(imagetotal.exposuremap,"finalresultexp.jpg");
    return 0;
}