#ifndef ASTROIO_H
#define ASTROIO_H
#include "struct.cuh"
template<typename D, unsigned int N> inline void saveastro(npp::ImageNPP<D,N> &image,std::string outputfilename);
template<typename D, unsigned int N> 
inline void saveastro(npp::ImageNPP<D,N> &image,std::string outputfilename){
    npp::ImageCPU<D,N,npp::ImageAllocatorCPU<D,N>> oHostdest(image.size());
    image.copyTo(oHostdest.data(),oHostdest.pitch());
    if (std::is_same<D,Npp8u>::value){
        if (N == 1){
        cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_8UC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
        imwrite(outputfilename,outputimg);
        }
        else if (N==3){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_8UC3,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            imwrite(outputfilename,outputimg);
        }
    }
    else if(std::is_same<D,Npp32s>::value){
        if (N == 1){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_32SC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            imwrite(outputfilename,outputimg);
            }
        }
    else if(std::is_same<D,Npp32f>::value){
        if (N == 1){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_32FC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            imwrite(outputfilename,outputimg);
            }
        else if (N == 3){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_32FC3,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            imwrite(outputfilename,outputimg);
        }
    }
}
#endif