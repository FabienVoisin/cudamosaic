#ifndef ASTROIO_H
#define ASTROIO_H
#include "struct.cuh"
template<typename D, unsigned int N> inline void saveastro(npp::ImageNPP<D,N> &image,std::string outputfilename);
template<typename D, unsigned int N> inline void printastro(npp::ImageNPP<D,N> &image,cv::Point_<int> beginpos, cv::Point_<int> endpos);
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

template<typename D, unsigned int N> 
inline void printastro(npp::ImageNPP<D,N> &image,cv::Point_<int> beginpos, cv::Point_<int> endpos){
    int i,j;
    npp::ImageCPU<D,N,npp::ImageAllocatorCPU<D,N>> oHostdest(image.size());
    image.copyTo(oHostdest.data(),oHostdest.pitch());
    if(std::is_same<D,Npp8u>::value){
        if (N == 1){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_8UC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            std::cout<<"[";
            for(i = beginpos.y; i<endpos.y  ;i++){
                std::cout<<"i="<<i<<"[";
                for(j = beginpos.x;j < endpos.x ;j++){
                    float grey=outputimg.data[outputimg.cols * i + j ] ;
                  if(grey>80)  std::cout<<grey<<",";
                }
                std::cout<<"]"<<std::endl;
                //std::cout<<i<<","<<j<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
        else if (N == 3){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_32FC3,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            

            for(i= beginpos.y; i<endpos.y  ;i++){
                for(j = beginpos.x;j < endpos.x ;j+=3){
                    float b = outputimg.data[outputimg.cols * i + j ] ;
                    float g = outputimg.data[outputimg.cols * i + j + 1];
                    float r = outputimg.data[outputimg.cols * i + j + 2];
                    std::cout<<"r="<<r<<",g="<<g<<",b="<<b<<std::endl;
                }
            std::cout<<i<<","<<j<<std::endl;
            }
        }
    }
    else if(std::is_same<D,Npp32f>::value){
        if (N == 1){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_32FC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            std::cout<<"[";
            for(i = beginpos.y; i<endpos.y  ;i++){
                std::cout<<"i="<<i<<"[";
                for(j = beginpos.x;j < endpos.x ;j++){
                    float grey=outputimg.data[outputimg.cols * i + j ] ;
                  if(grey>80)  std::cout<<grey<<",";
                }
                std::cout<<"]"<<std::endl;
                //std::cout<<i<<","<<j<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
        else if (N == 3){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_32FC3,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            

            for(i= beginpos.y; i<endpos.y  ;i++){
                for(j = beginpos.x;j < endpos.x ;j+=3){
                    float b = outputimg.data[outputimg.cols * i + j ] ;
                    float g = outputimg.data[outputimg.cols * i + j + 1];
                    float r = outputimg.data[outputimg.cols * i + j + 2];
                    std::cout<<"r="<<r<<",g="<<g<<",b="<<b<<std::endl;
                }
            std::cout<<i<<","<<j<<std::endl;
            }
        }
    }
}
#endif