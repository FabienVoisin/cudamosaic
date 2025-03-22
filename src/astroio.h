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
    else if(std::is_same<D,Npp16u>::value){
        if (N == 1){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_16UC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            imwrite(outputfilename,outputimg);
            }
        if (N == 3){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_16UC3,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
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
                    unsigned int  grey=outputimg.data[outputimg.cols * i + j ] ;
                    std::cout<<grey<<",";
                }
                std::cout<<"]"<<std::endl;
                //std::cout<<i<<","<<j<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
        else if (N == 3){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_8UC3,(void *)oHostdest.data(),(size_t)oHostdest.pitch());


            for(i=beginpos.y; i<endpos.y  ;i++){
                for(j = beginpos.x;j < endpos.x ;j+=3){
                    unsigned int b = outputimg.data[outputimg.cols * i + j ] ;
                    unsigned int g = outputimg.data[outputimg.cols * i + j + 1];
                    unsigned int r = outputimg.data[outputimg.cols * i + j + 2];
                    if (b>0) std::cout<<"("<<j<<","<<i<<"):r="<<r<<",g="<<g<<",b="<<b<<std::endl;
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
                    //_Float32 grey=(_Float32)outputimg.data[outputimg.cols * i + j ] ;
                    _Float32 grey=(_Float32)oHostdest.pixels(j,i)[0].x;
                    if(grey>0)  std::cout<<grey<<",";
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
                    _Float32 b = (_Float32)oHostdest.pixels(j,i)[0].x ;
                    _Float32 g = (_Float32)oHostdest.pixels(j,i)[1].x ;
                    _Float32 r = (_Float32)oHostdest.pixels(j,i)[2].x ;
                    if (b>0) std::cout<<"("<<j<<","<<i<<"):r="<<r<<",g="<<g<<",b="<<b<<std::endl;
                }
            std::cout<<i<<","<<j<<std::endl;
            }
        }
    }

    else if(std::is_same<D,Npp16u>::value){
        if (N == 1){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_16UC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
            std::cout<<"[";
            for(i = beginpos.y; i<endpos.y  ;i++){
                std::cout<<"i="<<i<<"[";
                for(j = beginpos.x;j < endpos.x ;j++){
                    uint16_t grey=(uint16_t)oHostdest.pixels(j,i)[0].x;
                   if (grey>0) std::cout<<grey<<",";
                }
                std::cout<<"]"<<std::endl;
                //std::cout<<i<<","<<j<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
        else if (N == 3){
            cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_16UC3,(void *)oHostdest.data(),(size_t)oHostdest.pitch());

            for(i= beginpos.y; i<endpos.y  ;i++){
                for(j = beginpos.x;j < endpos.x ;j+=3){
                    uint16_t b = oHostdest.pixels(j,i)[0].x;
                    uint16_t g = oHostdest.pixels(j,i)[1].x;
                    uint16_t r = oHostdest.pixels(j,i)[2].x;
                    std::cout<<"r="<<r<<",g="<<g<<",b="<<b<<std::endl;
                }
            std::cout<<i<<","<<j<<std::endl;
            }
        }
    }
}
#endif