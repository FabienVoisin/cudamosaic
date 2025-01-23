/*Author : Fabien Voisin*/
#include <iostream>
#include <Common/UtilNPP/ImagesCPU.h>
#include <Common/UtilNPP/ImagesNPP.h>
#include <nppdefs.h>
//#include "/mnt/sdd/cuda-samples/Common/UtilNPP/ImageIO.h"
//#include <opencv2\opencv.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <vector>

int main(int argc, char **argv){
cv::Mat img = imread("data/20241209_230358.jpg", cv::IMREAD_COLOR);

int imagewidth=img.size().width;
int imageheight=img.size().height;
//int bytes=img.imageSize;
//ROWS and COLUMNS ARE NOT IN BYTES 
//RGB is 8u thus 1 byte per CHANNELS

std::cout<<img.total()<<std::endl;
std::cout<<img.step1()<<":"<<img.size().width*img.channels()<<","<<img.step[1]<<","<<img.step[2]<<std::endl;
std::cout<<img.rows<<","<<img.cols<<std::endl;
//unsigned char *dSrc, *dDst;
//dSrc=img.data;
unsigned char *input = (unsigned char*)(img.data);
int i,j,r,g,b;
    /*for(int i = 0;i < img.rows  ;i++){
			for(int j = 0;j < img.cols*3 ;j+=3){
                b = img.data[img.step[0] * i + j*img.step[1] ] ;
                g = img.data[img.step[0] * i + j*img.step[1] + 1];
                r = img.data[img.step[0] * i + j*img.step[1] + 2];
                //img.at<cv::Vec3b>(i,j)[0];
                if(i>1600 && i< 1700 && b> 100) std::cout<<"("<<i<<","<<j/3<<"):"<<"r="<<r<<",g="<<g<<",b="<<b<<std::endl;
                //if(i>1600 && i< 1700) std::cout<<"("<<i<<","<<j/3<<"):"<<"b="<<img.at<cv::Vec3b>(i,j)[0]<<",g="<<img.at<cv::Vec3b>(i,j)[1]<<",b="<<img.at<cv::Vec3b>(i,j)[2]<<std::endl;
            }
        }*/

//memcpy( &dst[dstIdx], &src[srcIdx], numElementsToCopy * sizeof( Element ) );
std::cout<<"end"<<std::endl;
//The next step is to understand the data structure of the image on a per channel basis.
//std::cout<<dSrc[8]<<std::endl;

//cudaMalloc<unsigned char>(&dSrc,bytes);
npp::ImageCPU_8u_C3 inputfile((unsigned int)img.size().width,(unsigned int)img.size().height);
for (size_t i=0;i<img.rows;i++){
    //memcpy(inputfile.data(0,i),&input[img.rows*i],img.step1()*sizeof(Npp8u));
    memcpy(inputfile.data(0,i),&input[img.step[0]*i],img.step1());
}


/*for(int i=0;i<inputfile.height();i++){
    for(int j=0;j<inputfile.width();j++){
        inputfile.pixels(j,i)[0].x+=0;
        inputfile.pixels(j,i)[1].y+=0;
        inputfile.pixels(j,i)[2].z+=0;
        
    }

}*/


int linestep;
std::cout<<"test"<<std::endl;

/*Now create a NPP image output*/
npp::ImageNPP_8u_C3 nppinputfile(inputfile);
/*Create a grey image*/

npp::ImageNPP_8u_C1 nppgreyfile(nppinputfile.width(),nppinputfile.height());



std::cout<<"newtest"<<std::endl;


int devicewidth=linestep;
int deviceheight=img.size().height;

NppiSize osizeROI;
osizeROI.width=(int)nppinputfile.width();
osizeROI.height=(int)nppinputfile.height();

std::cout<<"nppinputfile:width="<<(int)nppinputfile.width()<<",pitch="<<(int)nppinputfile.pitch()<<std::endl;
nppiRGBToGray_8u_C3C1R(nppinputfile.data(), nppinputfile.pitch(), nppgreyfile.data(), nppgreyfile.pitch(),osizeROI);

Npp8u* maxbuffer;
size_t  maxbufferhostsize;
int *positionx;
int *positiony;
cv::Point_<int> hostmaxpixelposition;
Npp8u *nppmaxvalues;

nppiMaxIndxGetBufferHostSize_8u_C1R(osizeROI, &maxbufferhostsize);
cudaMalloc((void**)&maxbuffer,maxbufferhostsize);
cudaMalloc((void**)&nppmaxvalues, sizeof(Npp8u) * 1);
cudaMalloc((void**)&positionx,sizeof(int));
cudaMalloc((void**)&positiony,sizeof(int));

nppiMaxIndx_8u_C1R(nppgreyfile.data(), nppgreyfile.pitch(),osizeROI,maxbuffer, nppmaxvalues, positionx, positiony);
Npp8u hostmaxvalues;
cudaMemcpy(&hostmaxvalues,nppmaxvalues,sizeof(Npp8u) * 1,cudaMemcpyDeviceToHost);
cudaMemcpy(&(hostmaxpixelposition.x),positionx,sizeof(int),cudaMemcpyDeviceToHost);
cudaMemcpy(&(hostmaxpixelposition.y),positiony,sizeof(int),cudaMemcpyDeviceToHost);
std::cout<<"maxpixel(x="<<hostmaxpixelposition.x<<",y="<<hostmaxpixelposition.y<<")="<<(int)hostmaxvalues<<std::endl;

npp::ImageNPP_8u_C1 nppdestfile(nppinputfile.width(),nppinputfile.height());
const Npp8u threshold=10;
std::cout<<"newtest2"<<std::endl;
//nppiThreshold_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u rThresholds[3], NppCmpOp eComparisonOperation);
//NppStatus status=nppiThreshold_8u_C3IR(nppthreshold.data(),nppthreshold.pitch(), osizeROI, threshold, NPP_CMP_GREATER_EQ);
nppiCompareC_8u_C1R(nppgreyfile.data(),(int)nppgreyfile.pitch(), threshold, nppdestfile.data(),(int)nppdestfile.pitch(),osizeROI,NPP_CMP_GREATER_EQ);
//std::cout<<"status="<<status<<std::endl;

npp::ImageCPU_8u_C1 oHostdest(nppdestfile.size());
/*Npw we need to get the value back to the inputfile */
nppdestfile.copyTo(oHostdest.data(),oHostdest.pitch());

//nppinputfile.copyTo(oHostdest.data(),oHostdest.pitch());


std::cout<<"newtest4"<<std::endl;

cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_8UC1,(void *)oHostdest.data(),(size_t)oHostdest.pitch());
//cv::Mat outputimg((int)oHostdest.height(),(int)oHostdest.width(),CV_8UC1);
std::cout<<outputimg.step1()<<":"<<oHostdest.pitch()<<","<<outputimg.step[0]<<","<<std::endl;

;
unsigned char *output = (unsigned char*)(outputimg.data);
/*for (size_t i=0;i<outputimg.rows;i++){
    //std::cout<<"iteration="<<i<<std::endl;
    memcpy(&output[img.step[0]*i],oHostdest.data(0,i),oHostdest.pitch());
    //memcpy(&output[img.step[0]*i],oHostdest.data(0,i),outputimg.step[0]);
}*/
std::cout<<"final"<<std::endl;
bool check=imwrite("newimg4.jpg",outputimg);
std::cout<<outputimg.total()<<std::endl;

/*for(int i = 0;i < outputimg.rows  ;i++){
    for(int j = 0;j < outputimg.cols*3 ;j+=3){
        b = output[img.step[0] * i + j*img.step[1] ] ;
        g = output[img.step[0] * i + j*img.step[1] + 1];
        r = output[img.step[0] * i + j*img.step[1] + 2];
        std::cout<<"r="<<r<<",g="<<g<<",b="<<b<<std::endl;
    }
   // std::cout<<i<<","<<j<<std::endl;
}*/
//nppiAddC_8u_C3IRSfs(values,inputfile,linestep,osizeROI, 0);
/*Set outcome to host image*/
//npp::ImageCPU_8u_C3 ohostdest(&inputfile.size());

/*odevgrey.copyTo(ohostdest.data(),oHosdest.pitch());

SaveImage("testgrey.jpg",ohostdest);*/





return 0;

}