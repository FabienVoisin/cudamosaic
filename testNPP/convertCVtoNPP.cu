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
const Npp8u *pDstLine=inputfile.data();
unsigned int nDstPitch = inputfile.pitch();
for (size_t i=0;i<img.rows;i++){
    //memcpy(inputfile.data(0,i),&input[img.rows*i],img.step1()*sizeof(Npp8u));
    memcpy(inputfile.data(0,i),&input[img.step[0]*i],img.step1());
    pDstLine+=nDstPitch;
}

for(int i=0;i<inputfile.height();i++){
    for(int j=0;j<inputfile.width();j++){
        inputfile.pixels(j,i)[0].x+=10;
        inputfile.pixels(j,i)[1].y+=0;
        inputfile.pixels(j,i)[2].z+=50;
        
    }

}

// This will create a specific pitch 
//std::cout<<"value15="<<(unsigned char)inputfile.pixels(1,5)[0].x<<std::endl;
npp::ImageCPU_8u_C3 osrchost; // first input data;

int linestep;
std::cout<<"test"<<std::endl;



int devicewidth=linestep;
int deviceheight=img.size().height;
NppiSize osizeROI={devicewidth,deviceheight};


const Npp8u values[3]={1,2,3};


cv::Mat outputimg((int)inputfile.height(),(int)inputfile.width(),CV_8UC3,(void *)inputfile.data(),(size_t)inputfile.pitch());
std::cout<<outputimg.step1()<<":"<<outputimg.size().width*outputimg.channels()<<","<<outputimg.step[1]<<","<<outputimg.step[2]<<std::endl;

;
unsigned char *output = (unsigned char*)(outputimg.data);
/*for (size_t i=0;i<outputimg.rows;i++){
    memcpy(&output[outputimg.cols*i],inputfile.data(0,i),inputfile.pitch());
}*/

bool check=imwrite("newimg2.jpg",outputimg);
std::cout<<outputimg.total()<<std::endl;

for(int i = 0;i < outputimg.rows  ;i++){
    for(int j = 0;j < outputimg.cols ;j+=3){
        b = output[outputimg.cols * i + j ] ;
        g = output[outputimg.cols * i + j + 1];
        r = output[outputimg.cols * i + j + 2];
        //if(b>16) std::cout<<"r="<<r<<",g="<<g<<",b="<<b<<std::endl;
    }
   // std::cout<<i<<","<<j<<std::endl;
}
//nppiAddC_8u_C3IRSfs(values,inputfile,linestep,osizeROI, 0);
/*Set outcome to host image*/
//npp::ImageCPU_8u_C3 ohostdest(&inputfile.size());

/*odevgrey.copyTo(ohostdest.data(),oHosdest.pitch());

SaveImage("testgrey.jpg",ohostdest);*/





return 0;

}