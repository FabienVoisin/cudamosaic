#pragma once
#include <iostream>
#include <Common/UtilNPP/ImagesCPU.h>
#include <Common/UtilNPP/ImagesNPP.h>
#include <Common/UtilNPP/Exceptions.h>
#include <npp.h>
//#include <nppdefs.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>



template <typename D, unsigned int N>
class Nppop{
    public :
    cv::Point_<int> maxpixelposition;
    cv::Point_<int> maxcorrposition;
    npp::ImageNPP_32s_C1 maskimage;
    npp::ImageNPP<D,1> signalimage;
    npp::ImageNPP<D,1> correlationimage;

    Nppop(){}
    
    void getsignalimage(npp::ImageNPP<D,1> &nppgreyimage,const D threshold){
            
        npp::ImageNPP_8u_C1 nppdestfile(nppgreyimage.width(),nppgreyimage.height());
        NppiSize osizeROI={nppgreyimage.width(),nppgreyimage.height()};
        nppiCompareC_8u_C1R(nppgreyimage.data(),(int)nppgreyimage.pitch(), threshold, nppdestfile.data(),(int)nppdestfile.pitch(),osizeROI,NPP_CMP_GREATER_EQ);
        this->signalimage=nppdestfile;
    }

    void getmaxpixel(npp::ImageNPP<D,1> &image,cv::Point_<int> &maxpixelposition, D *maxbuffer){
        int *positionx, *positiony; //device values
        //host values to return
        Npp8u *nppmaxvalues;
        NppiSize osizeROI={(int)image.width(),(int)image.height()};
        Npp8u hostmaxvalues;
        cudaMalloc((void**)&nppmaxvalues, sizeof(D) * 1);
        cudaMalloc((void**)&positionx,sizeof(int));
        cudaMalloc((void**)&positiony,sizeof(int));
        if (std::is_same<D,Npp8u>::value){
            nppiMaxIndx_8u_C1R(image.data(), image.pitch(),osizeROI,maxbuffer, nppmaxvalues, positionx, positiony);
            
        }
        
        cudaMemcpy(&hostmaxvalues,nppmaxvalues,sizeof(D)*1,cudaMemcpyDeviceToHost);
        cudaMemcpy(&(maxpixelposition.x),positionx,sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(&(maxpixelposition.y),positiony,sizeof(int),cudaMemcpyDeviceToHost);        
        std::cout<<(unsigned int)hostmaxvalues<<","<<maxpixelposition.x<<","<<maxpixelposition.y<<std::endl;
        cudaFree(nppmaxvalues);
        cudaFree(positionx);
        cudaFree(positiony);
    }

    void createROIdata(int squaresize){
        NppiSize osizeROI={squaresize,squaresize}; //Setup the osizeROI

        int initpositionx=this->maxpixelposition.x-squaresize/2+1; 
        int initpositiony=this->maxpixelposition.y-squaresize/2+1;
        std::cout<<initpositionx<<","<<initpositiony<<std::endl;
        npp::ImageNPP_8u_C1 outputROIimage(squaresize,squaresize);
        npp::ImageNPP_8u_C1 outputmirrorimage(squaresize,squaresize);
        npp::ImageNPP_32s_C1 outputfinalimage(squaresize,squaresize);
        cudaError_t eResult;

        eResult=cudaMemcpy2D(outputROIimage.data(),outputROIimage.pitch(),this->signalimage.data(initpositionx,initpositiony),this->signalimage.pitch(),outputROIimage.width()*sizeof(Npp8u),outputROIimage.height(),cudaMemcpyDeviceToDevice);
        NPP_ASSERT(cudaSuccess == eResult);
        /*nppiMirror will flip the image so that the last values become the first, necessary for the convolution*/
        nppiMirror_8u_C1R(outputROIimage.data(),(int)outputROIimage.pitch(), outputmirrorimage.data(),(int)outputmirrorimage.pitch(), osizeROI, NPP_BOTH_AXIS);
        /*Finally we need to convert to 32s*/
        nppiConvert_8u32s_C1R(outputmirrorimage.data(),outputmirrorimage.pitch(),outputfinalimage.data(),outputfinalimage.pitch(),osizeROI);
        this->maskimage=outputfinalimage;
        //nppiFree(outputROIimage.data());
        //nppiFree(outputmirrorimage.data());
    }

    void Correlationimage(npp::ImageNPP_32s_C1 &maskimage,Npp8u *sumbuffer){
        NppiSize osizeROI={(int)this->signalimage.width(),(int)this->signalimage.height()};
        NppiSize omaskROI={(int)maskimage.width(),(int)maskimage.height()};
        NppiSize okernelSize={(int)maskimage.width(),(int)maskimage.height()};
        NppiPoint oAnchor={(int)(maskimage.width()/2),(int)(maskimage.height()/2)};
        std::cout<<"testsignal"<<std::endl;
        npp::ImageNPP_32f_C1 maskimagesum(maskimage.width(),maskimage.height());
      
        npp::ImageNPP_8u_C1 outputimage(this->signalimage.width(),this->signalimage.height());
        NppStatus status=nppiConvert_32s32f_C1R(maskimage.data(),(int) maskimage.pitch(), maskimagesum.data(),(int) maskimagesum.pitch(), omaskROI);
        NPP_ASSERT(NPP_SUCCESS  == status);

        Npp64f *nsum;
        Npp64f hostnsum;
        cudaMalloc((void **)&nsum,sizeof(Npp64f));
        nppiSum_32f_C1R(maskimagesum.data(),maskimagesum.pitch(),omaskROI, sumbuffer, nsum);  
        cudaDeviceSynchronize();
        cudaMemcpy(&hostnsum,nsum,sizeof(Npp64f),cudaMemcpyDeviceToHost);
        Npp32s ndivisor=(Npp32s) hostnsum;
        std::cout<<"hostnsum="<<ndivisor<<std::endl; 
        status=nppiFilter_8u_C1R(this->signalimage.data(), this->signalimage.pitch(), outputimage.data(), outputimage.pitch(),osizeROI, maskimage.data(), okernelSize, oAnchor,  ndivisor);
        NPP_ASSERT(NPP_SUCCESS  == status);
        this->correlationimage=outputimage;
        //nppiFree((void*)maskimagesum.data());
    }


    /*Addimage(){
        nppiAdd_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    }*/
    

};

class astrojpg_8u_rgb : public Nppop<Npp8u, 3> 
{
    public: 
        npp::ImageNPP_8u_C3 nppinputimage;
        npp::ImageNPP_8u_C1 nppgreyimage;
        
        
        
        astrojpg_8u_rgb(std::string filename){ //Constructor to load image to NPP
            
            cv::Mat img=imread(filename,cv::IMREAD_COLOR);
            npp::ImageCPU_8u_C3 localhostimg((unsigned int)img.size().width,(unsigned int)img.size().height);
            for (size_t i=0;i<img.rows;i++){
                unsigned char *input = (unsigned char*)(img.data);
                memcpy(localhostimg.data(0,i),&input[img.step[0]*i],img.step1());
            }
        npp::ImageNPP_8u_C3 nppinputfile(localhostimg); //Creation of the NPP
        this->nppinputimage=nppinputfile;
        }

        astrojpg_8u_rgb(unsigned int imagewidth, unsigned int imageheight){
            npp::ImageNPP_8u_C3 nppinputfile(imagewidth,imageheight);
            this->nppinputimage=nppinputfile;
        }

        void getgreyimage(){
        npp::ImageNPP_8u_C1 nppgreyfile(this->nppinputimage.width(),this->nppinputimage.height());
        NppiSize osizeROI={(int)this->nppinputimage.width(),(int)this->nppinputimage.height()};
        nppiRGBToGray_8u_C3C1R(this->nppinputimage.data(), this->nppinputimage.pitch(), nppgreyfile.data(), nppgreyfile.pitch(),osizeROI);
        this->nppgreyimage=nppgreyfile;
        }
        

         
};
