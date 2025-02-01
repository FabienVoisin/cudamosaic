#pragma once




    astrojpg_8u_rgb::astrojpg_8u_rgb(std::string filename){ //Constructor to load image to NPP
            
        cv::Mat img=imread(filename,cv::IMREAD_COLOR);
        npp::ImageCPU_8u_C3 localhostimg((unsigned int)img.size().width,(unsigned int)img.size().height);
        for (size_t i=0;i<img.rows;i++){
            unsigned char *input = (unsigned char*)(img.data);
            memcpy(localhostimg.data(0,i),&input[img.step[0]*i],img.step1());
        }
    npp::ImageNPP_8u_C3 nppinputfile(localhostimg); //Creation of the NPP
    this->nppinputimage=nppinputfile;
    }

    astrojpg_8u_rgb::astrojpg_8u_rgb(unsigned int imagewidth, unsigned int imageheight){
        npp::ImageNPP_8u_C3 nppinputfile(imagewidth,imageheight);
        this->nppinputimage=nppinputfile;
    }

    void astrojpg_8u_rgb::getgreyimage(){
    npp::ImageNPP_8u_C1 nppgreyfile(this->nppinputimage.width(),this->nppinputimage.height());
    NppiSize osizeROI={(int)this->nppinputimage.width(),(int)this->nppinputimage.height()};
    nppiRGBToGray_8u_C3C1R(this->nppinputimage.data(), this->nppinputimage.pitch(), nppgreyfile.data(), nppgreyfile.pitch(),osizeROI);
    this->nppgreyimage=nppgreyfile;
    }
        
    void astrojpg_8u_rgb::getsignalimage(const Npp8u threshold){
            
        npp::ImageNPP_8u_C1 nppdestfile(this->nppgreyimage.width(),this->nppgreyimage.height());
        NppiSize osizeROI={this->nppgreyimage.width(),this->nppgreyimage.height()};
        nppiCompareC_8u_C1R(this->nppgreyimage.data(),(int)this->nppgreyimage.pitch(), threshold, nppdestfile.data(),(int)nppdestfile.pitch(),osizeROI,NPP_CMP_GREATER_EQ);
        this->signalimage=nppdestfile;
    }


template <typename D, unsigned int N>
Nppop<D,N>::Nppop(){}
template <typename D, unsigned int N>
void Nppop<D,N>::getmaxpixel(npp::ImageNPP<D,1> &image,D *maxbuffer){
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
        cudaMemcpy(&(this->maxpixelposition.x),positionx,sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(&(this->maxpixelposition.y),positiony,sizeof(int),cudaMemcpyDeviceToHost);        
    
        cudaFree(nppmaxvalues);
        cudaFree(positionx);
        cudaFree(positiony);
    }

#include "struct.tpp"