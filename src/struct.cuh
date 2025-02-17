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
    npp::ImageNPP<D,1> exposuremap;

    Nppop():maskimage(1,1),signalimage(1,1),correlationimage(1,1),exposuremap(1,1),maxpixelposition(-1,-1),maxcorrposition(-1,-1){}
    
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
    }
        /*I need to make sure this is changed to appropriate output*/
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

    


    template<typename D2, unsigned int N2>
    void resizeimage(npp::ImageNPP<D2,N2> &inputimage,unsigned int newimagewidth, unsigned int newimageheight,cv::Point_<int> imageposition){
        npp::ImageNPP<D2,N2> tempimage(newimagewidth,newimageheight);
        unsigned int offsetpositionx=(imageposition.x>=0)*imageposition.x+(imageposition.x<0)*0;
        unsigned int offsetpositiony=(imageposition.y>=0)*imageposition.y+(imageposition.y<0)*0;
        cudaError_t eResult;
        if (std::is_same<D2,Npp8u>::value){
           switch(N2){
                case 1:
                eResult=cudaMemcpy2D(tempimage.data(),tempimage.pitch(),inputimage.data(offsetpositionx,offsetpositiony),inputimage.pitch(),inputimage.width()*sizeof(Npp8u),inputimage.height(),cudaMemcpyDeviceToDevice);
                break;
                case 3:
               /*Need to multiply the width by 3 to include the number of channels*/
                eResult=cudaMemcpy2D(tempimage.data(),tempimage.pitch(),inputimage.data(offsetpositionx,offsetpositiony),inputimage.pitch(),inputimage.width()*sizeof(Npp8u)*3,inputimage.height(),cudaMemcpyDeviceToDevice);
                break;
            }
        }
        else if (std::is_same<D2,Npp32s>::value){
            switch(N2){
                 case 1:
                 eResult=cudaMemcpy2D(tempimage.data(),tempimage.pitch(),inputimage.data(offsetpositionx,offsetpositiony),inputimage.pitch(),inputimage.width()*sizeof(Npp32s),inputimage.height(),cudaMemcpyDeviceToDevice);
                 break;
                 case 3:
                /*Need to multiply the width by 3 to include the number of channels*/
                 eResult=cudaMemcpy2D(tempimage.data(),tempimage.pitch(),inputimage.data(offsetpositionx,offsetpositiony),inputimage.pitch(),inputimage.width()*sizeof(Npp32s)*3,inputimage.height(),cudaMemcpyDeviceToDevice);
                 break;
             }
         }
        /*Now we replace the original inputimage by the new tempimage*/
        inputimage=tempimage;
    }
    
    void shiftposition(cv::Point_<int> &position,cv::Point_<int> offsetposition){
        cv::Point_<int> newposition;
        newposition.x=position.x+(offsetposition.x>=0)*offsetposition.x;
        newposition.y=position.y+(offsetposition.y>=0)*offsetposition.y;
        position=newposition;
    }

    template<typename D2, unsigned int N2>
    void addimage(npp::ImageNPP<D2,N2> &inputimage,npp::ImageNPP<D2,N2> &extimage, cv::Point_<int> imageposition){
        unsigned int offsetpositionx=(imageposition.x<0)*std::abs(imageposition.x)+(imageposition.x>=0)*0;
        unsigned int offsetpositiony=(imageposition.y<0)*std::abs(imageposition.y)+(imageposition.y>=0)*0;
        npp::ImageNPP<D2,N2> tempimage(inputimage.width(),inputimage.height()); //get the same size properties of input data (previously updated through resize)
        nppiadd<N2>(inputimage,extimage,tempimage,offsetpositionx,offsetpositiony);    
        inputimage=tempimage;
    }


    private:
    template<unsigned int N2>
    void nppiadd(npp::ImageNPP<Npp32f,N2> &inputimage, npp::ImageNPP<Npp32f,N2> &extimage, npp::ImageNPP<Npp32f,N2> &tempimage,unsigned int offsetpositionx, unsigned int offsetpositiony){
        NppiSize AddROI={extimage.width(),extimage.height()};
        switch(N2){
            case 3:
                nppiAdd_32f_C3R(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(offsetpositionx,offsetpositiony), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI);
                break;
        }        

    }
    template<unsigned int N2>
    void nppiadd(npp::ImageNPP<Npp8u,N2> &inputimage, npp::ImageNPP<Npp8u,N2> &extimage, npp::ImageNPP<Npp8u,N2> &tempimage,unsigned int offsetpositionx, unsigned int offsetpositiony){
        NppiSize AddROI={extimage.width(),extimage.height()};
        switch(N2){
            case 1: 
                    nppiAdd_8u_C1RSfs(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(offsetpositionx,offsetpositiony), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI, 1);
                    break;
            case 3:
                    nppiAdd_8u_C3RSfs(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(offsetpositionx,offsetpositiony), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI, 1);
                    break;    
        }        

    }
    
};


template <typename D>
class astrojpg_rgb_ : public Nppop<D, 3> 
{
    public: 
        npp::ImageNPP<D,3> nppinputimage;
        npp::ImageNPP<D,1> nppgreyimage;
        
        
        
        astrojpg_rgb_(std::string filename):
        nppinputimage(1,1),nppgreyimage(1,1)
        { //Constructor to load image to NPP
            
            cv::Mat img=imread(filename,cv::IMREAD_COLOR);
            npp::ImageCPU_8u_C3 localhostimg((unsigned int)img.size().width,(unsigned int)img.size().height);
            for (size_t i=0;i<img.rows;i++)       
            {
                unsigned char *input = (unsigned char*)(img.data);
                memcpy(localhostimg.data(0,i),&input[img.step[0]*i],img.step1());
            }
        npp::ImageNPP_8u_C3 nppinputfile(localhostimg); //Creation of the NPP
        setinputNPP(nppinputfile,this->nppinputimage);
        
        /*I need to add the exposure initialisation*/
        addexposure(this->nppinputimage);
        }

        astrojpg_rgb_(unsigned int imagewidth, unsigned int imageheight):
        nppinputimage(1,1),nppgreyimage(1,1)
        {
            npp::ImageNPP<D,3> nppinputfile(imagewidth,imageheight);
            this->nppinputimage=nppinputfile;
            addexposure(nppinputfile);
        }

        void getgreyimage(){
        npp::ImageNPP<D,1> nppgreyfile(this->nppinputimage.width(),this->nppinputimage.height());
        rgbtogray(this->nppinputimage,nppgreyfile);
        this->nppgreyimage=nppgreyfile;
        }
        
        void resize(unsigned int newimagewidth, unsigned int newimageheight,cv::Point_<int> imageposition){
            if (this->nppinputimage.width()*this->nppinputimage.height()!=1) Nppop<D,3>::template resizeimage<D,3>(this->nppinputimage,newimagewidth,newimageheight,imageposition);
            if (this->nppgreyimage.width()*this->nppgreyimage.height()!=1) Nppop<D,3>::template resizeimage<D,1>(this->nppgreyimage,newimagewidth,newimageheight,imageposition);
            if (this->maskimage.width()*this->maskimage.height()!=1) Nppop<D,3>::template resizeimage<Npp32s,1>(this->maskimage,newimagewidth,newimageheight,imageposition);
            if (this->signalimage.width()*this->signalimage.height()!=1) Nppop<D,3>::template resizeimage<D,1>(this->signalimage,newimagewidth,newimageheight,imageposition);
            if (this->correlationimage.width()*this->correlationimage.height()!=1) Nppop<D,3>::template resizeimage<D,1>(this->correlationimage,newimagewidth,newimageheight,imageposition);
            if (this->exposuremap.width()*this->exposuremap.height()!=1) Nppop<D,3>::template resizeimage<D,1>(this->exposuremap,newimagewidth,newimageheight,imageposition);
            if(this->maxpixelposition.x>-1 && this->maxpixelposition.y>-1) Nppop<D,3>::shiftposition(this->maxpixelposition,imageposition);
            if(this->maxcorrposition.x>-1 && this->maxcorrposition.y>-1)  Nppop<D,3>::shiftposition(this->maxcorrposition,imageposition);
 
        }


        /*I need to add exposure based on the ROI from the reference and target image auto-correlation */
        void stackimage(astrojpg_rgb_<Npp8u> &addedimage,cv::Point_<int> targetmaxcorrposition){

            unsigned int newimagewidth=this->nppinputimage.width()+std::abs(targetmaxcorrposition.x);
            unsigned int newimageheight=this->nppinputimage.height()+std::abs(targetmaxcorrposition.y);
            resize(newimagewidth,newimageheight,targetmaxcorrposition);

            /*I then need to create a 32f value of the targetimage*/
            NppiSize convertimageROI={(int)addedimage.nppinputimage.width(),(int)addedimage.nppinputimage.height()};

            npp::ImageNPP<Npp32f,3> converted32faddedimage(addedimage.nppinputimage.size());
            nppiConvert_8u32f_C3R(addedimage.nppinputimage.data(), addedimage.nppinputimage.pitch(), converted32faddedimage.data(), converted32faddedimage.pitch(), convertimageROI);
            Nppop<D,3>::template addimage<D,3>(this->nppinputimage,converted32faddedimage,targetmaxcorrposition);
            addexposure(this->nppinputimage);
            /* Now add values from other image properties */
            npp::ImageNPP<Npp32f,1> converted32faddedgreyimage(addedimage.nppgreyimage.size());
            nppiConvert_8u32f_C1R(addedimage.nppgreyimage.data(), addedimage.nppgreyimage.pitch(),converted32faddedgreyimage.data(),converted32faddedgreyimage.pitch(),convertimageROI);
            Nppop<D,3>::template addimage<D,1>(this->nppgreyimage,converted32faddedgreyimage,targetmaxcorrposition);

        }
        /*I need to first use the previous exposure map and add it onto the new map with the specific offset*/

        //nppiAdd_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
        //} 
    private:
        void setinputNPP(npp::ImageNPP_8u_C3 &nppinputfile,npp::ImageNPP_32f_C3 &nppinputimage){
            npp::ImageNPP_32f_C3 nppconvertedfile(nppinputfile.width(),nppinputfile.height());
                /*We need to convert from 8u to 32s*/
            NppiSize convertsizeROI={nppinputfile.width(),nppinputfile.height()};
            nppiConvert_8u32f_C3R(nppinputfile.data(), nppinputfile.pitch(), nppconvertedfile.data(), nppconvertedfile.pitch(),convertsizeROI);
            nppinputimage=nppconvertedfile;
        }
        void setinputNPP(npp::ImageNPP_8u_C3 &nppinputfile,npp::ImageNPP_8u_C3 &nppinputimage){
            nppinputimage=nppinputfile;
        }

        void rgbtogray(npp::ImageNPP_8u_C3 &nppinputimage, npp::ImageNPP_8u_C1 &nppgreyfile){
            NppiSize osizeROI={(int)nppinputimage.width(),(int)nppinputimage.height()};
            nppiRGBToGray_8u_C3C1R(nppinputimage.data(),nppinputimage.pitch(), nppgreyfile.data(), nppgreyfile.pitch(),osizeROI);
        }

        void rgbtogray(npp::ImageNPP_32f_C3 &nppinputimage, npp::ImageNPP_32f_C1 &nppgreyfile){
            NppiSize osizeROI={(int)nppinputimage.width(),(int)nppinputimage.height()};
            nppiRGBToGray_32f_C3C1R(this->nppinputimage.data(), this->nppinputimage.pitch(), nppgreyfile.data(), nppgreyfile.pitch(),osizeROI);
        }

        void addexposure(npp::ImageNPP_8u_C3 &nppinputfile){
            npp::ImageNPP_8u_C1 tempexposuremap((int)nppinputfile.width(),(int)nppinputfile.height());
            NppiSize osizeROI={(int)nppinputfile.width(),(int)nppinputfile.height()};
            nppiAddC_8u_C1IRSfs(1,tempexposuremap.data(), (int)tempexposuremap.pitch(),osizeROI,1);
            this->exposuremap=tempexposuremap;
        }
        void addexposure(npp::ImageNPP_32f_C3 &nppinputfile){
            npp::ImageNPP_32f_C1 tempexposuremap((int)nppinputfile.width(),(int)nppinputfile.height());
            NppiSize osizeROI={(int)nppinputfile.width(),(int)nppinputfile.height()};
            nppiAddC_32f_C1IR(1,tempexposuremap.data(), (int)tempexposuremap.pitch(),osizeROI);
            this->exposuremap=tempexposuremap;
        }

};
