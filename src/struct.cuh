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
    npp::ImageNPP<Npp8u,1> signalimage;
    npp::ImageNPP<Npp8u,1> correlationimage;
    npp::ImageNPP<D,3> exposuremap;

    Nppop():maskimage(1,1),signalimage(1,1),correlationimage(1,1),exposuremap(1,1),maxpixelposition(-1,-1),maxcorrposition(-1,-1){}
    
    void getsignalimage(npp::ImageNPP<D,1> &nppgreyimage,const D threshold){
            
        npp::ImageNPP_8u_C1 nppdestfile(nppgreyimage.width(),nppgreyimage.height());
        NppiSize osizeROI={nppgreyimage.width(),nppgreyimage.height()};
        nppicompare(nppgreyimage,nppdestfile,threshold,osizeROI);
        //nppiCompareC_8u_C1R(nppgreyimage.data(),(int)nppgreyimage.pitch(), threshold, nppdestfile.data(),(int)nppdestfile.pitch(),osizeROI,NPP_CMP_GREATER_EQ);
        this->signalimage=nppdestfile;
    }
    template <typename D2>
    void getmaxpixel(npp::ImageNPP<D2,1> &image,cv::Point_<int> &maxpixelposition, Npp8u *maxbuffer){
        int *positionx, *positiony; //device values
        cudaError_t eresult;
        //host values to return
        D2 *nppmaxvalues;
        NppiSize osizeROI={(int)image.width(),(int)image.height()};
        D2 hostmaxvalues;
        eresult=cudaMalloc((void**)&nppmaxvalues, sizeof(D2) * 1);
        eresult=cudaMalloc((void**)&positionx,sizeof(int));
        eresult=cudaMalloc((void**)&positiony,sizeof(int));
        nppimaxidx(image,osizeROI,maxbuffer,nppmaxvalues,positionx,positiony);
        
        eresult=cudaMemcpy(&hostmaxvalues,nppmaxvalues,sizeof(D2)*1,cudaMemcpyDeviceToHost);
        eresult=cudaMemcpy(&(maxpixelposition.x),positionx,sizeof(int),cudaMemcpyDeviceToHost);
        eresult=cudaMemcpy(&(maxpixelposition.y),positiony,sizeof(int),cudaMemcpyDeviceToHost);      
        
        cudaFree(nppmaxvalues);
        cudaFree(positionx);
        cudaFree(positiony);
    }


    void createROIdata(int squaresize){
        NppiSize osizeROI={squaresize,squaresize}; //Setup the osizeROI
        NppStatus status;
        int initpositionx=this->maxpixelposition.x-squaresize/2+1; 
        int initpositiony=this->maxpixelposition.y-squaresize/2+1;
        
        npp::ImageNPP_8u_C1 outputROIimage(squaresize,squaresize);
        npp::ImageNPP_8u_C1 outputmirrorimage(squaresize,squaresize);
        npp::ImageNPP_32s_C1 outputfinalimage(squaresize,squaresize);
        cudaError_t eResult;

        eResult=cudaMemcpy2D(outputROIimage.data(),outputROIimage.pitch(),this->signalimage.data(initpositionx,initpositiony),this->signalimage.pitch(),outputROIimage.width()*sizeof(Npp8u),outputROIimage.height(),cudaMemcpyDeviceToDevice);
        
        //NPP_ASSERT(cudaSuccess == eResult);
        /*nppiMirror will flip the image so that the last values become the first, necessary for the convolution*/
        status=nppiMirror_8u_C1R(outputROIimage.data(),(int)outputROIimage.pitch(), outputmirrorimage.data(),(int)outputmirrorimage.pitch(), osizeROI, NPP_BOTH_AXIS);
        cudaDeviceSynchronize();
        
        /*Finally we need to convert to 32s*/
        status=nppiConvert_8u32s_C1R(outputmirrorimage.data(),outputmirrorimage.pitch(),outputfinalimage.data(),outputfinalimage.pitch(),osizeROI);
        
        this->maskimage=outputfinalimage;
    }
        /*I need to make sure this is changed to appropriate output*/
    void Correlationimage(npp::ImageNPP_32s_C1 &maskimage,Npp8u *sumbuffer){
        cudaError_t err;
        NppiSize osizeROI={(int)this->signalimage.width(),(int)this->signalimage.height()};
        NppiSize omaskROI={(int)maskimage.width(),(int)maskimage.height()};
        NppiSize okernelSize={(int)maskimage.width(),(int)maskimage.height()};
        NppiPoint oAnchor={(int)(maskimage.width()/2),(int)(maskimage.height()/2)};
        
        npp::ImageNPP_32f_C1 maskimagesum(maskimage.width(),maskimage.height());
        npp::ImageNPP_8u_C1 outputimage(this->signalimage.width(),this->signalimage.height());
        NppStatus status=nppiConvert_32s32f_C1R(maskimage.data(),(int) maskimage.pitch(), maskimagesum.data(),(int) maskimagesum.pitch(), omaskROI);
        //NPP_ASSERT(NPP_SUCCESS  == status);

        Npp64f *nsum;
        Npp64f hostnsum;
        cudaMalloc((void **)&nsum,sizeof(Npp64f));
        
        status=nppiSum_32f_C1R(maskimagesum.data(),maskimagesum.pitch(),omaskROI, sumbuffer, nsum); 
        
        cudaDeviceSynchronize();
        err=cudaMemcpy(&hostnsum,nsum,sizeof(Npp64f),cudaMemcpyDeviceToHost);
        
        Npp32s ndivisor=(Npp32s) hostnsum;
        
        
        nppifilter(signalimage,outputimage,osizeROI,maskimage,okernelSize,oAnchor,ndivisor);
        //status=nppiFilter_8u_C1R(this->signalimage.data(), this->signalimage.pitch(), outputimage.data(), outputimage.pitch(),osizeROI, maskimage.data(), okernelSize, oAnchor,  ndivisor);
        //NPP_ASSERT(NPP_SUCCESS  == status);
        cudaDeviceSynchronize();
        //td::cout<<"outputimage:width="<<outputimage.width()<<",height="<<outputimage.height()<<std::endl;
        this->correlationimage=outputimage;
        //nppiFree((void*)maskimagesum.data());
    }

    


    template<typename D2, unsigned int N2>
    void resizeimage(npp::ImageNPP<D2,N2> &inputimage,unsigned int newimagewidth, unsigned int newimageheight,cv::Point_<int> imageposition){
        
        npp::ImageNPP<D2,N2> tempimage(newimagewidth,newimageheight);
        unsigned int offsetpositionx=(imageposition.x>=0)*imageposition.x+(imageposition.x<0)*0;
        unsigned int offsetpositiony=(imageposition.y>=0)*imageposition.y+(imageposition.x<0)*0;
        
        
        cudaError_t eResult;
        if (std::is_same<D2,Npp8u>::value){
           switch(N2){
                case 1:
                eResult=cudaMemcpy2D(tempimage.data(offsetpositionx,offsetpositiony),tempimage.pitch(),inputimage.data(),inputimage.pitch(),inputimage.width()*sizeof(Npp8u),inputimage.height(),cudaMemcpyDeviceToDevice);
                break;
                case 3:
               /*Need to multiply the width by 3 to include the number of channels*/
                eResult=cudaMemcpy2D(tempimage.data(offsetpositionx,offsetpositiony),tempimage.pitch(),inputimage.data(),inputimage.pitch(),inputimage.width()*sizeof(Npp8u)*3,inputimage.height(),cudaMemcpyDeviceToDevice);
                break;
            }
        }
        else if (std::is_same<D2,Npp16u>::value){
            switch(N2){
                 case 1:
                 eResult=cudaMemcpy2D(tempimage.data(offsetpositionx,offsetpositiony),tempimage.pitch(),inputimage.data(),inputimage.pitch(),inputimage.width()*sizeof(Npp16u),inputimage.height(),cudaMemcpyDeviceToDevice); 
                 break;
                 case 3:
                /*Need to multiply the width by 3 to include the number of channels*/
                 eResult=cudaMemcpy2D(tempimage.data(offsetpositionx,offsetpositiony),tempimage.pitch(),inputimage.data(),inputimage.pitch(),inputimage.width()*sizeof(Npp16u)*3,inputimage.height(),cudaMemcpyDeviceToDevice);
                 break;
             }
        }
        else if (std::is_same<D2,Npp32f>::value){
            switch(N2){
                 case 1:
                 eResult=cudaMemcpy2D(tempimage.data(offsetpositionx,offsetpositiony),tempimage.pitch(),inputimage.data(),inputimage.pitch(),inputimage.width()*sizeof(Npp32f),inputimage.height(),cudaMemcpyDeviceToDevice);
                 break;
                 case 3:
                /*Need to multiply the width by 3 to include the number of channels*/
                 eResult=cudaMemcpy2D(tempimage.data(offsetpositionx,offsetpositiony),tempimage.pitch(),inputimage.data(),inputimage.pitch(),inputimage.width()*sizeof(Npp32f)*3,inputimage.height(),cudaMemcpyDeviceToDevice);
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

    template<typename D2, unsigned int N2>
    void normaliseimage(npp::ImageNPP<D2,N2> &inputimage){
        /*This function will divice the input image with the exposure map */
        npp::ImageNPP<D2,N2> outputimage(inputimage.width(),inputimage.height()); 
        nppidivide<N2>(inputimage, this->exposuremap,outputimage);
        inputimage=outputimage;
    }

    private:
    template<unsigned int N2>
    void nppiadd(npp::ImageNPP<Npp32f,N2> &inputimage, npp::ImageNPP<Npp32f,N2> &extimage, npp::ImageNPP<Npp32f,N2> &tempimage,unsigned int offsetpositionx, unsigned int offsetpositiony){
        NppiSize AddROI={extimage.width(),extimage.height()};
        NppStatus status;
        switch(N2){
            case 3:
                status=nppiAdd_32f_C3R(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI);
                
                break;
            case 1:
                status=nppiAdd_32f_C1R(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI);
                
                break;
        }        

    }
    template<unsigned int N2>
    void nppiadd(npp::ImageNPP<Npp8u,N2> &inputimage, npp::ImageNPP<Npp8u,N2> &extimage, npp::ImageNPP<Npp8u,N2> &tempimage,unsigned int offsetpositionx, unsigned int offsetpositiony){
        NppiSize AddROI={extimage.width(),extimage.height()};
        NppStatus status;
        switch(N2){
            case 1: 
                    nppiAdd_8u_C1RSfs(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI, 0);
                    break;
            case 3:
                    nppiAdd_8u_C3RSfs(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI, 0);
                    break;    
        }        

    
    }
    template<unsigned int N2>
    void nppiadd(npp::ImageNPP<Npp16u,N2> &inputimage, npp::ImageNPP<Npp16u,N2> &extimage, npp::ImageNPP<Npp16u,N2> &tempimage,unsigned int offsetpositionx, unsigned int offsetpositiony){
        NppiSize AddROI={extimage.width(),extimage.height()};
        NppStatus status;
        switch(N2){
            case 1: 
                    status=nppiAdd_16u_C1RSfs(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI,0);
                    break;
            case 3:
                    status=nppiAdd_16u_C3RSfs(inputimage.data(offsetpositionx,offsetpositiony), inputimage.pitch(), extimage.data(), extimage.pitch(), tempimage.data(offsetpositionx,offsetpositiony), tempimage.pitch(), AddROI,0);
                    break;    
        }
    }

    template <unsigned int N2>
    void nppidivide(npp::ImageNPP<Npp32f,N2> &inputimage, npp::ImageNPP<Npp32f,3> &exposuremap,npp::ImageNPP<Npp32f,N2> &outputimage){
        NppiSize divROI={inputimage.width(),inputimage.height()};
        npp::ImageNPP_32f_C1 singleexposuremap(exposuremap.width(),exposuremap.height());
        NppStatus status;
        switch(N2){
            case 1:
                nppiCopy_32f_C3C1R(exposuremap.data(),(int) exposuremap.pitch(), singleexposuremap.data(),(int)singleexposuremap.pitch() , divROI);
                nppiDiv_32f_C1R(singleexposuremap.data(),(int)singleexposuremap.pitch(), inputimage.data(), inputimage.pitch(), outputimage.data(), outputimage.pitch(), divROI);
                break;
            case 3:
                nppiDiv_32f_C3R(exposuremap.data(), exposuremap.pitch(), inputimage.data(), inputimage.pitch(), outputimage.data(), outputimage.pitch(), divROI);
                break;
        }
        
    }

    template <unsigned int N2>
    void nppidivide(npp::ImageNPP<Npp8u,N2> &inputimage, npp::ImageNPP<Npp8u,3> &exposuremap,npp::ImageNPP<Npp8u,N2> &outputimage){
        NppiSize divROI={inputimage.width(),inputimage.height()};
        npp::ImageNPP_8u_C1 singleexposuremap(exposuremap.width(),exposuremap.height());
        switch(N2){
            case 1:
                nppiCopy_8u_C3C1R(exposuremap.data(),(int) exposuremap.pitch(), singleexposuremap.data(),(int)singleexposuremap.pitch() , divROI);
                nppiDiv_8u_C1RSfs(singleexposuremap.data(),singleexposuremap.pitch(), inputimage.data(), inputimage.pitch(), outputimage.data(), outputimage.pitch(), divROI,0);
                break;
            case 3:
                nppiDiv_8u_C3RSfs(exposuremap.data(), exposuremap.pitch(), inputimage.data(), inputimage.pitch(), outputimage.data(), outputimage.pitch(), divROI,0);
                break;
        }
        
    }

    template <unsigned int N2>
    void nppidivide(npp::ImageNPP<Npp16u,N2> &inputimage, npp::ImageNPP<Npp16u,3> &exposuremap,npp::ImageNPP<Npp16u,N2> &outputimage){
        NppiSize divROI={inputimage.width(),inputimage.height()};
        npp::ImageNPP_16u_C1 singleexposuremap(exposuremap.width(),exposuremap.height());
        
        switch(N2){
            case 1:
                nppiCopy_16u_C3C1R(exposuremap.data(),(int) exposuremap.pitch(), singleexposuremap.data(),(int)singleexposuremap.pitch() , divROI);
                nppiDiv_16u_C1RSfs(singleexposuremap.data(),singleexposuremap.pitch(), inputimage.data(), inputimage.pitch(), outputimage.data(), outputimage.pitch(), divROI,0);
                break;
            case 3:
                nppiDiv_16u_C3RSfs(exposuremap.data(), exposuremap.pitch(), inputimage.data(), inputimage.pitch(), outputimage.data(), outputimage.pitch(), divROI,0);
                break;
        }
        
    }

    void nppifilter(npp::ImageNPP_8u_C1 &signalimage,npp::ImageNPP_8u_C1 &outputimage,NppiSize osizeROI,npp::ImageNPP_32s_C1 &maskimage, NppiSize okernelSize, NppiPoint oAnchor, Npp32s ndivisor){
        NppStatus status;
        NppiPoint oSrcoffset={0,0};
        //status=nppiFilter_8u_C1R(signalimage.data(), signalimage.pitch(), outputimage.data(), outputimage.pitch(),osizeROI, maskimage.data(), okernelSize, oAnchor, ndivisor);
        status=nppiFilterBorder_8u_C1R(signalimage.data(), signalimage.pitch(),osizeROI,oSrcoffset, outputimage.data(), outputimage.pitch(),osizeROI, maskimage.data(), okernelSize, oAnchor, ndivisor,NPP_BORDER_REPLICATE);
        
    }

    void nppifilter(npp::ImageNPP_32f_C1 &signalimage,npp::ImageNPP_32f_C1 &outputimage,NppiSize osizeROI,npp::ImageNPP_32s_C1 &maskimage, NppiSize okernelSize, NppiPoint oAnchor, Npp32s ndivisor){
        npp::ImageNPP_32f_C1 maskfloat(maskimage.size());
        NppiPoint oSrcoffset={0,0};
        nppiConvert_32s32f_C1R(maskimage.data(), maskimage.pitch(), maskfloat.data(), maskfloat.pitch(), okernelSize);
        nppiFilterBorder_32f_C1R(signalimage.data(), signalimage.pitch(), osizeROI,oSrcoffset, outputimage.data(), outputimage.pitch(),osizeROI, maskfloat.data(), okernelSize, oAnchor,NPP_BORDER_REPLICATE);
    } 

    void nppimaxidx(npp::ImageNPP_8u_C1 &image,NppiSize osizeROI,Npp8u* maxbuffer,Npp8u *nppmaxvalues,int *positionx, int *positiony){
        NppStatus status;
        status=nppiMaxIndx_8u_C1R(image.data(), image.pitch(),osizeROI,maxbuffer, nppmaxvalues, positionx, positiony);
    }

    void nppimaxidx(npp::ImageNPP_32f_C1 &image,NppiSize osizeROI,Npp8u* maxbuffer,Npp32f *nppmaxvalues,int *positionx, int *positiony){
        NppStatus status;
        status=nppiMaxIndx_32f_C1R(image.data(), image.pitch(),osizeROI,maxbuffer, nppmaxvalues, positionx, positiony);
    }


    void nppicompare(npp::ImageNPP_8u_C1 &nppgreyimage,npp::ImageNPP_8u_C1 &nppdestfile,Npp8u threshold, NppiSize osizeROI){
        nppiCompareC_8u_C1R(nppgreyimage.data(),(int)nppgreyimage.pitch(), threshold, nppdestfile.data(),(int)nppdestfile.pitch(),osizeROI,NPP_CMP_GREATER_EQ);
    }

    void nppicompare(npp::ImageNPP_32f_C1 &nppgreyimage,npp::ImageNPP_8u_C1 &nppdestfile,Npp32f threshold, NppiSize osizeROI){
        nppiCompareC_32f_C1R(nppgreyimage.data(),(int)nppgreyimage.pitch(), threshold, nppdestfile.data(),(int)nppdestfile.pitch(),osizeROI,NPP_CMP_GREATER_EQ);
    }
    
    void nppicompare(npp::ImageNPP_16u_C1 &nppgreyimage,npp::ImageNPP_8u_C1 &nppdestfile,Npp16u threshold, NppiSize osizeROI){
        /*nppicompare does not exist in 32s, so I need to convert it to 32f first*/
        nppiCompareC_16u_C1R(nppgreyimage.data(),(int)nppgreyimage.pitch(), threshold, nppdestfile.data(),(int)nppdestfile.pitch(),osizeROI,NPP_CMP_GREATER_EQ);
        
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
        addexposure(this->nppinputimage,{0,0});
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
            if (this->signalimage.width()*this->signalimage.height()!=1) Nppop<D,3>::template resizeimage<Npp8u,1>(this->signalimage,newimagewidth,newimageheight,imageposition);
            if (this->correlationimage.width()*this->correlationimage.height()!=1) Nppop<D,3>::template resizeimage<Npp8u,1>(this->correlationimage,newimagewidth,newimageheight,imageposition);
            if (this->exposuremap.width()*this->exposuremap.height()!=1) Nppop<D,3>::template resizeimage<D,3>(this->exposuremap,newimagewidth,newimageheight,imageposition);
            if(this->maxpixelposition.x>-1 && this->maxpixelposition.y>-1) Nppop<D,3>::shiftposition(this->maxpixelposition,imageposition);
            if(this->maxcorrposition.x>-1 && this->maxcorrposition.y>-1)  Nppop<D,3>::shiftposition(this->maxcorrposition,imageposition);
 
        }


        /*I need to add exposure based on the ROI from the reference and target image auto-correlation */
        void stackimage(astrojpg_rgb_<Npp8u> &addedimage){
            NppStatus status;
            
            unsigned int offsetsizex=0;
            cv::Point_<int> offsetposition;
            cv::Point_<int> outboundupper;
            cv::Point_<int> outboundlower;

            offsetposition.x=addedimage.maxcorrposition.x-this->maxcorrposition.x;
            offsetposition.y=addedimage.maxcorrposition.y-this->maxcorrposition.y;

            outboundlower=-offsetposition;
            outboundupper.x=outboundlower.x+addedimage.nppinputimage.width();
            outboundupper.y=outboundlower.y+addedimage.nppinputimage.height();
            
            
            //std::cout<<"outboundupper x="<<outboundupper.x<<",y="<<outboundupper.y<<std::endl;
            //std::cout<<"outboundlower x="<<outboundlower.x<<",y="<<outboundlower.y<<std::endl;
            unsigned int newimagewidth=this->nppinputimage.width()+(outboundupper.x>=this->nppinputimage.width())*std::abs(outboundupper.x-(int)this->nppinputimage.width())+(outboundupper.x<this->nppinputimage.width())*0\ 
                                      +(outboundlower.x<=0)*std::abs(outboundlower.x)+(outboundlower.x>0)*0;

            unsigned int newimageheight=this->nppinputimage.height()+(outboundupper.y>=this->nppinputimage.height())*std::abs(outboundupper.y-(int)this->nppinputimage.height())+(outboundupper.y>this->nppinputimage.height())*0\ 
                                      +(outboundlower.y<=0)*std::abs(outboundlower.y)+(outboundlower.y>0)*0;

            //if (targetmaxcorrposition.x+addedimage.nppinputimage.width()>this->nppinputimage.width()) offsetsizex +=std::abs(targetmaxcorrposition.x+addedimage.nppinputfile.width()-this->nppinputimage.width());
            //if (targetmaxcorrposition.x+2*this->nppinputimage.maxcorrposition.x-addedimage.nppinputimage.width()<0) offsetsizex +=std::abs(targetmaxcorrposition.x+2*this->nppinputimage.maxcorrposition.x-addedimage.nppinputfile.width());

            resize(newimagewidth,newimageheight,offsetposition);
            
            /*I then need to create a 32f value of the targetimage*/
            //std::cout<<targetmaxcorrposition.x<<","<<targetmaxcorrposition.y<<std::endl;

            //npp::ImageNPP<Npp32f,3> converted32faddedimage(addedimage.nppinputimage.size());
            npp::ImageNPP<D,3> convertedaddedimage(addedimage.nppinputimage.size());
            convert8utoother<3>(addedimage.nppinputimage,convertedaddedimage);
            
            
            Nppop<D,3>::template addimage<D,3>(this->nppinputimage,convertedaddedimage,offsetposition);
            
            addexposure(convertedaddedimage,offsetposition);
            /* Now add values from other image properties */
            npp::ImageNPP<D,1> convertedaddedgreyimage(addedimage.nppgreyimage.size());
            convert8utoother<1>(addedimage.nppgreyimage,convertedaddedgreyimage);
            //nppiConvert_8u32f_C1R(addedimage.nppgreyimage.data(), addedimage.nppgreyimage.pitch(),converted32faddedgreyimage.data(),converted32faddedgreyimage.pitch(),convertimageROI);
            Nppop<D,3>::template addimage<D,1>(this->nppgreyimage,convertedaddedgreyimage,offsetposition);

        }
        /*I need to first use the previous exposure map and add it onto the new map with the specific offset*/

        //nppiAdd_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
        //} 
    private:
        void setinputNPP(npp::ImageNPP_8u_C3 &nppinputfile,npp::ImageNPP_32f_C3 &nppinputimage){
            npp::ImageNPP_32f_C3 nppconvertedfile(nppinputfile.width(),nppinputfile.height());
                /*We need to convert from 8u to 32f*/
            NppiSize convertsizeROI={nppinputfile.width(),nppinputfile.height()};
            //nppiConvert_8u32f_C3R(nppinputfile.data(), nppinputfile.pitch(), nppconvertedfile.data(), nppconvertedfile.pitch(),convertsizeROI);
            nppiConvert_8u32f_C3R(nppinputfile.data(), (int)nppinputfile.pitch(), nppconvertedfile.data(), (int)nppconvertedfile.pitch() , convertsizeROI);
            nppinputimage=nppconvertedfile;
        }

        void setinputNPP(npp::ImageNPP_8u_C3 &nppinputfile,npp::ImageNPP_16u_C3 &nppinputimage){
            NppStatus status;
            npp::ImageNPP_16u_C3 nppconvertedfile(nppinputfile.width(),nppinputfile.height());
                /*We need to convert from 8u to 32s*/
            NppiSize convertsizeROI={nppinputfile.width(),nppinputfile.height()};
            status=nppiConvert_8u16u_C3R(nppinputfile.data(),(int) nppinputfile.pitch(), nppconvertedfile.data(),(int) nppconvertedfile.pitch(),convertsizeROI);
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
            nppiRGBToGray_32f_C3C1R(nppinputimage.data(),nppinputimage.pitch(), nppgreyfile.data(), nppgreyfile.pitch(),osizeROI);
        }

        void rgbtogray(npp::ImageNPP_16u_C3 &nppinputimage, npp::ImageNPP_16u_C1 &nppgreyfile){
            NppStatus status;
            NppiSize osizeROI={(int)nppinputimage.width(),(int)nppinputimage.height()};
            status=nppiRGBToGray_16u_C3C1R(nppinputimage.data(), (int)nppinputimage.pitch(), nppgreyfile.data(),(int)nppgreyfile.pitch(),osizeROI);
        }

        void addexposure(npp::ImageNPP_8u_C3 &nppinputfile,cv::Point_<int> offsetposition){
            if (this->exposuremap.width()==1 && this->exposuremap.height()==1){
                npp::ImageNPP_8u_C3 tempexposuremap((int)nppinputfile.width(),(int)nppinputfile.height());
                this->exposuremap=tempexposuremap;
            }
            unsigned int offsetpositionx=(offsetposition.x<0)*std::abs(offsetposition.x)+(offsetposition.x>=0)*0;
            unsigned int offsetpositiony=(offsetposition.y<0)*std::abs(offsetposition.y)+(offsetposition.y>=0)*0;
            NppiSize osizeROI={(int)nppinputfile.width(),(int)nppinputfile.height()};
            const Npp8u ones[3]={1,1,1};
            nppiAddC_8u_C3IRSfs(ones,this->exposuremap.data(offsetpositionx,offsetpositiony), (int)this->exposuremap.pitch(),osizeROI,0);
            
        }
        void addexposure(npp::ImageNPP_32f_C3 &nppinputfile,cv::Point_<int> offsetposition){
            if (this->exposuremap.width()==1 && this->exposuremap.height()==1){
                npp::ImageNPP_32f_C3 tempexposuremap((int)nppinputfile.width(),(int)nppinputfile.height());
                this->exposuremap=tempexposuremap;
            }
            unsigned int offsetpositionx=(offsetposition.x<0)*std::abs(offsetposition.x)+(offsetposition.x>=0)*0;
            unsigned int offsetpositiony=(offsetposition.y<0)*std::abs(offsetposition.y)+(offsetposition.y>=0)*0;
            NppiSize osizeROI={(int)nppinputfile.width(),(int)nppinputfile.height()};
            const Npp32f ones[3]={1,1,1};
            nppiAddC_32f_C3IR(ones,this->exposuremap.data(offsetpositionx,offsetpositiony), (int)this->exposuremap.pitch(),osizeROI);
            
        }

        void addexposure(npp::ImageNPP_16u_C3 &nppinputfile,cv::Point_<int> offsetposition){
            if (this->exposuremap.width()==1 && this->exposuremap.height()==1){
                npp::ImageNPP_16u_C3 tempexposuremap((int)nppinputfile.width(),(int)nppinputfile.height());
                this->exposuremap=tempexposuremap;
            }
            NppStatus status;
            unsigned int offsetpositionx=(offsetposition.x<0)*std::abs(offsetposition.x)+(offsetposition.x>=0)*0;
            unsigned int offsetpositiony=(offsetposition.y<0)*std::abs(offsetposition.y)+(offsetposition.y>=0)*0;
            NppiSize osizeROI={(int)nppinputfile.width(),(int)nppinputfile.height()};
            const Npp16u ones[3]={1,1,1};
            status=nppiAddC_16u_C3IRSfs(ones,this->exposuremap.data(offsetpositionx,offsetpositiony), (int)this->exposuremap.pitch(),osizeROI,0);
            
        }

        template<unsigned int N2>
        void convert8utoother(npp::ImageNPP<Npp8u,N2> &addedimageinput,npp::ImageNPP<Npp32f,N2> &convertedimageinput){
            NppiSize convertimageROI={(int)addedimageinput.width(),(int)addedimageinput.height()};
            if(N2==1){
                nppiConvert_8u32f_C1R(addedimageinput.data(), addedimageinput.pitch(), convertedimageinput.data(), convertedimageinput.pitch(), convertimageROI);
            }
            else if (N2==3){
                nppiConvert_8u32f_C3R(addedimageinput.data(), addedimageinput.pitch(), convertedimageinput.data(), convertedimageinput.pitch(), convertimageROI);
            }
        }

        template<unsigned int N2>
        void convert8utoother(npp::ImageNPP<Npp8u,N2> &addedimageinput,npp::ImageNPP<Npp16u,N2> &convertedimageinput){
            NppiSize convertimageROI={(int)addedimageinput.width(),(int)addedimageinput.height()};
            NppStatus status;
            if (N2==1){
                status=nppiConvert_8u16u_C1R(addedimageinput.data(), addedimageinput.pitch(), convertedimageinput.data(), convertedimageinput.pitch(), convertimageROI);
                
            }
            else if(N2==3){
                status=nppiConvert_8u16u_C3R(addedimageinput.data(), addedimageinput.pitch(), convertedimageinput.data(), convertedimageinput.pitch(), convertimageROI);
            }
        }
        template<unsigned int N2>
        void convert8utoother(npp::ImageNPP<Npp8u,N2> &addedimageinput,npp::ImageNPP<Npp8u,N2> &convertedimageinput){
            convertedimageinput=addedimageinput;
        }
};
