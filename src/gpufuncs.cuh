#ifndef GPUFUNC_H
#define GPUFUNC_H
#include "struct.cuh"
inline void CreateROIdata(npp::ImageNPP_8u_C1 &greyimage,cv::Point_<int> &maxpixelposition, int squaresize,npp::ImageNPP_32s_C1 &outputfinalimage);
inline void Correlationimage(npp::ImageNPP_8u_C1 &referenceimage,npp::ImageNPP_32s_C1 &maskimage,Npp8u *sumbuffer,npp::ImageNPP_8u_C1 &outputimage);

inline void CreateROIdata(npp::ImageNPP_8u_C1 &greyimage,cv::Point_<int> &maxpixelposition, int squaresize,npp::ImageNPP_32s_C1 &outputfinalimage){
    NppiSize osizeROI={squaresize,squaresize}; //Setup the osizeROI
    /*The two values will help find the appropriate address of the first value*/
    int initpositionx=maxpixelposition.x-squaresize/2+1; 
    int initpositiony=maxpixelposition.y-squaresize/2+1;
    npp::ImageNPP_8u_C1 outputROIimage(squaresize,squaresize);
    npp::ImageNPP_8u_C1 outputmirrorimage(squaresize,squaresize);
    cudaError_t eResult;
    
    eResult=cudaMemcpy2D(outputROIimage.data(),outputROIimage.pitch(),greyimage.data(initpositionx,initpositiony),greyimage.pitch(),outputROIimage.width()*sizeof(Npp8u),outputROIimage.height(),cudaMemcpyDeviceToDevice);
    NPP_ASSERT(cudaSuccess == eResult);
    
    /*nppiMirror will flip the image so that the last values become the first, necessary for the convolution*/
    nppiMirror_8u_C1R(outputROIimage.data(),(int)outputROIimage.pitch(), outputmirrorimage.data(),(int)outputmirrorimage.pitch(), osizeROI, NPP_BOTH_AXIS);
    
    /*Finally we need to convert to 32s*/
    nppiConvert_8u32s_C1R(outputmirrorimage.data(),outputmirrorimage.pitch(),outputfinalimage.data(),outputfinalimage.pitch(),osizeROI);
    std::cout<<"Am I here now"<<std::endl;
    
    }
    
    
    inline void Correlationimage(npp::ImageNPP_8u_C1 &referenceimage,npp::ImageNPP_32s_C1 &maskimage,Npp8u *sumbuffer,npp::ImageNPP_8u_C1 &outputimage){
        
        
        NppiSize osizeROI={(int)referenceimage.width(),(int)referenceimage.height()};
        NppiSize omaskROI={(int)maskimage.width(),(int)maskimage.height()};
        NppiSize okernelSize={(int)maskimage.width(),(int)maskimage.height()};
        NppiPoint oAnchor={(int)(maskimage.width()/2),(int)(maskimage.height()/2)};
        
        npp::ImageNPP_32f_C1 maskimagesum(maskimage.width(),maskimage.height());
        NppStatus status=nppiConvert_32s32f_C1R(maskimage.data(),(int) maskimage.pitch(), maskimagesum.data(),(int) maskimagesum.pitch(), omaskROI);
        
        NPP_ASSERT(NPP_SUCCESS  == status);
        
        Npp64f *nsum;
        Npp64f hostnsum;
        cudaMalloc((void **)&nsum,sizeof(Npp64f));
        std::cout<<"testtest"<<std::endl;
        nppiSum_32f_C1R(maskimagesum.data(),maskimagesum.pitch(),omaskROI, sumbuffer, nsum);  
        cudaDeviceSynchronize();
        cudaMemcpy(&hostnsum,nsum,sizeof(Npp64f),cudaMemcpyDeviceToHost);
        Npp32s ndivisor2=(Npp32s) hostnsum; 
        //Npp32s ndivisor2=10000;  
        std::cout<<"hostnsum="<<ndivisor2<<std::endl;
        status=nppiFilter_8u_C1R(referenceimage.data(), referenceimage.pitch(), outputimage.data(), outputimage.pitch(),osizeROI, maskimage.data(), okernelSize, oAnchor,  ndivisor2);
        NPP_ASSERT(NPP_SUCCESS  == status);
        std::cout<<"last test"<<std::endl;
        cudaDeviceSynchronize();
        
    }
#endif