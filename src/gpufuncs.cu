#include "gpufuncs.cuh"
/*CreateROIdata  will create an array in the region of interest surrounding the max values of the greyimage
 which will be needed for the autoconvolution*/

npp::ImageCPU_32s_C1 &CreateROIdata(npp::ImageNPP_8u_C1 &greyimage,cv::Point_<int> &maxpixelposition, int squaresize){
NppiSize osizeROI={squaresize,squaresize}; //Setup the osizeROI
/*The two values will help find the appropriate address of the first value*/
int initpositionx=maxpixelposition.x-squaresize/2+1; 
int initpositiony=maxpixelposition.y-squaresize/2+1;
npp::ImageCPU_8u_C1 outputROIimage(squaresize,squaresize);
npp::ImageCPU_8u_C1 outputmirrorimage(squaresize,squaresize);
npp::ImageCPU_32s_C1 outputfinalimage(squaresize,squaresize);
cudaError_t eResult;
eResult=cudaMemcpy2D(outputROIimage.data(),outputROIimage.pitch(),greyimage.data(initpositionx,initpositiony),greyimage.pitch(),outputROIimage.width()*sizeof(Npp8u),outputROIimage.height(),cudaMemcpyDeviceToHost);
NPP_ASSERT(cudaSuccess == eResult);
/*nppiMirror will flip the image so that the last values become the first, necessary for the convolution*/
nppiMirror_8u_C1R(outputROIimage.data(),(int)outputROIimage.pitch(), outputmirrorimage.data(),(int)outputmirrorimage.pitch(), osizeROI, NPP_BOTH_AXIS);

/*Finally we need to convert to 32s*/
nppiConvert_8u32s_C1R(outputmirrorimage.data(),outputmirrorimage.pitch(),outputfinalimage.data(),outputfinalimage.pitch(),osizeROI);


return outputfinalimage;

}


npp::ImageNPP_8u_C1 &Correlationimage(npp::ImageNPP_8u_C1 &referenceimage,npp::ImageCPU_32s_C1 &maskimage){
    
    npp::ImageNPP_8u_C1 outputimage(referenceimage.width(),referenceimage.height());
    NppiSize osizeROI={(int)referenceimage.width(),(int)referenceimage.height()};
    NppiSize okernelSize={(int)maskimage.width(),(int)maskimage.height()};
    NppiPoint oAnchor={(int)(maskimage.width()/2),(int)(maskimage.height()/2)};
    Npp32s ndivisor=maskimage.width()*maskimage.height();
    NppStatus status=nppiFilter_8u_C1R(referenceimage.data(), referenceimage.pitch(), outputimage.data(), outputimage.pitch(),osizeROI, maskimage.data(), okernelSize, oAnchor, ndivisor);

    return outputimage;
}