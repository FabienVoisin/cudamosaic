//int nsrcstep=(int)odevsrc.width();
//npp::ImageNPP_8u_C1 odevgrey;
//int nostep=(int)odevsrc.width();
//NPP_CHECK_NPP(nppiRGBToGray_8u_AC4C1R(odevsrc,nsrcstep,odevgrey,nostep,osizeROI));

//Npp8u *inputfile=nppiMalloc_8u_C3(img.size().width,img.size().height*3,&linestep);
//size_t bstep=linestep;
//std::cout<<"linestep="<<linestep<<std::endl;
/* need to create a ROI */
//cudaError_t err=cudaMemcpy2D((void**)&inputfile,bstep,(void**)&input,img.step1(),img.step1(),img.size().height,cudaMemcpyHostToDevice);

//std::cout<<"err="<<err<<std::endl;
//if(err!=cudaSuccess){
//    printf("There is an error here %s\n",cudaGetErrorName(err));
//    exit(1);
//}