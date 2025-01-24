#ifndef HEADER_H
#define HEADER_H
#include <iostream>
#include <Common/UtilNPP/ImagesCPU.h>
#include <Common/UtilNPP/ImagesNPP.h>
#include <nppdefs.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <vector>
#endif

template<typename D, unsigned int N>  void saveastro(npp::ImageNPP<D,N> &image,std::string outputfilename);