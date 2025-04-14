#ifndef ASTROIO_H
#define ASTROIO_H
#include <iostream>
#include <string.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <npp.h>
#include <filesystem>
//forward declaration of npp::ImageNPP
namespace npp {
    template <typename D, unsigned int N> class ImageNPP;
    template <typename D, unsigned int N, class A> class ImageCPU;
    template <typename D, size_t N> class ImageAllocatorCPU;
}
template<typename D, unsigned int N> void saveastro(npp::ImageNPP<D,N> &image,std::string outputfilename);
template<typename D, unsigned int N> void printastro(npp::ImageNPP<D,N> &image,cv::Point_<int> beginpos, cv::Point_<int> endpos);

void listfiles(std::string directorypath,std::vector<std::string> &list_of_files);
#include "astroio.tpp"
#endif

