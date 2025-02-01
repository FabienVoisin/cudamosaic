#ifndef CLASS_H
#define CLASS_H
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
    cv::Point_<int> maxpixelposition={0,0}; 
    Nppop();
    
    void getmaxpixel(npp::ImageNPP<D,1> &image,D *maxbuffer);

};

class astrojpg_8u_rgb : public Nppop<Npp8u, 3> 
{
    public: 
        npp::ImageNPP_8u_C3 nppinputimage;
        npp::ImageNPP_8u_C1 nppgreyimage;
        npp::ImageNPP_8u_C1 signalimage;
        cv::Point_<int> hostmaxpixelposition={0,0};
        astrojpg_8u_rgb(std::string filename);

        astrojpg_8u_rgb(unsigned int imagewidth, unsigned int imageheight);
        
        void getgreyimage();
        
        void getsignalimage(const Npp8u threshold);

         

         
};


#endif