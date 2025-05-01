#include "astroio.h"

void listfiles(std::string directorypath,std::vector<std::string> &list_of_files){
    for (const auto& entry : std::filesystem::directory_iterator(directorypath)) {
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        //std::cout<<outfilename_str<<std::endl;
        list_of_files.push_back(outfilename_str);
    }
}
