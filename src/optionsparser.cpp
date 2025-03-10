#include "optionsparser.h"
#include "global.h"
#include <getopt.h>
#include <iostream>
void optionparser(int argc, char **argv){
    int c;
    int option_index=0;
    static struct option long_options[]=
    {
        {"directory",required_argument,NULL,'d'},
        {"output",required_argument,NULL,'o'}
    };

    
    while ((c = getopt_long(argc, argv, "d:o:",long_options, &option_index)) != -1) {
        int this_option_optind = optind ? optind : 1;
        switch (c) {
            case 'd':
                std::cout<<"directory has value "<<optarg<<std::endl;
                directorypath=optarg;
                break;
            case 'o':
                std::cout<<"Outputname is "<<optarg<<std::endl;
                outputfilename=optarg;
                break;
        }
    }
}