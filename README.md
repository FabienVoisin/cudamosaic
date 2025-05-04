# cudamosaic
This project aims to stack a few astronomical images together and come up with a Mosaic image.

# installation of cudamosaic and prerequisite
## Pre-requisites
### CUDA
cudamosaic has been tested with Ubuntu 22.04 and cuda 11 and above. We first suggest you install the cuda packages.
Please see https://developer.nvidia.com/cuda-downloads for latest cuda installation.

Make sure that nvcc is also in the PATH directory
```bash
ubuntu@ip-10-255-9-77:~/cudamosaic$ which nvcc
/usr/local/cuda-12.8/bin//nvcc
```
If not please add the cuda folder in your PATH environment variable.
```bash
export PATH=/usr/local/cuda-<version>/bin:${PATH}
```
where `version` is in our case 12.8.

### OpenCV
We also need access to `opencv`
```
sudo apt-get install libopencv-dev
```
Once this is done, we are ready to install `cudamosaic` 

## Clone and install cudamosaic

```
git clone https://github.com/FabienVoisin/cudamosaic.git
```
Then one needs to clone the NPP utils repo 
```
cd cudamosaic
git submodule init
git submodule update
```
Then type 
```
make
```
to compile the code

In order to test the code, one can use the Orion folder or testimage folders via the following command
```
./mosaic -d testimage -o testimagemosaic.png
```
The output image should in this case be  `testimagemosaic.png`. The image `finalresultexp.jpg` also provides the final exposure map from aggregating all the input images. 

## Command line parameters
There are currently two command line parameters:

`-d` (required): path of folders which has the input images

`-o` (optional): filename of the output mosaic image. In the case, this is not explicitely parsed, then `defaultfilename.jpg` would be used.

![](doc\inputimages.png)
# How it works
The code makes use of the NPP utils library to perform analysis and calculation on input images.
The code first creates a greyscale image of the first image. Based on a specific threshold, the code then highlight pixels above certain values. It will then select a box of the signals around the pixel with the highest intensity.
The code will use this box to perform correlation maps with other images. We expect the maximum value of the correlation maps will determine the offset position between the new input images and the mosaic images. Once that offset position is calculated, the code add the values of the new input images onto the mosaic output image. The final output image is then normalised by dividing the output image by the exposure map.
