# cudamosaic
This project aims to stack a few astronomical images together and come up with a Mosaic image.
# vision
The code will first look at specific pixels of the image above a specific threshold. We will call these pixels "signals" 
As a first step, the code will create a copy of this image with a binary value (0 for background, 1 for signal). 
It will then iterate through the next couple of images and proceed to the same step.
The algorithm will then compare the image and come up with a "difference" value. In short term, it will perform a substraction of all the binary values between the two images. The higher the value, the more difference there will be between the two images.
The goal is then to perform a translation (and eventually rotation) of the image to minimize this difference. Once we have found out the local minimum, we will create a combined "averaged" image where the second image is offset by that optimized value. 
We finally repeat the same step.
# translation algorithm and challenges
One of the difficult challenge to consider is that moving the second image by a pixel may not have any effect. 
A parallel search for that difference across the various tranlations.
One brainstorm idea would be to sum the value across the image to give us an idea of how much signal there is. We can potentially look for the difference between the two image "pixel by pixel" and pick the one whose final value is below 5% of the total signal found (we still need some head room as we can easily imagine there will be some artefacts.)
Let's consider an image as a matrix of binary values. We will perform a regression by rows and columns.

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


