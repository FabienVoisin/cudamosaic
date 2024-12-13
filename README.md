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
