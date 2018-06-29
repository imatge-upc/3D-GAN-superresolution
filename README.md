# 3D-GAN-superresolution
Here we present the implementation in TensorFlow of our work to generate high resolution MRI scans from low resolution images using Generative Adversarial Networks (GANs), accepted in the [Medical Imaging with Deep Learning Conference – Amsterdam. 4 - 6th July 2018.](https://midl.amsterdam/)

Discriminator network
![alt text](https://github.com/imatge-upc/3D-GAN-superresolution/blob/master/images/3D%20SRGAN(D).png)

Generator network
![alt text](https://github.com/imatge-upc/3D-GAN-superresolution/blob/master/images/3D%20SRGAN(G).png)


In this work we propose an architecture for MRI super-resolution that completely exploits the available volumetric information contained in MRI scans, using 3D convolutions to process the volumes and taking advantage of an adversarial framework, improving the realism of the generated volumes.
The model is based on the [SRGAN network](https://arxiv.org/abs/1609.04802). The adversarial loss uses least squares to stabilize the training and the generator loss, in addition to the adversarial term contains a content term based on mean square error and image gradients in order to improve the quality of the generated images. We explore three different methods for the upsampling phase: an upsampling layer which uses nearest neighbors to replicate consecutive pixels followed by a convolutional layer to improve the approximation, sub-pixel convolution layers as proposed in [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) and a modification of this method [Checkerboard artifact free sub-pixel convolution](https://arxiv.org/pdf/1707.02937.pdf) that alleviates checkbock artifacts produced by sub-pixel convolution layers (Check: [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) for more information).

Comparison of the upsampling methods used
![alt text](https://github.com/imatge-upc/3D-GAN-superresolution/blob/master/images/Upsamplings.png)
