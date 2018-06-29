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

### Data
We used a set of normal control T1-weighted images from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (see www.adni-info.org for details). Skull stripping is performed in all volumes and part of the background is removed. Final volumes have dimensions 224x224x152. Due to memory constraints the training is patch-based; for each volume we extract patches of size 128x128x92, with a step of 112x112x76, so there are 8 patches per volume, with an overlap of 16x16x16. We have a total number of 589 volumes, 470 are used for training while 119 are used for testing. We use batches of two patches, thus for each volume we perform 4 iterations. This code is prepared to do experiments with the processing of images and dimensions explained.

The code expects that the database is inside the folder specified by the data_path in the Train_dataset script. Inside there should be a folder for each of the patients containing a 'T1_brain_extractedBrainExtractionMask.nii.gz' file. This file was created taking the original images from ADNI and performing a skull-stripping processing of them. We use the nibabel library to load the images. 

### Training
To train the network the model.py script is used. When calling the script you should specify:
+ -path_prediction: Path to save training predictions.
+ -checkpoint_dir: Path to save checkpoints.
+ -residual_blocks: Number of residual blocks.
+ -upsampling_factor: Upsampling factor.
+ -subpixel_NN: Use subpixel nearest neighbour.
+ -nn: Use Upsampling3D + nearest neighbour, RC.
+ -feature_size: Number of filters.

By default it will use the sub-pixel convolution layers, 32 filters, 6 residual blocks and an umpsaling factor of 4.

If you want to restore the training, when calling the script you have to define the checkpoint to use using the restore argument:
⋅⋅* -restore: Checkpoint path to restore training

```
python model.py -path_prediction YOURPATH -checkpoint_dir YOURCHECKPOINTPATH -residual_blocks 8 -upsampling_factor 2 -subpixel_NN True -feature_size 64
```

### Testing
To test the network the model.py script is also used. When calling the script you should specify the same arguments as before for the configuration of the model and the new paths used. Also, the argument evaluate should be True:
+ -path_volumes: Path to save test volumes.
+ -checkpoint_dir_restore: Path to restore checkpoints.
+ -residual_blocks: Number of residual blocks.
+ -upsampling_factor: Upsampling factor.
+ -subpixel_NN: Use subpixel nearest neighbour.
+ -nn: Use Upsampling3D + nearest neighbour, RC.
+ -feature_size: Number of filters.
+ -evaluate: Test the model.

```
python model.py -path_volumes YOURPATH -checkpoint_dir_restore YOURCHECKPOINTPATH -residual_blocks 8 -upsampling_factor 2 -subpixel_NN True -feature_size 64 -evaluate True
```

# Contact
If you have any general doubt about our work or code which may be of interest for other researchers, please use the public issues section on this github repo.
