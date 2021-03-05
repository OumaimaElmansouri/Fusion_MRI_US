# Fusion of Magnetic Resonance and Ultrasound Images

This code introduces a new fusion method for magnetic resonance (MR) and ultrasound (US) images, which aims at combining the advantages of each modality, i.e., good contrast and signal to noise ratio for the MR image and good spatial resolution for the US image. The proposed algorithm is based on an inverse problem, performing a  super-resolution of the MR image and a denoising of the US image. A polynomial function is introduced to model the relationships between the gray levels of the MR and US images. The resulting inverse problem is solved using a proximal alternating linearized minimization algorithm.

For more details, see the paper: 
EL MANSOURI, Oumaima, VIDAL, Fabien, BASARAB, Adrian, et al. Fusion of magnetic resonance and ultrasound images for endometriosis detection. IEEE Transactions on Image Processing, 2020, vol. 29, p. 5324-5335.

NB: in this version of code, PALM algoritm was implemented without backtraking step.

#### How to run this code: Edit and run the demo script. In image folder, you can find two images MRI and US images (phantom images decribed in the article below)

This code is in development stage, thus any comments or bug reports are very welcome.

#### Contact: oumaima.el-mansouri@irit.fr
