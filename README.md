# SMiLEConverter

### Introduction
This repo implements an [Auxiliary Conditional GAN (ACGAN)](https://arxiv.org/pdf/1610.09585.pdf) model, a [Wasserstain](https://arxiv.org/pdf/1701.07875.pdf) ACGAN model, and an image encoder to build and train a system for converting human smiling expressions to non-smiling expressions and the other way around.

### Dataset
The system is trained using the [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
A quick image sample and attributes exploration is shown in **notebook/ExploreCelebA**

### Programming language
Python 2.7

### Library Dependencies
* TensorFlow 1.0
* Numpy 1.12.0
* Scipy 0.18.1
* Pandas 0.19.2
* Matplotlib 2.0.0
* tqdm 4.11.2

### Credits
The initial vanilla GAN and inital Wasserstain GAN implementations are adapted from and credited to [Sarath Shekkizhar](https://github.com/shekkizh/WassersteinGAN.tensorflow). The ACGAN and WACGAN model frameworks are built on top of the inital vanilla GAN.

### Prerequisites & Expeirments Explained
* Model was trained in Paperspace Linux system with Nvidia GPU
* CelebA dataset should be downloaded and unzipped manually
* Large portion of the implementations, settings, and trainings follows the [improved GAN training techniques](https://arxiv.org/pdf/1606.03498.pdf) and the ["How to train a GAN" talk](https://github.com/soumith/ganhacks) at NIPS2016
* Various experiments with different hyperparameter settings were explored and published in the **bash_file/** directory:
  * [ACGAN](https://arxiv.org/pdf/1610.09585.pdf) with feature matching - L1 distance (manually replace the `tf.nn.l2_loss` with `tf.abs` at line 548 in the **models/GAN_model.py** file)
  * [ACGAN](https://arxiv.org/pdf/1610.09585.pdf) with feature matching - L2 distance
  * [WACGAN](https://arxiv.org/pdf/1701.07875.pdf) without feature matching
  * [WACGAN](https://arxiv.org/pdf/1701.07875.pdf) with feature matching - L1 distance (manually replace the `tf.nn.l2_loss` with `tf.abs` at line 698 in the **models/GAN_model.py** file)
  * [WACGAN](https://arxiv.org/pdf/1701.07875.pdf) with feature matching - L2 distance
  * Train an image encoder with a frozen generator using L1 distance as loss metric
  * Train an image encoder with a frozen generator using L2 distance as loss metric (manually replace the `tf.abs` with `tf.nn.l2_loss` at line 183 in the **models/Encoder.py** file)

### Observations
* ACGAN with feature matching (both L1 and L2 distance) is able to achieve generating decent qualify of human face images with relatively pure class based generation
* WACGAN takes longer to train, its generated image quality is not as good as ACGAN, however it produces purer class based generation
* Neither L1 distance nor L2 distance turns out to be good metrics to train an image encoder for performing the perserving identity task
* Without regulating the distribution of the image encoder's output, mode collapose could happen at the second stage even though it didn't happen during the first stage of training the ACGAN and the generator

### Results
A more detailed information regarding the architectures of the models and the results of the trained models are aggregated into [this demo presentation](https://zhongyuk.github.io/)

### Future Work & Improvement
* Modify the image encoder to using precepture loss/image embeding space distance
* Regulate the distribution of the latent Z vectors produced by the image encoder to match with the Generator's input noise vector distribution 
