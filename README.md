# Caffe_Matlab_GAN

This repository tries to re-implement (Wasserstein GAN)[https://arxiv.org/abs/1701.07875] using Caffe and Matlab. 

Sorry that it only support [the ms branch of my Caffe](https://github.com/happynear/caffe-windows/tree/ms) and can only be runned in Windows now.

I don't quite know why, the weights are always trained to be 0.01 or -0.01. I don't think there is anything wrong with my implementation, but the way of realizing Lipschitz constraint may be wrong in the original paper.
