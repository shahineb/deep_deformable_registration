# Diffeomorphic Transformer Layer

This is an tensorflow implementation of the "Diffeomorphic" Transformer Layer that was presented in 2D at:

> Shu Z, Sahasrabudhe M, Guler A, Samaras D, Paragios N, Kokkinos I. Deforming Autoencoders: Unsupervised Disentangling of Shape and Appearance. arXiv preprint arXiv:1806.06503. 2018 Jun 18.

and in 3D at:

> (to be updated) Christodoulidis S, Sahasrabudhe M, Vakalopoulou M, ..., Paragios N. Linear and Deformable Image Registration with 3D Convolutional Neural Networks. RAMBO 2018.

In case of any questions, please do not hesitate to contact us.

## Environment:

This code was tested on a Linux machine with Ubuntu (16.04.4 LTS) using the following setup:

- CUDA (9.0)
- CUDNN (v7)
- python (3.5.2)
    * tensorflow (1.8.0)
    * keras (2.1.6)
    * matplotlib (2.2.2)

## How to use:

```
pip install -r requirements{-gpu}.txt
python example-2d.py
python example-3d.py
```

## TODO:

Add a better 3D dataset with visualization results


