# Building a Convolutional Neural Network Using Only Numpy

## Problem Statement
What are the most critical components of building an image classifier using only Numpy?

In order to answer this question, I will develop a python package--a collection of modules--that allows you to build convolutional neural networks to classify images. Model performance will be evaluated using accuracy.  Model performance will be compared to the performance of the model with the closest network architecture (e.g. LeNet, AlexNet, ResNet-50).

## Executive Summary
The four basic steps of a neural network are sampling a batch of labeled training data, forward propagating it through a network architecture and calculating the loss, backpropagating the gradient with respect to the loss through the network, and updating parameters based on the calculated gradients.

Every forward calculation must have a corresponding backward--derivative--calculation during backpropagation. Outputs calculated during forward propagation must be stored in memory, the incrementally consumed during backpropagation steps. 

To achieve this structure, the network is designed as a series of layers. The layers are represented as classes. Within each class is a forward and backward function. A final model module stacks these layers together in the appropriate order. 

In the simplest case, a convolutional neural network architecture is a list of layers that transform an image into an output volume holding the class scores.

Layer Summary:  
- InputLayer: holds raw pixel values of image
- Convolutionlayer: computes output of neurons connected to local regions in the input, changes output size
    - Has parameters: weights and biases
    - Has 4 hyperparameters: num_filters (K), filter_size (F), stride (S), zero-padding(P)
- ReluLayer: applies element-wise activation function max(0, x), no change in output size
- PoolingLayer: downsamples along width and height, changes output size
    - Has 2 hyperparameters: filter_size (F), stride (S)
- DenseLayer: computes class scores, resulting in volume [1 x 1 x 10], where each of 10 numbers correspond to CIFAR-10 class score.
    - Has parameters: weights and biases
    - Has hyperparameters: units

## Conclusions

## Future Work

## Resources and Inspiration:
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [Deeplearning.ai: Deep Learning Specialization](https://www.deeplearning.ai/deep-learning-specialization/)
- [ConvNetJS: Andrej Karpathy's Javascript Deep Learning framework](https://cs.stanford.edu/people/karpathy/convnetjs/)
- [tinygrad: Small deep learning framework](https://github.com/geohot/tinygrad)