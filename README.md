# Building a Convolutional Neural Network Using Only Numpy

## Problem Statement
What are the most critical components of building an image classifier using only Numpy?

In order to answer this question, I will develop a python package--a collection of modules--that allows you to build convolutional neural networks to classify images. Model performance will be evaluated using accuracy.  Model performance will be compared to the performance of the model with the closest network architecture (e.g. LeNet, AlexNet, ResNet-50).

## Executive Summary
The four basic steps of a neural network are sampling a batch of labeled training data, forward propagating it through a network architecture and calculating the loss, backpropagating the gradient with respect to the loss through the network, and updating parameters based on the calculated gradients.

Every forward calculation must have a corresponding backward--derivative--calculation during backpropagation. Outputs calculated during forward propagation must be stored in memory, the incrementally consumed during backpropagation steps. 

To achieve this structure, the network is designed as a series of layers. The layers are represented as classes. Within each class is a forward and backward function. A final model module stacks these layers together in the appropriate order. 

## Conclusions

## Future Work

## Resources and Inspiration:
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [Deeplearning.ai: Deep Learning Specialization](https://www.deeplearning.ai/deep-learning-specialization/)
- [ConvNetJS: Andrej Karpathy's Javascript Deep Learning framework](https://cs.stanford.edu/people/karpathy/convnetjs/)
- [tinygrad: Small deep learning framework](https://github.com/geohot/tinygrad)