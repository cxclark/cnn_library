# Building a Convolutional Neural Network Using Only Numpy

## Problem Statement
What are the most critical components of building an image classifier using only Numpy?

The tools for developing neural networks are becoming increasingly streamlined and robust. This has the benefit of of making deep learning very accessible, but it abstracts away neural network processes and detail from the practitioner. Even with robust tools, training neural networks is still very challenging. A good understanding of how neural nets work and converge may lead to greater project success.

This library implements an image classifier step-by-step using only NumPy. The task is to classify correctly images from the CIFAR-10 dataset. The goal of the build process is to gain intuitions and test concepts. Code comments and explanations are numerous. There will likely be mistakes, which should decrease over time. Feedback is welcome.

This work is inspired by Andrej Karpathy's library ConvnetJS and Andrew Ng's deeplearning.ai courses, whose works and open-source materials have helped countless people break into AI.

## Executive Summary
Steps of optimizing a neural network:
1. Sample data
2. Compute forward calculations and calculate loss
3. Compute backward derivative calculations
4. Update parameters based on local gradients
5. Repeat

Every forward calculation must have a corresponding backward--derivative--calculation during backpropagation. Outputs calculated during forward propagation must be stored in memory, then incrementally consumed during backpropagation steps.

To achieve this structure, the network is designed as a series of layers. In Python, the layers are represented as classes. Within each class is a forward and backward function. A final network class stacks these layers together for computation.

This library currently support these layers:
- Convolutions
- ReLU
- Max Pooling
- Flatten
- Dense or Fully-Connected
- Softmax

## Conclusions
The core components of a convolutional neural network were successfully assembled and executed. In the simplest sense, a convolutional neural network architecture is a list of layers that transform an image into an output volume holding the class scores. This was accomplished, as output probability vectors were produced.

Understanding the linear algebra and shape transformations between steps was critical to gettng the layers to work together.

However, the model was not able to optimize and converge. While working with small subsets of the data produce results, errors emerged when training on larger subsets of the data. This could be due to an error in the mathematical formulas used, suboptimal weight initializations, or something else.

## Future Work
Roadmap for the library:
1. Debug issues with scaling to be able to train on larger datasets
1. Publish to PyPi
1. Modularize parameter update methods
1. Modularize loss and accuracy 
1. Build out data preprocessing methods
1. Regularization layers
1. Batch normalization layers
1. Input layers to handle different image types
1. Generalize to other classification tasks

## References:
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [Deeplearning.ai: Deep Learning Specialization](https://www.deeplearning.ai/deep-learning-specialization/)
- [ConvNetJS: Andrej Karpathy's Javascript Deep Learning framework](https://cs.stanford.edu/people/karpathy/convnetjs/)
- [tinygrad: Small deep learning framework](https://github.com/geohot/tinygrad)