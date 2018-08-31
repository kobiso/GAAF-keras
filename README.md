# GAAF-Keras
This is a Keras implementation of ["Gradient Acceleration in Activation Functions (GAAF)"](https://arxiv.org/abs/1806.09783).
This repository includes *ResNet_v1* and *ResNet_v2* implementation to test and compare GAAF with original activation functions.

## Gradient Acceleration in Activation Functions
"Gradient Acceleration in Activation Functions" proposes a new technique for activations functions, *gradient acceeration in activation function (GAAF)*, that accelerates gradients to flow even in the saturation area.
Then, input to the activation function can climb onto the saturation area which makes the network more robust because the model converges on a flat region.

### GAAF
<div align="center">
  <img src="https://github.com/kobiso/GAAF-keras/blob/master/figures/gaaf.png"  width="600">
</div>

<div align="center">
  <img src="https://github.com/kobiso/GAAF-keras/blob/master/figures/gx.png"  width="350">
</div>

<div align="center">
  <img src="https://github.com/kobiso/GAAF-keras/blob/master/figures/fx.png"  width="350">
</div>

### Experimental Results in the Paper
<div align="center">
  <img src="https://github.com/kobiso/GAAF-keras/blob/master/figures/result.png"  width="700">
</div>

## Prerequisites
- Python 3.x
- Keras (Tensorflow backend)

## Prepare Data set
This repository use [*Cifar10*](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
When you run the training script, the dataset will be automatically downloaded.

### Change Activation Function
This repository supports two GAAF: **GAAF_relu** and **GAAF_tanh**.
For the *GAAF_relu*, It uses shifted sigmoid function as shape function and you can set the shift parameter.
For the *GAAF_tanh*, It uses modified Gaussian function with a peak point at y=1.

To change activation function, you can set `activation` on `main.py` with `GAAF_relu`, `GAAF_tanh`, and `relu`.

## Supportive CNN Models
You can train and test with base CNN model listed below.

- ResNet_v1 (e.g. ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet164, ResNet1001)
- ResNet_v2 (e.g. ResNet20, ResNet56, ResNet110, ResNet164, ResNet1001)

## Train a Model
You can simply train a model with `main.py`.

1. Set depth for ResNet model
    - e.g. `depth=20`  
2. Define a model you want to train.
    - e.g. `model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth, activation=activation)`  
3. Set other parameter such as *batch_size*, *epochs*, *data_augmentation* and so on.
4. Run the `main.py` file
    - e.g. `python main.py`
    
## Test Results
I conducted some experiments on ResNet20_v1 by replacing the original activation (*relu*) with *GAAF_relu* and the results is described as below.

num | data | backbone | activation | shift | steps | acc | batch_size | optimizer | lr
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
baseline | cifar10 | resnet20_v1 | relu | - | 200 | **0.8084** | 128 | adam | 0.001
ex1 | cifar10 | resnet20_v1 | GAAF_relu | 5 | 200 | 0.6517 | 128 | adam | 0.001
ex2 | cifar10 | resnet20_v1 | GAAF_relu | 4 | 200 | 0.6924 | 128 | adam | 0.001
ex3 | cifar10 | resnet20_v1 | GAAF_relu | 3 | 200 | 0.7042 | 128 | adam | 0.001
ex4 | cifar10 | resnet20_v1 | GAAF_relu | 2 | 200 | 0.7441 | 128 | adam | 0.001
ex5 | cifar10 | resnet20_v1 | GAAF_relu | 1 | 200 | 0.78 | 128 | adam | 0.001
ex6 | cifar10 | resnet20_v1 | GAAF_relu | 0 | 200 | 0.7886 | 128 | adam | 0.001
ex7 | cifar10 | resnet20_v1 | GAAF_relu | -0.5 | 200 | 0.7945 | 128 | adam | 0.001
ex8 | cifar10 | resnet20_v1 | GAAF_relu | -1 | 200 | **0.7948** | 128 | adam | 0.001
ex9 | cifar10 | resnet20_v1 | GAAF_relu | -2 | 200 | 0.78 | 128 | adam | 0.001
ex10 | cifar10 | resnet20_v1 | GAAF_relu | -3 | 200 | 0.7768 | 128 | adam | 0.001
ex11 | cifar10 | resnet20_v1 | GAAF_relu | -4 | 200 | 0.7733 | 128 | adam | 0.001
ex12 | cifar10 | resnet20_v1 | GAAF_relu | no s(x) | 200 | 0.6603 | 128 | adam | 0.001
ex13 | cifar10 | resnet20_v1 | GAAF_relu(K.round) | -1 | 200 | **0.8054** | 128 | adam | 0.001

*GAAF_relu* actually **did not give any improvement for every experiments.**
At the beginning of the experiment, I thought `shift=4` will give the best performance as shifted sigmoid shape is very similar with the shape function that GAAF paper suggested.
However, `shift=-1` gives the best performance but still lower than the baseline.
Interesting thing is, when I changed `mut-tf.floor(mut)` with `K.abs(mut-K.round(mut))`, it gives the best performance but less than the baseline. (Keras backend function does not have *floor* operation, so I attempted to use `K.round`)

**If there is any implementation error in this repository, please let me know.**

## Related Works
- Blog: [Gradient Acceleration in Activation Functions](https://kobiso.github.io//research/research-gradient-acceleration/)
- Repository: [CBAM-keras](https://github.com/kobiso/CBAM-keras)

## Reference
- Paper: [Gradient Acceleration in Activation Functions](https://arxiv.org/abs/1806.09783)
- Repository: [Cifar10 ResNet example in Keras](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
  
## Author
Byung Soo Ko / kobiso62@gmail.com
