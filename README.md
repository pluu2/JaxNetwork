![Jax Logo](/jax_logo_250px.png)


# JaxNetwork
## Practice with implementing trainable NNs in python using Jax. 

The following is very rough flexible implementation of NN using Jax. This jax implementation allows the user to put together a fully connected network with an arbitrary number of layers and neurons. Jax is used to calculate the gradient of the loss function. 

"Basic Jax Network" is a jupyter notebook with a very basic implementation of a network. In the example a network with an input layer, 2 hidden layer and a output layer is implemented. The network takes only 2 inputs, and outputs a value between 0 and 1. The network is meant to learn only the simple rule of any logic gate, including OR, AND, XOR, XNOR etc. 


### Notes: 
----

- The optimizer used in the basic implementation is small-batch-SGD. 
- ~~I have coded a much more barebones implementation of a ANN with VMAP. It will be uploaded soon.~~
- A barebones implementation of MNIST is uploaded. I ran into an issue with a OOM warning on GPU. According to Jax documentation it happens when you import tensorflow with GPU enabled. Tf automatically allocates memory. Since the example only uses tf to dataload mnist, I have forced tf to utilize CPU only. 
- I have completed a implementation of a CNN, and have successfully trained on CIFAR-10(61.75%), Fashion_MNIST(90.53%), and MNIST (99.03%) datasets


### To do: 
[ ] Difficulty in implementing VMAP within class objects. I am sure I do not understand how VMAP works. Attempt to implement VMAP to allow batching.

