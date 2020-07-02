
import jax.numpy as np
from jax import grad,jit,vmap
import random
import numpy as onp

class jax_base_model(): 
  def __init__(self): 
    self.layer=[]
    

  def relu (self,x,axis=0): 
    return np.max([np.zeros((np.size(x,axis=0),np.size(x,axis=1))) ,x],axis=0)

  def sigmoid_act(self,x): 
    return (np.exp(x)/(np.exp(x)+1))

  def tanh_act(self,x): 
    return np.tanh(x)

  def trainable_variables(self):  
    trainable_v={}
    for i in range(self.num_layers): 
      trainable_v[str(i)+' weights'] = self.layer[i].weights
      trainable_v[str(i)+' bias'] = self.layer[i].bias
    return trainable_v

  def binary_crossentropy(self,x,y): #x=input, y= target
    return -y*np.log(x)-(1-y)*np.log(1-x)

  def update_weights (self,parameters,lr=1.0):
    for i in range(self.num_layers):
      self.layer[i].weights=self.layer[i].weights - lr*parameters[str(i) + ' weights'] 
      self.layer[i].bias=self.layer[i].bias - lr*parameters[str(i) + ' bias']
  
  def set_input(self,x): 
    self.input=x
  def set_targets(self,y): 
    self.target=y

  def gradient(self,parameters): 
    return grad(self.backwards)(parameters)

      
