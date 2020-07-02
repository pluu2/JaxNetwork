class network_dense(): 
  def __init__(self,inputs, neurons): #initialize the weights.
    self.weights=onp.random.randn(neurons,inputs)
    self.bias = onp.random.rand(neurons)
  def forward(self,input_,weights='false',bias='false'): 
    if weights=='false' or bias=='false':
      return np.dot(np.array(input_),np.array(self.weights).T)+np.array(self.bias)
    else: 
      return np.dot(np.array(input_),np.array(weights).T)+np.array(bias)
