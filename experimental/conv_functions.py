def conv_layer(input,parameter,mode='same'):
  #This is general convolution 


  output=[]

  for i in range(len(parameter[0])):  #filters
    temp=0
    for j in range (input.shape[0]): #input
      temp=temp+np.convolve(input[j],parameter[0][i],mode=mode)
    output.append(temp+parameter[1])

  return np.array(output)
  
  
  
  def max_pool(image, nrows, ncols):
    #Splits into submatrix and output max value, however this hast obe done on
    output=[]
    for i in range(image.shape[0]):
      array=image[i].reshape(28,28)
      r, h = array.shape
      temp=(array.reshape(h//nrows, nrows, -1, ncols)
                  .swapaxes(1, 2)
                  .reshape(-1, nrows, ncols))
      
      output.append(np.max(temp.reshape(-1,4),axis=1))
    return np.array(output)
