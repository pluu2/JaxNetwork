def adj_matrix(row,height,padding):  #row_length is WITH padding. and assumes stride is 1
  padding=padding*2
  row=row+padding
  height=height+padding
  am =onp.zeros(((row)*(height),(row-padding)*(height-padding))) #adj matrix with 11*11 inputs, with 1 kernel doing an entire convolution for now. Imagine the first diemsnion like a linear line (For now only works on equal sized images. 
  
  #one row has (max length-kernel_size) (so 30-3 or 27 strides per row, with 27 stides in columbs 
  #This builds the adjacency matrix. 


  #the jumps will be (11) it just appears you jump 11 ahead on a linear scale. 
  start=0
  end=3
  jump = row #The jump is a row length
  mod=0
  ctr=0
  for i in range(row-padding): #assigning the pixels to a given stride, which is maxed out at 28*28 due to assumed 'same' padding. 
    for j in range(height-padding):
      am[start+mod:end+mod,ctr]=1
      am[start+mod+jump:end+mod+jump,ctr]=1
      am[start+mod+jump+jump:end+mod+jump+jump,ctr]=1
      mod+=1
      ctr+=1 
    start=end-1
    end =start+3

  return am 

def im2col(row,height,depth,padding):
  
  padded_image=onp.ones(((row+(padding*2))*(row+(padding*2)))).astype('float16')
  am=adj_matrix(row,height,padding)
  indices=am.T*padded_image #this is interesting as it can theoretically be used to make any shaped kernels.

  window_indices=[]
  image_indices=[]
  for i in range(row*height):
    window_indices.append(onp.nonzero(indices[i])[0])
  window_indices=np.array(window_indices).T
  image_indices.append(window_indices)
  for i in range(1,depth): #how many layers. 
    window_indices=window_indices+((row+(padding*2))*(row+(padding*2)))
    image_indices.append(window_indices)

  return onp.array(image_indices)

def add_pad (image_,kernelw_,kernelh_):  #input image are channel,h,w
  #new add_pad
  d,h,w=image_.shape
  #d=1
  padw=onp.zeros((d,w,onp.divmod(kernelw_,2)[0]))

  padded=onp.concatenate([padw,image_],axis=2) #you want to add, at the start if a 
  padded=onp.concatenate([padded,padw[:]],axis=2)

  padh=onp.zeros((d,onp.divmod(kernelh_,2)[0],padded.shape[2]))

  padded=onp.concatenate([padh,padded],axis=1)
  padded=onp.concatenate([padded,padh],axis=1)
  return padded

#input_image must be image (h,w,d), even if zero it needs one. 
def conv (input_image,parameter,im2col_matrix): #parameter will be shaped. [layer,weight/bias,filter#], on input the parameter will be just [weight/bias,filter#]
  d,h,w = input_image.shape
  kh,kw=parameter[0][0].shape #fetches the kernel size
  weights=parameter[0].reshape(-1,kh*kw) #changes the kernel into [filter,3*3] to linearlize. 
  weights=np.tile(weights,d) #this value will be number of channels as it has to repeat.  #tile the weights, depth times, as the same weights are used for every depth. 
  #print(weights.shape) #this worked
  #print(kh) #correct
  #print(kw)#correct
  input_image=add_pad(input_image,kh,kw) #add padding 
  #print(input_image.shape)
  input_image = input_image.flatten()
  #print(input_image.shape) #if this is multi-channel it will also put out a a h*w*d sized vector. 
  
  conv_image=input_image[im2col_matrix].T

  conv_image=conv_image.reshape(w*h,kh*kh*d)
  convolved=(np.dot(conv_image,weights.T) + parameter[1]).T
  convolved=convolved.reshape(len(parameter[0]),h,w)

  return convolved
