#Convoludes a random matrix using the torch Conv2d method

import torch

#Define input matrix of quantity 1, 3 layers, and 3x3
inputMatrix = torch.rand(1, 3, 3, 3)

#Create convolution for 3 input channels, 1 output channel, and 2x2 kernal
conv = torch.nn.Conv2d(3, 1, 2)
#Convolude the matrix
convoluted = conv(inputMatrix)

print(convoluted)
print(conv.__dict__)
