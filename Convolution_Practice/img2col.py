#Mirrors the visual on slide 4 of Convolutions - https://docs.google.com/presentation/d/1Tmhu4L5vTkYqR71-BspQeXWymGKoRcmAlvqhX_zREek/edit?usp=sharing

import torch

#Create IFmap matrix with 1 image, 1 layer(consisting of 3 flattened layers), and 12x4 per image
inputMatrix = torch.rand(8, 12)
#Create Filter matrix with 1 kernals, 1 layer(consisting of 2 flattened kernals), and 1x12 per kernal
kernal = torch.rand(12, 2)

#preform matrix multipication on the maps to form a 1x4 OFmap matrix
product = torch.matmul(inputMatrix, kernal)

print(product)