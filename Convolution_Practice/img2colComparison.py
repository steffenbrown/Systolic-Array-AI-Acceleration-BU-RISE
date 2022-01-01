import torch

inputMatrix = torch.rand(1, 3, 3, 3)

conv = torch.nn.Conv2d(3, 1, 2, bias=False)
convoluted = conv(inputMatrix)

kernal = conv.weight

xAccum = 0;
yAccum = 0;

flattenedInputMatrix = torch.empty(1, 1, (inputMatrix.shape[2] - kernal.shape[2] + 1)**2 * inputMatrix.shape[0], kernal.shape[2]**2 * kernal.shape[1])
flattenedKernal = torch.empty(1, 1, kernal.shape[2]**2 * kernal.shape[1], kernal.shape[0])

for channel in range(inputMatrix.shape[1]):
	for xShift in range(inputMatrix.shape[3] - kernal.shape[2] + 1):
		for yShift in range(inputMatrix.shape[2] - kernal.shape[2] + 1):
			for xPos in range(kernal.shape[3]):
				for yPos in range(kernal.shape[2]):
					flattenedInputMatrix[0][0][xAccum][yAccum] = inputMatrix[0][channel][xShift+xPos][yShift+yPos]
					xAccum+=1

					if xAccum == kernal.shape[2]**2:
						xAccum = 0
						yAccum+=1

kernAccum = 0
for z in range(kernal.shape[1]):
	for x in range(kernal.shape[2]):
		for y in range(kernal.shape[3]):
			flattenedKernal[0][0][kernAccum][0] = kernal[0][z][x][y]
			kernAccum+=1

for images in range(1, kernal.shape[0]):
	flattenedKernal[0][0][:][images] = flattenedKernal[0][0][:][0]

product = flattenedInputMatrix @ flattenedKernal

reshapedProduct = torch.empty(1, 1, convoluted.shape[2], convoluted.shape[2])
reshapeAccum = 0
for x in range(convoluted.shape[2]):
	for y in range(convoluted.shape[2]):
		reshapedProduct[0][0][x][y] = product[0][0][reshapeAccum][0]
		reshapeAccum+=1

#print(kernal)
#print(flattenedKernal)
#print(flattenedInputMatrix)
#print(inputMatrix)

print(convoluted)
print(reshapedProduct)


