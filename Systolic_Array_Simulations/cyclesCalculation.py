def weightStationaryCycles(inputHeight, inputWidth, weightHeight, weightWidth):
	if(weightHeight == inputHeight):
		return weightHeight + weightWidth + (weightHeight - 1) + (inputWidth - 1)
	elif(weightWidth == inputWidth):
		return weightWidth + weightHeight + (weightWidth - 1) + (inputHeight - 1)
	else:
		return "Error: 2 similar dimensions must match"


def inputStationaryCycles(inputHeight, inputWidth, weightHeight, weightWidth):
	if(inputHeight == weightHeight):
		return inputHeight + inputWidth + (inputHeight - 1) + (weightWidth - 1)
	elif(inputWidth == weightWidth):
		return inputWidth + inputHeight + (inputWidth - 1) + (weightHeight - 1)
	else:
		return "Error: 2 similar dimensions must match"



def outputStationaryCycles(inputHeight, inputWidth, weightHeight, weightWidth):
	if(inputHeight == weightWidth):
		return weightWidth + (inputWidth - 1) + (weightHeight - 1)
	elif(inputWidth == weightHeight):
		return inputWidth + (weightWidth - 1) + (inputHeight - 1)
	else:
		return "Error: 2 opposite dimensions must match"


print(weightStationaryCycles(3, 2, 1, 3))

