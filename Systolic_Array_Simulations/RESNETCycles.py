import torchvision.models as models
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def convolutionSizeCalculate(input_size, kernel_size, padding, stride):
    outputHeight = math.floor(((input_size[-2] + 2 * padding - kernel_size[-2]) / stride) + 1)
    outputWidth = math.floor(((input_size[-1] + 2 * padding - kernel_size[-1]) / stride) + 1)
    output_size = (input_size[0], kernel_size[0], outputHeight, outputWidth)
    return output_size



def im2col2DSizeCalculate(batch_size, num_channels, input_height, input_width, num_filters, kernel_height, kernel_width , stride, padding):
    input_size = (batch_size, num_channels, input_height, input_width)
    kernel_size = (num_filters, num_channels, kernel_height, kernel_width)
    rand_input = torch.rand(input_size)

    assert (kernel_height==kernel_width), 'Only square kernels are allowed!'

    conv = torch.nn.Conv2d(num_channels, num_filters, kernel_height, stride=stride, padding=padding)
    expected_output_size = tuple(conv(rand_input).shape)
    
    calculated_output_size = convolutionSizeCalculate(input_size, kernel_size, padding, stride)
    assert (expected_output_size==calculated_output_size), 'Output dimension test is failed!'

    flattened_ifmap_size = (batch_size*calculated_output_size[-2]*calculated_output_size[-1], kernel_height*kernel_width*num_channels)
    flattened_kernel_size = (kernel_height*kernel_width*num_channels,num_filters)

    return flattened_ifmap_size, flattened_kernel_size

def weightStationaryCycles(inputHeight, inputWidth, weightHeight, weightWidth, arrayDimensions):
    if(inputWidth == weightHeight):
        return (arrayDimensions + arrayDimensions + (arrayDimensions - 1) + (inputHeight - 1)) * (math.ceil(weightWidth / arrayDimensions) * math.ceil(weightHeight / arrayDimensions))
        #      |    Calculates the number of cycles for weight tile and input tile maxtrises.  |                   Calculates the number of tiles that are created                    |
        #           The dimensions of the weight matrix match that of the systolic array and
        #           The input matrix has one dimension matching the systolic array and one
        #           mathing the height of the IFMap.

        # The 2 portions are multiplied together since the systlic array multipication cycles are done for each tile.
    else:
        return "Error: 2 similar dimensions must match"


def inputStationaryCycles(inputHeight, inputWidth, weightHeight, weightWidth, arrayDimensions):
    if(inputWidth == weightHeight):
        return (arrayDimensions + weightWidth + (arrayDimensions - 1) + (arrayDimensions - 1)) * (math.ceil(inputWidth / arrayDimensions) * math.ceil(inputHeight / arrayDimensions))
    else:
        return "Error: 2 similar dimensions must match"



def outputStationaryCycles(inputHeight, inputWidth, weightHeight, weightWidth, arrayDimensions):
    if(inputWidth == weightHeight):
        return (2 * arrayDimensions + (arrayDimensions - 1) + (weightHeight - 1)) * (math.ceil(weightWidth / arrayDimensions) * math.ceil(inputHeight / arrayDimensions))
    else:
        return "Error: 2 opposite dimensions must match"


def weightStationaryUtilization(inputHeight, inputWidth, weightHeight, weightWidth, arrayDimensions):
    if(inputWidth == weightHeight):
        utilizationDenom = 0;
        utilizationNum = 0;

        cyclesPerTile = arrayDimensions + arrayDimensions + (arrayDimensions - 1) + (inputHeight - 1)

        for y in range(1, math.ceil(weightHeight / arrayDimensions) + 1):
            for x in range(1, math.ceil(weightWidth / arrayDimensions) + 1):
                percentage = 0;

                if(x == (math.ceil(weightWidth / arrayDimensions)) and y == (math.ceil(weightHeight / arrayDimensions))):
                    percentage = 1 - ((((arrayDimensions * math.ceil(weightWidth / arrayDimensions)) - weightWidth)) * (((arrayDimensions * math.ceil(weightHeight / arrayDimensions)) - weightHeight)) / arrayDimensions**2)
                elif(x == (math.ceil(weightWidth / arrayDimensions))):
                    percentage = 1 - ((arrayDimensions * math.ceil(weightWidth / arrayDimensions)) - weightWidth) / arrayDimensions
                elif(y == (math.ceil(weightHeight / arrayDimensions))):
                    percentage = 1 - ((arrayDimensions * math.ceil(weightHeight / arrayDimensions)) - weightHeight) / arrayDimensions
                else:
                    percentage = 1

                utilizationNum+=cyclesPerTile * percentage
                utilizationDenom+=cyclesPerTile

        return utilizationNum/utilizationDenom
    else:
        return "Error: 2 similar dimensions must match"

def inputStationaryUtilization(inputHeight, inputWidth, weightHeight, weightWidth, arrayDimensions):
    if(inputWidth == weightHeight):
        utilizationDenom = 0;
        utilizationNum = 0;

        cyclesPerTile = arrayDimensions + weightWidth + (arrayDimensions - 1) + (arrayDimensions - 1)

        for y in range(1, math.ceil(inputHeight / arrayDimensions) + 1):
            for x in range(1, math.ceil(inputWidth / arrayDimensions) + 1):
                percentage = 0;

                if(x == (math.ceil(inputWidth / arrayDimensions)) and y == (math.ceil(inputHeight / arrayDimensions))):
                    percentage = 1 - ((((arrayDimensions * math.ceil(inputWidth / arrayDimensions)) - inputWidth)) * (((arrayDimensions * math.ceil(inputHeight / arrayDimensions)) - inputHeight)) / arrayDimensions**2)
                elif(x == (math.ceil(weightWidth / arrayDimensions))):
                    percentage = 1 - ((arrayDimensions * math.ceil(inputWidth / arrayDimensions)) - inputWidth) / arrayDimensions
                elif(y == (math.ceil(weightHeight / arrayDimensions))):
                    percentage = 1 - ((arrayDimensions * math.ceil(inputHeight / arrayDimensions)) - inputHeight) / arrayDimensions
                else:
                    percentage = 1

                utilizationNum+=cyclesPerTile * percentage
                utilizationDenom+=cyclesPerTile

        return utilizationNum/utilizationDenom
    else:
        return "Error: 2 similar dimensions must match"


def outputStationaryUtilization(inputHeight, inputWidth, weightHeight, weightWidth, arrayDimensions):
    if(inputWidth == weightHeight):
        utilizationDenom = 0;
        utilizationNum = 0;

        cyclesPerTile = 2 * arrayDimensions + (arrayDimensions - 1) + (weightHeight - 1)

        for y in range(1, math.ceil(inputHeight / arrayDimensions) + 1):
            for x in range(1, math.ceil(weightWidth / arrayDimensions) + 1):
                percentage = 0;

                if(x == (math.ceil(weightWidth / arrayDimensions)) and y == (math.ceil(inputHeight / arrayDimensions))):
                    percentage = 1 - ((((arrayDimensions * math.ceil(weightWidth / arrayDimensions)) - weightWidth)) * (((arrayDimensions * math.ceil(inputHeight / arrayDimensions)) - inputHeight)) / arrayDimensions**2)
                elif(x == (math.ceil(weightWidth / arrayDimensions))):
                    percentage = 1 - ((arrayDimensions * math.ceil(weightWidth / arrayDimensions)) - weightWidth) / arrayDimensions
                elif(y == (math.ceil(weightHeight / arrayDimensions))):
                    percentage = 1 - ((arrayDimensions * math.ceil(inputHeight / arrayDimensions)) - inputHeight) / arrayDimensions
                else:
                    percentage = 1

                utilizationNum+=cyclesPerTile * percentage
                utilizationDenom+=cyclesPerTile

        return utilizationNum/utilizationDenom
    else:
        return "Error: 2 similar dimensions must match"


def resNETValues(resnet):
	resnetMod = resnet._modules

	inputChannels = []
	numFilters = []
	kernalSizes = []
	paddings = []
	strides = []

	for layerNumber in range(1, 5):
		layer = "layer" + str(layerNumber)
		for block in range(0, 2):
			for convolutionNumber in range(1, 3):
				convolution = "conv" + str(convolutionNumber)
				inputChannels.append(resnetMod[layer][block]._modules[convolution].in_channels)
				numFilters.append(resnetMod[layer][block]._modules[convolution].out_channels)
				strides.append(resnetMod[layer][block]._modules[convolution].stride[0])
				paddings.append(resnetMod[layer][block]._modules[convolution].padding[0])
				kernalSizes.append(resnetMod[layer][block]._modules[convolution].kernel_size[0])

	return inputChannels, numFilters, kernalSizes, paddings, strides

resnet18 = models.resnet18()

#print(resnet18._modules)

inputChannels, numFilters, kernalSizes, paddings, strides = resNETValues(resnet18)

batch_size = 64
input_width = 16
input_height = 16
systolic_array_size = 80

totalWeightStationaryCycles = 0
totalInputStationaryCycles = 0
totalOutputStationaryCycles = 0

batchSizeList = []
inputWidthList = []
inputHeightList = []
systolicArraySizeList = []

weightStationaryCyclesList = []
inputStationaryCyclesList = []
outputStationaryCyclesList = []

totalWeightStationaryCyclesList = []
totalInputStationaryCyclesList = []
totalOutputStationaryCyclesList = []

weightStationaryUtilizationList = []
inputStationaryUtilizationList = []
outputStationaryUtilizationList = []

interations = []

weightStationaryNetworkUtilization = 0
inputStationaryNetworkUtilization = 0
outputStationaryNetworkUtilization = 0

for x in range(len(inputChannels)):
    f_input_size, f_kernel_size = im2col2DSizeCalculate(batch_size, inputChannels[x-1], input_height, input_width, numFilters[x-1], kernalSizes[x-1], kernalSizes[x-1], strides[x-1], paddings[x-1])

    assert (f_input_size[1] == f_kernel_size[0]), 'Dimensions are inconsistent!'

    print("_______________________________________________")
    print('Flattened input size:', f_input_size, '\nFlattened kernel size:', f_kernel_size)
    print("Weight stationary cycles:", weightStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    print("Input stationary cycles:", inputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    print("Output stationary cycles:", outputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    print("Weight stationary Ultilization:", weightStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    print("Input stationary  Ultilization:", inputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    print("Output stationary Ultilization:", outputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))

    totalWeightStationaryCycles += weightStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)
    totalInputStationaryCycles += inputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)
    totalOutputStationaryCycles += outputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)

    weightStationaryCyclesList.append(weightStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    inputStationaryCyclesList.append(inputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    outputStationaryCyclesList.append(outputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))

    totalWeightStationaryCyclesList.append(totalWeightStationaryCycles)
    totalInputStationaryCyclesList.append(totalInputStationaryCycles)
    totalOutputStationaryCyclesList.append(totalOutputStationaryCycles)

    batchSizeList.append(batch_size)
    inputWidthList.append(input_width)
    inputHeightList.append(input_height)
    systolicArraySizeList.append(systolic_array_size)

    weightStationaryUtilizationList.append(weightStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    inputStationaryUtilizationList.append(inputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))
    outputStationaryUtilizationList.append(outputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size))

    interations.append(x)

    weightStationaryNetworkUtilization+=weightStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)*weightStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)
    inputStationaryNetworkUtilization+=inputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)*inputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)
    outputStationaryNetworkUtilization+=outputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)*outputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)



print("_______________________________________________")
print("Total weight stationary cycles:", totalWeightStationaryCycles)
print("Total input stationary cycles:", totalInputStationaryCycles)
print("Total output stationary cycles:", totalOutputStationaryCycles)
print("Total weight stationary utilization:", weightStationaryNetworkUtilization/totalWeightStationaryCycles)
print("Total input stationary utilization:", inputStationaryNetworkUtilization/totalInputStationaryCycles)
print("Total output stationary utilization:", outputStationaryNetworkUtilization/totalOutputStationaryCycles)


tableValues = {'Input Width': inputWidthList, 'Input Height': inputHeightList, 'Batch Size': batchSizeList, 'Syst. Array Dims.': systolicArraySizeList, 'Input Channels': inputChannels, 'Filters': numFilters, 'Kernal Dims': kernalSizes, 'Padding': paddings, 'Stride': strides, "Weight St. Cycles": weightStationaryCyclesList, "Input St. Cycles": inputStationaryCyclesList, "Output St. Cycles": outputStationaryCyclesList, "Weight St. Utl.": weightStationaryUtilizationList, "Input St. Utl.": inputStationaryUtilizationList, "Output St. Utl.": outputStationaryUtilizationList}
table = pd.DataFrame(data=tableValues)
print("_______________________________________________")
print("Datatable for graphs 1 and 2")
print("_______________________________________________")
print(table)

fig = plt.figure()

x = interations

print("Loading Total Cycles Per Layer Graph")
ax1 = fig.add_subplot(231)

ax1.plot(x, totalOutputStationaryCyclesList, '-o', c="g", label="Output Stationary")
ax1.plot(x, totalWeightStationaryCyclesList, '-o', c="b", label="Weight Stationary")
ax1.plot(x, totalInputStationaryCyclesList, '-o', c="r", label="Input Stationary")

plt.legend(loc='upper left')
plt.xlabel("Convolution Layer")
plt.ylabel("Total Cycles Preformed")

print("Loading Cycles Per Layer Graph")
ax2 = fig.add_subplot(232)

ax2.plot(x, outputStationaryCyclesList, '-o', c="g", label="Output Stationary")
ax2.plot(x, weightStationaryCyclesList, '-o', c="b", label="Weight Stationary")
ax2.plot(x, inputStationaryCyclesList, '-o', c="r", label="Input Stationary")

plt.legend(loc='upper left')
plt.xlabel("Convolution Layer")
plt.ylabel("Cycles Preformed")

totalOutputStationaryCycles = 0;
totalInputStationaryCycles = 0;
totalWeightStationaryCycles = 0;

batchList = []
weightStationaryThroughput = []
inputStationaryThroughput = []
outputStationaryThroughput = []

print("Loading Throughput vs Batch Size Graph")
for batch in range(1, 32, 2):
    for x in range(len(inputChannels)):
        f_input_size, f_kernel_size = im2col2DSizeCalculate(batch, inputChannels[x-1], input_height, input_width, numFilters[x-1], kernalSizes[x-1], kernalSizes[x-1], strides[x-1], paddings[x-1])

        totalWeightStationaryCycles += weightStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)
        totalInputStationaryCycles += inputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)
        totalOutputStationaryCycles += outputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], systolic_array_size)

    batchList.append(batch)
    weightStationaryThroughput.append(batch/totalWeightStationaryCycles)
    inputStationaryThroughput.append(batch/totalInputStationaryCycles)
    outputStationaryThroughput.append(batch/totalOutputStationaryCycles)

    totalOutputStationaryCycles = 0;
    totalInputStationaryCycles = 0;
    totalWeightStationaryCycles = 0;

ax3 = fig.add_subplot(233)

ax3.plot(batchList, outputStationaryThroughput, '-o', c="g", label="Output Stationary")
ax3.plot(batchList, weightStationaryThroughput, '-o', c="b", label="Weight Stationary")
ax3.plot(batchList, inputStationaryThroughput, '-o', c="r", label="Input Stationary")

plt.legend(loc='upper left')
plt.xlabel("Batch Size")
plt.ylabel("Throughput")

totalOutputStationaryCycles = 0;
totalInputStationaryCycles = 0;
totalWeightStationaryCycles = 0;

weightStationaryThroughput = []
inputStationaryThroughput = []
outputStationaryThroughput = []
arraySizeList = []

print("Loading Throughput vs Array Size Graph")
for arsi in range(1, 7):
    arraySize = 4 * (2**arsi)
    for x in range(len(inputChannels)):
        f_input_size, f_kernel_size = im2col2DSizeCalculate(batch_size, inputChannels[x-1], input_height, input_width, numFilters[x-1], kernalSizes[x-1], kernalSizes[x-1], strides[x-1], paddings[x-1])

        totalWeightStationaryCycles += weightStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], arraySize)
        totalInputStationaryCycles += inputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], arraySize)
        totalOutputStationaryCycles += outputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], arraySize)

    arraySizeList.append(arraySize)
    weightStationaryThroughput.append(batch_size/totalWeightStationaryCycles)
    inputStationaryThroughput.append(batch_size/totalInputStationaryCycles)
    outputStationaryThroughput.append(batch_size/totalOutputStationaryCycles)

    totalOutputStationaryCycles = 0;
    totalInputStationaryCycles = 0;
    totalWeightStationaryCycles = 0;

ax4 = fig.add_subplot(234)

ax4.plot(arraySizeList, outputStationaryThroughput, '-o', c="g", label="Output Stationary")
ax4.plot(arraySizeList, weightStationaryThroughput, '-o', c="b", label="Weight Stationary")
ax4.plot(arraySizeList, inputStationaryThroughput, '-o', c="r", label="Input Stationary")

plt.legend(loc='upper left')
plt.xlabel("Array Size")
plt.ylabel("Throughput")

weightStationaryUtilizationList = []
inputStationaryUtilizationList = []
outputStationaryUtilizationList = []
arraySizeList = []

weightStationaryUtilizationListAverage = []
inputStationaryUtilizationListAverage = []
outputStationaryUtilizationListAverage = []

print("Loading Ultilzation Graph")

for arraySize in range(10, 256, 2):
    for x in range(len(inputChannels)):
        f_input_size, f_kernel_size = im2col2DSizeCalculate(batch_size, inputChannels[x-1], input_height, input_width, numFilters[x-1], kernalSizes[x-1], kernalSizes[x-1], strides[x-1], paddings[x-1])
        weightStationaryUtilizationList.append(weightStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], arraySize))
        inputStationaryUtilizationList.append(inputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], arraySize))
        outputStationaryUtilizationList.append(outputStationaryUtilization(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], arraySize))

    arraySizeList.append(arraySize)
    weightStationaryUtilizationListAverage.append(sum(weightStationaryUtilizationList)/len(weightStationaryUtilizationList))
    inputStationaryUtilizationListAverage.append(sum(inputStationaryUtilizationList)/len(inputStationaryUtilizationList))
    outputStationaryUtilizationListAverage.append(sum(outputStationaryUtilizationList)/len(outputStationaryUtilizationList))

    weightStationaryUtilizationList = []
    inputStationaryUtilizationList = []
    outputStationaryUtilizationList = []


ax5 = fig.add_subplot(235)

ax5.plot(arraySizeList, outputStationaryUtilizationListAverage, '-o', c="g", label="Output Stationary")
ax5.plot(arraySizeList, weightStationaryUtilizationListAverage, '-o', c="b", label="Weight Stationary")
ax5.plot(arraySizeList, inputStationaryUtilizationListAverage, '-o', c="r", label="Input Stationary")

plt.legend(loc='upper left')
plt.xlabel("Array Size")
plt.ylabel("Average Utilization")

plt.show()
