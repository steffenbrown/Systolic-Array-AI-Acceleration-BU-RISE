import torch
import math

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


batch_size = 4
num_channels = 3
input_height = 3
input_width = 3
num_filters = 10
kernel_height = 3
kernel_width = 3
stride = 2
padding = 1

f_input_size, f_kernel_size = im2col2DSizeCalculate(batch_size, num_channels, input_height, input_width, num_filters, kernel_height, kernel_width, stride, padding)

assert (f_input_size[1] == f_kernel_size[0]), 'Dimensions are inconsistent!'

print('Flattened input size:', f_input_size, '\nFlattened kernel size:', f_kernel_size)
print("Weight stationary cycles:", weightStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], 40))
print("Input stationary cycles:", inputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], 40))
print("Output stationary cycles:", outputStationaryCycles(f_input_size[0], f_input_size[1], f_kernel_size[0], f_kernel_size[1], 40))