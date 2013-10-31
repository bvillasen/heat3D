# 3D simulation of heat dissipation 
# made by Bruno Villasenor
# contact me at: bvillasen@gmail.com
# personal web page:  https://bvillasen.webs.com
# github: https://github.com/bvillasen

#To run you need these complementary files: CUDAheat3D.cu, volumeRender.py, CUDAvolumeRender.cu, cudaTools.py
#you can find them in my github: 
#                               https://github.com/bvillasen/volumeRender
#                               https://github.com/bvillasen/tools
import sys, time, os
import numpy as np
import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )
globals()["stepFunc"] = None
import volumeRender
from cudaTools import  gpuArray3DtocudaArray, setCudaDevice, getFreeMemory

nPoints = 128
useDevice = None
for option in sys.argv:
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("device=") != -1: useDevice = int(option[-1]) 
 
#set simulation volume dimentions 
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth
Lx = 30.0
Ly = 30.0
Lz = 30.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )

#set time parameters
dt = 0.005
simulationTime = 0

#Convert parameters to float32
xMin = np.float32(xMin)
yMin = np.float32(yMin)
zMin = np.float32(zMin)
dx = np.float32(dx)
dy = np.float32(dy)
dz = np.float32(dz)
dt = np.float32(dt)


#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
grid3D = (gridx, gridy, gridz)
block3D = (block_size_x, block_size_y, block_size_z)



#Initialize openGL
volumeRender.initGL()    
#initialize pyCUDA context 
cudaDevice = setCudaDevice(devN=useDevice, usingAnimation=True )

#Read and compile CUDA code
print "Compiling CUDA code"
cudaCodeString_raw = open("CUDAheat3D.cu", "r").read()
cudaCodeString = cudaCodeString_raw % { "BLOCK_WIDTH":block3D[0], "BLOCK_HEIGHT":block3D[1], "BLOCK_DEPTH":block3D[2], }
cudaCode = SourceModule(cudaCodeString)
tex_tempIn = cudaCode.get_texref("tex_tempIn")
surf_tempOut = cudaCode.get_surfref("surf_tempOut")
eulerKernel = cudaCode.get_function("euler_kernel_texture" )
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
copyDtoD_float = ElementwiseKernel(arguments="float *input, float *output",
			      operation = "output[i] = input[i];")
########################################################################
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));")
########################################################################
multiplyByFloat = ElementwiseKernel(arguments="float a, float *input",
			      operation = "input[i] = a*input[i];")
########################################################################
def sendToScreen( plotData ):
  #maxVal = gpuarray.max(plotData).get() + 0.00005
  #multiplyByFloat( 1./maxVal, plotData )
  floatToUchar( plotData, plotData_d )
  copyToScreenArray()
########################################################################
def rk4Step():
  global simulationTime
  slopeCoef, weight = np.float32(1.0), np.float32(0.5)
  tex_tempIn.set_array(temp_dArray)
  surf_tempOut.set_array(k1Temp_dArray)
  eulerKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight, 
	      xMin,  yMin, zMin,  dx,  dy,  dz,  np.float32(simulationTime),  dt,
	      temp_d, tempRunge_d, np.int32(0),
	      grid=grid3D,block=block3D, texrefs=[tex_tempIn])
	      
  slopeCoef = np.float32(2.0)
  tex_tempIn.set_array(k1Temp_dArray)
  surf_tempOut.set_array(k2Temp_dArray)
  eulerKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight, 
	      xMin,  yMin, zMin,  dx,  dy,  dz,  np.float32(simulationTime+dt*0.5),  dt,
	      temp_d, tempRunge_d, np.int32(0),
	      grid=grid3D,block=block3D, texrefs=[tex_tempIn])
  
  weight = np.float32(1.0)
  tex_tempIn.set_array(k2Temp_dArray)
  surf_tempOut.set_array(k1Temp_dArray)
  eulerKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight, 
	      xMin,  yMin, zMin,  dx,  dy,  dz,  np.float32(simulationTime+dt*0.5),  dt,
	      temp_d, tempRunge_d, np.int32(0),
	      grid=grid3D,block=block3D, texrefs=[tex_tempIn])
  
  slopeCoef = np.float32(1.0)
  tex_tempIn.set_array(k1Temp_dArray)
  surf_tempOut.set_array(temp_dArray)
  eulerKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), slopeCoef, weight, 
	      xMin,  yMin, zMin,  dx,  dy,  dz,  np.float32(simulationTime),  dt,
	      temp_d, tempRunge_d, np.int32(1),
	      grid=grid3D,block=block3D, texrefs=[tex_tempIn])
  copyDtoD_float( tempRunge_d, temp_d )
  simulationTime += dt
########################################################################  
  
#Initialize all gpu data
print "Initializing Data"
initialMemory = getFreeMemory( show=True )  
#Set initial temperature
temp_h = np.zeros([nDepth, nHeight, nWidth], dtype = np.float32)
temp_d = gpuarray.to_gpu(temp_h)
tempRunge_d = gpuarray.to_gpu( np.zeros_like(temp_h) )
tempRunge_d.set(temp_h)
#For texture version
temp_dArray, copyTempArray = gpuArray3DtocudaArray( temp_d, allowSurfaceBind=True )
k1Temp_dArray, copyk1TempArray = gpuArray3DtocudaArray( temp_d, allowSurfaceBind=True )
k2Temp_dArray, copyk2TempArray = gpuArray3DtocudaArray( temp_d, allowSurfaceBind=True )
#For shared version

#memory for plotting
plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 


def stepFunction():
  sendToScreen( temp_d )
  [rk4Step() for i in range(10)]
  
#change volumeRender default step function for heat3D step function
volumeRender.stepFunc = stepFunction

#run volumeRender animation
#volumeRender.animate()


