// #include <pycuda-complex.hpp>
#include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>
// #define pi 3.14159265f

texture< float, cudaTextureType3D, cudaReadModeElementType> tex_tempIn;
surface< void, cudaSurfaceType3D> surf_tempOut;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float heatCore( int nWidth, int nHeight, float nDepth, float t, float xMin, float yMin, float zMin, 
				 float dx, float dy, float dz, int t_i, int t_j, int t_k ){
 
  float center, right, left, up, down, top, bottom, result, laplacian;
  center = tex3D(tex_tempIn, (float)t_j, (float)t_i, (float)t_k);
  up =     tex3D(tex_tempIn, (float)t_j, (float)t_i+1, (float)t_k);
  down =   tex3D(tex_tempIn, (float)t_j, (float)t_i-1, (float)t_k);
  right =  tex3D(tex_tempIn, (float)t_j+1, (float)t_i, (float)t_k);
  left =   tex3D(tex_tempIn, (float)t_j-1, (float)t_i, (float)t_k);
  top =    tex3D(tex_tempIn, (float)t_j, (float)t_i, (float)t_k+1);
  bottom = tex3D(tex_tempIn, (float)t_j, (float)t_i, (float)t_k-1);

  //Boundry Conditions
  if (t_j==0 ) left = 0.85f;
  if (t_j==nWidth-1) right = 0.f;
  if (t_i==0) down = 0.85f;
  if (t_i==nHeight-1) up = 0.f;
  if (t_k==0) bottom = 0.f;
  if (t_k==nDepth-1) top = 0.f;

  float dxInv = 1.0f/dx;
  float dyInv = 1.0f/dy;
  float dzInv = 1.0f/dz;

  laplacian = (up + down - 2.f*center )*dyInv*dyInv + (right + left - 2.f*center )*dxInv*dxInv + (top + bottom - 2.f*center )*dzInv*dzInv;
  result = laplacian;

  return result;
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////           EULER                //////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void euler_kernel_texture( int nWidth, int nHeight, int nDepth, float slopeCoef, float weight, 
				      float xMin, float yMin, float zMin, float dx, float dy, float dz, float t, float dt, 
				      float *psi1Real_d,
				      float *psiRungeReal,
				      int lastRK4Step){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  float incrementReal;  
  incrementReal = heatCore( nWidth, nHeight, nDepth, t, xMin, yMin, zMin, dx, dy, dz, t_i, t_j, t_k);
  incrementReal = dt*incrementReal;
 
  float valueReal;
  if (lastRK4Step ){
    valueReal = psiRungeReal[tid] + slopeCoef*incrementReal/6.0f;
    psiRungeReal[tid] = valueReal;
    surf3Dwrite(  valueReal, surf_tempOut,  t_j*sizeof(float), t_i, t_k,  cudaBoundaryModeClamp);
  }
  
  else{
    valueReal = psi1Real_d[tid] + weight*incrementReal;
    surf3Dwrite(  valueReal, surf_tempOut,  t_j*sizeof(float), t_i, t_k,  cudaBoundaryModeClamp);
    //add to rk4 final value
    psiRungeReal[tid] = psiRungeReal[tid] + slopeCoef*incrementReal/6.0f;
  }
}