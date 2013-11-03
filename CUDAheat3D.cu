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

__global__ void euler_kernel_shared( int nWidth, int nHeight, int nDepth, float slopeCoef, float weight, 
				      float xMin, float yMin, float zMin, float dx, float dy, float dz, float t, float dt, 
				      float *tempFirst,
				      float *inputTemp, float *outputTemp,
				      float *tempRunge,
				      int lastRK4Step){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  float dxInv = 1.0f/dx;
  float dyInv = 1.0f/dy;
  float dzInv = 1.0f/dz;
  
  //copy data to shared memory
  int tid2 = tid;
  const int nNeigh = 2;  //Number of neighbors for spatial derivatives
  int t_x = threadIdx.x + nNeigh;
  int t_y = threadIdx.y + nNeigh;
  int t_z = threadIdx.z + nNeigh;
  float val;
  __shared__ float shrd_temp[ %(BLOCK_WIDTH)s + 2*nNeigh ][ %(BLOCK_HEIGHT)s + 2*nNeigh ][ %(BLOCK_DEPTH)s + 2*nNeigh ];
  shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
  //fill x halo
  if ( t_x<2*nNeigh ){
    tid2 = (t_j-nNeigh) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    if (blockIdx.x==0) val = 0.5f;
    else val = inputTemp[tid2];
    shrd_temp[t_x-nNeigh][t_y][t_z] = val;
    tid2 = (t_j) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  }
  if ( t_x >= blockDim.x ){
    tid2 = (t_j+nNeigh) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    if (blockIdx.x == gridDim.x-1) val = 0.5f;
    else val = inputTemp[tid2];
    shrd_temp[t_x+nNeigh][t_y][t_z] = val;
    tid2 = (t_j) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  }
  //fill y halo
  if ( t_y<2*nNeigh ){
    tid2 = (t_j) + (t_i-nNeigh)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    if (blockIdx.y==0) val = 0.f;
    else val = inputTemp[tid2];
    shrd_temp[t_x][t_y-nNeigh][t_z] = val;
    tid2 = (t_j) + (t_i)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  }
  if ( t_y >= blockDim.y ){
    tid2 = (t_j) + (t_i+nNeigh)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    if (blockIdx.y == gridDim.y-1) val = 0.f;
    else val = inputTemp[tid2];
    shrd_temp[t_x][t_y+nNeigh][t_z] = val;
    tid2 = (t_j) + (t_i)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  }  
  //fill z halo
  if ( t_z<2*nNeigh ){
    tid2 = (t_j) + (t_i)*blockDim.x*gridDim.x + (t_k-nNeigh)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    if (blockIdx.z==0) val = 0.f;
    else val = inputTemp[tid2];
    shrd_temp[t_x][t_y][t_z-nNeigh] = val;
    tid2 = (t_j) + (t_i)*blockDim.x*gridDim.x + (t_k)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  }
  if ( t_z >= blockDim.z ){
    tid2 = (t_j) + (t_i)*blockDim.x*gridDim.x + (t_k+nNeigh)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    if (blockIdx.z == gridDim.z-1) val = 0.f;
    else val = inputTemp[tid2];
    shrd_temp[t_x][t_y][t_z+nNeigh] = val;
    tid2 = (t_j) + (t_i)*blockDim.x*gridDim.x + (t_k)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  }    
  __syncthreads(); 
  
  
//   //fill x halo
//   t_x -= 1;
//   t_j -= 1;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if (t_x < 1){
//     if (blockIdx.x == 0 ) shrd_temp[t_x][t_y][t_z] = 0.5f;
//     else shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   }
//   t_x += 2;
//   t_j += 2;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if (t_x > blockDim.x ) {
//     if (blockIdx.x == gridDim.x-1 ) shrd_temp[t_x][t_y][t_z] = 0.5f;
//     else shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   }
//   t_x -= 1;
//   t_j -= 1;  
//   //fill y halo
//   t_y -= 1;
//   t_i -= 1;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if (t_y < 1){
//     if (blockIdx.y == 0 ) shrd_temp[t_x][t_y][t_z] = 0.f;
//     else shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   }
//   t_y += 2;
//   t_i += 2;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if (t_y > blockDim.y ) {
//     if (blockIdx.y == gridDim.y-1 ) shrd_temp[t_x][t_y][t_z] = 0.f;
//     else shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   }
//   t_y -= 1;
//   t_i -= 1;
//   //fill z halo
//   t_z -= 1;
//   t_k -= 1;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if (t_z < 1){
//     if (blockIdx.z == 0 ) shrd_temp[t_x][t_y][t_z] = 0.f;
//     else shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   }
//   t_z += 2;
//   t_k += 2;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if (t_z > blockDim.z ){
//     if (blockIdx.z == gridDim.z-1 ) shrd_temp[t_x][t_y][t_z] = 0.f;
//     else shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   }
//   t_z -= 1;
//   t_k -= 1;
//   __syncthreads(); 


//   float val1, val2;
  float center = shrd_temp[t_x][t_y][t_z];
  float laplacian = 0.f;
  //Add x derivative
  laplacian += ( shrd_temp[t_x-1][t_y][t_z] + shrd_temp[t_x+1][t_y][t_z] - 2.f*center)*dxInv*dxInv;
  //Add y derivative
  laplacian += ( shrd_temp[t_x][t_y-1][t_z] + shrd_temp[t_x][t_y+1][t_z] - 2.f*center)*dyInv*dyInv;
  //Add z derivative
  laplacian += ( shrd_temp[t_x][t_y][t_z-1] + shrd_temp[t_x][t_y][t_z+1] - 2.f*center)*dzInv*dzInv; 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
//   //copy data to shared memory
//   int tid2;
//   __shared__ float shrd_temp[ %(BLOCK_WIDTH)s + 2 ][ %(BLOCK_HEIGHT)s + 2 ][ %(BLOCK_DEPTH)s + 2 ];
//   int t_x = threadIdx.x;
//   int t_y = threadIdx.y;
//   int t_z = threadIdx.z;
//   t_j -= 1;
//   t_i -= 1;
//   t_k -= 1;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   t_j += 2;
//   t_x += 2;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if (t_x > nWidth-2) shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   t_i += 2;
//   t_y += 2;
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if ( (t_y > nHeight-2) and (t_x > nWidth-2) ) shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   t_k += 2;
//   t_z += 2;  
//   tid2 = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   if ( (t_z > nDepth-2) and (t_y > nHeight-2) and (t_x > nWidth-2) ) shrd_temp[t_x][t_y][t_z] = inputTemp[tid2];
//   __syncthreads();
  

   
//   //GLOBAL MEMORY
//   float center = inputTemp[tid];
//   float laplacian = 0.f;
//   int tid_1, tid_2;
//   float val1, val2;
//   //Add x derivative
//   tid_1 = (t_j+1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   tid_2 = (t_j-1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   val1 = inputTemp[tid_1];
//   val2 = inputTemp[tid_2];
//   if (t_j == 0) val2 = 0.8f;
//   if (t_j == nWidth -1 ) val1 = 0.0f;
//   laplacian += ( val1 + val2 - 2.f*center)*dxInv*dxInv;
//   //Add y derivative
//   tid_1 = t_j + (t_i+1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   tid_2 = t_j + (t_i-1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   val1 = inputTemp[tid_1];
//   val2 = inputTemp[tid_2];
//   if (t_i == 0) val2 = 0.8f;
//   if (t_i == nHeight -1 ) val1 = 0.0f;
//   laplacian += ( val1 + val2 - 2.f*center)*dyInv*dyInv;
//   //Add z derivative
//   tid_1 = t_j + t_i*blockDim.x*gridDim.x + (t_k+1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   tid_2 = t_j + t_i*blockDim.x*gridDim.x + (t_k-1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   val1 = inputTemp[tid_1];
//   val2 = inputTemp[tid_2];
//   if (t_k == 0) val2 = 0.8f;
//   if (t_k == nDepth -1 ) val1 = 0.0f;
//   laplacian += ( val1 + val2 - 2.f*center)*dzInv*dzInv;
  
  float increment = dt * laplacian;
  

  if (lastRK4Step )
    tempRunge[tid] = tempRunge[tid] + slopeCoef*increment/6.0f;
  else{
    outputTemp[tid] = tempFirst[tid] + weight*increment;
    tempRunge[tid] = tempRunge[tid] + slopeCoef*increment/6.0f;
  }
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
  incrementReal = dt * heatCore( nWidth, nHeight, nDepth, t, xMin, yMin, zMin, dx, dy, dz, t_i, t_j, t_k);
//   incrementReal = dt*incrementReal;
 
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