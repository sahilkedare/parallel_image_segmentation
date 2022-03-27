#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "util.h"
#include "quickshift.h"

// use texture to simplize memory access
texture<float, cudaTextureType3D, cudaReadModeElementType> texImg;
texture<float, cudaTextureType2D, cudaReadModeElementType> texEnrg;

__global__ void getTree(
	int w, int h, int d,
	float sigma3, float tau2, float ratio,
	float* parent, float * dist
){
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	if(globalY >= w || globalX >= h) return;

	// pos of pixels to be checked
	int x_start = globalX < sigma3 ? 0 : globalX - sigma3;
	int x_end = globalX + sigma3 > w ? w : globalX + sigma3;
	int y_start = globalY < sigma3 ? 0 : globalY - sigma3;
	int y_end = globalY + sigma3 > h ? h : globalY + sigma3;

	// get the vector for current pixel
	float c[3];
	for(int k = 0; k < d; ++k){
		c[k] = Img(globalX, globalY, k);
	}

	// init temp variables
	float currE = Enrg(globalX, globalY);
	float minDist = INF;
	float atom;
	float temp;
	int x = globalX;
	int y = globalY;

	for(int i = x_start; i < x_end; ++i){
		for(int j = y_start; j < y_end; ++j){
			if(Enrg(i, j) > currE){
				temp = 0;
				for(int k = 0; k < d; ++k){
					atom = Img(i, j, k) - c[k];
					temp += atom * atom;
				}
				atom = ratio * (globalX - i);
				temp += atom * atom;
				atom = ratio * (globalY - j);
				temp += atom * atom;
				if(temp < minDist && temp < tau2){
					x = i;
					y = j;
					minDist = temp;
				}
			}
		}
	}

	parent[globalX * w + globalY] = (x * w + y); // parent[.] % width = y
	dist[globalX * w + globalY] = ((x != globalX || y != globalY) ?
																sqrt(minDist) : INF);
}

__global__ void getDist(
	int w, int h, int d,
	float sigma3, float ratio,
	float* enrg
){
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	if(globalY >= w || globalX >= h) return;

	// pos of pixels to be checked
	int x_start = globalX < sigma3 ? 0 : globalX - sigma3;
	int x_end = globalX + sigma3 > w ? w : globalX + sigma3;
	int y_start = globalY < sigma3 ? 0 : globalY - sigma3;
	int y_end = globalY + sigma3 > h ? h : globalY + sigma3;

	// cache the vector for current pixel
	float c[3];
	for(int k = 0; k < d; ++k){
		c[k] = Img(globalX, globalY, k);
	}

	// init temp variables
	float E = 0;
	float temp;
	float atom;
	for(int i = x_start; i < x_end; ++i){
		for(int j = y_start; j < y_end; ++j){
			temp = 0;
			// get distance
			for(int k = 0; k < d; ++k){
				atom = Img(i, j, k) - c[k];
				temp += atom * atom;
			}
			atom = ratio * (globalX - i);
			temp += atom * atom;
			atom = ratio * (globalY - j);
			temp += atom * atom;
			// add into energy
			E += exp(-temp * 4.5 / sigma3 / sigma3);
		}
	}
	// normalize
	enrg[globalX * w + globalY] = E / (x_end - x_start) / (y_end - y_start);
}

void quickshift(
	float* img,
	int w, int h, int d,
	float sigma3, float tau2, float ratio,
	float* parent, float* dist
){
	//////////////////////////////////////////////////////////////////////////////
	// generate texture for image
	//////////////////////////////////////////////////////////////////////////////
	// allocate device array
	cudaArray* arr_img;
	cudaChannelFormatDesc des_img = cudaCreateChannelDesc<float>();
	cudaExtent const ext = {w, h, d};
	gpuErrChk( cudaMalloc3DArray(&arr_img, &des_img, ext));

	// copy img from host to device
	cudaMemcpy3DParms cpyParms = {0};
	cpyParms.dstArray = arr_img;
	cpyParms.kind = cudaMemcpyHostToDevice;
	cpyParms.extent = make_cudaExtent(w, h, d);
	cpyParms.srcPtr = make_cudaPitchedPtr((void*) &img[0],
				ext.width * sizeof(float), ext.width, ext.height);
	gpuErrChk( cudaMemcpy3D(&cpyParms) );

	// bind texture to img array
	gpuErrChk( cudaBindTextureToArray(texImg, arr_img, des_img));
	// set soecial parameters for texture
	texImg.normalized = false;
	texImg.filterMode = cudaFilterModePoint;

	//////////////////////////////////////////////////////////////////////////////
	// compute pixel energy
	//////////////////////////////////////////////////////////////////////////////
	// malloc for device variables
	float *img_d;
	float *dist_d;
	float *parent_d;
	float *enrg_d;
	size_t size = w * h * sizeof(float);
	gpuErrChk( cudaMalloc( (void**) &img_d, size * d) ); // malloc for img
	gpuErrChk( cudaMalloc( (void**) &dist_d, size) ); // for pixel distance
	gpuErrChk( cudaMalloc( (void**) &parent_d, size) ); // for pixel parent
	gpuErrChk( cudaMalloc( (void**) &enrg_d, size) ); // for energy
	// prepare data
	gpuErrChk( cudaMemcpy(img_d, img, size * d, cudaMemcpyHostToDevice) );
	gpuErrChk( cudaMemset(enrg_d, 0, size) );

	// launch kernel to compute energy
	dim3 dimBlock(B_X, B_Y, 1);
	dim3 dimGrid((h % B_X != 0) ? h / B_X + 1 : h / B_X,
		(w % B_Y != 0) ? w / B_Y + 1 : w / B_Y, 1);
	getDist<<<dimGrid, dimBlock>>>(w, h, d, sigma3, ratio, enrg_d);
	gpuErrChk( cudaPeekAtLastError() );
	gpuErrChk( cudaThreadSynchronize() );

	// copy energy from device to host;
	// cudaMemcpy(enrg, enrg_d, size, cudaMemcpyDeviceToHost);

	//////////////////////////////////////////////////////////////////////////////
	// generate texture for energy
	//////////////////////////////////////////////////////////////////////////////
	cudaArray* arr_enrg;
	cudaChannelFormatDesc des_enrg = cudaCreateChannelDesc<float>();
	gpuErrChk( cudaMallocArray(&arr_enrg, &des_enrg, w, h) );
	size_t size_enrg = w * h * sizeof(float);
	gpuErrChk( cudaMemcpyToArray(arr_enrg, 0, 0, enrg_d, size_enrg,
		cudaMemcpyDeviceToDevice) );
	gpuErrChk( cudaBindTextureToArray(texEnrg, arr_enrg, des_enrg) );
	texEnrg.normalized = false;
	texEnrg.filterMode = cudaFilterModePoint;

	gpuErrChk( cudaThreadSynchronize() );

	//////////////////////////////////////////////////////////////////////////////
	// compute distance
	//////////////////////////////////////////////////////////////////////////////
	// launch kernel to compute distance
	getTree<<<dimGrid, dimBlock>>>(w,h,d, sqrt(tau2), tau2, ratio, parent_d, dist_d);
	gpuErrChk( cudaPeekAtLastError() );
	gpuErrChk( cudaThreadSynchronize() );

	// copy parent & distance from device to host
	gpuErrChk( cudaMemcpy(parent, parent_d, size, cudaMemcpyDeviceToHost) );
	gpuErrChk( cudaMemcpy(dist, dist_d, size, cudaMemcpyDeviceToHost) );

	//////////////////////////////////////////////////////////////////////////////
	// free memory
	//////////////////////////////////////////////////////////////////////////////
	gpuErrChk( cudaFree(img_d) );
	gpuErrChk( cudaFree(dist_d) );
	gpuErrChk( cudaFree(parent_d) );
	gpuErrChk( cudaFree(enrg_d) );
	gpuErrChk( cudaFreeArray(arr_img) );
	gpuErrChk( cudaFreeArray(arr_enrg) );
	gpuErrChk( cudaUnbindTexture(texImg) );
	gpuErrChk( cudaUnbindTexture(texEnrg) );
}

__global__ void getDist_shared(
  int w, int h, int sigma3, float ratio2,
  float* enrg_d
){
  extern __shared__ float color[];
  int globalX = blockIdx.x + threadIdx.x - sigma3;
  int globalY = blockIdx.y + threadIdx.y - sigma3;
  if(globalX < 0 || globalY < 0 || globalX >= h || globalY >= w){ return; }
  /*
  int globalZ = blockIdx.z;
  int localX = threadIdx.x;
  int localY = threadIdx.y;
  int tarX = blockIdx.x;
  int tarY = blockIdx.y;
  int tarZ = blockIdx.z;
  */
  color[threadIdx.x * w + threadIdx.y] =
    img_d[globalX * w + globalY + w * h * blockIdx.z];
  __syncthreads();
  float center = color[sigma3 * width + sigma3];
  float d = color[threadIdx.x * w + threadIdx.y] - center;
  d = d * d;
  if(blockIdx.z == 2)
  {
    d += ratio2 * (sigma3 - threadIdx.x) * (sigma3 - threadIdx.x);
  }
  else if(blockIdx.z == 1)
  {
    d += ratio2 * (sigma3 - threadIdx.y) * (sigma3 - threadIdx.y);
  }
  /*
  exclusive_single_thread
  {
    dist_d[globalX * w + globalY] += d;
  }*/
  atomicAdd(dist_d + blockIdx.x * w + blockIdx.y, d);
}

__global__ void getTree_shared(
  float* dist_d,
  int w, int h, int sigma3, float tau2,
  float* parent_d
){
  extern __shared__ float enrg[];
  int globalX = blockIdx.x + threadIdx.x - sigma3;
  int globalY = blockIdx.y + threadIdx.y - sigma3;
  if(globalX < 0 || globalY < 0 || globalX >= h || globalY >= w){ return; }
  if(globalX == 0 && globalY == 0)
  {
    enrg[(2 * sigma3 + 1) * (2 * sigma3 + 1)] = INF;
  }
  enrg[threadIdx.x * w + threadIdx.y] = exp(-dist_d[globalX * w + globalY] *
    4.5 / sigma3 / sigma3);
  int x_start = globalX > sigma3 ? globalX - sigma3 : 0;
  int x_end = globalX + sigma3 < h ? globalX + sigma3 : h;
  int y_start = globalY > sigma3 ? globalY - sigma3 : 0;
  int y_end = globalY + sigma3 < w ? globalY + sigma3 : w;
  enrg[threadIdx.x * w + threadIdx.y] /= (x_end - x_start) * (y_end - y_start);
  __syncthreads();

  float center = enrg[sigma3 * width + sigma3];
  if(enrg[threadIdx.x * w + threadIdx.y] < center){ return; }
  if(enrg[threadIdx.x * w + threadIdx.y] >
    exp(-tau2 * 4.5 / sigma3 / sigma3) / (x_end - x_start) / (y_end - y_start))
  { return; }
  exclusive_single_thread
  {
    if(enrg[threadIdx.x * w + threadIdx.y] <
       enrg[(2 * sigma3 + 1) * (2 * sigma3 + 1)])
    {
      enrg[(2 * sigma3 + 1) * (2 * sigma3 + 1)] =
      enrg[threadIdx.x * w + threadIdx.y];
      parent_d[blockIdx.x * w + blockIdx.y] = globalX * w + globalY;
    }
  }
}

void quickshift_shared(
  float* img,
  int w, int h, int d,
  float sigma3, float tau2, float ratio,
  float* parent
){
  if(sigma3 > 54)
  {
    printf("No support for large sigma(>18)!\nEnd Program.");
    return;
  }

  size_t size = w * h;
  gpuErrChk( cudaMalloc(img_d, size * d * sizeof(float)) );
  gpuErrChk( cudaMemcpy(img_d, img, size * d * sizeof(float),
             cudaMemcpyHostToDevice) );
  gpuErrChk( cudaMalloc(dist_d, size * sizeof(float)) );
  gpuErrChk( cudaMemset(dist_d, 0, size * sizeof(float)) );

  dim3 dimGrid(w, h, 3);
  size_t shared = 2 * (int)sigma3 + 1;
  dim3 dimBlock(shared, shared, 1);
  getDist_shared<<<dimGrid, dimBlock, shared * shared * sizeof(float)>>>(w, h,
    (int)sigma3, ratio * ratio, dist_d);
  gpuErrChk( cudaPeekAtLastError() );

  gpuErrChk( cudaMalloc(parent_d, size * sizeof(float)) );
  // gpuErrChk( cudaMalloc(enrg_d, size * sizeof(float)) );
  dim3 dimGrid2(w, h, 1);
  getTree_shared<<<dimGrid2, dimBlock, (shared*shared+1)*sizeof(float)>>>(w, h,
    sigma3, tau2, parent_d);
  gpuErrChk( cudaPeekAtLastError() );
  gpuErrChk( cudaMemcpy(parent, parent_d, size * sizeof(float),
             cudaMemcpyHostToDevice) );

  gpuErrChk( cudaFree(img_d) );
	gpuErrChk( cudaFree(dist_d) );
	gpuErrChk( cudaFree(parent_d) );
	// gpuErrChk( cudaFree(enrg_d) );
}
