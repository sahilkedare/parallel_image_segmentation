#ifndef __QUICKSHIFT_H__
#define __QUICKSHIFT_H__

#include <cuda.h>
#include <cuda_runtime.h>

#define B_X 32
#define B_Y 32

#define Img(y,x,z) tex3D(texImg, x + 0.5f, y + 0.5f, z + 0.5f)
#define Enrg(y,x) tex2D(texEnrg, x + 0.5f, y + 0.5f)

#define gpuErrChk(ans) { gpuAssert( (ans) ); }
inline void gpuAssert(cudaError_t code, bool abort=true){
	if(code != cudaSuccess){
		printf("GPU Assert: %s.\n", cudaGetErrorString(code));
		if(abort) exit(code);
	}
}

void quickshift(float* img, int w, int h, int d,
                float sigma3, float tau2, float ratio,
                float* parent, float* dist);
#endif
