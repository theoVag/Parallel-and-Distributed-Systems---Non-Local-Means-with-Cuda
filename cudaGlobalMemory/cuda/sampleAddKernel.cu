#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>


// Array access macros
#define im(i,j) A[(i) + (j)*(m)]
#define f(i,j) f[(i) + (j)*(m)]
#define Z(i,j) Z[(i) + (j)*m]

__global__ void Zev(float const * const A, float *Z,float const * const H, int m, int n,int patch,float patchSigma,float filtsigma){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
 if(x<m-(patch-1)/2 && y<n-(patch-1)/2){
		
		int i,j,p,l,count=0;
		patch=(patch-1) /2;
		float temp=0.0,sum=0;
		
	for(p=patch;p<m-patch;p++){
		for(l=patch;l<n-patch;l++){
			  for(i=-patch;i<=patch;i++){
				for(j=-patch;j<=patch;j++){
					temp=(im(x+patch+i,y+patch+j)-im(p+i,l+j))*H[count];
					sum=sum+temp*temp;
					count++;
					
				}       
			  }
			  
			Z(x+patch,y+patch)=Z(x+patch,y+patch)+exp(-(sum/(filtsigma)));
			sum=0;
			count=0;
		}
	}
 }
}

__global__ void fev(float const * const A,float const * const Z, float *f,float const * const H, int m, int n,int patch,float patchSigma,float filtsigma){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
  if(x<m-(patch-1)/2 && y<n-(patch-1)/2){
	  		patch=(patch-1) /2;
    int i,j;
	float temp,sum=0.0;
	int p,l,count=0;

	  for(p=patch;p<m-patch;p++){
	  	for(l=patch;l<n-patch;l++){
	  	  for(i=-patch;i<=patch;i++){
			for(j=-patch;j<=patch;j++){
			  temp=(im(x+patch+i,y+patch+j)-im(p+i,l+j))*H[count];
			  sum=sum+temp*temp;
			  count++;
							
	  		}       
	  	 }
	     count=0;

	  	 f(x+patch,y+patch)=f(x+patch,y+patch)+((1/Z(x+patch,y+patch))*exp(-(sum/filtsigma)))*im(p,l);
	  	 sum=0;
	  	}
	  }
  }
}

