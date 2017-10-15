#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define f(i,j) f[(i) + (j)*(m)]
#define Z(i,j) Z[(i) + (j)*m]

__global__ void Zev(float const * const Ag,float const * const A, float *Z,float const * const H, int m, int n,int patch,float filtsigma){
	
	int x = blockDim.x * blockIdx.x + threadIdx.x; 
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x_local=threadIdx.x;
	int y_local = threadIdx.y;
	int fix =(patch-1)/2;//temporary value
	int dim_sh=blockDim.x+patch-1;//dimensions of shared arrays = threadsPerBlock + patch-1
	
	extern __shared__ float block[];//dynamic allocation of shared memory
	float *shared_A=&block[0];//shared array to keep values for calculations
	float *shared_g=&block[dim_sh*dim_sh];//shared array to keep values for calculations
	float *shared_h=&block[dim_sh*dim_sh*2];//shared array to keep gaussian
	
	if(x<m-2*fix && y<n-2*fix){//  check to avoid exceeding arrays limits
		int th=blockDim.x;//block dimension=number of threads
		//fill shared_h with H values
		if(x_local<patch && y_local<patch){
		  	shared_h[x_local +y_local*patch]=H[x_local + y_local*patch];
		}
		//save in shared array the 2nd part for calculations (from a block of image array)
		__syncthreads();
		//filling shared memory is described in the report
		shared_A[x_local +y_local*dim_sh]=A[x_local+y_local*dim_sh];
		__syncthreads();
		if(x_local<(patch-1)){
			shared_A[(x_local+th) + y_local*dim_sh]=A[x_local+th +y_local*dim_sh];
		}
		__syncthreads();
		if(y_local<(patch-1)){
			shared_A[x_local + (y_local+th)*dim_sh]=A[x_local + (y_local+th)*dim_sh];
		}
		__syncthreads();
		if(x_local>blockDim.x-patch && y_local>blockDim.y-patch ){
			shared_A[x_local+patch-1 + (y_local+patch-1)*dim_sh]=A[x_local+patch-1 + (y_local+patch-1)*dim_sh];
		}
		__syncthreads();
		//save in shared array the 1st part for calculations (from image array)
		
		shared_g[x_local +y_local*dim_sh]=Ag[x+y*m];
		__syncthreads();
		if(x_local<(patch-1)){
			shared_g[(x_local+th) + y_local*dim_sh]=Ag[x+th +y*m];
		}
		__syncthreads();
		if(y_local<(patch-1)){
			shared_g[x_local + (y_local+th)*dim_sh]=Ag[x + (y+th)*m];
		}
		__syncthreads();
		if(x_local>blockDim.x-patch && y_local>blockDim.y-patch ){
			shared_g[x_local+patch-1 + (y_local+patch-1)*dim_sh]=Ag[x+patch-1 + (y+patch-1)*m];
		}
		__syncthreads();
		
	}
	patch=(patch-1)/2;
	if(x<m-2*patch && y<n-2*patch){
		int counter=0;//counter for gaussian values
		float temp=0,sum=0;
		float z_l=Z(x+patch,y+patch);//load into local z value from global Z
		
		for(int i=patch;i<dim_sh-patch;i++){//make calculations for the shared block wihtout pads
			for(int j=patch;j<dim_sh-patch;j++){
				for(int p=-patch;p<=patch;p++){//calculate neighborhood
					for(int l=-patch;l<=patch;l++){
						temp=(shared_g[(x_local+patch +l)+(y_local+patch + p)*dim_sh]-shared_A[(i+l) + (j+p)*dim_sh])*shared_h[counter];
						sum=sum+temp*temp;
						counter++;
					}
				}
				z_l=z_l+expf(-(sum/filtsigma));
				sum=0;
				counter=0;
			}
		}
		//__syncthreads();
		Z[x+patch + (y+patch)*m]=z_l;//return calculated value (with padding) matlab is used to remove pads from f
	}
	
}

__global__ void fev(float const * const Ag,float const * const A,float const * const Z,float const * const H,float *f, int m, int n,int patch,float filtsigma){
	int x = blockDim.x * blockIdx.x + threadIdx.x; 
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x_local=threadIdx.x;
	int y_local = threadIdx.y;
	int fix =(patch-1)/2;//temporary variable
	int dim_sh=blockDim.x+patch-1;//dimensions of shared arrays = threadsPerBlock + patch-1
	
	extern __shared__ float block[]; //dynamic allocation of shared memory
	float *shared_A=&block[0];//shared array to keep values for calculations
	float *shared_g=&block[dim_sh*dim_sh];//shared array to keep values for calculations
	float *shared_h=&block[dim_sh*dim_sh*2];//shared array to keep gaussian
	
	if(x<m-2*fix && y<n-2*fix){ //  check to avoid exceeding arrays limits
		int th=blockDim.x;//block dimension=number of threads
		//fill shared_h with H values
		if(x_local<patch && y_local<patch){
		 	shared_h[x_local +y_local*patch]=H[x_local + y_local*patch];
		}
		__syncthreads();
		//save in shared array the 2nd part for calculations (from a block of image array)
		shared_A[x_local +y_local*dim_sh]=A[x_local+y_local*dim_sh];
		__syncthreads();
		if(x_local<(patch-1)){
			shared_A[(x_local+th) + y_local*dim_sh]=A[x_local+th +y_local*dim_sh];
		}
		__syncthreads();
		if(y_local<(patch-1)){
			shared_A[x_local + (y_local+th)*dim_sh]=A[x_local + (y_local+th)*dim_sh];
		}
		__syncthreads();
		if(x_local>blockDim.x-patch && y_local>blockDim.y-patch ){
			shared_A[x_local+patch-1 + (y_local+patch-1)*dim_sh]=A[x_local+patch-1 + (y_local+patch-1)*dim_sh];
		}
		__syncthreads();
		//save in shared array the 1st part for calculations (from image array)
		
		shared_g[x_local +y_local*dim_sh]=Ag[x+y*m];
		__syncthreads();
		if(x_local<(patch-1)){
			shared_g[(x_local+th) + y_local*dim_sh]=Ag[x+th +y*m];
		}
		__syncthreads();
		if(y_local<(patch-1)){
			shared_g[x_local + (y_local+th)*dim_sh]=Ag[x + (y+th)*m];
		}
		__syncthreads();
		if(x_local>blockDim.x-patch && y_local>blockDim.y-patch ){
			shared_g[x_local+patch-1 + (y_local+patch-1)*dim_sh]=Ag[x+patch-1 + (y+patch-1)*m];
		}
		__syncthreads();
	}
	patch=(patch-1)/2; //patch for checking neighbourhood
	if(x<m-2*patch && y<n-2*patch){
		int counter=0;//counter for gaussian
		float temp=0,sum=0,z_l=Z(x+patch,y+patch),f_l=f(x+patch,y+patch);//load into local z(or f) value from global Z(or f)
		
		for(int i=patch;i<dim_sh-patch;i++){//make calculations for the shared block wihtout pads
			for(int j=patch;j<dim_sh-patch;j++){
				for(int p=-patch;p<=patch;p++){//calculate neighborhood
					for(int l=-patch;l<=patch;l++){
						temp=(shared_g[(x_local+patch +l)+(y_local+patch + p)*dim_sh]-shared_A[(i+l) + (j+p)*dim_sh])*shared_h[counter];
						sum=sum+temp*temp;
						counter++;
					}
				}
				f_l=f_l+(1/z_l)*(expf(-(sum/filtsigma)))*shared_A[i+j*dim_sh];
				sum=0;
				counter=0;
			}
		}
		//__syncthreads();
		f[x+patch + (y+patch)*m]=f_l;//return calculated value
	}
	
}


