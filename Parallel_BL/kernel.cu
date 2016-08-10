#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
//#include <helper_functions.h>
//#include <helper_image.h>
//#include <helper_timer.h>
#include <helper_cuda.h>
#include <math_functions.h>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
//#include <time.h>

/* Block dimensions and sigma(intensity)*/
#define BLOCK 32
#define sigr 20

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace cv;
using namespace std;

typedef unsigned char uchar;

/*	KR kernel radius		*
 *	KS is kernel width		*
 *	sigs is Sigma(distance)	*/
const int KR=8;
const int KS=17;
double sigs=20;

/* Kernel for Parallel Bilateral Filter */

/*	gdist_d		Gaussian Kernel for Distance
	img_data	Source image row array
	output		Processed array
	debug		Error checking matrix
	*/
__global__ void parallel_BL( double *gdist_d , uchar * img_data, uchar * output, int rows, int cols, int step, double *debug){
	
	int x = threadIdx.x + blockIdx.x*BLOCK;
	int y = threadIdx.y + blockIdx.y*BLOCK;
	
	//if(x>step-KS||y>rows-KS) return;

	int curr_pix = x + step*y;			//gives current value of the pixel which will be processed
	int channels = step/cols;			

	const int kr = 8,					
			  ks = 2*kr+1;

	double sum = 0, norm=0, gint[ks][ks];
	uchar img_ker[ks][ks]; 
	char I[ks][ks];

	int l = step*rows;
	//////////loads image kernel///////////
	for(int i=0; i<ks; i++){				
		for(int j=0; j<ks; j++){
			img_ker[i][j] = img_data[(x+j*3)+(y+i)*step];
			I[i][j]= img_data[kr+kr*ks] - I[i][j];

		}
	}
	__syncthreads();

	double t, mult;
	uchar ker_sum=0;
	/*
	gint	Gaussian of the intensity
	sum		output of Tranformation function
	norm	Normalizing factor for sum
	*/
	for(int i=0; i<ks; i++){
		for(int j=0; j<ks; j++){
			t = -((I[i][j]*I[i][j]))/(2*sigr*sigr);
			mult = pow(sigr*sqrt(2*3.142) ,-1);
			gint[i][j]=(mult)*expf(t);

			sum += gdist_d[i*ks+j]*gint[i][j]*img_ker[i][j];
			norm += gdist_d[i*ks+j]*gint[i][j];
			//gdist_d[i*ks+j]=img_ker[i][j];
		}
	}
	__syncthreads();
	double den = pow(norm, -1);
	double db = sum*den;
	uchar result = (uchar)db;
	
	//output[x+step*(y)] = result;			//Unable to identify the cause.
	
	/*
	I can calculate output result but once I change the required with this value everything goes haywire
	*/

	/*Error checking */

	if(x==100&&y==100) 
		{
			debug[1]=result;
			debug[2]=den;
			debug[3]=(sum*den);
			debug[4]=(uchar)debug[3];
			/*
			for(int i=0; i<ks; i++){
				for(int j=0; j<ks; j++){
					debug[i*ks+j] = gint[i][j];
				}
			}
			*/
	}
	__syncthreads();

}

int main(int argc, char** argv){
	
	/*
	Using OpenCV to read and store iamge data
	Mat object
	object.data gives image matrix in a single row form
	*/
	const Mat img = cv::imread("FILE NAME",CV_LOAD_IMAGE_UNCHANGED);
	Mat img2 = img.clone();
	Mat img3 = img.clone();
	/* Sanity check */
	if(img.empty()){
		cout<<"Error: Image not supported"<<endl;
		return -1;
	}

	else{

		uchar *input = img.data,
			  *output, *input_d,
			  *input2 = img2.data;

		int rows = img.rows,
			cols = img.cols,
			step = img.step;

		double gdist[KS][KS], 
				dist[KS][KS], 
				norm_gdist=0,
				*gdist_d,
				*debug;
		/*
		Calculates Gaussian kernel of distance
		*/
		for(int i=0; i<KS; i++){
			for(int j=0; j<KS; j++){
				
				dist[i][j] = (0-i)*(0-i)+(0-j)*(0-j);
				
				double den = 1/(sigs*sqrt(2*3.142));
				double exp_=-dist[i][j]/(2*sigs*sigs);
				
				gdist[i][j]=den*exp(exp_);
				
				norm_gdist += gdist[i][j];
				//cout<<d[i][j]<<"   ";
			}
			//cout<<endl;
		}
		int l = rows*step;
		
		size_t kernel = KS*KS*sizeof(double);
		size_t total = l*sizeof(uchar);
		/* CUDA initialize */
		cudaMalloc((void**)&gdist_d, kernel);
		cudaMalloc((void**)&debug, kernel);
		cudaMalloc((void**)&output, total);
		cudaMalloc((void**)&input_d, total);

		cudaMemcpy(gdist_d, gdist, kernel,cudaMemcpyHostToDevice);
		cudaMemcpy(input_d, input, total,cudaMemcpyHostToDevice);
		
		dim3 block(BLOCK,BLOCK);
		dim3 grid(step/block.x,rows/block.y);
		//////////////////////////
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate (&start);
		cudaEventCreate (&stop);
		cudaEventRecord (start, 0);
		//////////////////////////
		parallel_BL<<<grid, block>>>( gdist_d, input_d, output, rows, cols, step, debug);
		//////////////////////////
		cudaEventRecord (stop, 0);
		cudaEventSynchronize (stop);
		cudaEventElapsedTime (&time, start, stop);
		cudaEventDestroy (start);
		cudaEventDestroy (stop);
		//////////////////////////

		printf("Time to generate:  %f ms \n", time);
		cudaMemcpy(input2, output, total, cudaMemcpyDeviceToHost);
		cudaMemcpy(dist, debug, kernel, cudaMemcpyDeviceToHost);

		

		/*/////////////////Keeping it commented because output gets error is updated(explained above)													
		cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
		cv::imshow("Original",img);
		cv::namedWindow("Changed", CV_WINDOW_AUTOSIZE);
		cv::imshow("Changed",img2);
		cv::waitKey();
		*/
	}
}