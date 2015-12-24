

#include "getSADCUDA.cuh"
#include "cuda_runtime.h"

	__global__
	void GetSADGPU_kernel(uchar* leftImage, uchar* rightImage, uchar* laplacianL, uchar* laplacianR, int nstep, int startJ, int row_start, 
		int blockSize, int disparity, int sobelLimit,
		int *sadArray, int threadsPerBlock  )
	{
		// init parameters
		//int blockSize = state.blockSize;
		//int disparity = state.disparity;
		//int sobelLimit = state.sobelLimit;

		int x = threadIdx.x;
		int y = blockIdx.x;

		if( x >= threadsPerBlock )
			return;

		int pxX = startJ + x * blockSize;
		int pxY = row_start + y * blockSize;

		// top left corner of the SAD box
		int startX = pxX;
		int startY = pxY;

		// bottom right corner of the SAD box
		int endX = pxX + blockSize - 1;
		int endY = pxY + blockSize - 1;

		//printf("startX = %d, endX = %d, disparity = %d, startY = %d, endY = %d, rows = %d, cols = %d\n", startX, endX, disparity, startY, endY, leftImage.rows, leftImage.cols);

		int leftVal = 0, rightVal = 0;

		int sad = 0;

		for (int i=startY;i<=endY;i++) {
			// get a pointer for this row
			uchar *this_rowL = leftImage + i * nstep;
			uchar *this_rowR = rightImage + i * nstep;

			uchar *this_row_laplacianL = laplacianL + i * nstep;
			uchar *this_row_laplacianR = laplacianR + i * nstep;


				for (int j=startX;j<=endX;j++) {
					// we are now looking at a single pixel value
					/*uchar pxL = leftImage.at<uchar>(i,j);
					uchar pxR = rightImage.at<uchar>(i,j + disparity);

					uchar sL = laplacianL.at<uchar>(i,j);
					uchar sR = laplacianR.at<uchar>(i,j + disparity);
					*/


					uchar sL = this_row_laplacianL[j];//laplacianL.at<uchar>(i,j);
					uchar sR = this_row_laplacianR[j + disparity]; //laplacianR.at<uchar>(i,j + disparity);

					leftVal += sL;
					rightVal += sR;

					uchar pxL = this_rowL[j];
					uchar pxR = this_rowR[j + disparity];

					sad += abs(pxL - pxR);
				}
		}

		//cout << "(" << leftVal << ", " << rightVal << ") vs. (" << leftVal2 << ", " << rightVal2 << ")" << endl;

		int laplacian_value = leftVal + rightVal;

		//cout << "sad with neon: " << sad << " without neon: " << sad2 << endl;


		if (leftVal < sobelLimit || rightVal < sobelLimit)// || diff_score > state.interest_diff_limit)
		{
			sadArray[ y * threadsPerBlock + x] =  -1;
		}
		else 
			sadArray[ y * threadsPerBlock + x] =  NUMERIC_CONST*(float)sad/(float)laplacian_value;
	}

	void GetSadCUDA::runGetSAD( int row_start, int row_end, int startJ, int stopJ, int * sadArray, uchar* leftImage, uchar* rightImage, uchar* laplacianL, uchar* laplacianR, int nstep, int blockSize, int disparity, int sobelLimit )
	{
#if 0
		int gridY = (row_end - row_start)/blockSize;
		int blockDim = (stopJ - startJ)/blockSize;
		for (int y=0; y< gridY; y++)
		{
			for (int x=0; x< blockDim; x++)
			{
				int i = row_start + y * blockSize;
				int j = startJ + x * blockSize;
				GetSAD_kernel(leftImage, rightImage, laplacianL, laplacianR, nstep, j, i, 
					blockSize, disparity, sobelLimit,
					x, y, blockDim, sadArray );
			}
		}

#else
		cudaMemcpy( d_leftImage, leftImage, nSizeBuffer, cudaMemcpyHostToDevice );
		cudaMemcpy( d_rightImage, rightImage, nSizeBuffer, cudaMemcpyHostToDevice );
		cudaMemcpy( d_laplacianL, laplacianL, nSizeBuffer, cudaMemcpyHostToDevice );
		cudaMemcpy( d_laplacianR, laplacianR, nSizeBuffer, cudaMemcpyHostToDevice );

		int blocksPerGrid = (row_end - row_start)/blockSize;
		int threadsPerBlock = (stopJ - startJ)/blockSize;

		GetSADGPU_kernel<<<blocksPerGrid, threadsPerBlock+10>>> ( d_leftImage, d_rightImage, d_laplacianL, d_laplacianR, nstep, startJ, row_start, 
					blockSize, disparity, sobelLimit, 
					d_sadArray, threadsPerBlock );

		cudaMemcpy( sadArray, d_sadArray, blocksPerGrid*threadsPerBlock*sizeof(int), cudaMemcpyDeviceToHost );
		//cudaDeviceSynchronize();
		cudaError_t errorCode = cudaGetLastError();
		if( cudaSuccess != errorCode )
			printf("failed to run GetSAD_kernel \n");
#endif
	}

	void GetSadCUDA::initialize(int width, int height, int nstep)
	{
		nSizeBuffer = nstep * height;
		cudaMalloc( &d_leftImage, nSizeBuffer );
		cudaMalloc( &d_rightImage, nSizeBuffer );
		cudaMalloc( &d_laplacianL, nSizeBuffer );
		cudaMalloc( &d_laplacianR, nSizeBuffer );

		cudaMalloc( &d_sadArray, nSizeBuffer*sizeof(int) );
	}

	void GetSadCUDA::release()
	{
		if( NULL == d_leftImage)
			return;

		cudaFree( d_leftImage );
		cudaFree( d_rightImage );
		cudaFree( d_laplacianL );
		cudaFree( d_laplacianR );

		cudaFree( d_sadArray );

		d_leftImage = NULL;

		cudaDeviceReset();
	}

	GetSadCUDA::GetSadCUDA()
	{
		d_leftImage = NULL;
		d_rightImage = NULL;
		d_laplacianL = NULL;
		d_laplacianR = NULL;
		d_sadArray = NULL;
	}

	GetSadCUDA::~GetSadCUDA()
	{
		release();
	}