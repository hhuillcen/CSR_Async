#define _CRT_SECURE_NO_DEPRECATE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> 

#define ITERATIONS 10000000
#define BLOCK_SIZE 32

//captura de errores para las funciones de cuda
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ 
void spmvAsync(double *y,
                        const double *A,
                        const int *IA,
                        const int *JA,
                        const int M,
                        const double *x,
						const int offset)
{
        __shared__ float t_sum[BLOCK_SIZE]; // thread sum
        int j;

        int t_id = offset + threadIdx.x + blockDim.x * blockIdx.x; // thread id
        int row = t_id / 32; // one warp per row
        int t_warp = t_id & 31; // thread number within a given warp
        
        // boundary condition
        if (row < M){

                // compute running sum per thread in warp
                t_sum[threadIdx.x] = 0;

                for (j = IA[row] + t_warp; j < IA[row+1]; j += 32)
                        t_sum[threadIdx.x] += A[j] * x[JA[j]];

                // Parallel reduction of result in shared memory for one warp
                if (t_warp < 16) t_sum[threadIdx.x] += t_sum[threadIdx.x+16];
                if (t_warp < 8) t_sum[threadIdx.x] += t_sum[threadIdx.x+8];
                if (t_warp < 4) t_sum[threadIdx.x] += t_sum[threadIdx.x+4];
                if (t_warp < 2) t_sum[threadIdx.x] += t_sum[threadIdx.x+2];
                if (t_warp < 1) t_sum[threadIdx.x] += t_sum[threadIdx.x+1];
                
                // first thread within warp contains desired y[row] result so write it to y
                if (t_warp == 0)
                        y[row] = t_sum[threadIdx.x];
        }
}


void cootocsr(int *rowoff, int *row, int size) {
	
	rowoff[0] = 0;
	int prev = 0, accu = 1, j = 1;

	for (int i = 1; i < size; i++) {
		
		if (row[i] - row[prev] > 1) {
			for (int k = 0; k < row[i] - row[prev]; k++) {
				rowoff[j++] = accu;
			}
			prev = i;
		}

		else
			
			if (row[prev] != row[i]) {
			rowoff[j++] = accu;
			prev = i;
		}
		
		accu += 1;
	}

	rowoff[j] = accu;
}



int main() {
	 const int nStreams = 4;

	//************ 1) Leer archivos de dataset ************//	

	//FILE *pToMFile = fopen("mat5_5.txt", "r");   //5 5 13
	//FILE *pToMFile = fopen("mat20_20.txt", "r");  //20 20 34
	//FILE *pToMFile = fopen("cop20k_A.mtx", "r");  //121192 121192 1362087
	//FILE *pToMFile = fopen("cant.mtx", "r");  //62451 62451 2034917
	//FILE *pToMFile = fopen("consph.mtx", "r");  //83334 83334 3046907 
	//FILE *pToMFile = fopen("mac_econ_fwd500.mtx", "r");  //206500 206500 1273389
	//FILE *pToMFile = fopen("mc2depi.mtx", "r");  //525825 525825 2100225
	//FILE *pToMFile = fopen("pdb1HYS.mtx", "r");  //36417 36417 2190591
	//FILE *pToMFile = fopen("pwtk.mtx", "r");  //217918 217918 5926171
	//FILE *pToMFile = fopen("scircuit.mtx", "r");  //170998 170998 958936
	//FILE *pToMFile = fopen("shipsec1.mtx", "r");  //140874 140874 3977139
	FILE *pToMFile = fopen("webbase-1M.mtx", "r");  //1000005 1000005 3105536


	//************ 2) Extraer tamaños del vector/matriz_cuadradda y elementos no cero (NNZ ************//		
	int matsize, veclen,temp1;
	fscanf(pToMFile, "%d", &veclen); //tamaño del vector
	fscanf(pToMFile, "%d", &temp1); //saltar.
	fscanf(pToMFile, "%d", &matsize); //tamaño de NNZ
	
	
	//************ 3) Crear vectores host en pinned memory para capturar datos del archivo dataset ************//
	int mintsize = matsize * sizeof(int);
	double mdoublesize = matsize * sizeof(double);
	int *h_row;
	HANDLE_ERROR(cudaMallocHost((void **)&h_row, mintsize));
	double *h_mvalue;
	HANDLE_ERROR(cudaMallocHost((void **)&h_mvalue, mdoublesize));
	//int *h_col = (int *)malloc(mintsize);
	int *h_col;
	HANDLE_ERROR(cudaMallocHost((void **)&h_col, mintsize));


	//************ 4) Capturar elementos de la matriz sparse ************//
	for (int i = 0; i < matsize; i++)
	{
		fscanf(pToMFile, "%d", &h_col[i]);
		fscanf(pToMFile, "%d", &h_row[i]);
		fscanf(pToMFile, "%lf", &h_mvalue[i]);
	}

	fclose(pToMFile);

	
	//************ 5) Crear y poblar el vector en pinned memory ************//
	int vecbytes = veclen * sizeof(double);
	double *h_vec;
	HANDLE_ERROR(cudaMallocHost((void **)&h_vec, vecbytes));

	srand((long)time(NULL));
	for (int i = 0; i < veclen; i++) {
		h_vec[i]=rand()/(double)RAND_MAX;
	}


	//************ 6) convertir matriz sparced de COO a CSR convitiendo el vector h_row en h_rowoff ************//
	int rownum = h_row[matsize - 1] + 1; //sale 5 para una matriz de 5 x 5
	int rowoffsize = (rownum) * sizeof(int); //sale 6 para el tamaño del vector row_ofsset
	int *h_rowoff;
	HANDLE_ERROR(cudaMallocHost((void **)&h_rowoff, rowoffsize));
	cootocsr(h_rowoff, h_row, matsize);
	 

	//************ 7) Crear y localizar vectores en device  ************//
	double outputsize = veclen * sizeof(double);
	double *d_mvalue;
	int *d_col;
	int *d_rowoff;
	double *d_vec;
	double *d_output_sm;
	HANDLE_ERROR(cudaMalloc((void **)&d_mvalue, mdoublesize));
	HANDLE_ERROR(cudaMalloc((void **)&d_col, mintsize));
	HANDLE_ERROR(cudaMalloc((void **)&d_rowoff, rowoffsize));
	HANDLE_ERROR(cudaMalloc((void **)&d_vec, vecbytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_output_sm, outputsize));
	double *h_output_sm;
	HANDLE_ERROR(cudaMallocHost((void **)&h_output_sm, outputsize));	//pinned memory
	
	
	//************ 8) Crear streams y eventos para medir el tiempo  ************//
	cudaEvent_t startEvent, stopEvent, dummyEvent;
	cudaStream_t stream[nStreams];
	HANDLE_ERROR( cudaEventCreate(&startEvent) );
	HANDLE_ERROR( cudaEventCreate(&stopEvent) );
	HANDLE_ERROR( cudaEventCreate(&dummyEvent) );
	for (int i = 0; i < nStreams; ++i)
	HANDLE_ERROR( cudaStreamCreate(&stream[i]) );

	const int streamSize = matsize / nStreams;

	  
	//************ 9) Copiar de manera asyncrona vectores del host al device  ************//
	HANDLE_ERROR( cudaEventRecord(startEvent,0) );
	for (int i = 0; i < nStreams; ++i)
	  {
		int offset = i * streamSize;
		HANDLE_ERROR( cudaMemcpyAsync(&d_mvalue[offset], &h_mvalue[offset], streamSize * sizeof(double), cudaMemcpyHostToDevice, stream[i]) );
		HANDLE_ERROR( cudaMemcpyAsync(&d_col[offset], &h_col[offset], streamSize * sizeof(int), cudaMemcpyHostToDevice, stream[i]) );
	  }
	
	const int streamSize_rowoff = (rownum ) / nStreams;
	for (int i = 0; i < nStreams; ++i)
	  {
		int offset_rowoff = i * streamSize_rowoff;
		HANDLE_ERROR( cudaMemcpyAsync(&d_rowoff[offset_rowoff], &h_rowoff[offset_rowoff], streamSize_rowoff * sizeof(int), cudaMemcpyHostToDevice, stream[i]) );
	  }

	const int streamSize_vec = veclen / nStreams;
	for (int i = 0; i < nStreams; ++i)
	  {
		int offset_vec = i * streamSize_vec;
		HANDLE_ERROR( cudaMemcpyAsync(&d_vec[offset_vec], &h_vec[offset_vec], streamSize_vec * sizeof(double), cudaMemcpyHostToDevice, stream[i]) );
	  }


	//************ 10) Ejecutar de manera asyncrona le kernel code haciendo uso de shared memory  ************//
	  for (int i = 0; i < nStreams; ++i)
	  {
		int offset_vec = i * streamSize_vec;
		spmvAsync<<<(streamSize_vec/BLOCK_SIZE),BLOCK_SIZE,0,stream[i]>>>(d_output_sm, d_mvalue, d_rowoff, d_col, rownum, d_vec,offset_vec);
	  }



	//************ 11) Copiar de manera asyncrona vectores resultante del device al host  ************//

	  for (int i = 0; i < nStreams; ++i)
	  {
		int offset_vec = i * streamSize_vec;
		HANDLE_ERROR( cudaMemcpyAsync(&h_output_sm[offset_vec], &d_output_sm[offset_vec], streamSize_vec * sizeof(double), cudaMemcpyDeviceToHost, stream[i]));
	  }

	  
	//************ 12) Cálcular metricas de tiempo y Gflops  ************//	  
	  float ms;
	  HANDLE_ERROR( cudaEventRecord(stopEvent, 0) );	
	  HANDLE_ERROR( cudaEventSynchronize(stopEvent) );
	  HANDLE_ERROR( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	  printf("Tiempo de ejecución de transferir y ejecutar (segundos): %f\n", ms/1e3);
	  double Flops_sm=(ITERATIONS)/(double)(ms/1e3);
	  double gFlops_sm=(double)Flops_sm/1e9;// Calculate Giga Flops Formula: F lops * 10raised to (-9).
	  printf("GFLOPS con async : %f\n",gFlops_sm);
	  //calculo del ancho de banda
	  double cant_bytes=0;
	  cant_bytes+=matsize*8;  //vector elementos NNZ
	  cant_bytes+=matsize*4;  //vector de columnas
	  cant_bytes+=(rownum+1)*4; //vector de filas rowoff
	  cant_bytes+=veclen*8;  //vector a multiplicar
	  cant_bytes+=veclen*8; //vector resultante
	  printf("Effective Bandwidth (GB/s): %f \n", (cant_bytes/1e9)/(ms/1e3));



	//************ 13) Ejecutar y calcular tiempo para un proceso spmv serial  ************//	  
	double *temp = (double *)malloc(outputsize);
	clock_t ts; 
    ts = clock(); 
	for (int i = 0; i < rownum; i++) {
		temp[i] = 0;
		for (int j = h_rowoff[i]; j < h_rowoff[i + 1]; j++) {
			temp[i] += h_mvalue[j] * h_vec[h_col[j]];
		}
	}
    ts = clock() - ts; 
    double time_serial = (((double)ts)/CLOCKS_PER_SEC); // in seconds
    printf("El tiempo serial tomo %f segundos \n", time_serial); 
	printf("El SpeedUp es %f segundos \n", ms/time_serial); 
	system("PAUSE()"); 

	for (int i = 0; i < veclen; i++) {
		printf("Matriz resultante, elemento: %d = %lf\n", i, temp[i]);
	}
	
		system("PAUSE()"); 

	for (int i = 0; i < veclen; i++) {
		printf("Matriz resultante, elemento: %d = %lf\n", i, h_output_sm[i]);
	}

	//************ 14) Liberar vectores de memoria global y pinned  ************//	
	HANDLE_ERROR(cudaFreeHost(h_row));
	HANDLE_ERROR(cudaFreeHost(h_rowoff));
	HANDLE_ERROR(cudaFreeHost(h_col));
	HANDLE_ERROR(cudaFreeHost(h_mvalue));
	HANDLE_ERROR(cudaFreeHost(h_vec));
	free(temp);
	HANDLE_ERROR(cudaFreeHost(h_output_sm));
	//Deallocate GPU memory:
	HANDLE_ERROR(cudaFree(d_mvalue));
	HANDLE_ERROR(cudaFree(d_col));
	HANDLE_ERROR(cudaFree(d_rowoff));
	HANDLE_ERROR(cudaFree(d_vec));
	HANDLE_ERROR(cudaFree(d_output_sm));

	system("PAUSE()"); 

	return 0;

}