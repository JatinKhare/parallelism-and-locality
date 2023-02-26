#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <vector>
#include <time.h>
#include "saxpy.h"
#include "common.h"
#include "CycleTimer.h"
double timeKernelAvg = 0.0;
double timeCopyH2DAvg = 0.0;
double timeCopyD2HAvg = 0.0;
double totalTimeAvg = 0.0;
double totalCPUTimeAvg = 0.0;
// return GB/s
float toBW(double bytes, float sec) {
    return (static_cast<float>((bytes) / (1024. * 1024. * 1024.)) / sec);
}

void saxpyCpu(long N, float alpha, float* x, float* y, float* result) {
    for (long index=0; index<N; index++) 
       result[index] = alpha * x[index] + y[index];
}

bool check_saxpy(long N, float* a, float* b) {
    printf("%s\n", __func__);
    std::vector<long> diffs;
    for (long index=0; index<N; index++) {
       if (a[index] != b[index]){
        diffs.push_back(index);
       }
    }
    if (diffs.size() > 0) {
        MYDEBUG("%s done\n", __func__);
        for (unsigned int i=0; i<diffs.size(); i++) {
            int idx = diffs[i];
            MYDEBUG("[%16d] %10.3f != %10.3f (%e)\n", idx, a[idx], b[idx], a[idx] - b[idx]);
        }
        MYDEBUG(" failed #: %zu\n", diffs.size());
        return false;
    } else {
      return true;
    }
}

void usage(const char* progname) {
     printf("Usage: %s [options]\n", progname);
     printf("Program Options:\n");
     printf("  -n  --arraysize  <INT>  Number of elements in arrays\n");
     printf("  -p  --partitions <INT>  Number of partitions for the array\n");
     printf("  -i  --iterations <INT>  Number of iterations for statistics\n");
     printf("  -?  --help             This message\n");
}


int main(int argc, char** argv)
{

	double startmain = CycleTimer::currentSeconds();
    long total_elems = 512 * 1024 * 1024; 
    int partitions = 1;
    int iterations = 1;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"partitions",  1, 0, 'p'},
        {"iterations",  1, 0, 'i'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };
 
    while ((opt = getopt_long(argc, argv, "?n:p:i:", long_options, NULL)) != EOF) {
 
        switch (opt) {
        case 'n':
            total_elems = atol(optarg);
            break;
        case 'p':
            partitions = atoi(optarg);
            break;
        case 'i':
            iterations = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////
 
    const float alpha = 2.0f;
    float* xarray      = NULL;
    float* yarray      = NULL;
    float* resultarray = NULL;
    //
    // TODO: allocate host-side memory
    //
    getArrays(total_elems, &xarray, &yarray, &resultarray); 
    //
    // Initialize input arrays
    //
    //
    srand(time(NULL));
    for (long i=0; i<total_elems; i++) {
        xarray[i] = rand() / 100;
        yarray[i] = rand() / 100;
    }
    printCudaInfo();

    for (int i=0; i<iterations; i++) { 
        saxpyCuda(total_elems, alpha, xarray, yarray, resultarray, partitions);
    }
    totalCPUTimeAvg /= iterations;
    totalTimeAvg /= iterations;
    timeKernelAvg /= iterations;
    timeCopyH2DAvg /= iterations;
    timeCopyD2HAvg /= iterations;
 
    const int totalBytes = sizeof(float) * 3 * total_elems;
    const long int totalBytes_our = sizeof(float) * 3 * total_elems;
    long double transfer1 =  static_cast<long double>(2*totalBytes_our)/ static_cast<long double>(3);
    long double transfer2 =  static_cast<long double>(totalBytes_our)/3;
    printf("Overall time : %8.3f ms [%8.3f GB/s ]\n", 1000.f * totalTimeAvg, toBW(totalBytes_our, totalTimeAvg));
    printf("GPU Kernel   : %8.3f ms [%8.3f Ops/s]\n", 1000.f * timeKernelAvg, toBW(transfer2, timeKernelAvg));
    //printf("Copy CPU->GPU: %8.3f ms [%8.3f GB/s ]\n", 1000.f * timeCopyH2DAvg, toBW(totalBytes, timeCopyH2DAvg));
    printf("Copy CPU->GPU: %8.3f ms [%8.3f GB/s ]\n", 1000.f * timeCopyH2DAvg, toBW(transfer1, timeCopyH2DAvg));
    //printf("Copy CPU<-GPU: %8.3f ms [%8.3f GB/s ]\n", 1000.f * timeCopyD2HAvg, toBW(total_elems, timeCopyD2HAvg));
    printf("Copy CPU<-GPU: %8.3f ms [%8.3f GB/s ]\n", 1000.f * timeCopyD2HAvg, toBW(transfer2, timeCopyD2HAvg));
    
    if (resultarray != NULL) {
        float* resultrefer = new float[total_elems]();
        double startCPUtime = CycleTimer::currentSeconds();
        saxpyCpu(total_elems, alpha, xarray, yarray, resultrefer);
        double endCPUtime = CycleTimer::currentSeconds(); 
        totalCPUTimeAvg = endCPUtime - startCPUtime;
        if (check_saxpy(total_elems, resultarray, resultrefer)) {
            printf("Test succeeded\n");
        } else {
            printf("Test failed\n");
        }
    }
    //printf("CPU time : %8.3f ms [%8.3f GB/s ]\n", 1000.f * totalCPUTimeAvg, toBW(totalBytes, totalCPUTimeAvg));

    //
    // TODO: deallocate host-side memory
    //
    freeArrays(xarray, yarray, resultarray);
	double endmain = CycleTimer::currentSeconds(); 
	//printf("Code Running Time: %f\n", endmain - startmain);
    return 0;
}
