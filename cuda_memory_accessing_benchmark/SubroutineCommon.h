#ifndef _SUBROUNTINECOMMON_H_
#define _SUBROUNTINECOMMON_H_


#define ROUND								(10)
#define DATA_LENGTH							(1024 * 1024 * 4)
#define ASYNC_NUM_DATA_SET					(8)



#ifdef _USE_PINNED_HOST_MEMORY
#define MALLOC(SIZE)						(MallocPinned(SIZE))
#define FREE(PTR)							(FreePinned(PTR))
#else
#define MALLOC(SIZE)						(malloc(SIZE))
#define FREE(PTR)							(free(PTR))
#endif

#ifdef _ASYNCHRONOUS
#define GPUSAXPY(VAR1, VAR2, VAR3, VAR4, VAR5)		GPUSAXPYAsynchronous(VAR1, VAR2, VAR3, VAR4, VAR5)
#else
#define GPUSAXPY(VAR1, VAR2, VAR3, VAR4, VAR5)		GPUSAXPYSynchronous(VAR1, VAR2, VAR3, VAR4, VAR5)
#endif

#if defined( _WIN32 ) 

#define CLOCK_REALTIME 0
struct timespec
{
	long tv_sec;
	long tv_nsec;
};    //header part
#endif

#include "GPUCompute.cuh"

typedef struct
{
	CUDAHandle cuda_handle;
	float *g_p_array[2];
} GPUProcess;

#ifdef __cplusplus
extern "C"
{
#endif

#include <Windows.h>

int clock_gettime(int, struct timespec *spec);

void InitInputBuffer(float **pp_input1, float **pp_input2, 
	int length, BOOL is_pinned_memory);

void FreeBuffer(float *p_input1, float *p_input2, BOOL is_pinned_memory);
#ifdef __cplusplus
}
#endif




#endif /*_SUBROUNTINECOMMON_H_ */