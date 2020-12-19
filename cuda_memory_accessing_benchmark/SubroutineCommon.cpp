#include <Windows.h>
#include "GPUCompute.cuh"

#include "SubroutineCommon.h"
#include "cuda_runtime.h"

#ifdef _WIN32

/**********************************************************************/

int clock_gettime(int, struct timespec *spec)      //C-file part
{
	__int64 wintime; GetSystemTimeAsFileTime((FILETIME*)&wintime);
	wintime -= 116444736000000000i64;  //1jan1601 to 1jan1970
	spec->tv_sec = wintime / 10000000i64;           //seconds
	spec->tv_nsec = wintime % 10000000i64 * 100;      //nano-seconds
	return 0;
}

#endif

/**********************************************************************/

void* MallocPinned(size_t size)
{
	void *ptr;
	cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
	return ptr;
}

/**********************************************************************/

void FreePinned(void *ptr)
{
	cudaFreeHost(&ptr);
}

/**********************************************************************/

void InitInputBuffer(float **pp_input1, float **pp_input2, int length, BOOL is_pinned_memory)
{
	float *p_input1, *p_input2;

	if (FALSE == is_pinned_memory)
	{
		p_input1 = (float*)malloc(length * sizeof(float));
		p_input2 = (float*)malloc(length * sizeof(float));
	}
	else
	{
		p_input1 = (float*)MallocPinned(length * sizeof(float));
		p_input2 = (float*)MallocPinned(length * sizeof(float));
	}

	float value = 1;

	for (int i = 0; i < length; i++) {
		p_input1[i] = value;
		value += 1.0;
	}

	value = 1.0;
	for (int i = 0; i < length; i++) {
		p_input2[i] = value;
	}

	*pp_input1 = p_input1;
	*pp_input2 = p_input2;
}

/**********************************************************************/

void FreeBuffer(float *p_input1, float *p_input2, BOOL is_pinned_memory)
{
	if (FALSE == is_pinned_memory)
	{
		free(p_input1);
		free(p_input2);
	}
	else
	{
		FreePinned(p_input1);
		FreePinned(p_input2);
	}
}

/**********************************************************************/