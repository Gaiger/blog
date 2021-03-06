#include <Windows.h>
#include <stdio.h>

#include "GPUCompute.cuh"
#include "SubroutineCommon.h"
#include "SynchronousSubroutines.h"


/**********************************************************************/

void SyncSubroutineTransmissionDataByDeepCopy(BOOL is_pinned_memory)
{
	CUDAHandle cuda_handle;
	cuda_handle = InitGPUCompute(DATA_LENGTH);

	float *data_ptr_array[2];
	InitInputBuffer(&data_ptr_array[0], &data_ptr_array[1],
		DATA_LENGTH, is_pinned_memory);

	struct timespec t1, t2;

	clock_gettime(CLOCK_REALTIME, &t1);

	float elasped_time_including_copy_in_ms = 0;
	float elasped_time_excluding_copy_in_ms = 0;

	for (int j = 0; j < ROUND; j++) {
		GPUSAXPYSynchronousDeepCopy(cuda_handle, DATA_LENGTH, 2.0,
			data_ptr_array[1], data_ptr_array[1]);

		float inlcuding_time, excluding_time;

		GetElaspedTime(cuda_handle,
			&inlcuding_time, &excluding_time);
		elasped_time_including_copy_in_ms += inlcuding_time;
		elasped_time_excluding_copy_in_ms += excluding_time;

	}

	clock_gettime(CLOCK_REALTIME, &t2);


	CloseGPUCompute(cuda_handle); cuda_handle = NULL;
	FreeBuffer(data_ptr_array[0], data_ptr_array[1], is_pinned_memory);
	data_ptr_array[0] = data_ptr_array[1] = NULL;

	elasped_time_including_copy_in_ms /= ROUND;
	elasped_time_excluding_copy_in_ms /= ROUND;

	double averge_whole_process_time_in_ms = (t2.tv_sec - t1.tv_sec) * 1E3
		+ (t2.tv_nsec - t1.tv_nsec) * 1E-6;
	averge_whole_process_time_in_ms /= ROUND;

	printf("%s, is_pinned_memory = %d\r\n", __FUNCTION__,
		is_pinned_memory);

	printf("average elasped_time_excluding_copy_in_ms = %f ms \r\n",
		elasped_time_excluding_copy_in_ms);
	printf("average elasped_time_including_copy_in_ms = %f ms \r\n",
		elasped_time_including_copy_in_ms);

	printf("---averge_whole_process_time_in_ms = %f ms ---\r\n",
		averge_whole_process_time_in_ms);

	printf("\r\n");
}

/**********************************************************************/

void SyncSubroutineTransmissionDataByZeroCopy(void)
{
	CUDAHandle cuda_handle;
	cuda_handle = InitGPUCompute(DATA_LENGTH);

	float *data_ptr_array[2];
	InitInputBuffer(&data_ptr_array[0],
		&data_ptr_array[1], DATA_LENGTH, TRUE);

	struct timespec t1, t2;

	clock_gettime(CLOCK_REALTIME, &t1);

	float elasped_time_including_copy_in_ms = 0;
	float elasped_time_excluding_copy_in_ms = 0;

	for (int j = 0; j < ROUND; j++) {
		GPUSAXPYSynchronousZeroCopy(cuda_handle, DATA_LENGTH, 2.0,
			data_ptr_array[0], data_ptr_array[1]);

		float inlcuding_time, excluding_time;

		GetElaspedTime(cuda_handle,
			&inlcuding_time, &excluding_time);
		elasped_time_including_copy_in_ms += inlcuding_time;
		elasped_time_excluding_copy_in_ms += excluding_time;

	}

	clock_gettime(CLOCK_REALTIME, &t2);


	CloseGPUCompute(cuda_handle); cuda_handle = NULL;
	FreeBuffer(data_ptr_array[0], data_ptr_array[1], TRUE);
	data_ptr_array[0] = data_ptr_array[1] = NULL;

	elasped_time_including_copy_in_ms /= ROUND;
	elasped_time_excluding_copy_in_ms /= ROUND;

	double averge_whole_process_time_in_ms = (t2.tv_sec - t1.tv_sec) * 1E3
		+ (t2.tv_nsec - t1.tv_nsec) * 1E-6;
	averge_whole_process_time_in_ms /= ROUND;

	printf("%s\r\n", __FUNCTION__);

	printf("average elasped_time_excluding_copy_in_ms = %f ms \r\n",
		elasped_time_excluding_copy_in_ms);
	printf("average elasped_time_including_copy_in_ms = %f ms \r\n",
		elasped_time_including_copy_in_ms);

	printf("---averge_whole_process_time_in_ms = %f ms ---\r\n",
		averge_whole_process_time_in_ms);

	printf("\r\n");

	return;
}

/**********************************************************************/