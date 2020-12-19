#include <Windows.h>
#include <process.h>

#include <stdio.h>

#include "GPUCompute.cuh"
#include "SubroutineCommon.h"
#include "AsynchronousSubroutines.h"



/**********************************************************************/

void AsyncSubroutineDepthFirst(BOOL is_pinned_memory)
{
	GPUProcess gpu_process[ASYNC_NUM_DATA_SET];

	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
		gpu_process[i].cuda_handle = InitGPUCompute(DATA_LENGTH);
		InitInputBuffer(&gpu_process[i].g_p_array[0],
			&gpu_process[i].g_p_array[1],
			DATA_LENGTH, is_pinned_memory);
	}

	
	struct timespec t1, t2;

	unsigned int all_done_flag = 0;
	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++)
		all_done_flag |= (0x01 << i);

	float elasped_time_including_copy_in_ms = 0;
	float elasped_time_excluding_copy_in_ms = 0;

	clock_gettime(CLOCK_REALTIME, &t1);
	int k = 0; 

	while (k < ROUND)
	{
		unsigned int done_flag = 0;

		for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
			GPUSAXPYAsynchronous(gpu_process[i].cuda_handle,
				DATA_LENGTH, 2.0, gpu_process[i].g_p_array[0],
				gpu_process[i].g_p_array[1]);
		}

		while (1)
		{
			for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
				if (0x00 == (0x01 & (done_flag >> i)))
				{
					if (true == IsComputeDone(
						gpu_process[i].cuda_handle))
					{
						done_flag |= (0x1 << i);

						if (0 == i)
						{
							float inlcuding_time, excluding_time;
							GetElaspedTime(gpu_process[i].cuda_handle,
								&inlcuding_time, &excluding_time);

							elasped_time_including_copy_in_ms += inlcuding_time;
							elasped_time_excluding_copy_in_ms += excluding_time;
						}

					}
					else
					{
						Sleep(1);
					}
				}
			}

			if (all_done_flag == done_flag)
				break;
		}
		k++;
	}

	clock_gettime(CLOCK_REALTIME, &t2);

	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
		CloseGPUCompute(gpu_process[i].cuda_handle);
		FreeBuffer(gpu_process[i].g_p_array[0],
			gpu_process[i].g_p_array[1], is_pinned_memory);
	}

	elasped_time_including_copy_in_ms /= ROUND;
	elasped_time_excluding_copy_in_ms /= ROUND;

	double whole_process_time_in_ms
		= (t2.tv_sec - t1.tv_sec) * 1E3 + (t2.tv_nsec - t1.tv_nsec) * 1E-6;

	printf("%s, is_pinned_memory = %d\r\n", __FUNCTION__,
		is_pinned_memory);

	printf("average elasped_time_excluding_copy_in_ms = %f ms \r\n",
		elasped_time_excluding_copy_in_ms);
	printf("average elasped_time_including_copy_in_ms = %f ms \r\n",
		elasped_time_including_copy_in_ms);

	printf("---averge one ROUND process_time_in_ms = %f ms ---\r\n",
		whole_process_time_in_ms / ROUND);

	printf("---averge_whole_process_time_in_ms = %f ms ---\r\n",
		whole_process_time_in_ms / ROUND / ASYNC_NUM_DATA_SET);

	printf("\r\n");

}

/**********************************************************************/

void AsyncSubroutineBreadthFirst(BOOL is_pinned_memory)
{
	GPUProcess gpu_process[ASYNC_NUM_DATA_SET];

	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
		gpu_process[i].cuda_handle = InitGPUCompute(DATA_LENGTH);
		InitInputBuffer(&gpu_process[i].g_p_array[0],
			&gpu_process[i].g_p_array[1],
			DATA_LENGTH, is_pinned_memory);
	}


	struct timespec t1, t2;

	unsigned int all_done_flag = 0;
	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++)
		all_done_flag |= (0x01 << i);

	float elasped_time_including_copy_in_ms = 0;
	float elasped_time_excluding_copy_in_ms = 0;

	clock_gettime(CLOCK_REALTIME, &t1);
	int k = 0;

	while (k < ROUND)
	{
		unsigned int done_flag = 0;

		for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
			GPUSAXPYAsynchronousCopyHostToDevice(gpu_process[i].cuda_handle, 
				DATA_LENGTH, 2.0, gpu_process[i].g_p_array[0], 
				gpu_process[i].g_p_array[1]);

			GPUSAXPYAsynchronousCompute(gpu_process[i].cuda_handle,
				DATA_LENGTH, 2.0, gpu_process[i].g_p_array[0],
				gpu_process[i].g_p_array[1]);
		}

		for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
			GPUSAXPYAsynchronousCopyDeviceToHost(gpu_process[i].cuda_handle,
				DATA_LENGTH, 2.0, gpu_process[i].g_p_array[0],
				gpu_process[i].g_p_array[1]);
		}

		while (1)
		{
			for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
				if (0x00 == (0x01 & (done_flag >> i)))
				{
					if (true == IsComputeDone(
						gpu_process[i].cuda_handle))
					{
						done_flag |= (0x1 << i);

						if (ASYNC_NUM_DATA_SET / 2 == i)
						{
							float inlcuding_time, excluding_time;
							GetElaspedTime(gpu_process[i].cuda_handle,
								&inlcuding_time, &excluding_time);

							elasped_time_including_copy_in_ms += inlcuding_time;
							elasped_time_excluding_copy_in_ms += excluding_time;
						}

					}
					else
					{
						Sleep(1);
					}
				}
			}

			if (all_done_flag == done_flag)
				break;
		}
		k++;
	}

	clock_gettime(CLOCK_REALTIME, &t2);

	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
		CloseGPUCompute(gpu_process[i].cuda_handle);
		FreeBuffer(gpu_process[i].g_p_array[0],
			gpu_process[i].g_p_array[1], is_pinned_memory);
	}

	elasped_time_including_copy_in_ms /= ROUND;
	elasped_time_excluding_copy_in_ms /= ROUND;

	double whole_process_time_in_ms
		= (t2.tv_sec - t1.tv_sec) * 1E3 + (t2.tv_nsec - t1.tv_nsec) * 1E-6;

	printf("%s, is_pinned_memory = %d\r\n", __FUNCTION__,
		is_pinned_memory);

	printf("average elasped_time_excluding_copy_in_ms = %f ms \r\n",
		elasped_time_excluding_copy_in_ms);
	printf("average elasped_time_including_copy_in_ms = %f ms \r\n",
		elasped_time_including_copy_in_ms);

	printf("---averge one ROUND process_time_in_ms = %f ms ---\r\n",
		whole_process_time_in_ms / ROUND);

	printf("---averge_whole_process_time_in_ms = %f ms ---\r\n",
		whole_process_time_in_ms / ROUND / ASYNC_NUM_DATA_SET);

	printf("\r\n");

}

/**********************************************************************/

typedef struct {
	GPUProcess gpu_process[ASYNC_NUM_DATA_SET];
	HANDLE sending_done_semaphore;
	HANDLE receiving_done_semaphore;

	float elasped_time_including_copy_in_ms;
	float elasped_time_excluding_copy_in_ms;
}ThreadArgs;



unsigned __stdcall AsyncSenderThread(void *args)
{
	//printf("%s\r\n", __FUNCTION__);
	ThreadArgs *p_thread_args = (ThreadArgs*)args;

	int k = 0;
	
	while (k < ROUND)
	{
		WaitForSingleObject(p_thread_args->receiving_done_semaphore, 
			INFINITE);

		for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
			GPUSAXPYAsynchronousCopyHostToDevice(
				p_thread_args->gpu_process[i].cuda_handle,
				DATA_LENGTH, 2.0, p_thread_args->gpu_process[i].g_p_array[0],
				p_thread_args->gpu_process[i].g_p_array[1]);

			GPUSAXPYAsynchronousCompute(p_thread_args->gpu_process[i].cuda_handle,
				DATA_LENGTH, 2.0, p_thread_args->gpu_process[i].g_p_array[0],
				p_thread_args->gpu_process[i].g_p_array[1]);
		}


		for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
			GPUSAXPYAsynchronousCopyDeviceToHost(
				p_thread_args->gpu_process[i].cuda_handle,
				DATA_LENGTH, 2.0, p_thread_args->gpu_process[i].g_p_array[0],
				p_thread_args->gpu_process[i].g_p_array[1]);
		}

		ReleaseSemaphore( p_thread_args->sending_done_semaphore,
			1, NULL);
		//printf("kk = %d\r\n", k);
		k++;
	}

	return 0;
}

/**********************************************************************/

unsigned __stdcall AsyncReceiverThread(void *args)
{
	//printf("%s\r\n", __FUNCTION__);

	ThreadArgs *p_thread_args = (ThreadArgs*)args;

	unsigned int all_done_flag = 0;
	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++)
		all_done_flag |= (0x01 << i);



	int k = 0;
	while (k < ROUND)
	{
		WaitForSingleObject(p_thread_args->sending_done_semaphore,
			INFINITE);

		unsigned int done_flag = 0;

		while (1)
		{
			for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
				if (0x00 == (0x01 & (done_flag >> i)))
				{
					if (true == IsComputeDone(
						p_thread_args->gpu_process[i].cuda_handle))
					{
						done_flag |= (0x1 << i);

						if (ASYNC_NUM_DATA_SET/2 == i)
						{
							float inlcuding_time, excluding_time;

							GetElaspedTime(
								p_thread_args->gpu_process[i].cuda_handle,
								&inlcuding_time, &excluding_time);

							p_thread_args->elasped_time_including_copy_in_ms 
								+= inlcuding_time;
							p_thread_args->elasped_time_excluding_copy_in_ms 
								+= excluding_time;
						}

					}
					else
					{
						Sleep(1);
					}
				}
			}

			if (all_done_flag == done_flag)				
				break;
		}

		ReleaseSemaphore(p_thread_args->receiving_done_semaphore,
			1, NULL);
		k++;
	}

	return 0;
}

/**********************************************************************/

void AsyncSubroutineBreadthFirstSendReceiveThread(BOOL is_pinned_memory)
{
	HANDLE sender_thread_handle;
	HANDLE receiver_thread_handle;

	ThreadArgs thread_args;
	ZeroMemory(&thread_args, sizeof(ThreadArgs));

	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
		thread_args.gpu_process[i].cuda_handle = InitGPUCompute(DATA_LENGTH);
		InitInputBuffer(&thread_args.gpu_process[i].g_p_array[0],
			&thread_args.gpu_process[i].g_p_array[1],
			DATA_LENGTH, is_pinned_memory);
	}

	thread_args.sending_done_semaphore 
		= CreateSemaphore(NULL, 0, 1, NULL);
	thread_args.receiving_done_semaphore
		= CreateSemaphore(NULL, 1, 1, NULL);


	sender_thread_handle = (HANDLE)_beginthreadex(NULL, 0, AsyncSenderThread,
		(void*)&thread_args, CREATE_SUSPENDED, NULL);
	receiver_thread_handle = (HANDLE)_beginthreadex(NULL, 0, AsyncReceiverThread,
		(void*)&thread_args, CREATE_SUSPENDED, NULL);

	struct timespec t1, t2;
	clock_gettime(CLOCK_REALTIME, &t1);

	ResumeThread(sender_thread_handle);
	ResumeThread(receiver_thread_handle);


	WaitForSingleObject(sender_thread_handle, INFINITE);
	WaitForSingleObject(receiver_thread_handle, INFINITE);

	clock_gettime(CLOCK_REALTIME, &t2);

	thread_args.elasped_time_including_copy_in_ms /= ROUND;
	thread_args.elasped_time_excluding_copy_in_ms /= ROUND;

	double whole_process_time_in_ms;
	whole_process_time_in_ms
		= (t2.tv_sec - t1.tv_sec) * 1E3 + (t2.tv_nsec - t1.tv_nsec) * 1E-6;


	CloseHandle(sender_thread_handle); sender_thread_handle = NULL;
	CloseHandle(receiver_thread_handle); receiver_thread_handle = NULL;

	CloseHandle(thread_args.sending_done_semaphore);
	CloseHandle(thread_args.receiving_done_semaphore);

	for (int i = 0; i < ASYNC_NUM_DATA_SET; i++) {
		CloseGPUCompute(thread_args.gpu_process[i].cuda_handle);
		FreeBuffer(thread_args.gpu_process[i].g_p_array[0],
			thread_args.gpu_process[i].g_p_array[1], is_pinned_memory);
	}


	printf("%s, is_pinned_memory = %d\r\n", __FUNCTION__,
		is_pinned_memory);

	printf("average elasped_time_excluding_copy_in_ms = %f ms \r\n",
		thread_args.elasped_time_excluding_copy_in_ms);
	printf("average elasped_time_including_copy_in_ms = %f ms \r\n",
		thread_args.elasped_time_including_copy_in_ms);

	printf("---averge one ROUND process_time_in_ms = %f ms ---\r\n",
		whole_process_time_in_ms / ROUND);

	printf("---averge_whole_process_time_in_ms = %f ms ---\r\n",
		whole_process_time_in_ms / ROUND / ASYNC_NUM_DATA_SET);

	printf("\r\n");
}

/**********************************************************************/