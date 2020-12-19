#ifndef _GPUCOMPUTE_CUH_
#define _GPUCOMPUTE_CUH_


#define _ASYNCHRONOUS

#define _USE_PINNED_HOST_MEMORY

#ifdef _USE_PINNED_HOST_MEMORY
//#define _ZEROCOPY
#endif

#ifdef __cplusplus
extern "C" {
#endif

	typedef void* CUDAHandle;

	CUDAHandle InitGPUCompute(int length);

	/*p_input2[i] = A * p_input1[i] + p_input2[i] */
	int GPUSAXPYSynchronousDeepCopy(CUDAHandle handle, int length, float A,
		float *p_input1, float *p_input2);

	int GPUSAXPYSynchronousZeroCopy(CUDAHandle handle, int length, float A,
		float *p_input1, float *p_input2);

	int GPUSAXPYAsynchronous(CUDAHandle handle, 
		int length, float A, float *p_input1, float *p_input2);

	int GPUSAXPYAsynchronousCopyHostToDevice(CUDAHandle handle, 
		int length, float A,float *p_input1, float *p_input2);
	int GPUSAXPYAsynchronousCompute(CUDAHandle handle, 
		int length, float A, float *p_input1, float *p_input2);
	int GPUSAXPYAsynchronousCopyDeviceToHost(CUDAHandle handle, 
		int length, float A, float *p_input1, float *p_input2);


	int WaitComputingDone(CUDAHandle handle);
	bool IsComputeDone(CUDAHandle handle);
	void CloseGPUCompute(CUDAHandle handle);
	

	void GetElaspedTime(CUDAHandle handle, float *p_elasped_time_including_copy_in_ms, 
		float *p_elasped_time_excluding_copy_in_ms);

	void* MallocPinned(size_t size);
	void FreePinned(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /*_GPUCOMPUTE_CUH_*/