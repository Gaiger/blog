#ifndef _COMMON_H_
#define _COMMON_H_




#define _HOST_PIN
//#define _USE_READ_ONLY_CACHE
//#define _ROW_DATA_IN_CONSECUTIVE_SHARED_MEN


#ifdef _USE_READ_ONLY_CACHE
	#define  LDG(VAR)				__ldg(&(VAR))
#else
	#define  LDG(VAR)				(VAR)
#endif

#ifdef _DEBUG
	#define ROUND							(1)

	#define WIDTH							(128)
	#define HEIGHT							(130)
	#define KERNEL_RADIUS					(5)

	#define X_NUM_THREADS						(4)
	#define Y_NUM_THREADS						(5)
	#define NUM_BLOCKS							(4)
#else
	#define ROUND							(100)

	#define WIDTH							(1400)
	#define HEIGHT							(1400)
	#define KERNEL_RADIUS					(15)
	#define X_NUM_THREADS					(28)
	#define Y_NUM_THREADS					(28)

	#define NUM_BLOCKS						(4)
#endif

#define KERNEL_LENGTH					(2 * KERNEL_RADIUS + 1)

#endif /*COMMON*/
