#ifndef _COMMON_H_
#define _COMMON_H_


#define _SWAP_KERNEL_AND_WIDTH
//#define _USE_FMA

#ifndef _SWAP_KERNEL_AND_WIDTH
	#define _KERNEL_ALIGNED16
#endif


#ifdef _DEBUG

	#define ROUND							(1)

	#define WIDTH							(64)
	#define HEIGHT							(64)
	#define KERNEL_RADIUS					(10)
#else
	#define ROUND						(10)

	#define WIDTH						(1000)
	#define HEIGHT						(1000)
	#define KERNEL_RADIUS				(15)	
#endif

#define KERNEL_LENGTH					(2 * KERNEL_RADIUS + 1)

#endif /*COMMON*/
