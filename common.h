#ifndef _COMMON_H_
#define _COMMON_H_



#ifdef _DEBUG
	#define ROUND						(1)

	#define WIDTH						(64)
	#define HEIGHT						(64)
	#define KERNEL_RADIUS				(15)


#else
	#define ROUND						(10)

	#define WIDTH						(1000)
	#define HEIGHT						(1000)
	#define KERNEL_RADIUS				(7)

#endif

#define KERNEL_LENGTH				(2 * KERNEL_RADIUS + 1)


#endif /*COMMON*/
