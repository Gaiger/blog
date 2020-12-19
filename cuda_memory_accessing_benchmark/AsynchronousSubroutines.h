#ifndef _ASYNCHRONOUSSUBROUTINES_H_
#define _ASYNCHRONOUSSUBROUTINES_H_

#include <Windows.h>

#ifdef __cplusplus
extern "C"
{
#endif

void AsyncSubroutineDepthFirst(BOOL is_pinned_memory);
void AsyncSubroutineBreadthFirst(BOOL is_pinned_memory);

void AsyncSubroutineBreadthFirstSendReceiveThread(BOOL is_pinned_memory);
#ifdef __cplusplus
}
#endif


#endif /*_ASYNCHRONOUSSUBROUTINES_H_*/