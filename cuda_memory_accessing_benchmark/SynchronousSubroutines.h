#ifndef _SYNCHRONOUSSUBROUTINES_H_
#define _SYNCHRONOUSSUBROUTINES_H_

#ifdef __cplusplus
extern "C"
{
#endif

void SyncSubroutineTransmissionDataByDeepCopy(BOOL is_pinned_memory);
void SyncSubroutineTransmissionDataByZeroCopy(void);

#ifdef __cplusplus
}
#endif


#endif /*_SYNCHRONOUSSUBROUTINES_H_*/