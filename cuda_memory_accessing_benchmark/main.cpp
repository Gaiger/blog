#include <stdio.h>
#include <windows.h>

#include "SubroutineCommon.h"
#include "SynchronousSubroutines.h"
#include "AsynchronousSubroutines.h"


/**********************************************************************/

int main(int argc, char *argv[])
{
	printf("ROUND = %d\r\n", ROUND);
	printf("\r\n");


	SyncSubroutineTransmissionDataByDeepCopy(FALSE);
	SyncSubroutineTransmissionDataByDeepCopy(TRUE);
	SyncSubroutineTransmissionDataByZeroCopy();

	AsyncSubroutineDepthFirst(FALSE);
	AsyncSubroutineDepthFirst(TRUE);

	AsyncSubroutineBreadthFirst(FALSE);
	AsyncSubroutineBreadthFirst(TRUE);

	AsyncSubroutineBreadthFirstSendReceiveThread(FALSE);
	AsyncSubroutineBreadthFirstSendReceiveThread(TRUE);

	printf("\r\n");
	return 0;
}

/**********************************************************************/