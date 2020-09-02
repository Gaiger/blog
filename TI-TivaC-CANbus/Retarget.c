/******************************************************************************/
/* RETARGET.C: 'Retarget' layer for target-dependent low level functions      */
/******************************************************************************/
/* This file is part of the uVision/ARM development tools.                    */
/* Copyright (c) 2005 Keil Software. All rights reserved.                     */
/* This software may only be used under the terms of a valid, current,        */
/* end user licence from KEIL for a compatible version of KEIL software       */
/* development tools. Nothing else gives you the right to use this software.  */
/******************************************************************************/


#include <stdio.h>
#include <time.h>
#include <rt_misc.h>

#include "utils/uartstdio.h"

#pragma import(__use_no_semihosting_swi)




struct __FILE { int handle; /* Add whatever you need here */ };


FILE __stdout;
FILE __stdin;


int fputc(int ch, FILE *f) {
  return UARTwrite((char*)&ch, 1);
}

int fgetc(FILE *f) {
	return UARTgetc();
}


int ferror(FILE *f) {
  /* Your implementation of ferror */
  return EOF;
}


void _ttywrch(int ch) {
  UARTwrite((char*)&ch, 1);
}


void _sys_exit(int return_code) {
  while (1);    /* endless loop */
}
