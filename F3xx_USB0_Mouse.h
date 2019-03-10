//-----------------------------------------------------------------------------
// F3xx_USB0_Mouse.h
//-----------------------------------------------------------------------------
// Copyright 2010 Silicon Laboratories, Inc.
// http://www.silabs.com
//
// Program Description:
//
// Header file for mouse routines.
//
//
// How To Test:    See Readme.txt
//
//
// FID:
// Target:         C8051F3xx
// Tool chain:     Keil / Raisonance
//                 Silicon Laboratories IDE version 2.6
// Command Line:   See Readme.txt
// Project Name:   F3xx_MouseExample
//
// Release 1.2 (ES)
//    -Added support for Raisonance
//    -No change to this file
//    -02 APR 2010
// Release 1.1
//    -Minor code comment changes
//    -16 NOV 2006
// Release 1.0
//    -Initial Revision (PD)
//    -07 DEC 2005
//
#ifndef  _MOUSE_H_
#define  _MOUSE_H_

extern signed char MOUSE_VECTOR;
extern unsigned char MOUSE_AXIS;
extern unsigned char MOUSE_BUTTON;
unsigned char xdata IN_PACKET[];


#define X_Axis 0
#define Y_Axis 1




void System_Init(void);
void USB0_Init(void);

void UART0_Init ( 
	void (*uart_received_callback_function_ptr)(unsigned char received_datum) 
		);

char putchar (char c);
	
#define REPORT_IN_BUFFER_SIZE					(32)
#define REPORT_OUT_BUFFER_SIZE					(8)
	
#define KEYBOARD_IN_REPORT_SIZE					(9)
#define KEYBOARD_OUT_REPORT_SIZE				(2)	
#define MOUSE_IN_REPORT_SIZE					(5)
	

#endif  /* _MOUSE_H_ */