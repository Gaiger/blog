//-----------------------------------------------------------------------------
// F3xx_USB_Main.c
//-----------------------------------------------------------------------------
// Copyright 2010 Silicon Laboratories, Inc.
// http://www.silabs.com
//
// Program Description:
//
// This application will communicate with a PC across the USB interface.
// The device will appear to be a mouse, and will manipulate the cursor
// on screen.
//
// How To Test:    See Readme.txt
//
//
// FID:
// Target:         C8051F32x/C8051F340
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

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------

#include "c8051f3xx.h"
#include "F3xx_USB0_InterruptServiceRoutine.h"
#include "F3xx_USB0_Mouse.h"
#include "F3xx_USB0_ReportHandler.h"

#include <stdio.h>
#include <string.h>
#define MAX_UART_RX_DATA_BUFFER_SIZE				(24)

unsigned char xdata uart_rx_data_buffer[MAX_UART_RX_DATA_BUFFER_SIZE];// = {0};
unsigned char uart_rx_data_size = 0;

void ReportParser(void)
{
	if(uart_rx_data_size >= 2)
	{
		if( 0x0d == uart_rx_data_buffer[uart_rx_data_size - 2] 
			&& 0x0a == uart_rx_data_buffer[uart_rx_data_size - 1] )
		{
			if(uart_rx_data_size >= MOUSE_IN_REPORT_SIZE + 2)
			{
				memcpy(&IN_PACKET[0], &uart_rx_data_buffer[0], uart_rx_data_size - 2);
			}/*if */
			
			memset(&uart_rx_data_buffer[0], 0, MAX_UART_RX_DATA_BUFFER_SIZE);
			uart_rx_data_size = 0;					
		}/*if end of line */

	}/*if \r\n exist*/

	if(MAX_UART_RX_DATA_BUFFER_SIZE == uart_rx_data_size)
	{
		memset(&uart_rx_data_buffer[0], 0, MAX_UART_RX_DATA_BUFFER_SIZE);
		uart_rx_data_size = 0;
	}/*if MAX_DATA_BUFFER_SIZE == g_received_data_size */

}/*ReportParser*/


void UARTReceived(unsigned char c)
{	
	uart_rx_data_buffer[uart_rx_data_size] = c;
	uart_rx_data_size++;
	ReportParser();
}/*UARTReceived*/

//-----------------------------------------------------------------------------
// Main Routine
//-----------------------------------------------------------------------------
void main(void)
{

   System_Init ();
   UART0_Init(UARTReceived);
   USB0_Init ();

   EA = 1;
   while (1)
   {
	  if(KEYBOARD_REPORT_ID == IN_PACKET[0] 
			|| MOUSE_REPORT_ID == IN_PACKET[0])
	  {
		SendPacket (IN_PACKET[0]);
#if(0)		
		  {
			int i;
			for(i = 0; i< 9; i++)
				printf("0x%02x ", (unsigned int)IN_PACKET[i]);
			printf("\r\n");
		  }
#endif
		  IN_PACKET[0] = 0x00;
	  }
   }
}

