

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_gpio.h"

#include "inc/hw_can.h"

#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"

#include "driverlib/watchdog.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "driverlib/timer.h"
#include "driverlib/systick.h"
#include "driverlib/interrupt.h"
#include "driverlib/can.h"

#include "utils/uartstdio.h"


/*

https://www.ti.com/lit/ug/spmu365c/spmu365c.pdf?ts=1597807520128&ref_url=https%253A%252F%252Fwww.google.com%252F

Two user switches are provided for input and control of the TM4C1294NCPDTI software. The switches
are connected to GPIO pins PJ0 and PJ1.

Four user LEDs are provided on the board. D1 and D2 are connected to GPIOs PN1 and PN0. These
LEDs are dedicated for use by the software application. D3 and D4 are connected to GPIOs PF4 and
PF0, which can be controlled by user’s software or the integrated Ethernet module of the microcontroller.

*/

#define MILLI_SEC_PER_SEC							(1000)

/**********************************************************************/

uint32_t g_ui32SysClock;

int configure_uart(void)
{
	//
	// Enable the GPIO Peripheral used by the UART.
	//
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

	//
	// Enable UART0
	//
	SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

	//
	// Configure GPIO Pins for UART mode.
	//
	GPIOPinConfigure(GPIO_PA0_U0RX);
	GPIOPinConfigure(GPIO_PA1_U0TX);
	GPIOPinTypeUART(GPIO_PORTA_BASE,  GPIO_PIN_0 | GPIO_PIN_1);

	 
	UARTStdioConfig(0, 115200, g_ui32SysClock);
	return 0;
}/*configure_uart*/

/**********************************************************************/

void watchdog_interrupt_handler(void)
{
	//
	// Clear the watchdog interrupt.
	//
	WatchdogIntClear(WATCHDOG0_BASE);
}

/**********************************************************************/
unsigned long long g_uptime_in_msec = 0;

void timer0A_interrupt_handler(void)
{
	TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
	g_uptime_in_msec++;

}/*timer0_int_handler*/

/**********************************************************************/


unsigned int g_is_canbus_error_occurred = 0;
unsigned int g_controller_status = 0;

unsigned int g_is_canbus_sent = 0;
unsigned int g_sent_canbus_message_count = 0;

unsigned int g_received_canbus_message_count = 0;
unsigned int g_is_canbus_received = 0;


#define SEND_MESSAGE_OBJ					(2)
#define RECEIVE_MESSAGE_OBJ					(1)

void canbus1_interrupt_handler(void)
{
	uint32_t interrupt_cause;

	interrupt_cause = CANIntStatus(CAN1_BASE, CAN_INT_STS_CAUSE);
#if(1)
	if(CAN_INT_INTID_STATUS == interrupt_cause)
	{
		g_controller_status = CANStatusGet(CAN1_BASE, CAN_STS_CONTROL);
		if(CAN_STATUS_TXOK == g_controller_status)
		{
			interrupt_cause = SEND_MESSAGE_OBJ;
		}
		
		if(CAN_STATUS_RXOK == g_controller_status)
		{
			interrupt_cause = RECEIVE_MESSAGE_OBJ;
		}
		
		g_controller_status &= ~CAN_STATUS_TXOK;
		g_controller_status &= ~CAN_STATUS_RXOK;
		
		if(CAN_STATUS_LEC_NONE != g_controller_status)
			g_is_canbus_error_occurred = 1;
	}
	
	if(0 == g_is_canbus_error_occurred)
	{
		switch(interrupt_cause)
		{
		case SEND_MESSAGE_OBJ:
			g_sent_canbus_message_count++;
			g_is_canbus_sent = 1;
			CANIntClear(CAN1_BASE, SEND_MESSAGE_OBJ);
			break;
		case RECEIVE_MESSAGE_OBJ:
			g_received_canbus_message_count++;
			g_is_canbus_received = 1;
			CANIntClear(CAN1_BASE, RECEIVE_MESSAGE_OBJ);
			break;
		}

	}
	
#else

	if(CAN_INT_INTID_STATUS == interrupt_cause)
	{
		g_controller_status = CANStatusGet(CAN1_BASE, CAN_STS_CONTROL);

		if(CAN_STATUS_TXOK == g_controller_status)
		{
			g_sent_canbus_message_count++;
			g_is_canbus_sent = 1;
			CANIntClear(CAN1_BASE, SEND_MESSAGE_OBJ);
		}
		
		if(CAN_STATUS_RXOK == g_controller_status)
		{
			g_received_canbus_message_count++;
			g_is_canbus_received = 1;
			CANIntClear(CAN1_BASE, RECEIVE_MESSAGE_OBJ);
		}
		
		g_controller_status &= ~CAN_STATUS_TXOK;
		g_controller_status &= ~CAN_STATUS_RXOK;
		
		
		if(CAN_STATUS_LEC_NONE != g_controller_status)
			g_is_canbus_error_occurred = 1;
	}
#endif
	
}

/**********************************************************************/

int main(void)
{
	unsigned int previous_uptime_in_ms;
	
	// Run from the PLL at 120 MHz.
	//
#if(1)
	g_ui32SysClock = SysCtlClockFreqSet( 
		SYSCTL_XTAL_25MHZ | SYSCTL_OSC_MAIN | SYSCTL_USE_PLL | SYSCTL_CFG_VCO_480, 
		120*1000*1000);
#else
	g_ui32SysClock = SysCtlClockFreqSet(
		SYSCTL_OSC_INT | SYSCTL_USE_PLL | SYSCTL_CFG_VCO_320, 
		8*1000*10000);
#endif


#if(0)
	SysTickPeriodSet(g_ui32SysClock/MILLI_SEC_PER_SEC);
	SysTickEnable();
#endif
	
	configure_uart();
	printf("TIVA C start @%dMHz...", g_ui32SysClock/1000/1000);

	printf("CANBUS \r\n");


	IntMasterEnable();

	{
		SysCtlPeripheralEnable(SYSCTL_PERIPH_WDOG0);
	
		//
		// Set the period of the watchdog timer to 2 second.
		//
		WatchdogReloadSet(WATCHDOG0_BASE, 2 * g_ui32SysClock);
		//
		// Enable reset generation from the watchdog timer.
		//
		WatchdogResetEnable(WATCHDOG0_BASE);
		//
		// Enable the watchdog timer.
		//
		IntEnable(INT_WATCHDOG);

		WatchdogEnable(WATCHDOG0_BASE);
	}


	{
		SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);
		
		TimerConfigure(TIMER0_BASE, TIMER_CFG_SPLIT_PAIR|TIMER_CFG_A_PERIODIC|TIMER_CFG_B_PERIODIC);

		TimerPrescaleSet(TIMER0_BASE, TIMER_A, g_ui32SysClock/1000/1000 - 1);
		TimerLoadSet(TIMER0_BASE, TIMER_A, 1000 - 1);
		TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
		IntEnable(INT_TIMER0A);

		TimerEnable(TIMER0_BASE, TIMER_A);
	}



	{
		SysCtlPeripheralEnable(SYSCTL_PERIPH_GPION);
		GPIOPinTypeGPIOOutput(GPIO_PORTN_BASE, GPIO_PIN_1|GPIO_PIN_0);
	}


	{	
		SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
		GPIOPinConfigure(GPIO_PB0_CAN1RX);
		GPIOPinConfigure(GPIO_PB1_CAN1TX);
	
		GPIOPinTypeCAN(GPIO_PORTB_BASE, GPIO_PIN_0 | GPIO_PIN_1);
		
		SysCtlPeripheralEnable(SYSCTL_PERIPH_CAN1);
		CANInit(CAN1_BASE);

		
		CANBitRateSet(CAN1_BASE, g_ui32SysClock, 500 * 1000);
		CANIntRegister(CAN1_BASE, canbus1_interrupt_handler);
		CANIntEnable(CAN1_BASE, CAN_INT_MASTER | CAN_INT_ERROR | CAN_INT_STATUS);
		IntEnable(INT_CAN1);
		CANEnable(CAN1_BASE);
	}


#if(1)
	#define SEND_MESSAGE_ID							(0x0A0)
	#define RECEIVE_MESSAGE_ID						(0x0B0)
#else
	#define SEND_MESSAGE_ID							(0x0B0)
	#define RECEIVE_MESSAGE_ID						(0x0A0)
#endif
	
#define MAX_DATA_FRAME_SIZE							(8)
	{
		
		tCANMsgObject canbus_send_message;
		unsigned char can_send_buffer[MAX_DATA_FRAME_SIZE];

		tCANMsgObject canbus_receive_message;
		unsigned char can_receive_buffer[MAX_DATA_FRAME_SIZE];
		int remain_print_time_in_ms;
		int remain_canbus_sending_time_in_ms;
	
		int remain_led_n0_blinking_time_in_ms;
		int remain_led_n1_blinking_time_in_ms;

		previous_uptime_in_ms = g_uptime_in_msec;

#define PRINT_UPTIME_INTERVAL_IN_MS					(1000)
		remain_print_time_in_ms = PRINT_UPTIME_INTERVAL_IN_MS;

#define CANBUS_SEND_INTERVAL_IN_MS					(2000)
		remain_canbus_sending_time_in_ms = CANBUS_SEND_INTERVAL_IN_MS;

#define LED_N0_BLINKING_INTERVAL_IN_MS				(30)
		remain_led_n0_blinking_time_in_ms = LED_N0_BLINKING_INTERVAL_IN_MS;

#define LED_N1_BLINKING_INTERVAL_IN_MS				(200)
		remain_led_n1_blinking_time_in_ms = LED_N1_BLINKING_INTERVAL_IN_MS;


		{
			canbus_send_message.ui32MsgID = SEND_MESSAGE_ID;
			canbus_send_message.ui32MsgIDMask = 0;
			canbus_send_message.ui32Flags = MSG_OBJ_TX_INT_ENABLE;
			canbus_send_message.pui8MsgData = &can_send_buffer[0];
		}

#define CAN_SFF_MASK 								(0x000007FFU) /* standard frame format (SFF) */
		{
			canbus_receive_message.ui32MsgID = RECEIVE_MESSAGE_ID;
			canbus_receive_message.ui32MsgIDMask = CAN_SFF_MASK;
			canbus_receive_message.ui32Flags = MSG_OBJ_RX_INT_ENABLE | MSG_OBJ_USE_ID_FILTER;
			canbus_receive_message.ui32MsgLen = sizeof(can_receive_buffer);
			canbus_receive_message.pui8MsgData = &can_receive_buffer[0];
		}
		
		CANMessageSet(CAN1_BASE, RECEIVE_MESSAGE_OBJ, &canbus_receive_message, MSG_OBJ_TYPE_RX);


		while(1)
		{
			int delta_time;
			delta_time = 0;
			if( previous_uptime_in_ms != g_uptime_in_msec)
			{
				delta_time = g_uptime_in_msec - previous_uptime_in_ms;
				previous_uptime_in_ms = g_uptime_in_msec;
			}

			remain_print_time_in_ms -= delta_time;
			if(0 >= remain_print_time_in_ms)
			{
				printf("%u sec.\r\n", (unsigned int)(g_uptime_in_msec/1000));
				remain_print_time_in_ms = PRINT_UPTIME_INTERVAL_IN_MS;
			}/*if */
	
			remain_led_n0_blinking_time_in_ms -= delta_time;
			if(0 != GPIOPinRead(GPIO_PORTN_BASE, GPIO_PIN_0) 
				&& 0 >= remain_led_n0_blinking_time_in_ms)
			{
					GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0, ~GPIO_PIN_0);
			}

			remain_led_n1_blinking_time_in_ms -= delta_time;
			if(0 != GPIOPinRead(GPIO_PORTN_BASE, GPIO_PIN_1) 
				&& 0 >= remain_led_n1_blinking_time_in_ms)
			{
				GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_1, ~GPIO_PIN_1);
			}
			
#if(1)
			if(0 != g_is_canbus_error_occurred)
			{
				printf("canbus_error_occurred, controller_status = 0x%02x\r\n", g_controller_status);
				g_is_canbus_error_occurred = 0;
			}
#endif

			if(0 != g_is_canbus_sent)
			{
				printf("can message sent, sent count = %u\r\n", g_sent_canbus_message_count);
				g_is_canbus_sent = 0;
			}


			remain_canbus_sending_time_in_ms -= delta_time;
			if(0 >= remain_canbus_sending_time_in_ms)
			{
				int used_bytes;
				unsigned int uptime_in_sec;
				
				uptime_in_sec = (unsigned int)g_uptime_in_msec/1000;
				
				used_bytes = 0;
				
				memset(&can_send_buffer[0], 0, sizeof(can_send_buffer));
				memcpy(&can_send_buffer[0], &uptime_in_sec, sizeof(unsigned int));

				used_bytes +=  sizeof(uptime_in_sec);
				canbus_send_message.ui32MsgLen = used_bytes;
				
				CANMessageSet(CAN1_BASE, SEND_MESSAGE_OBJ, &canbus_send_message, MSG_OBJ_TYPE_TX);
				
				printf("send msg id = 0x%03X, message = ", canbus_send_message.ui32MsgID);

				{
					int i;
					for(i = 0; i < used_bytes; i++)
						printf("0x%02X ", can_send_buffer[i]);
				}
				printf("(decimal : %u)", uptime_in_sec);
				printf("\r\n");
				
				remain_canbus_sending_time_in_ms = CANBUS_SEND_INTERVAL_IN_MS;
				
				GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0, GPIO_PIN_0);
				remain_led_n0_blinking_time_in_ms = LED_N0_BLINKING_INTERVAL_IN_MS;
			}

			if(0 != g_is_canbus_received)
			{
				memset(&can_receive_buffer[0], 0, sizeof(can_receive_buffer));
				CANMessageGet(CAN1_BASE, RECEIVE_MESSAGE_OBJ, &canbus_receive_message, 0);

				g_is_canbus_received = 0;
				printf("can message received, received count = %u\r\n", g_received_canbus_message_count);
				
				if(canbus_receive_message.ui32Flags & MSG_OBJ_DATA_LOST)
					printf("CAN message loss detected\r\n");

				printf("received msg id = 0x%03X len=%u, message = ",
					canbus_receive_message.ui32MsgID, canbus_receive_message.ui32MsgLen);
				
				{
					int i;
					unsigned int value;
					for(i = 0; i < canbus_receive_message.ui32MsgLen; i++)
						printf("0x%02X ", can_receive_buffer[i]);
		
					memcpy(&value, &can_receive_buffer[0], sizeof(unsigned int));
					printf("(decimal : %u)", value);
				}
				printf("\r\n");
				
				GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_1, GPIO_PIN_1);
				remain_led_n1_blinking_time_in_ms = LED_N1_BLINKING_INTERVAL_IN_MS;
			}

		}
	
	}
}/*main*/

