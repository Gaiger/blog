

#include <stdio.h> 
#include <stdint.h>
#include <string.h>

#include <nrf24le1.h>

#include "hal_uart.h"
#include "hal_clk.h"
#include "hal_delay.h"

#include "hal_wdog.h"

#include "nordic_common.h"


#include "esb_app_ptx_noack.h"


#define _ENABLE_LED_AND_BEEP

#define ENTER_CRITICAL_SECTION(XX)				(EA = 0)
#define EXIT_CRITICAL_SECTION(XX)					(EA = 1)

char putchar(char c)
{		
	hal_uart_putchar((uint8_t)c);	
	return c;
}/*putchar*/


/*
	P00:out, D1	       				P12:in, button S1  
	P01:out, D2			   				P13:in,button S2
	P02: out, OLED timer			 P14:out,OLED MOSI
	P03: out, UART TXD			   P15: out,OLED chip select 
	P04: in, UART RXD			     P16:out, OLED command/data switcher
	P06: AIN6  AD detection/out Buzzer
*/

#ifdef _ENABLE_LED_AND_BEEP
	#define D1    			P00 
	#define D2					P01
	#define BEEP 				P06
#endif

void io_init(void)
{
#ifdef _ENABLE_LED_AND_BEEP	
	P0DIR &= ~BIT_0; // D1
	P0DIR &= ~BIT_1; // D2
	
	P0DIR &= ~0x40; //P06 BEEP		
#endif
	P0DIR &= ~BIT_3;    //P03: UART TXD
	P0DIR |= BIT_4;     //P04: UART RXD	
	
}/*io_init*/


void clock_init(void)
{		
	// Always run on 16MHz crystal oscillator
	hal_clk_set_16m_source(HAL_CLK_XOSC16M); 
	
	//32 KHz from 32KRCOSC
	hal_clklf_set_source(HAL_CLKLF_RCOSC32K);
		
	while(false == hal_clklf_ready()); 
}/*clock_init*/


#define MCU_CRYSTAL_FREQUENCY_IN_HZ							(16*1000*1000L)
#define CLOCK_NUMBER_PER_MACHINE_CYCLE					(12L)	


#define FEED_DOG_INTERVAL_IN_MS									(500L)
uint8_t g_is_need_to_feed_the_watchdog = 0;

#define PRINT_TIME_INTERVAL_IN_MS								(1000L)
uint8_t g_is_need_to_print_time = 0;

#ifndef _PRX
#define SEND_DATA_INTERVAL_IN_MS								(500L)
uint8_t g_is_need_to_send_data = 0;
#endif

xdata uint32_t g_elaspsed_time_in_ms = 0L;

#define ONE_MILLI_SEC_IN_US											(1000)
#define MINI_TIME_SCALE_IN_US										(100)

void timer0_irq() interrupt INTERRUPT_T0
{	
	static uint8_t timer0_overflow_counter = 0;		
	
	timer0_overflow_counter++;	
	
	if((ONE_MILLI_SEC_IN_US/MINI_TIME_SCALE_IN_US) <= timer0_overflow_counter)
	{					
		timer0_overflow_counter = 0;						
		g_elaspsed_time_in_ms++;		
		
		if(0 == g_elaspsed_time_in_ms % FEED_DOG_INTERVAL_IN_MS)
			g_is_need_to_feed_the_watchdog = 1;	
		
		if(0 == g_elaspsed_time_in_ms % PRINT_TIME_INTERVAL_IN_MS)
			g_is_need_to_print_time = 1;

		if(0 == g_elaspsed_time_in_ms % SEND_DATA_INTERVAL_IN_MS)
			g_is_need_to_send_data = 1;	
	}/*if */		
}/*timer0_irq*/


void timer0_init(void)
{	  			
  uint8_t timer0_overflow_counter = 0;
	
	timer0_overflow_counter = 
		MCU_CRYSTAL_FREQUENCY_IN_HZ/CLOCK_NUMBER_PER_MACHINE_CYCLE/1000L*MINI_TIME_SCALE_IN_US/1000L;		
	
	/*timer 0 be mode 2, 8 bit auto reload*/
	TMOD |= (BIT_1 & ~BIT_0);
	  
	
  TH0 = 0x100 - timer0_overflow_counter;	 
	TL0 = TH1; 	

  ET0  = 1;		 //timer0 interrupt enable
	TR0  = 1;		

}/*timer0_init*/


uint32_t get_elasped_time_in_ms(void)
{		
	return g_elaspsed_time_in_ms;
}/*get_leasped_time*/


void initialized_notification(void)
{
	printf("Welcome ESB TX NOACK!!\r\n");    
	
#ifdef _ENABLE_LED_AND_BEEP	
	{
		int i;	
			
		for(i = 0; i< 2; i++){
			BEEP	= 1;
			delay_ms(30);
			BEEP = 0;
			delay_ms(15);
		}/*for i*/	 
	}/*local variable*/
#endif
	
}/*initialized_notify*/


void rf_max_retry_count_reached(void)
{		
	//esb_max_retry_count_reached_has_been_handled();	
	printf("rf_max_retry_count_reached!!\r\n");	
}/*rf_max_retry_count_reached*/



uint8_t g_sending_packet_serial_number = 0;

void run_event_loop(void)
{		
	uint8_t buffer[ESB_MAX_PAYLOAD_LEN];
	uint8_t len;
	
#define LED_BLINKING_INTERVAL_IN_MS							(10)					
	
	if(0 != g_is_need_to_print_time)
	{							
		printf(" %lu sec\r\n", get_elasped_time_in_ms()/1000L);															
		g_is_need_to_print_time = 0;				
	}/*if */						

	
	if(0 != g_is_need_to_send_data)
	{								
		uint32_t elapsed_time_in_ms;
					
	  elapsed_time_in_ms = get_elasped_time_in_ms();	
						
		buffer[0] = 0;
		len  = 1;		
					
		memcpy(&buffer[len], &g_sending_packet_serial_number, 
			sizeof(g_sending_packet_serial_number));		
		len += sizeof(g_sending_packet_serial_number);
		
		memcpy(&buffer[len], &elapsed_time_in_ms, sizeof(elapsed_time_in_ms));									
		len += sizeof(elapsed_time_in_ms);				
		
		buffer[0] = len - 1;			
		
#ifdef _TX_FOR_ALL_CHANNEL
		{
			uint8_t j;
			for(j = HAL_NRF_PIPE0; j <= HAL_NRF_PIPE5; j++){
				esb_send_data(j, &buffer[0], len);
				printf("rf snd in pipe = %bu, len = %bu\r\n", j, len); 
				delay_ms(20);
			}/*for */		
		}/*local variable*/
#else
		{
			uint8_t i;		
			esb_send_data(TX_PIPE,  &buffer[0], len);						
			printf("rf snd in pipe = %bu, len = %bu::", 
				TX_PIPE, len);				
			for(i = 0; i< len; i++)
				printf(" %02bx",buffer[i]);
			printf("\r\n");
		}/*local variable*/
#endif				
		
#ifdef _ENABLE_LED_AND_BEEP		
		D1 = 0;
		delay_ms(LED_BLINKING_INTERVAL_IN_MS);	
		D1 = 1;
#endif
		g_sending_packet_serial_number++;	
		g_is_need_to_send_data = 0;
	}/*if g_is_need_to_send_data*/

	
	if(0 != g_is_need_to_feed_the_watchdog)
	{
		hal_wdog_restart();
		g_is_need_to_feed_the_watchdog = 0;
	}/*if */
	
}/*run_time_events*/


void main(void)
{		
	
	ENTER_CRITICAL_SECTION();		
	
	io_init();
	clock_init();

	hal_uart_init(UART_BAUD_38K4);		
		
	timer0_init();
	
#define WDOG_TIMEOUT_IN_SEC							(5*FEED_DOG_INTERVAL_IN_MS/1000)	
#ifdef _ENABLE_POWER_SAVING	
	#if WDOG_TIMEOUT_IN_SEC <= POWER_SAVING_TIME_IN_SEC
		#undef WDOG_TIMEOUT_IN_SEC
		#define WDOG_TIMEOUT_IN_SEC  			(POWER_SAVING_TIME_IN_SEC + 1)
	#endif
#endif
  /*watch dog timeout = SDSV*256/32768*/		
	hal_wdog_init(WDOG_TIMEOUT_IN_SEC*32768/256); 		
	
	while(hal_clk_get_16m_source() != HAL_CLK_XOSC16M); /*wait clock to be stable*/			
		
	esb_ptx_init();
	EXIT_CRITICAL_SECTION();		
	
	initialized_notification();			
	
	while(1)
	{
		run_event_loop();
	}/*while*/
	
}/*main*/
