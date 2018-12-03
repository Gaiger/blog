/* MAIN.C file
 * 
 * Copyright (c) 2002-2005 STMicroelectronics
 */

#include "STM8l10x_conf.h"
#include <stdio.h>
#include <string.h>
#include "esb_app_txrx_ack.h"

#define _ENABLE_LED


/*
	GPIO definition :
	
	GPIOC::GPIO_Pin_2 :: USART RX, IN
	GPIOC::GPIO_Pin_3 :: USART TX, OUT
	
	GPIOA::GPIO_Pin_2  -> LED RED, OUT
	GPIOA::GPIO_Pin_3  -> LED GREEN, OUT
	
	
	GPIOB::GPIO_Pin_7  -> SPI_MISO, IN
	GPIOB::GPIO_Pin_6  -> SPI_MOSI, OUT
	GPIOB::GPIO_Pin_5  -> SPI_SCK, OUT
	GPIOB::GPIO_Pin_4  -> SPI::NSS/CSN, OUT	
	GPIOB::GPIO_Pin_3  -> NRF24l01::CE, OUT
	GPIOC::GPIO_Pin_0  -> NRF24l01::IRQ, IN
	
*/
	
void io_init(void)
{
#ifdef _ENABLE_LED	
	GPIO_Init( GPIOA, GPIO_Pin_2 | GPIO_Pin_3, 
		GPIO_Mode_Out_PP_High_Fast);
	GPIO_ResetBits( GPIOA, GPIO_Pin_2);	
	GPIO_ResetBits( GPIOA, GPIO_Pin_3);
#endif

}/*io_init*/


void clock_init(void)
{
	CLK_DeInit();
	CLK_PeripheralClockConfig(CLK_Peripheral_TIM2
		|CLK_Peripheral_TIM3|CLK_Peripheral_TIM4
		|CLK_Peripheral_I2C|CLK_Peripheral_SPI
		|CLK_Peripheral_USART| CLK_Peripheral_AWU	
		,DISABLE);
		
#define MASTER_PRESCALER				(CLK_MasterPrescaler_HSIDiv8)

	CLK_MasterPrescalerConfig(MASTER_PRESCALER);
}/*clock_init*/


void usart_init(void)
{
	USART_DeInit();	
	
	GPIO_Init( GPIOC, GPIO_Pin_2, GPIO_Mode_In_PU_No_IT);
	GPIO_Init( GPIOC, GPIO_Pin_3, GPIO_Mode_Out_PP_High_Slow);
	
	CLK_PeripheralClockConfig(CLK_Peripheral_USART, ENABLE);
	
#define BAUDRATE														(38400)
	USART_Init(BAUDRATE, USART_WordLength_8D, USART_StopBits_1, 
	USART_Parity_No,USART_Mode_Tx|USART_Mode_Rx);		
	
	USART_ITConfig(USART_IT_RXNE, ENABLE);
	
	USART_Cmd(ENABLE);
}/*uart_init*/


char putchar(char c)
{
	USART_SendData8((uint8_t)c);
	
	while(SET != USART_GetFlagStatus(USART_FLAG_TXE) );
	
	return c;
}/*putchar*/


@far @interrupt void usart_received_interrupt(void)
{
	if(SET != USART_GetFlagStatus(USART_FLAG_RXNE))
		return;
		
	USART_ClearITPendingBit();
	GPIO_ToggleBits( GPIOA, GPIO_Pin_3);
}/*usart_received_interrupt*/


void iwdg_init(void)
{
	uint8_t iwdg_reload_count;
	while(RST_GetFlagStatus(RST_FLAG_IWDGF) != RESET)	
		RST_ClearFlag(RST_FLAG_IWDGF);
		
	/*				 
		IWDG_timeout = (iwdg_reload_count + 1)/(LSI_FREQ/IWDG_PRESCALER) 
		
		for iwdg_reload_count = 255, the IWDG_timeout = 1725 ms
	*/
#define LSI_FREQ							(38000L)
#define IWDG_PRESCALER				(IWDG_Prescaler_256)
	
	iwdg_reload_count = 255;
	
	IWDG_Enable();
	IWDG_WriteAccessCmd(IWDG_WriteAccess_Enable);
		
	IWDG_SetPrescaler(IWDG_Prescaler_256);
	IWDG_SetReload(iwdg_reload_count);
	IWDG_ReloadCounter();
	
}/*IWdg_Init*/



#define FEED_IWDG_INTERVAL_IN_MS								(1 << 9)
#define FAST_MOD_FOR_POWER_2(VAL, DIVISOR)			((VAL) & ((DIVISOR)- 1) )

uint8_t g_is_need_to_feed_iwdg = 0;

#define PRINT_TIME_INTERVAL_IN_MS								(1000)
uint8_t g_is_need_to_print_time = 0;

#define SEND_DATA_INTERVAL_IN_MS								(500)
uint8_t g_is_need_to_send_data = 0;


#define ONE_SEC_IN_MS														(1000)
#define ONE_MS_IN_US														(1000)


 /*use 16 bit integer as time counter to avoid 
 the calculation of remainder (e.g g_elaspsed_time_in_ms %
 PRINT_TIME_INTERVAL_IN_MS)  costs too much time bringing 
 system hanging*/
 
uint16_t g_elaspsed_time_in_ms = 0;
uint16_t g_elaspsed_min = 0;

@far @interrupt void timer3_interrupt_handler(void)
{
	TIM3_ClearFlag(TIM3_IT_Update);
	
	g_elaspsed_time_in_ms++;	

#if(0)
	if(0 == FAST_MOD_FOR_POWER_2(g_elaspsed_time_in_ms,
			FEED_IWDG_INTERVAL_IN_MS) )
	{
			g_is_need_to_feed_iwdg = 1;	
	}/*if feed iwdg */
#else
	if(0 == g_elaspsed_time_in_ms%FEED_IWDG_INTERVAL_IN_MS)
		g_is_need_to_feed_iwdg = 1;
#endif

	if(0 == g_elaspsed_time_in_ms % PRINT_TIME_INTERVAL_IN_MS)
			g_is_need_to_print_time = 1;	

	if(0 == g_elaspsed_time_in_ms % SEND_DATA_INTERVAL_IN_MS)
			g_is_need_to_send_data = 1;

	if((uint16_t)(60*ONE_SEC_IN_MS) <= g_elaspsed_time_in_ms)
	{
		g_elaspsed_time_in_ms %= (uint16_t)(60*ONE_SEC_IN_MS);
		g_elaspsed_min++;
	}/*to minute*/
	
	
}/*timer3_interrupt_handler*/


void timer3_init(void)
{	
	/*
	 ((timer0_coundown_value + 1)/(CRYSTAL_FREQ/TIMER_PRESCALER) 
		= interrupt_interval
		
		-> timer0_coundown_value 
		= interrupt_interval*CRYSTAL_FREQ/PRESCALER - 1
	*/	
	uint8_t timer3_load_value_for_one_ms;
	
	CLK_PeripheralClockConfig(CLK_Peripheral_TIM3, ENABLE);

#define TIMER3_PRESCALER					(TIM3_Prescaler_128)
	timer3_load_value_for_one_ms =
		1*CLK_GetClockFreq()/(1 << TIMER3_PRESCALER)
		/ONE_SEC_IN_MS - 1;
		
	TIM3_TimeBaseInit(TIMER3_PRESCALER, 
		TIM3_CounterMode_Down, timer3_load_value_for_one_ms);
	
	TIM3_ARRPreloadConfig(ENABLE);
	TIM3_ITConfig(TIM3_IT_Update, ENABLE);
	TIM3_Cmd(ENABLE);
}/*timer3_init*/


void delay_ms(uint16_t x) 
{
	/*
	 ((timer4_overflow_value + 1)/(CRYSTAL_FREQ/TIMER_PRESCALER)
		= interrupt_interval
		
		-> timer4_overflow_value 
		= interrupt_interval*CRYSTAL_FREQ/PRESCALER - 1
	*/	
	uint8_t timer4_load_value_for_one_ms;

	CLK_PeripheralClockConfig(CLK_Peripheral_TIM4 , ENABLE);

#define TIMER4_PRESCALER					(TIM4_Prescaler_128) 	
	timer4_load_value_for_one_ms =
		1*CLK_GetClockFreq()/(1 << TIMER4_PRESCALER)/ONE_SEC_IN_MS
		- 1;	
	
	TIM4_TimeBaseInit(TIMER4_PRESCALER, 
		timer4_load_value_for_one_ms);		

	TIM4_ARRPreloadConfig(ENABLE);
	TIM4_ClearFlag(TIM4_FLAG_Update); 	

	TIM4_Cmd(ENABLE);
	
	while(x)
  {
		while( RESET == TIM4_GetFlagStatus(TIM4_FLAG_Update));
		
    TIM4_ClearFlag(TIM4_FLAG_Update); 
		x--;
	}/* x */
	
	TIM4_Cmd(DISABLE);
	
}/*delay_ms*/


@far @interrupt void external_interrupt0_handler(void)
{
	EXTI_ClearITPendingBit(GPIO_Pin_0);	
	esb_irq();
}/*external_interrupt0_handler*/


void spi_init(void)
{
	//Config the GPIOs for SPI bus
	
	/*
		GPIOB::GPIO_Pin_7  -> SPI_MISO			
		GPIOB::GPIO_Pin_6  -> SPI_MOSI		
		GPIOB::GPIO_Pin_5  -> SPI_SCK			
	*/
	GPIO_Init( GPIOB, GPIO_Pin_7, GPIO_Mode_In_PU_No_IT);
	GPIO_Init( GPIOB, GPIO_Pin_5 | GPIO_Pin_6, 
		GPIO_Mode_Out_PP_High_Slow);
		
	SPI_DeInit();
	//enable clock for SPI bus
	CLK_PeripheralClockConfig(CLK_Peripheral_SPI, ENABLE);

/*SPI BAUDRATE should not be over 5MHz*/
#define SPI_BAUDRATEPRESCALER			(SPI_BaudRatePrescaler_4)

	//Set the priority of the SPI
	SPI_Init( SPI_FirstBit_MSB, SPI_BAUDRATEPRESCALER,
            SPI_Mode_Master, SPI_CPOL_Low, SPI_CPHA_1Edge,
            SPI_Direction_2Lines_FullDuplex, SPI_NSS_Soft);
	//Enable SPi
	SPI_Cmd(ENABLE);	
	
}/*spi_init*/


void nrf24l01_init(void)
{		
	spi_init();
	
	/*GPIOB::GPIO_Pin_4 -> SPI::NSS	-> NRF24l01::CSN*/
	GPIO_Init( GPIOB, GPIO_Pin_4, GPIO_Mode_Out_PP_High_Fast);
	
	/*GPIOB::GPIO_Pin_3 -> NRF24l01::CE*/
	GPIO_Init( GPIOB, GPIO_Pin_3, GPIO_Mode_Out_PP_High_Fast);

	/*GPIOC::GPIO_Pin_0 -> NRF24l01::IRQ*/	
	GPIO_Init(GPIOC, GPIO_Pin_0, GPIO_Mode_In_PU_IT);	
	EXTI_SetPinSensitivity(GPIO_Pin_0, EXTI_Trigger_Falling_Low);
	EXTI_ClearITPendingBit(GPIO_Pin_0);
}/*spi_init*/


void initialized_notification(void)
{	
	printf("Example nRF24L01 + STM101XX!!\r\n");
	printf("stm8l01F3P in freq = %lu Hz\r\n", 
		CLK_GetClockFreq());	
		
#ifdef _ENABLE_LED
	{
		int i;	
			
		for(i = 0; i< 2; i++){
			GPIO_SetBits( GPIOA, GPIO_Pin_2 | GPIO_Pin_3);
			delay_ms(50);
			GPIO_ResetBits(GPIOA, GPIO_Pin_2 | GPIO_Pin_3);
			delay_ms(50);
		}/*for i*/
		
		GPIO_SetBits( GPIOA, GPIO_Pin_2 | GPIO_Pin_3);
		delay_ms(250);
		GPIO_ResetBits(GPIOA, GPIO_Pin_2 | GPIO_Pin_3);
			
	}/*local variable*/
#endif
}/*initialized_notify*/


void rf_max_retry_count_reached(void)
{		
	esb_max_retry_count_reached_has_been_handled();	
	printf("rf_max_retry_count_reached!!\r\n");	

#ifdef _ENABLE_LED	
	{
		int i;	
		for(i = 0; i< 2; i++){
				GPIO_SetBits( GPIOA, GPIO_Pin_2);
				delay_ms(15);
				GPIO_ResetBits(GPIOA, GPIO_Pin_2);
				delay_ms(5);
		}/*for i*/	 
	}/*local variable*/
#endif

}/*rf_max_retry_count_reached*/


uint8_t g_sending_packet_serial_number = 0;

void run_event_loop(void)
{
	uint8_t buffer[ESB_MAX_PAYLOAD_LEN];	
	uint8_t len;
	
	
	if(	0 != g_is_need_to_print_time)
	{
		g_is_need_to_print_time = 0;
		
		/*
			if there is a variable of type uint32_t, note that
			the value to be printed must not be greater than S16_MAX,
			because there is no %llu in cosmic c stdlib, printing 
			the out of range value will lead hanging.		
		*/
		
		printf(" %u:%02u\r\n", g_elaspsed_min, 
			g_elaspsed_time_in_ms/ONE_SEC_IN_MS);
	}/*print time*/
		
	
	if(0 != g_is_need_to_send_data)
	{
		uint8_t i;
		uint8_t pipe;
		g_is_need_to_send_data = 0;	
		
		pipe = TX_PIPE;			
		len = 1;
	
		memcpy(&buffer[len], &pipe, sizeof(pipe));
		len += sizeof(pipe);
	
		memcpy(&buffer[len], &g_sending_packet_serial_number, 
			sizeof(g_sending_packet_serial_number));
		len += sizeof(g_sending_packet_serial_number);
	
		buffer[0] = len;

		esb_send_data(pipe, &buffer[0], len);
		
		printf("rf send in pipe = %u, len = %u::", 
			(uint16_t)pipe, (uint16_t)len);
		for(i = 0; i< len; i++)
			printf(" %02x", (uint16_t) buffer[i]);
		printf("\r\n");

#ifdef _ENABLE_LED
		GPIO_SetBits( GPIOA, GPIO_Pin_2);
		delay_ms(2);
		GPIO_ResetBits(GPIOA, GPIO_Pin_2);	
#endif
		g_sending_packet_serial_number++;				
	}/*send data*/


	if(0 != is_esb_received_data())
	{
		uint8_t i;						
		hal_nrf_address_t pipe;			
		
		esb_fetch_received_data(&pipe, &buffer[0], &len);
		esb_receiving_event_has_been_handled();		
			
		printf("rf rcv in pipe = %u, len = %u::", 
			(uint16_t)pipe, (uint16_t)len);			
		
		for(i = 0; i< len; i++)
			printf(" %02x", (uint16_t)buffer[i]);			
		printf("\r\n");						
		
#ifdef _ENABLE_LED
		GPIO_SetBits( GPIOA, GPIO_Pin_3);
		delay_ms(2);
		GPIO_ResetBits(GPIOA, GPIO_Pin_3);	
#endif


/*
 In tx/rx switching mode, 
 the ack-payload would not be received by the sender, 
 I do not know why.
*/
#if(0)
		for(i = 0; i< ESB_MAX_ACK_PAYLOAD_LEN; i++)
			buffer[i] = i;		
		esb_send_ack_data(&buffer[0], ESB_MAX_ACK_PAYLOAD_LEN);
		
		printf("ack len = %u:: ", (uint16_t)ESB_MAX_ACK_PAYLOAD_LEN);
		for(i = 0; i< ESB_MAX_ACK_PAYLOAD_LEN; i++)
			printf(" %02x", (uint16_t)buffer[i]);
		printf("\r\n");
#endif					
	}/*if received_data*/			


	if(0 != is_esb_max_retry_count_reached())
	{	
		rf_max_retry_count_reached();			
	}/*if is_max_retry_count_reached()*/
	
	if(0 != g_is_need_to_feed_iwdg)
	{
		IWDG_ReloadCounter();
		g_is_need_to_feed_iwdg = 0;
	}/*feed iwdg*/
	
}/*run_event_loop*/


void main(void)
{	
	disableInterrupts();
	
	clock_init();
	io_init();
	
	usart_init();	
	timer3_init();
	
	nrf24l01_init();
	
	iwdg_init();
	enableInterrupts();
	
	esb_txrx_init();
	initialized_notification();
	
	
	while (1)
	{						
		run_event_loop();
	}/*while 1*/
	
}/*main*/