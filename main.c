#include <stdio.h>
#include <string.h>

#include "stm32f10x_conf.h"
#include "esb_app.h"

#define ONE_SEC_IN_MS														(1000)
#define ONE_MS_IN_US														(1000)


void io_init(void)
{
	GPIO_InitTypeDef  gpio_init;
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA|RCC_APB2Periph_GPIOD, ENABLE);

	/*PA.8 */
	gpio_init.GPIO_Pin = GPIO_Pin_8;
	gpio_init.GPIO_Mode = GPIO_Mode_Out_PP;
	gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
	
	GPIO_Init(GPIOA, &gpio_init);
	GPIO_SetBits(GPIOA, GPIO_Pin_8);	
	

	gpio_init.GPIO_Pin = GPIO_Pin_2;
	gpio_init.GPIO_Mode = GPIO_Mode_Out_PP;
	gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
	
	GPIO_Init(GPIOD, &gpio_init);
	GPIO_SetBits(GPIOD, GPIO_Pin_2);	
	
	GPIO_SetBits(GPIOA, GPIO_Pin_8);	
	GPIO_SetBits(GPIOD, GPIO_Pin_2);	
		
}/*io_init*/



void clock_init(void)
{
	SysTick_CLKSourceConfig(SysTick_CLKSource_HCLK_Div8);
	//fac_us = SystemCoreClock/8/ONE_SEC_IN_MS/ONE_MS_IN_US;	
}/*clock_init*/


void delay_us(uint16_t x)
{
		
	SysTick->LOAD = x * (uint32_t)(SystemCoreClock/8/ONE_SEC_IN_MS/ONE_MS_IN_US); 
	
	SysTick->VAL = 0x00;
	SysTick->CTRL|=SysTick_CTRL_ENABLE_Msk ;
	
	{
		uint32_t sys_tick_ctrl_val;
	
		do
		{
			sys_tick_ctrl_val = SysTick->CTRL;
			
		}while((0x01&sys_tick_ctrl_val) 
			&& (0 == ((sys_tick_ctrl_val>>16) &0x01) ) );
		
	}/*local variable*/
	
	SysTick->CTRL&=~SysTick_CTRL_ENABLE_Msk;
	SysTick->VAL =0x00;
}/*delay_us*/


void delay_ms(uint16_t x)
{
	
	do
	{
		uint16_t xx;
				
		xx = x;
		if(1864 < xx) /*0xffffff*ONE_SEC_IN_MS/(SysTick/8) = 1864.135*/		
			xx = 1864;
				
		SysTick->LOAD = xx*(uint32_t)((SystemCoreClock)/8/ONE_SEC_IN_MS);	
		SysTick->VAL =0x00;
		
		SysTick->CTRL|=SysTick_CTRL_ENABLE_Msk ;
		
		{
			uint32_t sys_tick_ctrl_val;
		
			do
			{
				sys_tick_ctrl_val = SysTick->CTRL;
				
			}while((0x01 & sys_tick_ctrl_val)
				&& (0 == ((sys_tick_ctrl_val>>16) &0x01) ) );
			
		}/*local variable*/
		
		SysTick->CTRL&=~SysTick_CTRL_ENABLE_Msk;
		SysTick->VAL =0x00;
		
		x -= xx;
	}while(x > 0);
	
}/*delay_ms*/


#pragma import(__use_no_semihosting)             

struct __FILE 
{ 
	int handle; 
}; 

FILE __stdout;       

void _sys_exit(int x) { x = x;}


int fputc(int ch, FILE *f)
{

	USART_SendData(USART1, (uint16_t)ch);
	while(SET != USART_GetFlagStatus(USART1, USART_FLAG_TXE) );
	
	return ch;
}/*fputc*/


void usart_init(void)
{
	GPIO_InitTypeDef gpio_init;
	USART_InitTypeDef usart_init;
	NVIC_InitTypeDef nvic_init;
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1|RCC_APB2Periph_GPIOA, ENABLE);
	
	/*TX*/
	gpio_init.GPIO_Pin = GPIO_Pin_9; //PA.9
	gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
	gpio_init.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_Init(GPIOA, &gpio_init);
	
	
	/*RX*/
	gpio_init.GPIO_Pin = GPIO_Pin_10;
	gpio_init.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_Init(GPIOA, &gpio_init);
	
	
	nvic_init.NVIC_IRQChannel = USART1_IRQn;
	nvic_init.NVIC_IRQChannelPreemptionPriority = 3;
	nvic_init.NVIC_IRQChannelSubPriority = 3;
	nvic_init.NVIC_IRQChannelCmd = ENABLE;	
	NVIC_Init(&nvic_init);
	
#define BAUDRATE								(38400)

	usart_init.USART_BaudRate = BAUDRATE;
	usart_init.USART_WordLength = USART_WordLength_8b;
	usart_init.USART_StopBits = USART_StopBits_1;
	usart_init.USART_Parity = USART_Parity_No;
	usart_init.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	usart_init.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	
	USART_Init(USART1, &usart_init);
	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
	USART_Cmd(USART1, ENABLE);
	 
}/*uart_init*/


void USART1_IRQHandler(void)
{
	if(SET != USART_GetFlagStatus(USART1, USART_FLAG_RXNE))
		return;
	
	USART_ClearITPendingBit(USART1, USART_IT_RXNE);
	(void)USART_ReceiveData(USART1);
	
}/*USART1_IRQHandler*/


void timer7_init(void)
{
	TIM_TimeBaseInitTypeDef  time_base_init;
	NVIC_InitTypeDef nvic_init;
	
	RCC_ClocksTypeDef rcc_clock;
	
	RCC_GetClocksFreq(&rcc_clock);	
	
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM7, ENABLE);
	
	/*
	 ((timer0_coundown_value + 1)/(CRYSTAL_FREQ/(TIMER_PRESCALER + 1)) 
		= interrupt_interval
		
		-> timer0_coundown_value 
		= interrupt_interval*CRYSTAL_FREQ/(PRESCALER + 1) - 1
	*/	
//#define CRYSTAL_FREQ_IN_Hz							(72*1000*1000)
	
	time_base_init.TIM_Prescaler  =
		(rcc_clock.SYSCLK_Frequency/ONE_MS_IN_US/ONE_SEC_IN_MS) - 1;
	
	time_base_init.TIM_Period = ONE_MS_IN_US - 1;	
	time_base_init.TIM_ClockDivision = 0;
	time_base_init.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM7, &time_base_init);
	
	TIM_ITConfig( TIM7, TIM_IT_Update, ENABLE);
		
	nvic_init.NVIC_IRQChannel = TIM7_IRQn;
	nvic_init.NVIC_IRQChannelPreemptionPriority = 0;
	nvic_init.NVIC_IRQChannelSubPriority = 3;
	nvic_init.NVIC_IRQChannelCmd = ENABLE;
	
	NVIC_Init(&nvic_init);
	TIM_Cmd(TIM7, ENABLE);
	
}/*timer7_init*/




#define PRINT_TIME_INTERVAL_IN_MS								(1000)
uint32_t g_is_need_to_print_time = 0;

uint32_t g_elapsed_time_in_ms = 0;

#define FEED_IWDG_INTERVAL_IN_MS								(500)
uint8_t g_is_need_to_feed_iwdg = 0;

#define SEND_DATA_INTERVAL_IN_MS								(500)
uint8_t g_is_need_to_send_data = 0;


void TIM7_IRQHandler(void) 
{
	if (RESET == TIM_GetITStatus(TIM7, TIM_IT_Update) )
		return;
	
	TIM_ClearITPendingBit(TIM7, TIM_IT_Update  ); 
		
	
	g_elapsed_time_in_ms++;
	
	if(0 == g_elapsed_time_in_ms % FEED_IWDG_INTERVAL_IN_MS)
		g_is_need_to_feed_iwdg = 1;
	
	if(0 == g_elapsed_time_in_ms % PRINT_TIME_INTERVAL_IN_MS)
		g_is_need_to_print_time = 1;		
	
	
	if(0 == g_elapsed_time_in_ms % SEND_DATA_INTERVAL_IN_MS)
			g_is_need_to_send_data = 1;
	
}/*TIM7_IRQHandler*/


void iwdg_init(void)
{
	/*		
		iwdg timeout = value/(IWDG_FREQ/(4*2^PRESCALER_FACTOR))
		IWDG_FREQ = 10 KHz
	*/
	
	IWDG_WriteAccessCmd(IWDG_WriteAccess_Enable);
#define PRESCALER_FACTOR							(5)
	IWDG_SetPrescaler(PRESCALER_FACTOR);
	
#define RELOAD_VALUE									(625)// 2s
	IWDG_SetReload(RELOAD_VALUE);
	IWDG_ReloadCounter();
	IWDG_Enable();
	
}/*iwdg_init*/

void spi1_init(void)
{

	GPIO_InitTypeDef goio_init;
	SPI_InitTypeDef  spi_init;
	
	RCC_APB2PeriphClockCmd(	RCC_APB2Periph_GPIOA|RCC_APB2Periph_SPI1, ENABLE);	

#if(0)	
	goio_init.GPIO_Pin = GPIO_Pin_2 | GPIO_Pin_3 ;
	goio_init.GPIO_Mode = GPIO_Mode_Out_PP;
	goio_init.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &goio_init);
	
	GPIO_SetBits(GPIOA , GPIO_Pin_2|GPIO_Pin_3);
#endif	

	goio_init.GPIO_Pin = GPIO_Pin_5 | GPIO_Pin_6 | GPIO_Pin_7;
	goio_init.GPIO_Mode = GPIO_Mode_AF_PP;
	goio_init.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &goio_init);
	
	GPIO_SetBits(GPIOA , GPIO_Pin_5|GPIO_Pin_6|GPIO_Pin_7);


	SPI_Cmd(SPI1, DISABLE); 
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_SPI1, ENABLE);	
	
	spi_init.SPI_Direction = SPI_Direction_2Lines_FullDuplex;  
	spi_init.SPI_Mode = SPI_Mode_Master;	
	spi_init.SPI_DataSize = SPI_DataSize_8b;		
	spi_init.SPI_CPOL = SPI_CPOL_Low;		
	spi_init.SPI_CPHA = SPI_CPHA_1Edge;	
	spi_init.SPI_NSS = SPI_NSS_Soft;
	spi_init.SPI_CRCPolynomial = 7;
	
#define SPI_BAUDRATEPRESCALER			(SPI_BaudRatePrescaler_16)	
	spi_init.SPI_BaudRatePrescaler = SPI_BAUDRATEPRESCALER;	
	spi_init.SPI_FirstBit = SPI_FirstBit_MSB;	
	
	SPI_Init(SPI1, &spi_init); 
 
	SPI_Cmd(SPI1, ENABLE);

}/*spi1_init*/


void nrf24l01_init(void)
{
	GPIO_InitTypeDef goio_init;
	
	spi1_init();
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA|RCC_APB2Periph_GPIOC, ENABLE);
		
	/*CSN*/
	goio_init.GPIO_Pin = GPIO_Pin_4;
	goio_init.GPIO_Mode = GPIO_Mode_Out_PP;
	goio_init.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOC, &goio_init);
	
	
	/*CE*/
	goio_init.GPIO_Pin = GPIO_Pin_4;
	goio_init.GPIO_Mode = GPIO_Mode_Out_PP;
	goio_init.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &goio_init);
	
	GPIO_SetBits(GPIOC, GPIO_Pin_4);
	GPIO_SetBits(GPIOA, GPIO_Pin_4);

	
	/*IRQ*/
	goio_init.GPIO_Pin = GPIO_Pin_1;
	goio_init.GPIO_Mode = GPIO_Mode_IPU;   
	goio_init.GPIO_Speed = GPIO_Speed_10MHz;
	GPIO_Init(GPIOA, &goio_init);
	
	{		
		EXTI_InitTypeDef exti_init;
		NVIC_InitTypeDef nvic_init;
		
		RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO,ENABLE);
		GPIO_EXTILineConfig(GPIO_PortSourceGPIOA, GPIO_PinSource1);
		
		exti_init.EXTI_Line = EXTI_Line1;
  	exti_init.EXTI_Mode = EXTI_Mode_Interrupt;	
  	exti_init.EXTI_Trigger = EXTI_Trigger_Falling;
  	exti_init.EXTI_LineCmd = ENABLE;
  	EXTI_Init(&exti_init);
		
		nvic_init.NVIC_IRQChannel = EXTI1_IRQn;
		nvic_init.NVIC_IRQChannelPreemptionPriority = 1;
		nvic_init.NVIC_IRQChannelSubPriority = 0;
		nvic_init.NVIC_IRQChannelCmd = ENABLE;	
		NVIC_Init(&nvic_init);		
	}
	
	
	
}/*nrf24l01_init*/


void EXTI1_IRQHandler(void)
{
	EXTI_ClearITPendingBit(EXTI_Line1);	
	esb_irq();	
}/*EXTI1_IRQHandler*/


void initialized_notification(void)
{
	{
		RCC_ClocksTypeDef rcc_clock;
		RCC_GetClocksFreq(&rcc_clock);	
		printf("Alientek stm32f103RC %u Hz\r\n", 
			rcc_clock.SYSCLK_Frequency);		
	}/* system*/
	
	{
		int i;	
		for(i = 0; i< 2; i++){
				GPIO_ResetBits(GPIOA, GPIO_Pin_8);	
				GPIO_ResetBits(GPIOD, GPIO_Pin_2);	
				delay_ms(50);
				GPIO_SetBits(GPIOA, GPIO_Pin_8);
				GPIO_SetBits(GPIOD, GPIO_Pin_2);	
				delay_ms(50);
		}/*for i*/
	}/*local variable*/
	
}/*initialized_notification*/


void rf_max_retry_count_reached(void)
{		
	esb_max_retry_count_reached_has_been_handled();	
	printf("rf_max_retry_count_reached!!\r\n");	

	{
		int i;	
		for(i = 0; i< 2; i++){
				GPIO_ResetBits( GPIOA, GPIO_Pin_8);
				delay_ms(15);
				GPIO_SetBits(GPIOA, GPIO_Pin_8);
				delay_ms(5);
		}/*for i*/	 
	}/*local variable*/

}/*rf_max_retry_count_reached*/


uint8_t g_sending_packet_serial_number = 0;

void run_event_loop(void)
{
	uint8_t buffer[ESB_MAX_PAYLOAD_LEN];	
	uint8_t len;
	
	if(0 != g_is_need_to_print_time)
	{

		printf("%u sec\r\n", g_elapsed_time_in_ms/ONE_SEC_IN_MS);
		g_is_need_to_print_time = 0;
						
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

		esb_send_data((hal_nrf_address_t)pipe, &buffer[0], len);
		
		printf("rf send in pipe = %u, len = %u::", 
			(uint16_t)pipe, (uint16_t)len);
		for(i = 0; i< len; i++)
			printf(" %02x", (uint16_t) buffer[i]);
		printf("\r\n");

		GPIO_ResetBits(GPIOA, GPIO_Pin_8);	
		delay_ms(2);
		GPIO_SetBits(GPIOA, GPIO_Pin_8);

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
		
		GPIO_ResetBits(GPIOD, GPIO_Pin_2);	
		delay_ms(2);
		GPIO_SetBits(GPIOD, GPIO_Pin_2);	


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


int main(void)
{	
		
	clock_init();
	io_init();
	
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	
	usart_init();
	timer7_init();
	
	iwdg_init();
	
	nrf24l01_init();
	esb_init();
	
	initialized_notification();
	
	
	while(1)
	{
		run_event_loop();
	}/*while 1*/
		
}/*main*/


