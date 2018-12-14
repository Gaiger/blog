
#include <string.h>
#include "esb_app.h"



#define LOCAL																		static
	
#define ESB_DATA_RATE														(HAL_NRF_1MBPS)
#define ESB_RF_CHANNEL        									(50)
#define ESB_OUTPUT_POWER												(HAL_NRF_0DBM)


/*
	https://www.nordicsemi.com/eng/Nordic-FAQ/Silicon-Products/nRF24L01/How-to-choose-an-address
	Quick summary:

	Use at least 32bit address and enable 16bit CRC.
	Avoid addresses that start with 0x55, 0xAA. 
	Avoid adopting 0x00 and 0x55 as a part of addresses
*/


#define ADDR_PIPE0_0														(0x10)
#define ADDR_PIPE0_1														(0x11)
#define ADDR_PIPE0_2														(0x12)
#define ADDR_PIPE0_3														(0x13)
#define ADDR_PIPE0_4														(0x14)


#define ADDR_PIPE1TO5_1													(0xE1)
#define ADDR_PIPE1TO5_2													(0xE2)
#define ADDR_PIPE1TO5_3													(0xE3)
#define ADDR_PIPE1TO5_4													(0xE4)
									
#define ADDR_PIPE1_0														(0x81)
#define ADDR_PIPE2_0														(0x82)
#define ADDR_PIPE3_0														(0x83)
#define ADDR_PIPE4_0														(0x84)
#define ADDR_PIPE5_0														(0x85)


static xdata uint8_t l_pipe_addr[6][5] = 
{
	{ADDR_PIPE0_0, ADDR_PIPE0_1, ADDR_PIPE0_2, ADDR_PIPE0_3, ADDR_PIPE0_4},
	{ADDR_PIPE1_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4}, 
	{ADDR_PIPE2_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4},
	{ADDR_PIPE3_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4},
	{ADDR_PIPE4_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4},
	{ADDR_PIPE5_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4}
};										


void esb_init(void)
{	
	CE_LOW();

#if defined(NRF24LE1_H_) || defined(NRF24LU1P_H)
	RFCKEN = 1;	 //enable RF timer
#endif

	hal_nrf_get_clear_irq_flags();
	hal_nrf_flush_rx();
	hal_nrf_flush_tx();
	
	hal_nrf_close_pipe(HAL_NRF_ALL);  /*close all pipe*/
	
	hal_nrf_set_output_power(ESB_OUTPUT_POWER);		
	hal_nrf_set_rf_channel(ESB_RF_CHANNEL);		
	
	hal_nrf_set_datarate(ESB_DATA_RATE);	
	hal_nrf_set_address_width(HAL_NRF_AW_5BYTES);
	
	{
		uint8_t i;
		for(i = HAL_NRF_PIPE0; i <= HAL_NRF_PIPE5; i++)
			hal_nrf_set_address((hal_nrf_address_t)i, l_pipe_addr[i]);
	}/*set all addresss*/

	
#ifdef _TX_PIPE_FIXED	
	hal_nrf_set_address(HAL_NRF_PIPE0, l_pipe_addr[TX_PIPE]);
	hal_nrf_set_address(HAL_NRF_TX, l_pipe_addr[TX_PIPE]);
	
	hal_nrf_open_pipe(HAL_NRF_PIPE0, true);
	hal_nrf_open_pipe(TX_PIPE, true); 	
#endif
	
	hal_nrf_open_pipe(RX_PIPE, true); 
	
	
#if(1)	
	hal_nrf_setup_dynamic_payload(HAL_NRF_ALL);
	hal_nrf_enable_dynamic_payload(true);
#else
	{
		uint8_t i;
		for(i = HAL_NRF_PIPE0; i <= HAL_NRF_PIPE5; i++)
			hal_nrf_set_rx_payload_width(i, MAX_PAYLOAD_LEN);
	}/*hal_nrf_set_rx_payload_width(HAL_NRF_ALL, MAX_PAYLOAD_LEN) does not work*/
#endif		
	
	hal_nrf_enable_dynamic_ack(false);
	hal_nrf_enable_ack_payload(true);
	
#define MAX_RETRRANS_TIME							(3)	
#define RF_RETRANS_DELAY_IN_US				(250)

	hal_nrf_set_auto_retr(MAX_RETRRANS_TIME, RF_RETRANS_DELAY_IN_US);	
	
	hal_nrf_set_crc_mode(HAL_NRF_CRC_16BIT);	
	
	
	/*enable all rf-interrupt for properly working */
	hal_nrf_set_irq_mode(HAL_NRF_MAX_RT, true);
	hal_nrf_set_irq_mode(HAL_NRF_TX_DS, true);
	hal_nrf_set_irq_mode(HAL_NRF_RX_DR, true);	


	hal_nrf_set_operation_mode(HAL_NRF_PRX);				
	hal_nrf_set_power_mode(HAL_NRF_PWR_UP);
	
#if defined(NRF24LE1_H_) || defined(NRF24LU1P_H)
	RF = 1; //enable rf interrupt
#endif	

	CE_HIGH();	
}/*enhanced_shockburst_init*/


LOCAL xdata uint8_t  l_rx_payload[ESB_MAX_PAYLOAD_LEN];
LOCAL uint8_t l_rx_payload_len = 0;

LOCAL uint8_t l_is_radio_busy = 0;
LOCAL uint8_t l_is_rf_rcvd_flag = 0;
LOCAL uint8_t l_is_max_retry_count_reached = 0;

xdata uint8_t l_received_pipe = 0xff;


LOCAL uint8_t l_is_ack_reached_flag = 0;



void esb_irq(void)
{
	uint8_t irq_flags;
	
	// read and clear irq flag on register
  irq_flags = hal_nrf_get_clear_irq_flags();
	

	//re-transmission has reached max number
	if(irq_flags & ((1<<HAL_NRF_MAX_RT)))	
  {		
		l_is_radio_busy = 0; 		
		l_is_max_retry_count_reached	= 1;
  }/*HAL_NRF_MAX_RT*/


	 //transimmter finish
  if(irq_flags & ((1<<HAL_NRF_TX_DS)))
	{				
    l_is_radio_busy = 0;
		//hal_nrf_flush_tx();
	}/*HAL_NRF_TX_DS*/


	//this interruption is for receiving
	if(irq_flags & (1<<HAL_NRF_RX_DR))
	{		
		if(false == hal_nrf_rx_fifo_empty())
		{
			uint16_t pipe_and_len;			
			pipe_and_len = hal_nrf_read_rx_payload(l_rx_payload);
			l_received_pipe = (uint8_t)(pipe_and_len >> 8);
			l_rx_payload_len = (uint8_t)(0xff & pipe_and_len);
		}/*if*/
				
		hal_nrf_flush_rx();

		if(RX_PIPE == l_received_pipe)
			l_is_rf_rcvd_flag = 1; 
		
		if(HAL_NRF_PIPE0 == l_received_pipe)		
			l_is_ack_reached_flag = 1;
				
	}/*HAL_NRF_RX_DR*/

}/*rf_irq*/


#if defined(NRF24LE1_H_) || defined(NRF24LU1P_H)
static void rf_irq(void) interrupt INTERRUPT_RFIRQ
{
		esb_irq();
}/*esb_irq*/
#endif


void esb_send_data(hal_nrf_address_t tx_pipe_number, 
	uint8_t *p_data, uint8_t len)
{					
	CE_LOW();

#ifndef _TX_PIPE_FIXED
	hal_nrf_set_address(HAL_NRF_PIPE0, l_pipe_addr[tx_pipe_number]);
	hal_nrf_set_address(HAL_NRF_TX, l_pipe_addr[tx_pipe_number]);
	
	hal_nrf_open_pipe(HAL_NRF_PIPE0, true);
	hal_nrf_open_pipe(tx_pipe_number, true); 	
#endif	
	
	hal_nrf_set_operation_mode(HAL_NRF_PTX);	
	hal_nrf_write_tx_payload(p_data, len); 
	
	CE_PULSE();	         //emit
	
	CE_HIGH();
	
	l_is_radio_busy = 1;
	while(0 != l_is_radio_busy);

	hal_nrf_flush_tx();
	hal_nrf_set_operation_mode(HAL_NRF_PRX);   //switch to rx	
		
}/*esb_send_data*/


void esb_fetch_received_data(hal_nrf_address_t *p_pipe, 
	uint8_t *p_data, uint8_t *p_len)
{	
	*p_pipe = (hal_nrf_address_t)l_received_pipe;
	*p_len = l_rx_payload_len;
	memcpy(p_data, &l_rx_payload[0], l_rx_payload_len);
}/*rf_fetch_received_data*/


void esb_send_ack_data(uint8_t *p_data, uint8_t len)
{
	if(len > ESB_MAX_ACK_PAYLOAD_LEN)
		len = ESB_MAX_ACK_PAYLOAD_LEN;
	
	hal_nrf_write_ack_payload(3, &p_data[0], len);			
}/*esb_send_ack_data*/


void esb_fetch_ack_data(uint8_t *p_data, uint8_t *p_len)
{
	*p_len = l_rx_payload_len;
	memcpy(p_data, &l_rx_payload[0], l_rx_payload_len);
	l_is_ack_reached_flag = 0;
}/*esp_fetch_ack_data*/


uint8_t is_esb_received_data(void)
{return l_is_rf_rcvd_flag;}


uint8_t is_esb_max_retry_count_reached(void)
{return l_is_max_retry_count_reached;}


void esb_max_retry_count_reached_has_been_handled(void)
{l_is_max_retry_count_reached = 0;}


uint8_t  is_esb_ack_reached(void)
{return l_is_ack_reached_flag;}


void esb_receiving_event_has_been_handled(void)
{ l_is_rf_rcvd_flag = 0;}
