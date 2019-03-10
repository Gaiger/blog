
#include <string.h>
#include "esb_app_ptx_ack.h"

#include "hal_delay.h"
//#include "stdio.h"

#define LOCAL																		static
	

static xdata uint8_t l_pipe_addr[6][5] = 
{
	{ADDR_PIPE0_0, ADDR_PIPE0_1, ADDR_PIPE0_2, ADDR_PIPE0_3, ADDR_PIPE0_4},
	{ADDR_PIPE1_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4}, 
	{ADDR_PIPE2_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4},
	{ADDR_PIPE3_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4},
	{ADDR_PIPE4_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4},
	{ADDR_PIPE5_0, ADDR_PIPE1TO5_1, ADDR_PIPE1TO5_2, ADDR_PIPE1TO5_3, ADDR_PIPE1TO5_4}
};										


void esb_ptx_init(void)
{	
	CE_LOW();
	RFCKEN = 1;	 //enable RF timer
	
	hal_nrf_get_clear_irq_flags();
	hal_nrf_flush_rx();
	hal_nrf_flush_tx();
	
	hal_nrf_close_pipe(HAL_NRF_ALL);
	
	hal_nrf_set_output_power(ESB_OUTPUT_POWER);		
	hal_nrf_set_rf_channel(ESB_RF_CHANNEL);		
	
	hal_nrf_set_datarate(ESB_DATA_RATE);	
	hal_nrf_set_address_width(HAL_NRF_AW_5BYTES);
	
	{
		uint8_t i;
		for(i = HAL_NRF_PIPE0; i <= HAL_NRF_PIPE5; i++)
			hal_nrf_set_address(i, l_pipe_addr[i]);
	}/*set all addresss*/

#ifndef _TX_FOR_ALL_CHANNEL		
	hal_nrf_set_address(HAL_NRF_PIPE0, l_pipe_addr[TX_PIPE]);	
	hal_nrf_set_address(HAL_NRF_TX, l_pipe_addr[TX_PIPE]);
	
	hal_nrf_open_pipe(HAL_NRF_PIPE0, true);
	hal_nrf_open_pipe(TX_PIPE, true);
#endif
	
	
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
	
#define MAX_RETRRANS_TIME							(5)	
#define RF_RETRANS_DELAY_IN_US				(100)

	hal_nrf_set_auto_retr(MAX_RETRRANS_TIME, RF_RETRANS_DELAY_IN_US);		
	
	hal_nrf_set_crc_mode(HAL_NRF_CRC_16BIT);
		
	/*enable all rf-interrupt for properly working */
	hal_nrf_set_irq_mode(HAL_NRF_MAX_RT, true);
	hal_nrf_set_irq_mode(HAL_NRF_TX_DS, true);
	hal_nrf_set_irq_mode(HAL_NRF_RX_DR, true);	

	hal_nrf_set_operation_mode(HAL_NRF_PTX);			
	
	hal_nrf_set_power_mode(HAL_NRF_PWR_UP);
	
	RF = 1; //enable rf interrupt
	CE_HIGH();	
}/*enhanced_shockburst_init*/


LOCAL xdata uint8_t  l_rx_payload[ESB_MAX_PAYLOAD_LEN];
LOCAL uint8_t l_rx_payload_len = 0;

LOCAL uint8_t l_is_radio_busy = 0;
LOCAL uint8_t l_is_max_retry_count_reached = 0;

LOCAL uint8_t l_is_ack_reached_flag = 0;


static void esb_irq(void) interrupt INTERRUPT_RFIRQ
{
	uint8_t irq_flags;
	
  irq_flags = hal_nrf_get_clear_irq_flags();// read and clear irq flag on register

	
	if(irq_flags & ((1<<HAL_NRF_MAX_RT)))	//re-transmission has reached max number
  {		
		l_is_radio_busy = 0; 
		hal_nrf_flush_tx();			
		l_is_max_retry_count_reached	= 1;
  }/*HAL_NRF_MAX_RT*/

	if(irq_flags & ((1<<HAL_NRF_TX_DS))) //transimmter finish
	{				
    l_is_radio_busy = 0;				
	}/*HAL_NRF_TX_DS*/

	
	if(irq_flags & (1<<HAL_NRF_RX_DR)) //this interruption is for receiving
	{	
		//ack payload received
		uint8_t received_pipe;	
		if(false == hal_nrf_rx_fifo_empty())
		{					
			uint16_t pipe_and_len;						
			pipe_and_len = hal_nrf_read_rx_payload(&l_rx_payload[0]);			
			
			received_pipe = (uint8_t)(pipe_and_len >> 8);			
			l_rx_payload_len = (uint8_t)(0xff & pipe_and_len);						
		}/*if */
		
		hal_nrf_flush_rx();
		if(HAL_NRF_PIPE0 == received_pipe)
				l_is_ack_reached_flag = 1;
			
	}/*if */		
	
}/*rf_irq*/


void esb_send_data(hal_nrf_address_t tx_pipe_number, 
	uint8_t *p_data, uint8_t len)
{	
	uint32_t snd_k;
				
	CE_LOW();
	
#if defined(_TX_FOR_ALL_CHANNEL)
	hal_nrf_close_pipe(HAL_NRF_ALL);
			
	hal_nrf_set_address(HAL_NRF_TX, l_pipe_addr[tx_pipe_number]);
	hal_nrf_set_address(HAL_NRF_PIPE0, l_pipe_addr[tx_pipe_number]);
	hal_nrf_open_pipe(tx_pipe_number, true); 	
	hal_nrf_open_pipe(HAL_NRF_PIPE0, true);
#endif
	
	hal_nrf_write_tx_payload(p_data, len); 
	
	CE_PULSE();	         //emit 	
  l_is_radio_busy = 1;
	
	snd_k = 0;
	while(0 != l_is_radio_busy)
	{
		//snd_k++; 
		//delay_us(10);			
	};//wait done
	//printf("sent, cost = %ld us\r\n", snd_k*10);
		
	hal_nrf_flush_tx();
	CE_HIGH();	
}/*esb_send_data*/


void esb_fetch_ack_data(uint8_t *p_data, uint8_t *p_len)
{
	*p_len = l_rx_payload_len;
	memcpy(p_data, &l_rx_payload[0], l_rx_payload_len);
	l_is_ack_reached_flag = 0;
}/*esp_fetch_ack_data*/


uint8_t is_esb_max_retry_count_reached(void){ return l_is_max_retry_count_reached;}

void esb_max_retry_count_reached_has_been_handled(void){ l_is_max_retry_count_reached = 0;}

uint8_t  is_esb_ack_reached(void){	return l_is_ack_reached_flag;}

