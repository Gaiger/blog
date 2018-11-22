
#include <string.h>
#include "esb_app_prx_ack.h"

#include "hal_delay.h"
#include "stdio.h"

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

									
void esb_prx_init(void)
{	
	CE_LOW();
	RFCKEN = 1;	 //enable RF timer
	
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
			hal_nrf_set_address(i, l_pipe_addr[i]);
	}/*set all addresss*/

	
#ifdef _RX_FOR_ALL_CHANNEL	
	hal_nrf_open_pipe(HAL_NRF_ALL, true); 
#else
	hal_nrf_open_pipe(RX_PIPE, true); 
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
	
#define MAX_RETRRANS_TIME							(4)	
#define RF_RETRANS_DELAY_IN_US				(150)

	hal_nrf_set_auto_retr(MAX_RETRRANS_TIME, RF_RETRANS_DELAY_IN_US);	
	
	hal_nrf_set_crc_mode(HAL_NRF_CRC_16BIT);	
	
	
	/*enable all rf-interrupt for properly working */
	hal_nrf_set_irq_mode(HAL_NRF_MAX_RT, true);
	hal_nrf_set_irq_mode(HAL_NRF_TX_DS, true);
	hal_nrf_set_irq_mode(HAL_NRF_RX_DR, true);	

	hal_nrf_set_operation_mode(HAL_NRF_PRX);			
		
	hal_nrf_set_power_mode(HAL_NRF_PWR_UP);
	
	RF = 1; //enable rf interrupt
	CE_HIGH();	
}/*enhanced_shockburst_init*/


LOCAL xdata uint8_t  l_rx_payload[ESB_MAX_PAYLOAD_LEN];
LOCAL uint8_t l_rx_payload_len = 0;

LOCAL uint8_t l_is_radio_busy = 0;
LOCAL uint8_t l_is_rf_rcvd_flag = 0;

xdata uint8_t l_received_pipe = 0xff;


static void esb_irq(void) interrupt INTERRUPT_RFIRQ
{
	uint8_t irq_flags;
	
  irq_flags = hal_nrf_get_clear_irq_flags();// read and clear irq flag on register

	
	if(irq_flags & ((1<<HAL_NRF_MAX_RT)))	//re-transmission has reached max number
  {	
		/*never reach here*/
		l_is_radio_busy = 0; 
		hal_nrf_flush_tx();					
  }/*HAL_NRF_MAX_RT*/

	
  if(irq_flags & ((1<<HAL_NRF_TX_DS))) //transimmter finish
	{				
		/*never reach here*/
    l_is_radio_busy = 0;				
	}/*HAL_NRF_TX_DS*/

	
	if(irq_flags & (1<<HAL_NRF_RX_DR)) //this interruption is for receiving
	{		
		if(false == hal_nrf_rx_fifo_empty())
		{
			uint16_t pipe_and_len;			
			pipe_and_len = hal_nrf_read_rx_payload(l_rx_payload);			
			l_received_pipe = (uint8_t)(pipe_and_len >> 8);
			l_rx_payload_len = (uint8_t)(0xff & pipe_and_len);			
		}/*if*/
				
		hal_nrf_flush_rx();
		l_is_rf_rcvd_flag = 1;  	
								
	}/*HAL_NRF_RX_DR*/
	
}/*rf_irq*/


void esb_fetch_received_data(hal_nrf_address_t *p_pipe, 
	uint8_t *p_data, uint8_t *p_len)
{	
	*p_pipe = l_received_pipe;
	*p_len = l_rx_payload_len;
	memcpy(p_data, &l_rx_payload[0], l_rx_payload_len);
}/*rf_fetch_received_data*/


void esb_send_ack_data(uint8_t *p_data, uint8_t len)
{
	if(len > ESB_MAX_ACK_PAYLOAD_LEN)
		len = ESB_MAX_ACK_PAYLOAD_LEN;
	
	hal_nrf_write_ack_payload(l_received_pipe, &p_data[0], len);			
}/*esb_send_ack_data*/


uint8_t is_esb_received_data(void){	return l_is_rf_rcvd_flag;}


void esb_receiving_event_has_been_handled(void){ l_is_rf_rcvd_flag = 0;}
