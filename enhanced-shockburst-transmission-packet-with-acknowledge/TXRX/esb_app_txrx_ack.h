
#ifndef _ESB_APP_ACK_H_
#define _ESB_APP_ACK_H_

#include <stdint.h>

#include "hal_nrf.h"
#include "hal_nrf_hw.h"

#include "trscvr_para.h"


#define _TX_PIPE_FIXED

#ifdef _TX_PIPE_FIXED
	#define TX_PIPE													(HAL_NRF_PIPE1)
#endif
#define RX_PIPE														(HAL_NRF_PIPE3)

void esb_txrx_init(void);

/*for PTX*/
void esb_send_data(hal_nrf_address_t tx_pipe_number, 
	uint8_t *p_data, uint8_t len);

uint8_t  is_esb_ack_reached(void);

void esb_fetch_ack_data(uint8_t *p_data, uint8_t *p_len);

uint8_t is_esb_max_retry_count_reached(void);

void esb_max_retry_count_reached_has_been_handled(void);


/*for PRX*/
void esb_fetch_received_data(hal_nrf_address_t *p_pipe, 
	uint8_t *p_data, uint8_t *p_len);

uint8_t is_esb_received_data(void);

void esb_send_ack_data(uint8_t *p_data, uint8_t len);

void esb_receiving_event_has_been_handled(void);

#endif