
#ifndef _ESB_APP_ACK_H_
#define _ESB_APP_ACK_H_

#include <stdint.h>

#include "hal_nrf.h"
#include "hal_nrf_hw.h"

#include "trscvr_para.h"


//#define _RX_FOR_ALL_CHANNEL


#if !defined(_RX_FOR_ALL_CHANNEL)
	#define RX_PIPE																	(HAL_NRF_PIPE3)
#endif


void esb_prx_init(void);

uint8_t is_esb_received_data(void);
void esb_fetch_received_data(hal_nrf_address_t *p_pipe, 
	uint8_t *p_data, uint8_t *p_len);

void esb_send_ack_data(uint8_t *p_data, uint8_t len);
void esb_receiving_event_has_been_handled(void);


#endif