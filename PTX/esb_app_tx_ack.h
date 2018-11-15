
#ifndef _ESB_APP_TX_ACK_H_
#define _ESB_APP_TX_ACK_H_

#include <stdint.h>

#include "hal_nrf.h"
#include "hal_nrf_hw.h"

#include "trscvr_para.h"


/*enabling this macro leads PTX hanging*/

//#define _TX_FOR_ALL_CHANNEL

#if !defined(_TX_FOR_ALL_CHANNEL) 
	#define TX_PIPE																	(HAL_NRF_PIPE3)
#endif


void esb_ptx_init(void);

/*for PTX*/
void esb_send_data(hal_nrf_address_t tx_pipe_number, uint8_t *p_data, uint8_t len);

uint8_t  is_esb_ack_reached(void);
void esb_fetch_ack_data(uint8_t *p_data, uint8_t *p_len);

uint8_t is_esb_max_retry_count_reached(void);
void esb_max_retry_count_reached_has_been_handled(void);



#endif