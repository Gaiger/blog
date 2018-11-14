
#ifndef _ESP_APP_RX_NOACK_H_
#define _ESP_APP_RX_NOACK_H_

#include <stdint.h>

#include "hal_nrf.h"
#include "hal_nrf_hw.h"
#include "trscvr_para.h"


//#define _RX_FOR_ALL_CHANNEL

#ifndef _RX_FOR_ALL_CHANNEL
	#define RX_PIPE																(HAL_NRF_PIPE5)
#endif

/* the functions declaration
*/

void esb_prx_init(void);

uint8_t is_esb_received_data(void);
void esb_fetch_received_data(hal_nrf_address_t *p_pipe, uint8_t *p_data, uint8_t *p_len);
void esb_receiving_event_has_been_done(void);


#endif /*_ESP_APP_RX_NOACK_H_*/