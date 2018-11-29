
#ifndef _ESP_APP_TXRX_NOACK_H_
#define _ESP_APP_TXRX_NOACK_H_

#include <stdint.h>

#include "hal_nrf.h"
#include "hal_nrf_hw.h"
#include "trscvr_para.h"
//#define _RELAYER

#define TX_PIPE																	(HAL_NRF_PIPE3)
#define RX_PIPE																	(HAL_NRF_PIPE5)



/* the functions declaration
*/

void esb_txrx_init(void);

void esb_send_data(hal_nrf_address_t tx_pipe_number, uint8_t *p_data, uint8_t len);

void esb_fetch_received_data(hal_nrf_address_t *p_pipe, uint8_t *p_data, uint8_t *p_len);

uint8_t is_esb_received_data(void);

uint8_t is_esb_max_retry_count_reached(void);

void esb_max_retry_count_reached_has_been_handled(void);

void esb_receiving_event_has_been_done(void);


#endif /*_ESP_APP_NOACK_H_*/