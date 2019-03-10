
#ifndef _ESP_APP_PTX_NOACK_H_
#define _ESP_APP_PTX_NOACK_H_

#include <stdint.h>

#include "hal_nrf.h"
#include "hal_nrf_hw.h"

#include "trscvr_para.h"




//#define _TX_FOR_ALL_CHANNEL		


#if !defined(_TX_FOR_ALL_CHANNEL)
	#define TX_PIPE																(HAL_NRF_PIPE5)
#endif



/* the functions declaration
*/

void esb_ptx_init(void);

void esb_send_data(hal_nrf_address_t tx_pipe_number, uint8_t *p_data, uint8_t len);



#endif /*_ESP_APP_PTX_NOACK_H_*/