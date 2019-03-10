/* Copyright (c) 2009 Nordic Semiconductor. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic 
 * Semiconductor ASA.Terms and conditions of usage are described in detail 
 * in NORDIC SEMICONDUCTOR STANDARD SOFTWARE LICENSE AGREEMENT. 
 *
 * Licensees are granted free, non-transferable use of the information. NO
 * WARRENTY of ANY KIND is provided. This heading must NOT be removed from
 * the file.
 *              
 */



#include <stdint.h>
#include "hal_nrf.h"
#include "STM8l10x_conf.h"

uint8_t hal_nrf_rw(uint8_t value)
{
	while (RESET == SPI_GetFlagStatus(SPI_FLAG_TXE));
	SPI_SendData(value);
	
	//wait for receiving a byte
	while(RESET == SPI_GetFlagStatus(SPI_FLAG_RXNE));
	return SPI_ReceiveData();
}


