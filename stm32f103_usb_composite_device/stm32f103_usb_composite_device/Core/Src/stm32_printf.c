/*
 * usbdevice_print.c
 *
 *  Created on: Nov 21, 2022
 *      Author: gaiger.chen
 */


#include "stm32f1xx_hal.h"
#include <stdarg.h>
#include <stdio.h>

#include "stm32_printf.h"

extern UART_HandleTypeDef huart1;

#ifdef __GNUC__
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif


PUTCHAR_PROTOTYPE
{
	HAL_UART_Transmit(&huart1, (uint8_t*) &ch, 1, HAL_MAX_DELAY);
	return ch;
}

/**********************************************************************/

int ch340_printf(bool is_show_comport_name, const char *format, ...)
{
	int len = 0;
	if(true == is_show_comport_name)
		len += printf("%s","CH340:: ");

	va_list args;
	va_start(args, format);
	len += vprintf((char *)format, args);
	va_end(args);

	return len;
}

/**********************************************************************/

#ifdef _USB_CDC_AS_PRINTF
#define USBD_TX_DATA_BUFFER_SIZE						(128)
uint8_t usbd_tx_data_buffer[CDC_DATA_FS_MAX_PACKET_SIZE];

int usbd_printf(bool is_show_comport_name, const char *format, ...)
{
	int comport_name_len = 0;
	if(true == is_show_comport_name)
		comport_name_len = snprintf((char *)&usbd_tx_data_buffer[0], USBD_TX_DATA_BUFFER_SIZE, "%s", "USBD:: ");

    va_list args;
    uint32_t length;

    va_start(args, format);
    length = vsnprintf((char *)&usbd_tx_data_buffer[comport_name_len],
    		USBD_TX_DATA_BUFFER_SIZE - comport_name_len, (char *)format, args);
    va_end(args);
    CDC_Transmit_FS(&usbd_tx_data_buffer[0], comport_name_len + length);

    return comport_name_len + length;
}

#endif //__USBD_CDC_IF_H__
