#ifndef _STM32_PRINTF_H_
#define _STM32_PRINTF_H_
#include <stdbool.h>


//#define _USB_CDC_AS_PRINTF

#ifdef _USB_CDC_AS_PRINTF
#include "usbd_cdc_if.h"
#endif


int ch340_printf(bool is_show_comport_name, const char *format, ...);


#ifdef _USB_CDC_AS_PRINTF

int usbd_printf(bool is_show_comport_name, const char *format, ...);

#define PRINTF(FMT, ...)							do{	\
														ch340_printf(true, FMT, ##__VA_ARGS__); \
														usbd_printf(true, FMT, ##__VA_ARGS__); \
													}while(0)

#define PRINTF_NO_COMNAME(FMT, ...)					do{	\
														ch340_printf(false, FMT, ##__VA_ARGS__); \
														usbd_printf(false, FMT, ##__VA_ARGS__); \
													}while(0)

#else

#define PRINTF(FMT, ...)							do{	\
														ch340_printf(true, FMT, ##__VA_ARGS__); \
													}while(0)

#define PRINTF_NO_COMNAME(FMT, ...)					do{	\
														ch340_printf(false, FMT, ##__VA_ARGS__); \
													}while(0)
#endif //_USB_CDC_AS_PRINTF

#endif /* _STM32_PRINTF_H_ */
