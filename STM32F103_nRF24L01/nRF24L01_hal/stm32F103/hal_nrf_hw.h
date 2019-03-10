

/** @file
 * @brief Macros and hardware includes for STM8L101XX
 *
 * @{
 * @name Hardware dependencies
 * @{
 *
 */

#ifndef HAL_NRF_STM8L101XX_H__
#define HAL_NRF_STM8L101XX_H__
#include "stm32f10x_conf.h"

#define data	

#ifdef SET_BIT
	#undef SET_BIT
#endif



/** Macro that set radio's CSN line LOW.
 *
 */
#define CSN_LOW() do {  GPIO_ResetBits( GPIOC, GPIO_Pin_4);} while(false)

/** Macro that set radio's CSN line HIGH.
 *
 */
#define CSN_HIGH() do { GPIO_SetBits( GPIOC, GPIO_Pin_4);} while(false)

/** Macro that set radio's CE line LOW.
 *
 */
#define CE_LOW() do { GPIO_ResetBits(GPIOA, GPIO_Pin_4);} while(false)

/** Macro that set radio's CE line HIGH.
 *
 */
#define CE_HIGH() do { GPIO_SetBits(GPIOA, GPIO_Pin_4);} while(false)
/** Macro for writing the radio SPI data register.
 *
 */
#define HAL_NRF_HW_SPI_WRITE(d) do{ \
																	while (RESET == SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) ); \
																	SPI_I2S_SendData(SPI1, d); \
																} while(false)

/** Macro for reading the radio SPI data register.
 *
 */
#define HAL_NRF_HW_SPI_READ() 	SPI_I2S_ReceiveData(SPI1)
  
/** Macro specifyng the radio SPI busy flag.
 *
 */
#define HAL_NRF_HW_SPI_BUSY 		RESET == SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE)

/**
 * Pulses the CE to nRF24L01 for at least 10 us
 */

#define CE_PULSE() do { \
  uint8_t count; \
  count = 20; \
  CE_HIGH();  \
  while(count--){} \
  CE_LOW();  \
  } while(false)

	
#endif // HAL_NRF_STM8L101XX_H__

/** @} */
