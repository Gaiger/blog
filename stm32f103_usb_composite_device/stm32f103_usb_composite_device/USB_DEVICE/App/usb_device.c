/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : usb_device.c
  * @version        : v2.0_Cube
  * @brief          : This file implements the USB Device
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/

#include "usb_device.h"
#include "usbd_core.h"
#include "usbd_desc.h"
#include "usbd_customhid.h"
#include "usbd_custom_hid_if.h"
#include "usbd_customhid_cdc_msc_composite.h"
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/

/* USER CODE END PV */

/* USER CODE BEGIN PFP */
/* Private function prototypes -----------------------------------------------*/

/* USER CODE END PFP */

/* USB Device Core handle declaration. */
USBD_HandleTypeDef hUsbDeviceFS = {0};

/*
 * -- Insert your variables declaration here --
 */
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/*
 * -- Insert your external function declaration here --
 */
/* USER CODE BEGIN 1 */
static void USB_DEVICE_Renumerate(void)
{
	GPIO_InitTypeDef GPIO_InitStruct = { 0 };              // All zeroed out
	GPIO_InitStruct.Pin = GPIO_PIN_12;                     // Hardcoding this - PA12 is D+
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;            // Push-pull mode
	GPIO_InitStruct.Pull = GPIO_PULLDOWN;                  // Resetting so pull low
	GPIO_InitStruct.Speed = GPIO_SPEED_HIGH;               // Really shouldn't matter in this case
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);                // Initialize with above settings
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12, GPIO_PIN_RESET); // Yank low
	HAL_Delay(50);                                         // Enough time for host to disconnect device
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12, GPIO_PIN_SET);   // Back high - so host will enumerate
	HAL_GPIO_DeInit(GPIOA, GPIO_PIN_12);                   // Deinitialize the pin
}

/*******************************************************************************/

void USB_DEVICE_CustomHID_CDC_MSC_Composite_Init(void)
{
	if(0x00 != hUsbDeviceFS.pDesc)
		USBD_DeInit(&hUsbDeviceFS);
	USB_DEVICE_Renumerate();

	if(USBD_Init(&hUsbDeviceFS, &FS_Desc, DEVICE_FS)!= USBD_OK)
		Error_Handler();
	if(USBD_RegisterClass(&hUsbDeviceFS, &USBD_CustomHID_CDC_MSC_COMPOSITE) != USBD_OK)
		Error_Handler();
	if (USBD_Start(&hUsbDeviceFS) != USBD_OK)
	    Error_Handler();
}

/*******************************************************************************/

/* USER CODE END 1 */

/**
  * Init USB device Library, add supported class and start the library
  * @retval None
  */
void MX_USB_DEVICE_Init(void)
{
  /* USER CODE BEGIN USB_DEVICE_Init_PreTreatment */
  /* USER CODE END USB_DEVICE_Init_PreTreatment */

  /* USER CODE BEGIN USB_DEVICE_Init_PostTreatment */
  USB_DEVICE_CustomHID_CDC_MSC_Composite_Init();
  /* USER CODE END USB_DEVICE_Init_PostTreatment */
}

/**
  * @}
  */

/**
  * @}
  */

