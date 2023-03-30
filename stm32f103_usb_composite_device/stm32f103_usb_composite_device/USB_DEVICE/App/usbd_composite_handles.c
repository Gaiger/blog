/*
 * usbd_composite_class_handles.c
 *
 *  Created on: 2022年12月8日
 *      Author: gaiger.chen
 */

#include "usbd_composite_handles.h"

static USBD_CUSTOM_HID_HandleTypeDef s_p_custom_hid_handle;
static USBD_CDC_HandleTypeDef s_cdc_handle;
static USBD_MSC_BOT_HandleTypeDef s_msc_bot_handle;

/*******************************************************************************/

USBD_CUSTOM_HID_HandleTypeDef* GetCustomHIDHandlePtr(void){ return &s_p_custom_hid_handle;}
USBD_CDC_HandleTypeDef* GetCDCHandlePtr(void){ return &s_cdc_handle;}
USBD_MSC_BOT_HandleTypeDef* GetMSCBOTHandlePtr(void){ return &s_msc_bot_handle;}
/*******************************************************************************/

void SwitchHandleInterfaceToCustomHID(USBD_HandleTypeDef *pdev)
{
	pdev->pUserData = &USBD_CustomHID_fops_FS;
	pdev->pClassData = &s_p_custom_hid_handle;
}

/*******************************************************************************/

void SwitchHandleInterfaceToCDC(USBD_HandleTypeDef *pdev)
{
	pdev->pUserData = &USBD_CDC_Interface_fops_FS;
	pdev->pClassData = &s_cdc_handle;
}

/*******************************************************************************/

void SwitchHandleInterfaceToMSC(USBD_HandleTypeDef *pdev)
{
	pdev->pUserData = &USBD_Storage_Interface_fops_FS;
	pdev->pClassData = &s_msc_bot_handle;
}

