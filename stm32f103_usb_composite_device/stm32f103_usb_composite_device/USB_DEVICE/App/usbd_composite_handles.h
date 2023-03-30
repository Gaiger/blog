#ifndef _USBD_COMPOSITE_HANDLES_H_
#define _USBD_COMPOSITE_HANDLES_H_

#include "usbd_customhid.h"
#include "usbd_cdc.h"
#include "usbd_custom_hid_if.h"
#include "usbd_cdc_if.h"
#include "usbd_msc.h"
#include "usbd_storage_if.h"

USBD_CUSTOM_HID_HandleTypeDef* GetCustomHIDHandlePtr(void);
USBD_CDC_HandleTypeDef* GetCDCHandlePtr(void);
USBD_MSC_BOT_HandleTypeDef* GetMSCBOTHandlePtr(void);

void SwitchHandleInterfaceToCustomHID(USBD_HandleTypeDef *pdev);
void SwitchHandleInterfaceToCDC(USBD_HandleTypeDef *pdev);
void SwitchHandleInterfaceToMSC(USBD_HandleTypeDef *pdev);

#endif /* _USBD_COMPOSITE_HANDLES_H_ */
