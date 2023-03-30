#include "usbd_customhid.h"
#include "usbd_cdc.h"
#include "usbd_msc.h"

#include "usbd_composite_handles.h"
#include "usbd_customhid_cdc_msc_composite.h"

#define WBVAL(x)							(x & 0xFF), ((x >> 8) & 0xFF)
#define DBVAL(x)							(x & 0xFF), ((x >> 8) & 0xFF), ((x >> 16) & 0xFF), ((x >> 24) & 0xFF)

#define USBD_IAD_DESC_SIZE					0x08
#define USBD_IAD_DESCRIPTOR_TYPE			0x0B


#ifdef USBD_MAX_NUM_INTERFACES
	#undef USBD_MAX_NUM_INTERFACES
#endif
#define USBD_MAX_NUM_INTERFACES				4 /*1 for Custom HID, 2 for CDC, 1 for MSC */

#define USBD_CUSTOM_HID_INTERFACE_NUM		1	/* CUSTOM_HID Interface NUM */
#define USBD_CUSTOM_HID_INTERFACE 			0

#define USBD_CDC_INTERFACE_NUM				2	/* CDC Interface NUM */
#define USBD_CDC_FIRST_INTERFACE			1	/* CDC FirstInterface */
#define USBD_CDC_CMD_INTERFACE				1
#define USBD_CDC_DATA_INTERFACE				2

#define USBD_MSC_INTERFACE_NUM				1	/* MSC Interface NUM */
#define USBD_MSC_FIRST_INTERFACE			3
#define USBD_MSC_INTERFACE					3


#define CUSTOM_HID_INDATA_NUM				(CUSTOM_HID_EPIN_ADDR & 0x0F)
#define CUSTOM_HID_OUTDATA_NUM				(CUSTOM_HID_EPOUT_ADDR & 0x0F)

#define CDC_INDATA_NUM						(CDC_IN_EP & 0x0F)
#define CDC_OUTDATA_NUM						(CDC_OUT_EP & 0x0F)
#define CDC_OUTCMD_NUM						(CDC_CMD_EP & 0x0F)

#define MSC_INDATA_NUM						(MSC_EPIN_ADDR & 0x0F)
#define MSC_OUTDATA_NUM						(MSC_EPOUT_ADDR & 0x0F)


#define USBD_CUSTOMHID_CDC_MSC_COMPOSITE_DESC_SIZE    (9 \
											+ (USBD_IAD_DESC_SIZE + 9 + (5 + 5 + 4 + 5) + 7 + 9 + 7 + 7) \
											+ (9 + 9 + 7 + 7) \
											+ (9 + 7 + 7) \
											)


static uint8_t USBD_CustomHID_CDC_MSC_Composite_Init(USBD_HandleTypeDef *pdev,
		uint8_t cfgidx);

static uint8_t USBD_CustomHID_CDC_MSC_Composite_DeInit(USBD_HandleTypeDef *pdev,
		uint8_t cfgidx);

static uint8_t USBD_CustomHID_CDC_MSC_Composite_EP0_RxReady(USBD_HandleTypeDef *pdev);

static uint8_t USBD_CustomHID_CDC_MSC_Composite_Setup (USBD_HandleTypeDef *pdev,
		USBD_SetupReqTypedef *req);

static uint8_t USBD_CustomHID_CDC_MSC_Composite_DataIn(USBD_HandleTypeDef *pdev,
		uint8_t epnum);

static uint8_t USBD_CustomHID_CDC_MSC_Composite_DataOut(USBD_HandleTypeDef *pdev,
		uint8_t epnum);

static uint8_t *USBD_CustomHID_CDC_MSC_Composite_GetFSCfgDesc(uint16_t *length);

static uint8_t *USBD_CustomHID_CDC_MSC_Composite_GetDeviceQualifierDescriptor(uint16_t *length);


USBD_ClassTypeDef  USBD_CustomHID_CDC_MSC_COMPOSITE =
{
	USBD_CustomHID_CDC_MSC_Composite_Init,
	USBD_CustomHID_CDC_MSC_Composite_DeInit,
	USBD_CustomHID_CDC_MSC_Composite_Setup,
	NULL, /*EP0_TxSent*/
	USBD_CustomHID_CDC_MSC_Composite_EP0_RxReady,
	USBD_CustomHID_CDC_MSC_Composite_DataIn,
	USBD_CustomHID_CDC_MSC_Composite_DataOut,
	NULL,
	NULL,
	NULL,
	NULL,
	USBD_CustomHID_CDC_MSC_Composite_GetFSCfgDesc,
	NULL,
	USBD_CustomHID_CDC_MSC_Composite_GetDeviceQualifierDescriptor,
};


__ALIGN_BEGIN uint8_t USBD_CustomHID_CDC_MSC_Composite_CfgFSDesc[USBD_CUSTOMHID_CDC_MSC_COMPOSITE_DESC_SIZE] __ALIGN_END =
{
	0x09,   /* bLength: Configuation Descriptor size */
	USB_DESC_TYPE_CONFIGURATION,   /* bDescriptorType: Configuration */
	WBVAL(USBD_CUSTOMHID_CDC_MSC_COMPOSITE_DESC_SIZE),
	USBD_MAX_NUM_INTERFACES ,  /* bNumInterfaces: */
	0x01,   /* bConfigurationValue: */
	0x00,   /* iConfiguration: */
	0xC0,   /* bmAttributes: self powered */
	0x00,   /* MaxPower 0 mA */


	/***************************CUSTOM HID********************************/
	/*Interface Descriptor */
	0x09,         /*bLength: Interface Descriptor size*/
	USB_DESC_TYPE_INTERFACE,/*bDescriptorType: Interface descriptor type*/
	USBD_CUSTOM_HID_INTERFACE,         /*bInterfaceNumber: Number of Interface*/
	0x00,         /*bAlternateSetting: Alternate setting*/
	0x02,         /*bNumEndpoints*/
	0x03,         /*bInterfaceClass: CUSTOM_HID*/
	0x00,         /*bInterfaceSubClass : 1=BOOT, 0=no boot*/
	0x00,         /*nInterfaceProtocol : 0=none, 1=keyboard, 2=mouse*/
	0x00,  /*iInterface: Index of string descriptor*/

	/******************** Descriptor of CUSTOM_HID *************************/
	0x09,         /*bLength: CUSTOM_HID Descriptor size*/
	CUSTOM_HID_DESCRIPTOR_TYPE, /*bDescriptorType: CUSTOM_HID*/
	0x11,         /*bCUSTOM_HIDUSTOM_HID: CUSTOM_HID Class Spec release number*/
	0x01,
	0x00,         /*bCountryCode: Hardware target country*/
	0x01,         /*bNumDescriptors: Number of CUSTOM_HID class descriptors to follow*/
	0x22,         /*bDescriptorType*/
	USBD_CUSTOM_HID_REPORT_DESC_SIZE,/*wItemLength: Total length of Report descriptor*/
	0x00,

	/******************** Descriptor of Custom HID endpoints ********************/
	/*Endpoint IN Descriptor*/
	0x07,          /*bLength: Endpoint Descriptor size*/
	USB_DESC_TYPE_ENDPOINT, /*bDescriptorType:*/
	CUSTOM_HID_EPIN_ADDR,     /*bEndpointAddress: Endpoint Address (IN)*/
	0x03,          /*bmAttributes: Interrupt endpoint*/
	CUSTOM_HID_EPIN_SIZE, /*wMaxPacketSize: 2 Byte max */
	0x00,
	CUSTOM_HID_FS_BINTERVAL,          /*bInterval: Polling Interval */

	/*Endpoint OUT Descriptor*/
	0x07,          /* bLength: Endpoint Descriptor size */
	USB_DESC_TYPE_ENDPOINT, /* bDescriptorType: */
	CUSTOM_HID_EPOUT_ADDR,  /*bEndpointAddress: Endpoint Address (OUT)*/
	0x03, /* bmAttributes: Interrupt endpoint */
	CUSTOM_HID_EPOUT_SIZE,  /* wMaxPacketSize: 2 Bytes max  */
	0x00,
	CUSTOM_HID_FS_BINTERVAL,  /* bInterval: Polling Interval */


	/****************************CDC************************************/
	/* Interface Association Descriptor */
	USBD_IAD_DESC_SIZE,               // bLength
	USBD_IAD_DESCRIPTOR_TYPE,         // bDescriptorType
	USBD_CDC_FIRST_INTERFACE,         // bFirstInterface
	USBD_CDC_INTERFACE_NUM,           // bInterfaceCount
	0x02,                             // bFunctionClass
	0x02,                             // bFunctionSubClass
	0x01,                             // bInterfaceProtocol
	0x00,                             // iFunction

	/*Interface Descriptor */
	0x09,   /* bLength: Interface Descriptor size */
	USB_DESC_TYPE_INTERFACE,  /* bDescriptorType: Interface */
	/* Interface descriptor type */
	USBD_CDC_CMD_INTERFACE,   /* bInterfaceNumber: Number of Interface */
	0x00,   /* bAlternateSetting: Alternate setting */
	0x01,   /* bNumEndpoints: One endpoints used */
	0x02,   /* bInterfaceClass: Communication Interface Class */
	0x02,   /* bInterfaceSubClass: Abstract Control Model */
	0x01,   /* bInterfaceProtocol: Common AT commands */
	0x00,   /* iInterface: */

	/*Header Functional Descriptor*/
	0x05,   /* bLength: Endpoint Descriptor size */
	0x24,   /* bDescriptorType: CS_INTERFACE */
	0x00,   /* bDescriptorSubtype: Header Func Desc */
	0x10,   /* bcdCDC: spec release number */
	0x01,

	/*Call Management Functional Descriptor*/
	0x05,   /* bFunctionLength */
	0x24,   /* bDescriptorType: CS_INTERFACE */
	0x01,   /* bDescriptorSubtype: Call Management Func Desc */
	0x00,   /* bmCapabilities: D0+D1 */
	0x01,   /* bDataInterface: 1 */

	/*ACM Functional Descriptor*/
	0x04,   /* bFunctionLength */
	0x24,   /* bDescriptorType: CS_INTERFACE */
	0x02,   /* bDescriptorSubtype: Abstract Control Management desc */
	0x02,   /* bmCapabilities */

	/*Union Functional Descriptor*/
	0x05,   /* bFunctionLength */
	0x24,   /* bDescriptorType: CS_INTERFACE */
	0x06,   /* bDescriptorSubtype: Union func desc */
	USBD_CDC_CMD_INTERFACE,   /* bMasterInterface: Communication class interface */
	USBD_CDC_DATA_INTERFACE,   /* bSlaveInterface0: Data Class Interface */

	/*Endpoint CMD Descriptor*/
	0x07,                           /* bLength: Endpoint Descriptor size */
	USB_DESC_TYPE_ENDPOINT,   /* bDescriptorType: Endpoint */
	CDC_CMD_EP,                     /* bEndpointAddress */
	0x03,                           /* bmAttributes: Interrupt */
	LOBYTE(CDC_CMD_PACKET_SIZE),     /* wMaxPacketSize: */
	HIBYTE(CDC_CMD_PACKET_SIZE),
	0x01,                           /* bInterval: */

	/*Data class interface descriptor*/
	0x09,   /* bLength: Endpoint Descriptor size */
	USB_DESC_TYPE_INTERFACE,  /* bDescriptorType: */
	USBD_CDC_DATA_INTERFACE,   /* bInterfaceNumber: Number of Interface */
	0x00,   /* bAlternateSetting: Alternate setting */
	0x02,   /* bNumEndpoints: Two endpoints used */
	0x0A,   /* bInterfaceClass: CDC */
	0x02,   /* bInterfaceSubClass: */
	0x00,   /* bInterfaceProtocol: */
	0x00,   /* iInterface: */

	/*Endpoint OUT Descriptor*/
	0x07,   /* bLength: Endpoint Descriptor size */
	USB_DESC_TYPE_ENDPOINT,      /* bDescriptorType: Endpoint */
	CDC_OUT_EP,                        /* bEndpointAddress */
	0x02,                              /* bmAttributes: Bulk */
	LOBYTE(CDC_DATA_FS_MAX_PACKET_SIZE),  /* wMaxPacketSize: */
	HIBYTE(CDC_DATA_FS_MAX_PACKET_SIZE),
	0x01,                              /* bInterval: ignore for Bulk transfer */

	/*Endpoint IN Descriptor*/
	0x07,   /* bLength: Endpoint Descriptor size */
	USB_DESC_TYPE_ENDPOINT,      /* bDescriptorType: Endpoint */
	CDC_IN_EP,                         /* bEndpointAddress */
	0x02,                              /* bmAttributes: Bulk */
	LOBYTE(CDC_DATA_FS_MAX_PACKET_SIZE),  /* wMaxPacketSize: */
	HIBYTE(CDC_DATA_FS_MAX_PACKET_SIZE),
	0x01,                               /* bInterval: ignore for Bulk transfer */


	/********************  Mass Storage interface ********************/
	0x09,   /* bLength: Interface Descriptor size */
	USB_DESC_TYPE_INTERFACE,   /* bDescriptorType: */
	USBD_MSC_INTERFACE,   /* bInterfaceNumber: Number of Interface */
	0x00,   /* bAlternateSetting: Alternate setting */
	0x02,   /* bNumEndpoints*/
	0x08,   /* bInterfaceClass: MSC Class */
	0x06,   /* bInterfaceSubClass : SCSI transparent*/
	0x50,   /* nInterfaceProtocol */
	0x05,          /* iInterface: */

	/********************  Mass Storage Endpoints ********************/
	0x07,   /*Endpoint descriptor length = 7*/
	0x05,   /*Endpoint descriptor type */
	MSC_EPIN_ADDR,   /*Endpoint address (IN, address 1) */
	0x02,   /*Bulk endpoint type */
	LOBYTE(MSC_MAX_FS_PACKET),
	HIBYTE(MSC_MAX_FS_PACKET),
	0x01,   /*Polling interval in milliseconds */

	0x07,   /*Endpoint descriptor length = 7 */
	0x05,   /*Endpoint descriptor type */
	MSC_EPOUT_ADDR,   /*Endpoint address (OUT, address 1) */
	0x02,   /*Bulk endpoint type */
	LOBYTE(MSC_MAX_FS_PACKET),
	HIBYTE(MSC_MAX_FS_PACKET),
	0x01,     /*Polling interval in milliseconds*/
};


/* USB Standard Device Descriptor */
__ALIGN_BEGIN  uint8_t USBD_CustomHID_CDC_MSC_Composite_DeviceQualifierDesc[USB_LEN_DEV_QUALIFIER_DESC]  __ALIGN_END =
{
	USB_LEN_DEV_QUALIFIER_DESC,
	USB_DESC_TYPE_DEVICE_QUALIFIER,
	0x00,
	0x02,
	0x00,
	0x00,
	0x00,
	CDC_DATA_FS_MAX_PACKET_SIZE,
	0x01,
	0x00,
};


/*******************************************************************************/

static uint8_t USBD_CustomHID_CDC_MSC_Composite_Init (USBD_HandleTypeDef *pdev,
		uint8_t cfgidx)
{
	uint8_t res = 0;

	pdev->pUserData = &USBD_CustomHID_fops_FS;
	res += USBD_CUSTOM_HID.Init(pdev, cfgidx);

	pdev->pUserData = &USBD_CDC_Interface_fops_FS;
	res += USBD_CDC.Init(pdev, cfgidx);

	pdev->pUserData = &USBD_Storage_Interface_fops_FS;
	res +=  USBD_MSC.Init(pdev,cfgidx);

	return res;
}

/*******************************************************************************/

static uint8_t USBD_CustomHID_CDC_MSC_Composite_DeInit (USBD_HandleTypeDef *pdev,
		uint8_t cfgidx)
{
	uint8_t res = 0;

	SwitchHandleInterfaceToCustomHID(pdev);
	res += USBD_CUSTOM_HID.DeInit(pdev, cfgidx);

	SwitchHandleInterfaceToCDC(pdev);
	res += USBD_CDC.DeInit(pdev, cfgidx);

	SwitchHandleInterfaceToMSC(pdev);
	res +=  USBD_MSC.DeInit(pdev,cfgidx);

	return res;
}

/*******************************************************************************/

static uint8_t  USBD_CustomHID_CDC_MSC_Composite_Setup (USBD_HandleTypeDef *pdev,
		USBD_SetupReqTypedef *req)
{
	switch (req->bmRequest & USB_REQ_RECIPIENT_MASK)
	{
	case USB_REQ_RECIPIENT_INTERFACE:
		switch(req->wIndex)
		{
		case USBD_CUSTOM_HID_INTERFACE:
			SwitchHandleInterfaceToCustomHID(pdev);
			return(USBD_CUSTOM_HID.Setup(pdev, req));

		case USBD_CDC_DATA_INTERFACE:
		case USBD_CDC_CMD_INTERFACE:
			SwitchHandleInterfaceToCDC(pdev);
			return(USBD_CDC.Setup(pdev, req));

		case USBD_MSC_INTERFACE:
			SwitchHandleInterfaceToMSC(pdev);
			return(USBD_MSC.Setup (pdev, req));

		 default:
			break;
		}
		break;

	case USB_REQ_RECIPIENT_ENDPOINT:
		switch(req->wIndex)
		{
		case CUSTOM_HID_EPIN_ADDR:
		case CUSTOM_HID_EPOUT_ADDR:
			SwitchHandleInterfaceToCustomHID(pdev);
			return(USBD_CUSTOM_HID.Setup(pdev, req));

		case CDC_IN_EP:
		case CDC_OUT_EP:
		case CDC_CMD_EP:
			SwitchHandleInterfaceToCDC(pdev);
			return(USBD_CDC.Setup(pdev, req));

		case MSC_EPIN_ADDR:
		case MSC_EPOUT_ADDR:
			SwitchHandleInterfaceToMSC(pdev);
			return(USBD_MSC.Setup (pdev, req));

		default:
			break;
		}
		break;

	default:
		break;
	}
	return USBD_OK;
}

/*******************************************************************************/

static uint8_t USBD_CustomHID_CDC_MSC_Composite_EP0_RxReady(USBD_HandleTypeDef *pdev)
{
	SwitchHandleInterfaceToCustomHID(pdev);
	USBD_CUSTOM_HID.EP0_RxReady(pdev);

	SwitchHandleInterfaceToCDC(pdev);
	return USBD_CDC.EP0_RxReady(pdev);
}

/*******************************************************************************/

static uint8_t USBD_CustomHID_CDC_MSC_Composite_DataIn(USBD_HandleTypeDef *pdev,
		uint8_t epnum)
{
	switch(epnum)
	{
	case CUSTOM_HID_INDATA_NUM:
		SwitchHandleInterfaceToCustomHID(pdev);
		return(USBD_CUSTOM_HID.DataIn(pdev, epnum));

	case CDC_INDATA_NUM:
		SwitchHandleInterfaceToCDC(pdev);
		return(USBD_CDC.DataIn(pdev, epnum));

	case MSC_INDATA_NUM:
		SwitchHandleInterfaceToMSC(pdev);
		return(USBD_MSC.DataIn(pdev, epnum));

	default:
		break;
	}
	return USBD_FAIL;
}

/*******************************************************************************/

static uint8_t USBD_CustomHID_CDC_MSC_Composite_DataOut(USBD_HandleTypeDef *pdev,
		uint8_t epnum)
{
	switch(epnum)
	{
	case CUSTOM_HID_OUTDATA_NUM:
		SwitchHandleInterfaceToCustomHID(pdev);
		return(USBD_CUSTOM_HID.DataOut(pdev, epnum));

	case CDC_OUTDATA_NUM:
	case CDC_OUTCMD_NUM:
		SwitchHandleInterfaceToCDC(pdev);
		return(USBD_CDC.DataOut(pdev, epnum));

	case MSC_OUTDATA_NUM:
		SwitchHandleInterfaceToMSC(pdev);
		return(USBD_MSC.DataOut(pdev, epnum));

	default:
		break;
	}
	return USBD_FAIL;
}

/*******************************************************************************/

uint8_t	*USBD_CustomHID_CDC_MSC_Composite_GetFSCfgDesc(uint16_t *length)
{
	*length = sizeof(USBD_CustomHID_CDC_MSC_Composite_CfgFSDesc);
	return USBD_CustomHID_CDC_MSC_Composite_CfgFSDesc;
}

/*******************************************************************************/

uint8_t *USBD_CustomHID_CDC_MSC_Composite_GetDeviceQualifierDescriptor(uint16_t *length)
{
	*length = sizeof(USBD_CustomHID_CDC_MSC_Composite_DeviceQualifierDesc);
	return USBD_CustomHID_CDC_MSC_Composite_DeviceQualifierDesc;
}


