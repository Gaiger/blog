//-----------------------------------------------------------------------------
// F3xx_USB0_Descriptor.c
//-----------------------------------------------------------------------------
// Copyright 2010 Silicon Laboratories, Inc.
// http://www.silabs.com
//
// Program Description:
//
// Source file for USB firmware. Includes descriptor data.
//
//
// How To Test:    See Readme.txt
//
//
// FID
// Target:         C8051F32x/C8051F340
// Tool chain:     Keil / Raisonance
//                 Silicon Laboratories IDE version 2.6
// Command Line:   See Readme.txt
// Project Name:   F3xx_MouseExample
//
// Release 1.2 (ES)
//    -Added support for Raisonance
//    -Revomed 'const' from descriptor defitions
//    -02 APR 2010
// Release 1.1
//    -Minor code comment changes
//    -16 NOV 2006
// Release 1.0
//    -Initial Revision (PD)
//    -07 DEC 2005
//
//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include "F3xx_USB0_Register.h"
#include "F3xx_USB0_InterruptServiceRoutine.h"
#include "F3xx_USB0_Descriptor.h"
#include "F3xx_USB0_ReportHandler.h"

#define SWAP_UINT16(X) 					\
						((((unsigned short)(X)) >> 8) | (((unsigned short)(X)) << 8))
#define LITTLE_ENDIAN(VALUE)			(SWAP_UINT16(VALUE))

//-----------------------------------------------------------------------------
// Descriptor Declarations
//-----------------------------------------------------------------------------
#define VENDOR_ID						(0x10C4)
#define PRODUCT_ID						(0x81B9)
#define FIRMWARE_REVISION_NUMER			(0x0000)


const device_descriptor code DEVICEDESC =
{
   18,                                 // bLength
   0x01,                               // bDescriptorType
   0x1001,                             // bcdUSB
   0x00,                               // bDeviceClass
   0x00,                               // bDeviceSubClass
   0x00,                               // bDeviceProtocol
   EP0_PACKET_SIZE,                    // bMaxPacketSize0
   LITTLE_ENDIAN(VENDOR_ID),  			// idVendor
   LITTLE_ENDIAN(PRODUCT_ID), 			// idProduct
   LITTLE_ENDIAN(FIRMWARE_REVISION_NUMER), // bcdDevice
   0x01,                               // iManufacturer
   0x02,                               // iProduct
   0x00,                               // iSerialNumber
   0x01                                // bNumConfigurations
}; //end of DEVICEDESC

// From "USB Device Class Definition for Human Interface Devices (HID)".
// Section 7.1:
// "When a Get_Descriptor(Configuration) request is issued,
// it returns the Configuration descriptor, all Interface descriptors,
// all Endpoint descriptors, and the HID descriptor for each interface."
const this_configuration_descriptor_all code CONFIGDESC =
{

{ // configuration_descriptor hid_configuration_descriptor
   0x09,                               // Length
   0x02,                               // Type
   0x4200,                             // Totallength = 9 + (9+9+7+7) + (9+9+7)
   0x02,                               // NumInterfaces
   0x01,                               // bConfigurationValue
   0x00,                               // iConfiguration
   0x80,                               // bmAttributes
   0x20                                // MaxPower (in 2mA units)
},


{ // interface_descriptor hid_interface_descriptor
   0x09,                               // bLength
   0x04,                               // bDescriptorType
   KEYBOARD_REPORT_ID,                 // bInterfaceNumber
   0x00,                               // bAlternateSetting
   0x02,                               // bNumEndpoints
   0x03,                               // bInterfaceClass (3 = HID)
   0x01,                               // bInterfaceSubClass
   0x01,                               // bInterfaceProcotol //keyboard
   0x00                                // iInterface
},

{ // class_descriptor hid_descriptor
   0x09,                               // bLength
   0x21,                               // bDescriptorType
   0x0101,                             // bcdHID
   0x00,                               // bCountryCode
   0x01,                               // bNumDescriptors
   0x22,                               // bDescriptorType
   LITTLE_ENDIAN(sizeof(keyboard_report_descriptor)) // wItemLength (tot. len. of report descriptor)
},

// IN endpoint (mandatory for HID)
{ // endpoint_descriptor hid_endpoint_in_descriptor
   0x07,                               // bLength
   0x05,                               // bDescriptorType
   0x81,                               // bEndpointAddress
   0x03,                               // bmAttributes
   LITTLE_ENDIAN(EP1_IN_PACKET_SIZE),    // MaxPacketSize (LITTLE ENDIAN)
   10                                  // bInterval
},

// OUT endpoint (optional for HID)
{ // endpoint_descriptor hid_endpoint_out_descriptor
   0x07,                               // bLength
   0x05,                               // bDescriptorType
   0x01,                               // bEndpointAddress
   0x03,                               // bmAttributes
   LITTLE_ENDIAN(EP1_OUT_PACKET_SIZE),   // MaxPacketSize (LITTLE ENDIAN)
   10                                  // bInterval
},

{ // interface_descriptor hid_interface_descriptor
   0x09,                               // bLength
   0x04,                               // bDescriptorType
   MOUSE_REPORT_ID,                    // bInterfaceNumber
   0x00,                               // bAlternateSetting
   0x01,                               // bNumEndpoints
   0x03,                               // bInterfaceClass (3 = HID)
   0x01,                               // bInterfaceSubClass
   0x02,                               // bInterfaceProcotol
   0x00                                // iInterface
},

{ // class_descriptor hid_descriptor
   0x09,                               // bLength
   0x21,                               // bDescriptorType
   0x0101,                             // bcdHID
   0x00,                               // bCountryCode
   0x01,                               // bNumDescriptors
   0x22,                               // bDescriptorType
   LITTLE_ENDIAN(sizeof(mouse_report_descriptor))       // wItemLength (tot. len. of report
										// descriptor)
},

// IN endpoint (mandatory for HID)
{ // endpoint_descriptor hid_endpoint_in_descriptor
   0x07,                               // bLength
   0x05,                               // bDescriptorType
   0x82,                               // bEndpointAddress
   0x03,                               // bmAttributes
   LITTLE_ENDIAN(EP2_OUT_PACKET_SIZE),                 // MaxPacketSize (LITTLE ENDIAN)
   10                                  // bInterval
}

};


const keyboard_report_descriptor code KEYBOARDREPORTDESC =
{
	 0x05, 0x01, // USAGE_PAGE (Generic Desktop)
	 0x09, 0x06, // USAGE (Keyboard)
	 0xa1, 0x01, // COLLECTION (Application)
	 0x85, KEYBOARD_REPORT_ID, //Report ID 

	0x05, 0x07, //     USAGE_PAGE (Keyboard/Keypad)
	 0x19, 0xe0, //     USAGE_MINIMUM (Keyboard LeftControl)
	 0x29, 0xe7, //     USAGE_MAXIMUM (Keyboard Right GUI)
	 0x15, 0x00, //     LOGICAL_MINIMUM (0)
	 0x25, 0x01, //     LOGICAL_MAXIMUM (1)
   	0x95, 0x08, //     REPORT_COUNT (8)
	0x75, 0x01, //     REPORT_SIZE (1)
	0x81, 0x02, //     INPUT (Data,Var,Abs)
	0x95, 0x01, //     REPORT_COUNT (1)
	0x75, 0x08, //     REPORT_SIZE (8)
	0x81, 0x03, //     INPUT (Cnst,Var,Abs)

	0x95, 0x06, //   REPORT_COUNT (6)
	 0x75, 0x08, //   REPORT_SIZE (8)
	 0x15, 0x00, //   LOGICAL_MINIMUM (0)	
	0x25, 0xFF, //   LOGICAL_MAXIMUM (255)
	 
	 0x05, 0x07, //   USAGE_PAGE (Keyboard/Keypad)
	 0x19, 0x00, //   USAGE_MINIMUM (Reserved (no event indicated))
	 0x29, 0x65, //   USAGE_MAXIMUM (Keyboard Application)
	 0x81, 0x00, //     INPUT (Data,Ary,Abs)
	 0x25, 0x01, //     LOGICAL_MAXIMUM (1)
	 0x95, 0x05, //   REPORT_COUNT (5)
	 0x75, 0x01, //   REPORT_SIZE (1)
	 
	 
	 0x05, 0x08, //   USAGE_PAGE (LEDs)
	 0x19, 0x01, //   USAGE_MINIMUM (Num Lock)
	 0x29, 0x05, //   USAGE_MAXIMUM (Kana)
	 0x91, 0x02, //   OUTPUT (Data,Var,Abs)
	 0x95, 0x01, //   REPORT_COUNT (1)
	 0x75, 0x03, //   REPORT_SIZE (3)
	 0x91, 0x03, //   OUTPUT (Cnst,Var,Abs)
	 0xc0        // END_COLLECTION
 };

const mouse_report_descriptor code MOUSEREPORTDESC =
{
  0x05, 0x01, // USAGE_PAGE (Generic Desktop)
  0x09, 0x02, // USAGE (Mouse)
  0xa1, 0x01, // COLLECTION (Application)

  0x85, MOUSE_REPORT_ID, //Report ID (2)

  0x09, 0x01, //   USAGE (Pointer)
  0xa1, 0x00, //   COLLECTION (Physical)

  0x05, 0x09, //     USAGE_PAGE (Button)
  0x19, 0x01, //     USAGE_MINIMUM (Button 1)
  0x29, 0x03, //     USAGE_MAXIMUM (Button 3)
  0x15, 0x00, //     LOGICAL_MINIMUM (0)
  0x25, 0x01, //     LOGICAL_MAXIMUM (1)
  0x95, 0x05, //     REPORT_COUNT (5)
  0x75, 0x01, //     REPORT_SIZE (1)
  0x81, 0x02, //     INPUT (Data,Var,Abs)


  0x95, 0x01, //     REPORT_COUNT (1)
  0x75, 0x03, //     REPORT_SIZE (3)
  0x81, 0x03, //     INPUT (Cnst,Var,Abs)

  0x05, 0x01, //     USAGE_PAGE (Generic Desktop)
  0x09, 0x30, //     USAGE (X)
  0x09, 0x31, //     USAGE (Y)
  0x09, 0x38, //     USAGE (Wheel)
  0x15, 0x81, //     LOGICAL_MINIMUM (-127)
  0x25, 0x7f, //     LOGICAL_MAXIMUM (127)
  0x75, 0x08, //     REPORT_SIZE (8)
  0x95, 0x03, //     REPORT_COUNT (3)
  0x81, 0x06, //     INPUT (Data,Var,Rel)

  0xc0,       //   END_COLLECTION

  0xc0        // END_COLLECTION
};


#define STR0LEN 4

code unsigned char String0Desc [STR0LEN] =
{
   STR0LEN, 0x03, 0x09, 0x04
}; //end of String0Desc

#define STR1LEN sizeof ("SILICON LABORATORIES") * 2

code unsigned char String1Desc [STR1LEN] =
{
   STR1LEN, 0x03,
   'S', 0,
   'I', 0,
   'L', 0,
   'I', 0,
   'C', 0,
   'O', 0,
   'N', 0,
   ' ', 0,
   'L', 0,
   'A', 0,
   'B', 0,
   'O', 0,
   'R', 0,
   'A', 0,
   'T', 0,
   'O', 0,
   'R', 0,
   'I', 0,
   'E', 0,
   'S', 0
}; //end of String1Desc

#define STR2LEN sizeof ("C8051F320 Development Board") * 2

code unsigned char String2Desc [STR2LEN] =
{
   STR2LEN, 0x03,
   'C', 0,
   '8', 0,
   '0', 0,
   '5', 0,
   '1', 0,
   'F', 0,
   '3', 0,
   'x', 0,
   'x', 0,
   ' ', 0,
   'D', 0,
   'e', 0,
   'v', 0,
   'e', 0,
   'l', 0,
   'o', 0,
   'p', 0,
   'm', 0,
   'e', 0,
   'n', 0,
   't', 0,
   ' ', 0,
   'B', 0,
   'o', 0,
   'a', 0,
   'r', 0,
   'd', 0
}; //end of String2Desc

unsigned char* const STRINGDESCTABLE [] =
{
   String0Desc,
   String1Desc,
   String2Desc
};