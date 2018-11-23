#ifndef _TRRSCVR_PARA_H_
#define _TRRSCVR_PARA_H_

#define ESB_MAX_PAYLOAD_LEN											(32)
#define ESB_MAX_ACK_PAYLOAD_LEN									(4) //DOC said it is 32, but my tring revealed it being 4 

#define ESB_DATA_RATE														(HAL_NRF_1MBPS)
#define ESB_RF_CHANNEL        									(50)
#define ESB_OUTPUT_POWER												(HAL_NRF_0DBM)


/*
	https://www.nordicsemi.com/eng/Nordic-FAQ/Silicon-Products/nRF24L01/How-to-choose-an-address
	Quick summary:

	Use at least 32bit address and enable 16bit CRC.
	Avoid addresses that start with  0x55, 0xAA. 
	Avoid adopting 0x00 and 0x55 as a part of addresses
*/


#define ADDR_PIPE0_0														(0x10)
#define ADDR_PIPE0_1														(0x11)
#define ADDR_PIPE0_2														(0x12)
#define ADDR_PIPE0_3														(0x13)
#define ADDR_PIPE0_4														(0x14)


#define ADDR_PIPE1TO5_1													(0xE1)
#define ADDR_PIPE1TO5_2													(0xE2)
#define ADDR_PIPE1TO5_3													(0xE3)
#define ADDR_PIPE1TO5_4													(0xE4)
									
#define ADDR_PIPE1_0														(0x81)
#define ADDR_PIPE2_0														(0x82)
#define ADDR_PIPE3_0														(0x83)
#define ADDR_PIPE4_0														(0x84)
#define ADDR_PIPE5_0														(0x85)


#endif /*_TRRSCVR_PARA_H_*/