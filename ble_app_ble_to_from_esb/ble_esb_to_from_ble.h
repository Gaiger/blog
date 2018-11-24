
#ifndef _BLE_ESB_TO_FROM_BLE_H_
#define _BLE_ESB_TO_FROM_BLE_H_

#include "ble.h"
#include "ble_srv_common.h"
#include <stdint.h>
#include <stdbool.h>



#define BLE_UUID_BLE_TO_FROM_ESB_SERVICE 				0x1123                      /**< The UUID of the Nordic UART Service. */
#define BLE_NUS_MAX_DATA_LEN 									(GATT_MTU_SIZE_DEFAULT - 3) /**< Maximum length of data (in bytes) that can be transmitted to the peer by the Nordic UART service module. */


typedef struct ble_esb_to_from_ble_s ble_esb_to_from_ble_t;


typedef void (*ble_esb_to_from_ble_ble_data_receiving_handler_t) 
		(ble_esb_to_from_ble_t * p_to_from_esb_t, uint8_t * p_data, uint16_t len);

typedef struct
{
    ble_esb_to_from_ble_ble_data_receiving_handler_t  data_receving_handler;		
} ble_esb_to_from_ble_init_t;


struct ble_esb_to_from_ble_s
{
    uint8_t                  uuid_type;               /**< UUID type for Nordic UART Service Base UUID. */
    uint16_t                 service_handle;          /**< Handle of Nordic UART Service (as provided by the SoftDevice). */
    ble_gatts_char_handles_t to_esb_handles;              /**< Handles related to the TX characteristic (as provided by the SoftDevice). */
    ble_gatts_char_handles_t from_esb_handles;              /**< Handles related to the RX characteristic (as provided by the SoftDevice). */
    uint16_t                 conn_handle;             /**< Handle of the current connection (as provided by the SoftDevice). BLE_CONN_HANDLE_INVALID if not in a connection. */   
    ble_esb_to_from_ble_ble_data_receiving_handler_t   data_receving_handler;  /**< Event handler to be called for handling received data. */	
};

uint32_t ble_esb_to_from_ble_init(ble_esb_to_from_ble_t *p_esb_to_from_ble, 
	const ble_esb_to_from_ble_init_t *p_esb_to_from_ble_init);

		 
void ble_esb_to_from_ble_on_ble_evt(ble_esb_to_from_ble_t * p_esb_to_from_ble, 
	ble_evt_t * p_ble_evt);

uint32_t ble_esb_to_from_ble_send_to_ble
	(ble_esb_to_from_ble_t *p_esb_to_from_ble, uint8_t * p_data, uint16_t len);

#endif

