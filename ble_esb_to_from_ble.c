
#include "ble_esb_to_from_ble.h"

#include "ble_srv_common.h"
#include "sdk_common.h"


static void on_connect(ble_esb_to_from_ble_t * p_esb_to_from_ble, 
	ble_evt_t * p_ble_evt)
{
    p_esb_to_from_ble->conn_handle = p_ble_evt->evt.gap_evt.conn_handle;
}/*on_connect*/


static void on_disconnect(ble_esb_to_from_ble_t * p_esb_to_from_ble, 
	ble_evt_t * p_ble_evt)
{
    UNUSED_PARAMETER(p_ble_evt);
    p_esb_to_from_ble->conn_handle = BLE_CONN_HANDLE_INVALID;
}/*on_disconnect*/


static void on_write(ble_esb_to_from_ble_t * p_esb_to_from_ble, 
	ble_evt_t * p_ble_evt)
{
	ble_gatts_evt_write_t * p_evt_write;
	p_evt_write= &p_ble_evt->evt.gatts_evt.params.write;
	

	if(p_evt_write->handle != p_esb_to_from_ble->to_esb_handles.value_handle)
		return;
		
	if(NULL == p_esb_to_from_ble)
		return;
		
	if(NULL == p_esb_to_from_ble->data_receving_handler)
		return;
		
	p_esb_to_from_ble->data_receving_handler(p_esb_to_from_ble,  
		p_evt_write->data, p_evt_write->len);
	
}/*on_write*/


void ble_esb_to_from_ble_on_ble_evt(ble_esb_to_from_ble_t * p_esb_to_from_ble, 
	ble_evt_t * p_ble_evt)
{
	if((NULL == p_esb_to_from_ble ) 
			|| (NULL == p_ble_evt))
	{
			return;
	}/** if handle NULL*/

	switch (p_ble_evt->header.evt_id)
	{
	case BLE_GAP_EVT_CONNECTED:
			on_connect(p_esb_to_from_ble, p_ble_evt);
			break;

	case BLE_GAP_EVT_DISCONNECTED:
			on_disconnect(p_esb_to_from_ble, p_ble_evt);
			break;

	case BLE_GATTS_EVT_WRITE:
			on_write(p_esb_to_from_ble, p_ble_evt);
			break;

	default:
			// No implementation needed.
			break;
	}
		
	return ;
}/*ble_esb_to_from_ble_on_ble_evt*/


static uint32_t to_esb_char_add(ble_esb_to_from_ble_t *p_esb_to_from_ble, 
	const ble_esb_to_from_ble_init_t *p_esb_to_from_ble_init)
{
	ble_gatts_char_md_t char_md;
	ble_gatts_attr_t    attr_char_value;
	ble_uuid_t          ble_uuid;
	ble_gatts_attr_md_t attr_md;
	
	VERIFY_PARAM_NOT_NULL(p_esb_to_from_ble_init);
	
	memset(&char_md, 0, sizeof(char_md));

	char_md.char_props.write         = 1;
	//char_md.char_props.write_wo_resp = 1;
	char_md.p_char_user_desc         = NULL;
	char_md.p_char_pf                = NULL;
	char_md.p_user_desc_md           = NULL;
	char_md.p_cccd_md                = NULL;
	char_md.p_sccd_md                = NULL;

	

#define BLE_TO_FROM_ESB_UUID_TO_ESB_CHAR    	(0x2028)	
	BLE_UUID_BLE_ASSIGN(ble_uuid, BLE_TO_FROM_ESB_UUID_TO_ESB_CHAR);

	memset(&attr_md, 0, sizeof(attr_md));

	BLE_GAP_CONN_SEC_MODE_SET_OPEN(&attr_md.read_perm);
	BLE_GAP_CONN_SEC_MODE_SET_OPEN(&attr_md.write_perm);

	attr_md.vloc    = BLE_GATTS_VLOC_STACK;
	attr_md.rd_auth = 0;
	attr_md.wr_auth = 0;
	attr_md.vlen    = 1;

	
	memset(&attr_char_value, 0, sizeof(attr_char_value));

	attr_char_value.p_uuid    = &ble_uuid;
	attr_char_value.p_attr_md = &attr_md;
	attr_char_value.init_len  = 1;
	attr_char_value.init_offs = 0;
	attr_char_value.max_len   = GATT_MTU_SIZE_DEFAULT;
	attr_char_value.p_value      = NULL;
	
	return sd_ble_gatts_characteristic_add(p_esb_to_from_ble->service_handle,
		 &char_md,
		 &attr_char_value,
		 &p_esb_to_from_ble->to_esb_handles);

}/*to_esb_char_add*/


static uint32_t from_esb_char_add(ble_esb_to_from_ble_t *p_esb_to_from_ble, 
	const ble_esb_to_from_ble_init_t *p_esb_to_from_ble_init)
{
	ble_gatts_char_md_t char_md;
	ble_gatts_attr_md_t cccd_md;
	ble_gatts_attr_t    attr_char_value;
	ble_uuid_t          ble_uuid;
	ble_gatts_attr_md_t attr_md;
	
	
	VERIFY_PARAM_NOT_NULL(p_esb_to_from_ble_init);
	
	memset(&cccd_md, 0, sizeof(cccd_md));

	BLE_GAP_CONN_SEC_MODE_SET_OPEN(&cccd_md.read_perm);
	BLE_GAP_CONN_SEC_MODE_SET_OPEN(&cccd_md.write_perm);

	cccd_md.vloc = BLE_GATTS_VLOC_STACK;
	
	memset(&char_md, 0, sizeof(char_md));
	
	char_md.char_props.notify 			= 1;
	char_md.p_char_user_desc         = NULL;
	char_md.p_char_pf                = NULL;
	char_md.p_user_desc_md           = NULL;
	char_md.p_cccd_md                = &cccd_md;
	char_md.p_sccd_md                = NULL;


#define BLE_TO_FROM_ESB_UUID_FROM_ESB_CHAR    	(0x2058)	
	BLE_UUID_BLE_ASSIGN(ble_uuid, BLE_TO_FROM_ESB_UUID_FROM_ESB_CHAR);


	memset(&attr_md, 0, sizeof(attr_md));

	BLE_GAP_CONN_SEC_MODE_SET_OPEN(&attr_md.read_perm);
	BLE_GAP_CONN_SEC_MODE_SET_OPEN(&attr_md.write_perm);

	attr_md.vloc    = BLE_GATTS_VLOC_STACK;
	attr_md.rd_auth = 0;
	attr_md.wr_auth = 0;
	attr_md.vlen    = 1;


	memset(&attr_char_value, 0, sizeof(attr_char_value));
	
	attr_char_value.p_uuid    = &ble_uuid;
	attr_char_value.p_attr_md = &attr_md;
	attr_char_value.init_len  = sizeof(uint8_t);
	attr_char_value.init_offs = 0;
	attr_char_value.max_len   = GATT_MTU_SIZE_DEFAULT;
	attr_char_value.p_value   = NULL;
	
	return sd_ble_gatts_characteristic_add(p_esb_to_from_ble->service_handle,
		&char_md,
		&attr_char_value,
		&p_esb_to_from_ble->from_esb_handles);	
}/*from_esb_char_add*/


uint32_t ble_esb_to_from_ble_init(ble_esb_to_from_ble_t *p_esb_to_from_ble, 
	 const ble_esb_to_from_ble_init_t *p_esb_to_from_ble_init)
{

	uint32_t   err_code;
	ble_uuid_t ble_uuid;	

	VERIFY_PARAM_NOT_NULL(p_esb_to_from_ble);
	VERIFY_PARAM_NOT_NULL(p_esb_to_from_ble_init);

	p_esb_to_from_ble->conn_handle  = BLE_CONN_HANDLE_INVALID;

	p_esb_to_from_ble->data_receving_handler   
		= p_esb_to_from_ble_init->data_receving_handler;	

	BLE_UUID_BLE_ASSIGN(ble_uuid, BLE_UUID_BLE_TO_FROM_ESB_SERVICE);

	err_code = sd_ble_gatts_service_add(BLE_GATTS_SRVC_TYPE_PRIMARY,
									&ble_uuid,
									&p_esb_to_from_ble->service_handle);
	VERIFY_SUCCESS(err_code);


	to_esb_char_add(p_esb_to_from_ble, p_esb_to_from_ble_init);	
	if(NRF_SUCCESS != err_code)    
	return err_code;


	from_esb_char_add(p_esb_to_from_ble, p_esb_to_from_ble_init);
	if(NRF_SUCCESS != err_code)    
	return err_code;  
	
	return NRF_SUCCESS;
}/*ble_esb_to_from_ble_init*/



uint32_t ble_esb_to_from_ble_send_to_ble(ble_esb_to_from_ble_t *p_esb_to_from_ble, 
	uint8_t * p_data, uint16_t len)
{
	ble_gatts_hvx_params_t hvx_params;

	VERIFY_PARAM_NOT_NULL(p_esb_to_from_ble);

	if(BLE_CONN_HANDLE_INVALID == p_esb_to_from_ble->conn_handle)		
		return NRF_ERROR_INVALID_STATE;
		

	if(GATT_MTU_SIZE_DEFAULT < len)		
		return NRF_ERROR_INVALID_PARAM;		

	memset(&hvx_params, 0, sizeof(hvx_params));

	hvx_params.handle = p_esb_to_from_ble->from_esb_handles.value_handle;
	hvx_params.p_data = p_data;
	hvx_params.p_len  = &len;
	hvx_params.type   = BLE_GATT_HVX_NOTIFICATION;

	return sd_ble_gatts_hvx(p_esb_to_from_ble->conn_handle, &hvx_params);
		
}/*ble_esb_to_from_ble_send_to_ble*/
