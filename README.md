# nRF51/nRF52 concurrently runs ble and esb
In here, I demonstrate how to run nRF51822/nRF52832 as bluetooth low energy(BLE) peripheral and enhance shockburst(ESB) protocol at the same time. This code are based on nRF5X SDK 11 and nRF24LE1 with nRFgo SDK. 
nRF5X SDK you could download from here : https://developer.nordicsemi.com/nRF5_SDK/


The nRF24LE1 will keep sending data to nRF51822 via ESB. If there is a BLE central(ex: phone) connected with the nRF51822, the nRF51822 will convey the data from nRF24LE1 to the central.

On the other hand, once the BLE central send data to nRF51/nRF52, the nRF51/nRF52 will relay the data to nRF24LE1, which will print the received data in UART.

Download the code and place it under the SDK directory, nRF5_SDK_11.0.0_89a8197\examples\ble_peripheral. 
The folder structure be : 

config : boardchip function configuration, which is as the nRF5X default projects.
esb : here contents the esb code, which is modified, not as the same as nRF5_SDK_11.0.0_89a8197\components\properitary_rf\esb
esb_timeslot : the esb_timeslot code modifies the time-schedule mechanism to register a time-sliced for esb usage.
nRF24LE1 : an ESB sender/receiver, which would periodically send ESB data (to nRF5x) via designated pipe, and it would receive ESB data also.
project for pca10028 : for nRF51
project for pca10040 : for nRF52



