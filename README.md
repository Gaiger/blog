# nRF51-concurrently-runs-ble-and-esb
In here I demonstrate how to make nRF51822 running as bluetooth low energy(BLE) peripheral and Enhance shockburst(ESB) device at the same time. This code are based on nRF51 SDK 11 and nRF24LE1 with nRFgo SDK. 

The nRF24LE1 will keep sending data to nRF51822 via ESB. If there is a BLE central(ex: phone) connected with the nRF51822, the nRF51822 will convey the data from nRF24LE1 to the central.
On the other hand, once the BLE central send data to nRF51822, the nRF51822 will relay the data to nRF24LE1, which will print the received data in UART.
