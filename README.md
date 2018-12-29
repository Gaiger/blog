# STM32F103_nRF24L01
Use STM32F103 to control nRF24L01
This device is communicable with the nRF5X of https://github.com/Gaiger/nRF51-nRF52-concurrently-runs-ble-peripheral-and-esb, and the STM8L of https://github.com/Gaiger/STM8_nRF24L01.
The GPIOs are defined :

PA8 : led for sending data  
PD2 : led for received data.

PC4 : CSN
PA4 : CE
PA1 : IRQ

SPI1 :
PA5: SLK
PA7: MOSI
PA6: MISO
