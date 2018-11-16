# enhanced-ehockburst-designating-destinations-and-sources.
Explanation is here : https://gaiger-programming.blogspot.com/2018/11/make-nordic-enhanced-shockburst-esb.html

Demonstrate how to operate enhanced shockburst transceiver as a multi-sender and multi-receiver whilethe destination and source are designated.
This code is based on nRF24LE1, but is easy to be ported to the other microcontroller.

Folder PRX ::  An primary receiver receives the packets from the designated pipe(s). If you define the macro _RX_FOR_ALL_CHANNEL in file esb_app_prx_noack.h, the receiver would receive the packets from all(six) pipes.

Folder PTX ::  An primary transmitter sends the packets to  the designated pipe(s). If you define the macro _TX_FOR_ALL_CHANNEL in file esb_app_ptx_noack.h, the transmitter would send the packets to each pipe.

Folder TXRX :: this transceiver is able to receive and transmit the (designated) packets. By default, it will transmit the packet periodically. If you define the macro _RELAYER, it would become a relayer to receive the packets from pipe RX_PIPE and resend the packets to pipe TX_PIPE.

Note :: The addresses defined must be the same for all the devices to make the transmission to work. Otherwise, if the TX/RX address for transmit/receive is not consisted, tx's sending would not successfully deliver to rx. But it is not necessary for the pipe number of each address being the same.



  
