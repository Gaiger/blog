# enhanced-ehockburst-designating-destinations-and-sources.
Demonstrate how to operate nRF24LE1 as a multi-sender and multi-receiver, the destination and source are designated
This code is based on nRF24LE1, but is very easy to port to the other microcontroller.

Folder PRX ::  An primary receiver receives the packets from the designated pipe(s). If you define the macro _RX_FOR_ALL_CHANNEL in file esb_app_prx_noack.h, the receiver would receive the packets from all(six) pipes.

Folder PTX ::  An primary transmitter sends the packets to  the designated pipe(s). If you define the macro _TX_FOR_ALL_CHANNEL in file esb_app_ptx_noack.h, the transmitter would send the packets to each pipe.
  
The addresses defined in PTX are the same as the addresses defined in PPX. Hence, if the RX_PIPE in PRX is the same as TX_PIPE in PTX, the transmission would be work.
Otherwise, if the TX/RX address is not consisted, PTX's sending would not successfully deliver to PRX (the pipe number could be not the same, but the designated address must be the same for the tx/rx pair).
  
