# Enhanced-shockburst-with-acknowledge
Demonstrate how to enable nRF24LE1 acknowledge mechanism,  It brings the transmission being reliable.

The different between Enhanced Shockburst and Shockburst are, the  Enhanced Shockburst is within the functions of dynamical packet length and auto-acknowlege mechanism. dynamical packet length is very intuitive, the packet length is able to be not fixed. Auto-acknowledge mechanism is while the receiver gets a packet, the receiver would send back an ack signal to indicate this packet has been successfully received. If the ack does not be delivered to the sender after max retries count, the Max retries interrupt would occur. 

Besides of guarantee of transmission , there is a bonus function of aoto-acknowlege : the reciver is able to sendback customed packload, up to 4 bytes(based on my testing). 



