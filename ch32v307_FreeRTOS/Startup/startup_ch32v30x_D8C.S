/********************************** (C) COPYRIGHT *******************************
* File Name          : startup_ch32v30x.s
* Author             : WCH
* Version            : V1.0.0
* Date               : 2021/06/06
* Description        : CH32V30x vector table for eclipse toolchain.
*******************************************************************************/

	.section	.init,"ax",@progbits
	.global	_start
	.align	1
_start:
	j	handle_reset
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00000013
	.word 0x00100073
    .section    .vector,"ax",@progbits
    .align  1

_exception_base:
 	.option norvc;
	nop
    j Exception_Handler

_vector_base:
    .option norvc;
    nop
    j Default_Handler         	 /*   0: Reserved                     */
    lw t1, 0(sp)
    j Default_Handler            /*   1: Reserved                      */
    lw t1, 0(sp)
    j NMI_Handler                /*   2: NMI                           */
    lw t1, 0(sp)
    j Default_Handler            /*   3: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*   4: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*   5: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*   6: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*   7: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*   8: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*   9: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*  10: Reserved                      */
    lw t1, 0(sp)
    j Default_Handler            /*  11: Reserved                      */
    lw t1, 0(sp)
    j SysTick_Handler            /*  12: SysTick                       */
    lw t1, 0(sp)
    j Default_Handler            /*  13: Reserved                      */
    lw t1, 0(sp)
    j SW_Handler                 /*  14: Software                      */
    lw t1, 0(sp)
    j Default_Handler            /*  15: Reserved                      */
    lw t1, 0(sp)
    j WWDG_IRQHandler            /*  16: Window Watchdog               */
    lw t1, 0(sp)
    j PVD_IRQHandler             /*  17: PVD through EXTI Line detect  */
    lw t1, 0(sp)
    j TAMPER_IRQHandler          /*  18: TAMPER                        */
    lw t1, 0(sp)
    j RTC_IRQHandler             /*  19: RTC                           */
    lw t1, 0(sp)
    j FLASH_IRQHandler           /*  20: Flash                         */
    lw t1, 0(sp)
    j RCC_IRQHandler             /*  21: RCC                           */
    lw t1, 0(sp)
    j EXTI0_IRQHandler           /*  22: EXTI Line 0                   */
    lw t1, 0(sp)
    j EXTI1_IRQHandler           /*  23: EXTI Line 1                   */
    lw t1, 0(sp)
    j EXTI2_IRQHandler           /*  24: EXTI Line 2                   */
    lw t1, 0(sp)
    j EXTI3_IRQHandler           /*  25: EXTI Line 3                   */
    lw t1, 0(sp)
    j EXTI4_IRQHandler           /*  26: EXTI Line 4                   */
    lw t1, 0(sp)
    j DMA1_Channel1_IRQHandler   /*  27: DMA1 Channel 1                */
    lw t1, 0(sp)
    j DMA1_Channel2_IRQHandler   /*  28: DMA1 Channel 2                */
    lw t1, 0(sp)
    j DMA1_Channel3_IRQHandler   /*  29: DMA1 Channel 3                */
    lw t1, 0(sp)
    j DMA1_Channel4_IRQHandler   /*  30: DMA1 Channel 4                */
    lw t1, 0(sp)
    j DMA1_Channel5_IRQHandler   /*  31: DMA1 Channel 5                */
    lw t1, 0(sp)
    j DMA1_Channel6_IRQHandler   /*  32: DMA1 Channel 6                */
    lw t1, 0(sp)
    j DMA1_Channel7_IRQHandler   /*  33: DMA1 Channel 7                */
    lw t1, 0(sp)
    j ADC1_2_IRQHandler          /*  34: ADC1_2                        */
    lw t1, 0(sp)
    j USB_HP_CAN1_TX_IRQHandler  /*  35: USB HP and CAN1 TX            */
    lw t1, 0(sp)
    j USB_LP_CAN1_RX0_IRQHandler /*  36: USB LP and CAN1RX0            */
    lw t1, 0(sp)
    j CAN1_RX1_IRQHandler        /*  37: CAN1 RX1                      */
    lw t1, 0(sp)
    j CAN1_SCE_IRQHandler        /*  38: CAN1 SCE                      */
    lw t1, 0(sp)
    j EXTI9_5_IRQHandler         /*  39: EXTI Line 9..5                */
    lw t1, 0(sp)
    j TIM1_BRK_IRQHandler        /*  40: TIM1 Break                    */
    lw t1, 0(sp)
    j TIM1_UP_IRQHandler         /*  41: TIM1 Update                   */
    lw t1, 0(sp)
    j TIM1_TRG_COM_IRQHandler    /*  42: TIM1 Trigger and Commutation  */
    lw t1, 0(sp)
    j TIM1_CC_IRQHandler         /*  43: TIM1 Capture Compare          */
    lw t1, 0(sp)
    j TIM2_IRQHandler            /*  44: TIM2                          */
    lw t1, 0(sp)
    j TIM3_IRQHandler            /*  45: TIM3                          */
    lw t1, 0(sp)
    j TIM4_IRQHandler            /*  46: TIM4                          */
    lw t1, 0(sp)
    j I2C1_EV_IRQHandler         /*  47: I2C1 Event                    */
    lw t1, 0(sp)
    j I2C1_ER_IRQHandler         /*  48: I2C1 Error                    */
    lw t1, 0(sp)
    j I2C2_EV_IRQHandler         /*  49: I2C2 Event                    */
    lw t1, 0(sp)
    j I2C2_ER_IRQHandler         /*  50: I2C2 Error                    */
    lw t1, 0(sp)
    j SPI1_IRQHandler            /*  51: SPI1                          */
    lw t1, 0(sp)
    j SPI2_IRQHandler            /*  52: SPI2                          */
    lw t1, 0(sp)
    j USART1_IRQHandler          /*  53: USART1                        */
    lw t1, 0(sp)
    j USART2_IRQHandler          /*  54: USART2                        */
    lw t1, 0(sp)
    j USART3_IRQHandler          /*  55: USART3                        */
    lw t1, 0(sp)
    j EXTI15_10_IRQHandler       /*  56: EXTI Line 15..10              */
    lw t1, 0(sp)
    j RTCAlarm_IRQHandler        /*  57: RTC Alarm through EXTI Line   */
    lw t1, 0(sp)
    j USBWakeUp_IRQHandler       /*  58: USB Wakeup from suspend       */
    lw t1, 0(sp)
    j TIM8_BRK_IRQHandler        /*  59: TIM8 Break                    */
    lw t1, 0(sp)
    j TIM8_UP_IRQHandler         /*  60: TIM8 Update                   */
    lw t1, 0(sp)
    j TIM8_TRG_COM_IRQHandler    /*  61: TIM8 Trigger and Commutation  */
    lw t1, 0(sp)
    j TIM8_CC_IRQHandler         /*  62: TIM8 Capture Compare          */
    lw t1, 0(sp)
    j RNG_IRQHandler             /*  63: RNG                           */
    lw t1, 0(sp)
    j FSMC_IRQHandler            /*  64: FSMC                          */
    lw t1, 0(sp)
    j SDIO_IRQHandler            /*  65: SDIO                          */
    lw t1, 0(sp)
    j TIM5_IRQHandler            /*  66: TIM5                          */
    lw t1, 0(sp)
    j SPI3_IRQHandler            /*  67: SPI3                          */
    lw t1, 0(sp)
    j UART4_IRQHandler           /*  68: UART4                         */
    lw t1, 0(sp)
    j UART5_IRQHandler           /*  69: UART5                         */
    lw t1, 0(sp)
    j TIM6_IRQHandler            /*  70: TIM6                          */
    lw t1, 0(sp)
    j TIM7_IRQHandler            /*  71: TIM7                          */
    lw t1, 0(sp)
    j DMA2_Channel1_IRQHandler   /*  72: DMA2 Channel 1                */
    lw t1, 0(sp)
    j DMA2_Channel2_IRQHandler   /*  73: DMA2 Channel 2                */
    lw t1, 0(sp)
    j DMA2_Channel3_IRQHandler   /*  74: DMA2 Channel 3                */
    lw t1, 0(sp)
    j DMA2_Channel4_IRQHandler   /*  75: DMA2 Channel 4                */
    lw t1, 0(sp)
    j DMA2_Channel5_IRQHandler   /*  76: DMA2 Channel 5                */
    lw t1, 0(sp)
    j ETH_IRQHandler             /*  77: ETH                           */
    lw t1, 0(sp)
    j ETH_WKUP_IRQHandler        /*  78: ETH WakeUp                    */
    lw t1, 0(sp)
    j CAN2_TX_IRQHandler         /*  79: CAN2 TX                       */
    lw t1, 0(sp)
    j CAN2_RX0_IRQHandler        /*  80: CAN2 RX0                      */
    lw t1, 0(sp)
    j CAN2_RX1_IRQHandler        /*  81: CAN2 RX1                      */
    lw t1, 0(sp)
    j CAN2_SCE_IRQHandler        /*  82: CAN2 SCE                      */
    lw t1, 0(sp)
    j OTG_FS_IRQHandler          /*  83: OTGFS                         */
    lw t1, 0(sp)
    j USBHSWakeup_IRQHandler     /*  84: USBHS Wakeup                  */
    lw t1, 0(sp)
    j USBHS_IRQHandler           /*  85: USBHS                         */
    lw t1, 0(sp)
    j DVP_IRQHandler             /*  86: DVP                           */
    lw t1, 0(sp)
    j UART6_IRQHandler           /*  87: UART6                         */
    lw t1, 0(sp)
    j UART7_IRQHandler           /*  88: UART7                         */
    lw t1, 0(sp)
    j UART8_IRQHandler           /*  89: UART8                         */
    lw t1, 0(sp)
    j TIM9_BRK_IRQHandler        /*  90: TIM9 Break                    */
    lw t1, 0(sp)
    j TIM9_UP_IRQHandler         /*  91: TIM9 Update                   */
    lw t1, 0(sp)
    j TIM9_TRG_COM_IRQHandler    /*  92: TIM9 Trigger and Commutation  */
    lw t1, 0(sp)
    j TIM9_CC_IRQHandler         /*  93: TIM9 Capture Compare          */
    lw t1, 0(sp)
    j TIM10_BRK_IRQHandler       /*  94: TIM10 Break                   */
    lw t1, 0(sp)
    j TIM10_UP_IRQHandler        /*  95: TIM10 Update                  */
    lw t1, 0(sp)
    j TIM10_TRG_COM_IRQHandler   /*  96: TIM10 Trigger and Commutation */
    lw t1, 0(sp)
    j TIM10_CC_IRQHandler        /*  97: TIM10 Capture Compare         */
    lw t1, 0(sp)
    j DMA2_Channel6_IRQHandler   /*  98: DMA2 Channel 6                */
    lw t1, 0(sp)
    j DMA2_Channel7_IRQHandler   /*  99: DMA2 Channel 7                */
    lw t1, 0(sp)
    j DMA2_Channel8_IRQHandler   /* 100: DMA2 Channel 8                */
    lw t1, 0(sp)
    j DMA2_Channel9_IRQHandler   /* 101: DMA2 Channel 9                */
    lw t1, 0(sp)
    j DMA2_Channel10_IRQHandler  /* 102: DMA2 Channel 10               */
    lw t1, 0(sp)
    j DMA2_Channel11_IRQHandler  /* 103: DMA2 Channel 11               */
    .option rvc;

    .section    .text.vector_handler, "ax", @progbits
    .weak   NMI_Handler                /* NMI */
    .weak   HardFault_Handler          /* Hard Fault */
    .weak   Ecall_M_Mode_Handler       /* Ecall M Mode */
    .weak   Ecall_U_Mode_Handler       /* Ecall U Mode */
    .weak   Break_Point_Handler        /* Break Point */
    .weak   SysTick_Handler            /* SysTick */
    .weak   SW_Handler                 /* SW */
    .weak   WWDG_IRQHandler            /* Window Watchdog */
    .weak   PVD_IRQHandler             /* PVD through EXTI Line detect */
    .weak   TAMPER_IRQHandler          /* TAMPER */
    .weak   RTC_IRQHandler             /* RTC */
    .weak   FLASH_IRQHandler           /* Flash */
    .weak   RCC_IRQHandler             /* RCC */
    .weak   EXTI0_IRQHandler           /* EXTI Line 0 */
    .weak   EXTI1_IRQHandler           /* EXTI Line 1 */
    .weak   EXTI2_IRQHandler           /* EXTI Line 2 */
    .weak   EXTI3_IRQHandler           /* EXTI Line 3 */
    .weak   EXTI4_IRQHandler           /* EXTI Line 4 */
    .weak   DMA1_Channel1_IRQHandler   /* DMA1 Channel 1 */
    .weak   DMA1_Channel2_IRQHandler   /* DMA1 Channel 2 */
    .weak   DMA1_Channel3_IRQHandler   /* DMA1 Channel 3 */
    .weak   DMA1_Channel4_IRQHandler   /* DMA1 Channel 4 */
    .weak   DMA1_Channel5_IRQHandler   /* DMA1 Channel 5 */
    .weak   DMA1_Channel6_IRQHandler   /* DMA1 Channel 6 */
    .weak   DMA1_Channel7_IRQHandler   /* DMA1 Channel 7 */
    .weak   ADC1_2_IRQHandler          /* ADC1_2 */
    .weak   USB_HP_CAN1_TX_IRQHandler  /* USB HP and CAN1 TX */
    .weak   USB_LP_CAN1_RX0_IRQHandler /* USB LP and CAN1RX0 */
    .weak   CAN1_RX1_IRQHandler        /* CAN1 RX1 */
    .weak   CAN1_SCE_IRQHandler        /* CAN1 SCE */
    .weak   EXTI9_5_IRQHandler         /* EXTI Line 9..5 */
    .weak   TIM1_BRK_IRQHandler        /* TIM1 Break */
    .weak   TIM1_UP_IRQHandler         /* TIM1 Update */
    .weak   TIM1_TRG_COM_IRQHandler    /* TIM1 Trigger and Commutation */
    .weak   TIM1_CC_IRQHandler         /* TIM1 Capture Compare */
    .weak   TIM2_IRQHandler            /* TIM2 */
    .weak   TIM3_IRQHandler            /* TIM3 */
    .weak   TIM4_IRQHandler            /* TIM4 */
    .weak   I2C1_EV_IRQHandler         /* I2C1 Event */
    .weak   I2C1_ER_IRQHandler         /* I2C1 Error */
    .weak   I2C2_EV_IRQHandler         /* I2C2 Event */
    .weak   I2C2_ER_IRQHandler         /* I2C2 Error */
    .weak   SPI1_IRQHandler            /* SPI1 */
    .weak   SPI2_IRQHandler            /* SPI2 */
    .weak   USART1_IRQHandler          /* USART1 */
    .weak   USART2_IRQHandler          /* USART2 */
    .weak   USART3_IRQHandler          /* USART3 */
    .weak   EXTI15_10_IRQHandler       /* EXTI Line 15..10 */
    .weak   RTCAlarm_IRQHandler        /* RTC Alarm through EXTI Line */
    .weak   USBWakeUp_IRQHandler       /* USB Wakeup from suspend */
    .weak   TIM8_BRK_IRQHandler        /* TIM8 Break */
    .weak   TIM8_UP_IRQHandler         /* TIM8 Update */
    .weak   TIM8_TRG_COM_IRQHandler    /* TIM8 Trigger and Commutation */
    .weak   TIM8_CC_IRQHandler         /* TIM8 Capture Compare */
    .weak   RNG_IRQHandler             /* RNG */
    .weak   FSMC_IRQHandler            /* FSMC */
    .weak   SDIO_IRQHandler            /* SDIO */
    .weak   TIM5_IRQHandler            /* TIM5 */
    .weak   SPI3_IRQHandler            /* SPI3 */
    .weak   UART4_IRQHandler           /* UART4 */
    .weak   UART5_IRQHandler           /* UART5 */
    .weak   TIM6_IRQHandler            /* TIM6 */
    .weak   TIM7_IRQHandler            /* TIM7 */
    .weak   DMA2_Channel1_IRQHandler   /* DMA2 Channel 1 */
    .weak   DMA2_Channel2_IRQHandler   /* DMA2 Channel 2 */
    .weak   DMA2_Channel3_IRQHandler   /* DMA2 Channel 3 */
    .weak   DMA2_Channel4_IRQHandler   /* DMA2 Channel 4 */
    .weak   DMA2_Channel5_IRQHandler   /* DMA2 Channel 5 */
    .weak   ETH_IRQHandler             /* ETH */
    .weak   ETH_WKUP_IRQHandler        /* ETH WakeUp */
    .weak   CAN2_TX_IRQHandler         /* CAN2 TX */
    .weak   CAN2_RX0_IRQHandler        /* CAN2 RX0 */
    .weak   CAN2_RX1_IRQHandler        /* CAN2 RX1 */
    .weak   CAN2_SCE_IRQHandler        /* CAN2 SCE */
    .weak   OTG_FS_IRQHandler          /* OTGFS */
    .weak   USBHSWakeup_IRQHandler     /* USBHS Wakeup */
    .weak   USBHS_IRQHandler           /* USBHS */
    .weak   DVP_IRQHandler             /* DVP */
    .weak   UART6_IRQHandler           /* UART6 */
    .weak   UART7_IRQHandler           /* UART7 */
    .weak   UART8_IRQHandler           /* UART8 */
    .weak   TIM9_BRK_IRQHandler        /* TIM9 Break */
    .weak   TIM9_UP_IRQHandler         /* TIM9 Update */
    .weak   TIM9_TRG_COM_IRQHandler    /* TIM9 Trigger and Commutation */
    .weak   TIM9_CC_IRQHandler         /* TIM9 Capture Compare */
    .weak   TIM10_BRK_IRQHandler       /* TIM10 Break */
    .weak   TIM10_UP_IRQHandler        /* TIM10 Update */
    .weak   TIM10_TRG_COM_IRQHandler   /* TIM10 Trigger and Commutation */
    .weak   TIM10_CC_IRQHandler        /* TIM10 Capture Compare */
    .weak   DMA2_Channel6_IRQHandler   /* DMA2 Channel 6 */
    .weak   DMA2_Channel7_IRQHandler   /* DMA2 Channel 7 */
    .weak   DMA2_Channel8_IRQHandler   /* DMA2 Channel 8 */
    .weak   DMA2_Channel9_IRQHandler   /* DMA2 Channel 9 */
    .weak   DMA2_Channel10_IRQHandler  /* DMA2 Channel 10 */
    .weak   DMA2_Channel11_IRQHandler  /* DMA2 Channel 11 */


Default_Handler:  1:  j 1b
NMI_Handler:  1:  j 1b
HardFault_Handler:  1:  j 1b
Ecall_M_Mode_Handler:  1:  j 1b
Ecall_U_Mode_Handler:  1:  j 1b
Break_Point_Handler:  1:  j 1b
SysTick_Handler:  1:  j 1b
SW_Handler:  1:  j 1b
WWDG_IRQHandler:  1:  j 1b
PVD_IRQHandler:  1:  j 1b
TAMPER_IRQHandler:  1:  j 1b
RTC_IRQHandler:  1:  j 1b
FLASH_IRQHandler:  1:  j 1b
RCC_IRQHandler:  1:  j 1b
EXTI0_IRQHandler:  1:  j 1b
EXTI1_IRQHandler:  1:  j 1b
EXTI2_IRQHandler:  1:  j 1b
EXTI3_IRQHandler:  1:  j 1b
EXTI4_IRQHandler:  1:  j 1b
DMA1_Channel1_IRQHandler:  1:  j 1b
DMA1_Channel2_IRQHandler:  1:  j 1b
DMA1_Channel3_IRQHandler:  1:  j 1b
DMA1_Channel4_IRQHandler:  1:  j 1b
DMA1_Channel5_IRQHandler:  1:  j 1b
DMA1_Channel6_IRQHandler:  1:  j 1b
DMA1_Channel7_IRQHandler:  1:  j 1b
ADC1_2_IRQHandler:  1:  j 1b
USB_HP_CAN1_TX_IRQHandler:  1:  j 1b
USB_LP_CAN1_RX0_IRQHandler:  1:  j 1b
CAN1_RX1_IRQHandler:  1:  j 1b
CAN1_SCE_IRQHandler:  1:  j 1b
EXTI9_5_IRQHandler:  1:  j 1b
TIM1_BRK_IRQHandler:  1:  j 1b
TIM1_UP_IRQHandler:  1:  j 1b
TIM1_TRG_COM_IRQHandler:  1:  j 1b
TIM1_CC_IRQHandler:  1:  j 1b
TIM2_IRQHandler:  1:  j 1b
TIM3_IRQHandler:  1:  j 1b
TIM4_IRQHandler:  1:  j 1b
I2C1_EV_IRQHandler:  1:  j 1b
I2C1_ER_IRQHandler:  1:  j 1b
I2C2_EV_IRQHandler:  1:  j 1b
I2C2_ER_IRQHandler:  1:  j 1b
SPI1_IRQHandler:  1:  j 1b
SPI2_IRQHandler:  1:  j 1b
USART1_IRQHandler:  1:  j 1b
USART2_IRQHandler:  1:  j 1b
USART3_IRQHandler:  1:  j 1b
EXTI15_10_IRQHandler:  1:  j 1b
RTCAlarm_IRQHandler:  1:  j 1b
USBWakeUp_IRQHandler:  1:  j 1b
TIM8_BRK_IRQHandler:  1:  j 1b
TIM8_UP_IRQHandler:  1:  j 1b
TIM8_TRG_COM_IRQHandler:  1:  j 1b
TIM8_CC_IRQHandler:  1:  j 1b
RNG_IRQHandler:  1:  j 1b
FSMC_IRQHandler:  1:  j 1b
SDIO_IRQHandler:  1:  j 1b
TIM5_IRQHandler:  1:  j 1b
SPI3_IRQHandler:  1:  j 1b
UART4_IRQHandler:  1:  j 1b
UART5_IRQHandler:  1:  j 1b
TIM6_IRQHandler:  1:  j 1b
TIM7_IRQHandler:  1:  j 1b
DMA2_Channel1_IRQHandler:  1:  j 1b
DMA2_Channel2_IRQHandler:  1:  j 1b
DMA2_Channel3_IRQHandler:  1:  j 1b
DMA2_Channel4_IRQHandler:  1:  j 1b
DMA2_Channel5_IRQHandler:  1:  j 1b
ETH_IRQHandler:  1:  j 1b
ETH_WKUP_IRQHandler:  1:  j 1b
CAN2_TX_IRQHandler:  1:  j 1b
CAN2_RX0_IRQHandler:  1:  j 1b
CAN2_RX1_IRQHandler:  1:  j 1b
CAN2_SCE_IRQHandler:  1:  j 1b
OTG_FS_IRQHandler:  1:  j 1b
USBHSWakeup_IRQHandler:  1:  j 1b
USBHS_IRQHandler:  1:  j 1b
DVP_IRQHandler:  1:  j 1b
UART6_IRQHandler:  1:  j 1b
UART7_IRQHandler:  1:  j 1b
UART8_IRQHandler:  1:  j 1b
TIM9_BRK_IRQHandler:  1:  j 1b
TIM9_UP_IRQHandler:  1:  j 1b
TIM9_TRG_COM_IRQHandler:  1:  j 1b
TIM9_CC_IRQHandler:  1:  j 1b
TIM10_BRK_IRQHandler:  1:  j 1b
TIM10_UP_IRQHandler:  1:  j 1b
TIM10_TRG_COM_IRQHandler:  1:  j 1b
TIM10_CC_IRQHandler:  1:  j 1b
DMA2_Channel6_IRQHandler:  1:  j 1b
DMA2_Channel7_IRQHandler:  1:  j 1b
DMA2_Channel8_IRQHandler:  1:  j 1b
DMA2_Channel9_IRQHandler:  1:  j 1b
DMA2_Channel10_IRQHandler:  1:  j 1b
DMA2_Channel11_IRQHandler:  1:  j 1b

	.section	.text.handle_reset,"ax",@progbits
	.weak	handle_reset
	.align	1
handle_reset:
.option push
.option	norelax
   // csrw mepc, t0
	la gp, __global_pointer$
.option	pop
1:
	la sp, _eusrstack
2:
	/* Load data section from flash to RAM */
	la a0, _data_lma
	la a1, _data_vma
	la a2, _edata
	bgeu a1, a2, 2f
1:
	lw t0, (a0)
	sw t0, (a1)
	addi a0, a0, 4
	addi a1, a1, 4
	bltu a1, a2, 1b
2:
	/* Clear bss section */
	la a0, _sbss
	la a1, _ebss
	bgeu a0, a1, 2f
1:
	sw zero, (a0)
	addi a0, a0, 4
	bltu a0, a1, 1b
2:
    li t0, 0x1f
    csrw 0xbc0, t0

    /* Enable nested and hardware stack */
	li t0, 0x1f
	csrw 0x804, t0

    /* Enable floating point and interrupt */
#if(0)//GAIGER
   	li t0, 0x6088
#else
	li t0, 0x7800
#endif
   	csrs mstatus, t0

	la t0, _exception_base
	ori t0, t0, 0  /*exceptions mode*/
	csrw mtvec, t0

	jal  SystemInit
	jal main 

Exception_Handler:
    addi sp, sp, -8
    sw t0, 4(sp)
    sw t1, 8(sp)

    csrr t0, mcause
    ble t0, x0, interrupt_handler /* Check interrupt */
    li t1, 11 /* Find an M mode ecall (11) */
    beq t1, t0, ecall_m_handler
    li t1, 8
    beq t1, t0, ecall_u_handler
    j fault_handler

ecall_m_handler:
    lw t0, 4(sp)
    lw t1, 8(sp)
    addi sp, sp, 8
    j freertos_risc_v_trap_handler

ecall_u_handler:
    lw t0, 4(sp)
    lw t1, 8(sp)
    addi sp, sp, 8
    j freertos_risc_v_trap_handler

fault_handler:
    lw t0, 4(sp)
    lw t1, 8(sp)
    addi sp, sp, 8
    j Fault_Handler

interrupt_handler: /* Home made vector table */
    slli t0, t0, 3 /* t0 = t0 * 8 */

    la t1, _vector_base
    add t1, t1, t0

    lw t0, 4(sp)
    addi sp, sp, 8

    jr t1

Fault_Handler:  1:  j 1b
