/********************************** (C) COPYRIGHT *******************************
* File Name          : ch32v30x_it.c
* Author             : WCH
* Version            : V1.0.0
* Date               : 2021/06/06
* Description        : Main Interrupt Service Routines.
* Copyright (c) 2021 Nanjing Qinheng Microelectronics Co., Ltd.
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/
#include "ch32v30x_it.h"

#define __IRQ       __attribute__((interrupt()))
#define __NAKED     __attribute__((naked))
#define __IRQ_WEAK __attribute__((interrupt(), weak))

__NAKED void Ecall_M_Handler(void) {
    /* Use naked function to generate a short call, without saving stack. */
    asm("j freertos_risc_v_trap_handler");
}

__NAKED void Ecall_U_Handler(void) {
    for(;;) {
        /* Who called me? */
    }
}

__NAKED void Default_Handler(void) {
    for(;;) {
        /* Where are you from? */
    }
}

__IRQ void NMI_Handler(void)
{
    /* NMI handler */
}


extern void vPortSysTick_Handler(void);

__IRQ void SysTick_Handler(void)
{
    vPortSysTick_Handler();
}

__IRQ_WEAK void EXTI0_IRQHandler(void)
{
  if(EXTI_GetITStatus(EXTI_Line0)!=RESET)
  {
#if 1
    printf("Run at EXTI\r\n");

#endif
    EXTI_ClearITPendingBit(EXTI_Line0);     /* Clear Flag */
  }
}

