/* Copyright (c) 2007 Nordic Semiconductor. All Rights Reserved.
 *
 * The information contained herein is property of Nordic Semiconductor ASA.
 * Terms and conditions of usage are described in detail in NORDIC
 * SEMICONDUCTOR STANDARD SOFTWARE LICENSE AGREEMENT. 
 *
 * Licensees are granted free, non-transferable use of the information. NO
 * WARRENTY of ANY KIND is provided. This heading must NOT be removed from
 * the file.
 *
 * $LastChangedRevision: 186 $
 */

/** @file
 * Type definitions for firmware projects developed at Nordic Semiconductor.
 *
 * Standard storage classes in C, such as @c char, @c int, and @c long, are not always
 * interpreted in the same way by the compiler. The types here are defined by their
 * bit length and signed/unsigned property, as their names indicate. The correlation
 * between the name and properties of the storage class should be true, regardless of
 * the compiler being used.
 */

#ifndef __STDINT_H__
#define __STDINT_H__

#include "STM8l10x_conf.h"

#ifndef NULL
#define NULL (void*)0
#endif

#endif // __STDINT_H__
