#ifndef _app_FATFS_UTILITY_H_
#define _app_FATFS_UTILITY_H_

#include <stdint.h>
#include <time.h>
#include "ff.h"

struct tm app_fatfs_convert_FILINFO_to_tm(const FILINFO* p_file_info);

FRESULT app_fatfs_mount(void);
FRESULT app_fatfs_format(void);
FRESULT app_fatfs_get_free_space(uint32_t* p_total_size_in_KB, uint32_t* p_free_size_in_KB);
void app_fatfs_print_all_files_attribute(void);

#endif /* _app_FATFS_UTILITY_H_ */
