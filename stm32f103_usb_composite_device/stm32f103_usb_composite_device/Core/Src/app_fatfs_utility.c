
#define _CRT_SECURE_NO_WARNINGS		

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <stm32_printf.h>

#include "ff.h"
#include "app_fatfs_utility.h"

#ifndef UNUSED
#define UNUSED(x)									(void)(x)
#endif
static FATFS s_fatfs; 	//Fatfs handle


/**********************************************************************/

FRESULT app_fatfs_mount(void)
{
	PRINTF("%s\r\n", __FUNCTION__);
	FRESULT fres;

	//printf("spi_flash_scan = %u\r\n", app_spi_flash_scan());

	fres = f_mount(&s_fatfs, "", 1); //1=mount now
	return fres;
}

/**********************************************************************/

FRESULT app_fatfs_format(void)
{
	FRESULT fres;

	PRINTF("format the SD card!!!\r\n");
	PRINTF("formatting start...\r\n");
#define SFD											(1)
#define AUTO_SELECT_ALLOCATION_UNIT_SIZE			(0)
	fres = f_mkfs("", SFD, AUTO_SELECT_ALLOCATION_UNIT_SIZE);
	if (FR_OK != fres)
	{
		PRINTF("f_mkfs error (%i)\r\n", fres);
		return fres;
	}

	PRINTF("formatting completed...\r\n");
	return fres;
}

/**********************************************************************/

FRESULT app_fatfs_get_free_space(uint32_t* p_total_size_in_KB, uint32_t* p_free_size_in_KB)
{	
	*p_total_size_in_KB = *p_free_size_in_KB = 0xFFFFFFFF;

	FRESULT fres;
	uint32_t free_clusters;
	FATFS* p_free_fatfs;

	fres = f_getfree("", &free_clusters, &p_free_fatfs);
	if (FR_OK != fres)
		return fres;
	uint32_t total_sectors, free_sectors;
	total_sectors = (p_free_fatfs->n_fatent - 2) * p_free_fatfs->csize;
	free_sectors = free_clusters * p_free_fatfs->csize;
	*p_total_size_in_KB = total_sectors/2; // 2 means * 512/1024
	*p_free_size_in_KB = free_sectors/2;

	return fres;
}


/**********************************************************************/

struct tm app_fatfs_convert_FILINFO_to_tm(const FILINFO* const p_file_info)
{
	/*
	 * https://pubs.opengroup.org/onlinepubs/007908799/xsh/time.h.html
	 * http://elm-chan.org/fsw/ff/doc/fattime.html
	 */
	struct tm time;

	time.tm_sec = (0x1F & (p_file_info->ftime >> 0)) * 2;
	time.tm_min = (0x3F & (p_file_info->ftime >> 5));
	time.tm_hour = (0x1F & (p_file_info->ftime >> 11));

	time.tm_mday = (0x1F & (p_file_info->fdate >> 0));
	time.tm_mon = (0x0F & (p_file_info->fdate >> 5)) - 1;
	time.tm_year = (0x7F & (p_file_info->fdate >> 9)) + 1980;

	time.tm_wday = -1;
	time.tm_yday = -1;
	time.tm_isdst = -1;
	return time;
}

/**********************************************************************/

static int print_file_attribute(const FILINFO* p_file_info, const TCHAR* const p_path, const int dir_level)
{
	char string_buffer[_MAX_LFN + 64];
	char string_appendix[64] = { 0 };
	int ret = 0;

	if (AM_DIR & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0], sizeof(string_appendix), "%s", "Dir");
		ret = 1;
	}
	else
	{
		snprintf(&string_appendix[0], sizeof(string_appendix), "%10d bytes", (int)(p_file_info->fsize));
	}

	if (AM_RDO & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", R-ONLY");
	}

	if (AM_SYS & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", SYS");
	}

	if (AM_HID & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", HIDDEN");
	}

	if (AM_ARC & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", ARCHIVE");
	}

	struct tm time_info = app_fatfs_convert_FILINFO_to_tm(p_file_info);

	char string_time_info[16];
	strftime(&string_time_info[0], sizeof(string_time_info), "%y%m%d-%H:%M:%S", &time_info);

	const TCHAR* p_file_name = &p_file_info->fname[0];
#if _USE_LFN
	if(0 != p_file_info->lfname[0])
		p_file_name = p_file_info->lfname;
	UINT file_name_size = strlen(p_file_name);
#endif
	snprintf(&string_buffer[0], sizeof(string_buffer), "%*s%.*s %*s %s %s",
		dir_level * 4, "",
		file_name_size, p_file_name,
		12, "",
		&string_time_info[0],
		&string_appendix[0]);

	PRINTF("%s\r\n", &string_buffer[0]);
	return ret;
}

/**********************************************************************/

static FRESULT app_fatfs_print_folder_content(const TCHAR* const p_path, const int level)
{
	//PRINTF("__FUNCTION__ = %s\r\n", __FUNCTION__ );
	FRESULT fres;
	DIR dir;
	fres = f_opendir(&dir, p_path);
	if (FR_OK != fres)
	{
		//PRINTF("f_opendir error (%d)\r\n", fres);
		//PRINTF("error path = %s\r\n", p_path);
		return fres;
	}

	while (1)
	{
		FILINFO file_info;
#if _USE_LFN
		TCHAR lfn[_MAX_LFN];
		file_info.lfname = lfn;
		file_info.lfsize = sizeof(lfn);
#endif
		FRESULT readdir_fres = f_readdir(&dir, &file_info);
		if ((FR_OK != readdir_fres) || (0 == file_info.fname[0]))
		{
			break;
		}

		if (0 < print_file_attribute(&file_info, p_path, level))
		{
			const TCHAR* p_file_name = &file_info.fname[0];
#if _USE_LFN
			if(0 != file_info.lfname[0])
				p_file_name = file_info.lfname;
#endif
			TCHAR path[_MAX_LFN];
			if(0 == level)
				snprintf(&path[0], sizeof(path), "%s", p_file_name);
			else
				snprintf(&path[0], sizeof(path), "%s/%s", p_path, p_file_name);

			app_fatfs_print_folder_content(&path[0], level + 1);
		}
	}
	f_closedir(&dir);
	return fres;
}

/**********************************************************************/

void app_fatfs_print_all_files_attribute(void)
{
	app_fatfs_print_folder_content("", 0);
}

/**********************************************************************/

#if(0)
/**********************************************************************/

static FRESULT app_fatfs_access_each_file_recursively_level(const TCHAR* const p_path, const int level,
	int (*handle_file_func)(const FILINFO* const p_file_info, const TCHAR* const p_path, const int dir_level, 
		const TCHAR* const p_file_name, const UINT file_name_size, void* p_params),
	void* p_params)
{
	//PRINTF("__FUNCTION__ = %s\r\n", __FUNCTION__ );
	FRESULT fres;
	DIR dir;
	fres = f_opendir(&dir, p_path);
	if (FR_OK != fres)
	{
		PRINTF("f_opendir error (%d)\r\n", fres);
		return fres;
	}

	while (1)
	{
		FILINFO file_info;
#if _USE_LFN
		TCHAR lfn[_MAX_LFN];
		file_info.lfname = lfn;
		file_info.lfsize = sizeof(lfn);
#endif
		FRESULT readdir_fres = f_readdir(&dir, &file_info);
		if ((FR_OK != readdir_fres) || (0 == file_info.fname[0]))
		{
			break;
		}

		TCHAR *p_file_name = &file_info.fname[0];
		UINT file_name_size = strlen(&file_info.fname[0]);
#if _USE_LFN
		if(0 != file_info.lfname[0])
		{
			p_file_name = file_info.lfname;
			file_name_size = strlen(p_file_name);
		}
#endif

		if (0 < handle_file_func(&file_info, p_path, level, p_file_name, file_name_size, p_params))
		{
			TCHAR path[_MAX_LFN];
			if(0 == level)
				snprintf(&path[0], sizeof(path), "%s", p_file_name);
			else
				snprintf(&path[0], sizeof(path), "%s/%s", p_path, p_file_name);

			app_fatfs_access_each_file_recursively_level(&path[0], level + 1, handle_file_func, p_params);
		}
	}
	f_closedir(&dir);
	return fres;
}

/**********************************************************************/

FRESULT app_fatfs_access_each_file_recursively(const TCHAR* p_path,
	int (*handle_file_func)(const FILINFO* const p_file_info, const TCHAR* const p_path, const int dir_level, 
		const TCHAR* const p_file_name, const UINT file_name_size, void* p_params),
	void* p_params)
{
	return app_fatfs_access_each_file_recursively_level(p_path, 0, handle_file_func, p_params);
}

/**********************************************************************/

static int print_file_attribute(const FILINFO* p_file_info, const TCHAR* const p_path, const int dir_level, 
	const TCHAR* const p_file_name, const UINT file_name_size, void* p_params)
{
	UNUSED(p_params);

	char string_buffer[_MAX_LFN + 32];
	char string_appendix[64] = { 0 };
	int ret = 0;

	if (AM_DIR & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0], sizeof(string_appendix), "%s", "Dir");
		ret = 1;
	}
	else
	{
		snprintf(&string_appendix[0], sizeof(string_appendix), "%10d bytes", (int)(p_file_info->fsize));
	}

	if (AM_RDO & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", R-ONLY");
	}

	if (AM_SYS & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", SYS");
	}

	if (AM_HID & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", HIDDEN");
	}

	if (AM_ARC & p_file_info->fattrib)
	{
		snprintf(&string_appendix[0] + strlen(&string_appendix[0]),
			sizeof(string_appendix) - strlen(&string_appendix[0]), "%s", ", ARCHIVE");
	}

	struct tm time_info = app_fatfs_convert_FILINFO_to_tm(p_file_info);

	char string_time_info[16];
	strftime(&string_time_info[0], sizeof(string_time_info), "%y%m%d-%H:%M:%S", &time_info);

	snprintf(&string_buffer[0], sizeof(string_buffer), "%*s%.*s %*s %s %s",
		dir_level * 4, "",
		file_name_size, p_file_name,
		12, "",
		&string_time_info[0],
		&string_appendix[0]);

	PRINTF("%s\r\n", &string_buffer[0]);
	return ret;
}

void app_fatfs_print_all_files_attribute(void)
{
	app_fatfs_access_each_file_recursively("", print_file_attribute, NULL);
}

/**********************************************************************/
#endif

