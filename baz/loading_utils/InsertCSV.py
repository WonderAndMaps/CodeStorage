# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:03:46 2018

@author: yang
"""

import copy
import mmap
import xlrd
import csv
import statecrawler.postgres.objects_to_sql as objects_to_sql
#import os
import statecrawler.postgres.pg_utils as pg_utils
import local_settings as settings
import datetime
from itertools import chain


# This class runs a specific insert csv function depending on the incoming table name.
# WARNING - The xlxs reader is exceptionally slow for some reason.
# TODO - Make this more elegant
class InsertCSV:
    def insert_csv(self, csv_path, ots, load_id, sheet_index=0):
        return self.default_insert(csv_path, ots, load_id, sheet_index)

    @staticmethod
    def clean_string(title_string, strip_tbg=True):
        replace_array = [
                ("\xef\xbb\xbf", ""),
                (" ", "_"),
                ("&", "and"),
                ("-", "_"),
                ("@", "at_"),
                ("i1_", ""),
                ("/", "_"),
                ('"', ""),
                ("(","_"),
                (",","_"),
                (")",""),
                ("\n","_"),
                (".","_"),
                ("__", "_"),
                ("+","_plus_")
                ]
        return_string = copy.copy(title_string).lower()
        
        for a, b in replace_array:
            return_string = return_string.replace(a, b)
            
        if return_string.startswith(('1','2','3','4','5','6','7','8','9','0')):
            return_string = 'num_' + return_string
                
        if strip_tbg:
            return return_string.strip().replace("~tbg", "")
        else:
            return return_string.strip()
        
    @staticmethod
    def clean_rig_rows(row_str):
        try:
            dtime = datetime.datetime.strftime(row_str, "%Y-%m-%d %H:%M:%S")
            return dtime
        except:
            return row_str
        
    @staticmethod
    def clean_excel_time(x):
        try:
            time_str = xlrd.xldate_as_datetime(x,0)
            return time_str
        except:
            if x==u'':
                return(None)
            else:
                return(x)
                
    # newly added
    @staticmethod
    def clean_excel_blank(x):
        if x==u'':
            return(None)
        else:
            return(x)
                
    def XLSRowReader(self, f, sheet_index=0):
        """
        Returns dicts of rows from an xls sheet, ala CSV.DictReader
        Assumes first row are the column/dict names.
        :param  f: file_name,
                num_col: int, num of columns wanted. -1 indicates all cols
        :return: dict of a row
        """
        
        book = xlrd.open_workbook(file_contents=mmap.mmap(f.fileno(), 0,
                                                          access=mmap.ACCESS_READ))
        sheet = book.sheet_by_index(sheet_index)
        headers = [self.clean_string(self.clean_string(str(x))) for x in sheet.row_values(0)]       
        
        # TODO: there can be two or more types in a column, need a better solution
        coltypes = [max(set(sheet.col_types(j))) for j in range(sheet.ncols)]
        return_values = ([self.clean_excel_blank(sheet.cell_value(i,j)) if coltypes[j]!=3\
                          else self.clean_excel_time(sheet.cell_value(i,j))\
                          for j in range(sheet.ncols)] for i in range(1, sheet.nrows))
        
        return chain([headers],return_values)

    def default_insert(self, csv_path, ots, load_id, sheet_index=0):
        with open(csv_path, "rU") as f:
            if csv_path.endswith(".xlsx"):
                reader = self.XLSRowReader(f,sheet_index)
            if csv_path.endswith(".csv"):
                reader = csv.reader(f)
            else:
                # TODO - add json file loader
                
                
                pass
                
                
            header = None
            
            for count, row in enumerate(reader):
                if not header:
                    header  = ['load_id'] + [self.clean_string(str(x)) for x in row]
                    
                else:
                    # Avoid blank rows
                    blank = True
                    for column in row:
                        if column:
                            blank = False
                            break
                    if blank:
                        continue

                    row = [load_id] + [self.clean_rig_rows(x) for x in row]

                    object = dict(zip(header, row))
                    ots.write(object)
                    
                    
#-------------------------------------------testing
if __name__ == '__main__':
    csv_path = 'C:/Users/dell/Desktop/6-11-18/Master_Collected_List_20180611 - OIv2.xlsx'                    
    destination_schema = 'target_data'
    destination_table = 'sky_observations'
    which_server = 'nightheron'
    server_links = {"nightheron": settings.NIGHTHERON_STRING,
                    "premium": settings.POSTGRES_STRING}

    # be careful about what's in this .sql
    create_sql = 'C:/Users/dell/Desktop/sky_observations.sql'
    full_table_name = ".".join([destination_schema, destination_table])



    conn = pg_utils.open_pg(server_links[which_server], True)
    cursor = conn.cursor()

    ots = objects_to_sql.objects_to_sql()
    ots.upcase = False
    ots.open(conn, cursor, full_table_name, create_sql)

    insert = InsertCSV()
    insert.insert_csv(csv_path,ots,load_id='b36fae3a-5a37-4865-81ff-b19735d25a99')
    ots.close()

    conn.commit()
    conn.close()
