# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:23:31 2018

@author: yang
"""
import csv
import xlrd
import copy
import mmap
import os


class CreateRawSql:
    '''
    raw sql file will be stored as root_path/destination_schema/destination_table.sql
    '''
    
    def __init__(self, csv_path, destination_schema, destination_table, server, root_path='magpie/loading_utils/sql_scripts'):
        self.csv_path = csv_path
        self.destination_schema = destination_schema
        self.destination_table = destination_table
        self.sql_file_path = os.path.join(root_path, server, destination_schema)
        self.mkdir(self.sql_file_path)
    
    # copied from InsertCSV class with few things added
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
                # still under construction
                # what if the col name is just a number?
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
    def mkdir(path):
        path=path.strip()
        path=path.rstrip("\\")
        path=path.rstrip("/")
 
        isExists=os.path.exists(path)
 

        if not isExists:
            os.makedirs(path)
            #print ('%s created'%path)
            #return True
            return None
        else:
            #print ('%s exists'%path)
            #return False
            return None

    # copied from InsertCSV class with few things added
    def XLSRowReader(self, f, sheet_index=0):
        """
        Returns dicts of rows from an xls sheet, ala CSV.DictReader
        Assumes first row are the column/dict names.
        :param filename: file
        :return: dict of a row, headers, coltypes
        """
        book = xlrd.open_workbook(file_contents=mmap.mmap(f.fileno(), 0,
                                                          access=mmap.ACCESS_READ))
        sheet = book.sheet_by_index(sheet_index)
        headers = [self.clean_string(self.clean_string(str(x))) for x in sheet.row_values(0)]
        
        # TODO: there can be two or more types in a column, try a better solution
        coltypes = [max(set(sheet.col_types(i))) for i in range(len(headers))]
        return headers,coltypes


    # create sql table from the header of one excel/csv file
    def run(self):
        """
        Creates a postgres sql file from the header of one excel/csv file
        Writes out 'raw' sql to create the tables.
        """
        map_excel_types = {
        5: 'text', # type 'error' as string
        4: 'text', # type 'boolean' as string
        3: 'timestamp',
        2: 'float',
        1: 'text',
        0: 'text' # type 'NULL' as string
        }


        with open(self.csv_path, 'rU') as f:
            if self.csv_path.endswith('.xlsx'):
                headers,coltypes = self.XLSRowReader(f)
            if self.csv_path.endswith('.csv'):
                reader = csv.reader(f)
                headers = None
                for count, row in enumerate(reader):
                    if not headers:
                        headers = [self.clean_string(str(x)) for x in row]
                        break
                # using datatype 'string' for csv files
                coltypes = [1] * len(headers)
            # TODO - .json file
            else:
                pass
                
        
        file_path = os.path.join(self.sql_file_path,self.destination_table+'.sql')
        with open(file_path, 'w') as sql_file:
            full_table_name = '%s.%s' % (self.destination_schema, self.destination_table)
            sql_file.write('CREATE TABLE IF NOT EXISTS %s (\n' % full_table_name)
            sql_file.write('%s %s, \n' % ('load_id','uuid'))
            
            count = 0
            for header,coltype in zip(headers,coltypes):
                count += 1
                
                # fix the last row ','
                if count == len(headers):
                    sql_file.write('%s %s \n' % (header,map_excel_types[coltype]))
                    break
                
                sql_file.write('%s %s, \n' % (header,map_excel_types[coltype]))
                # TODO: add date_created (last saved date)?
            sql_file.write('); \n')
            sql_file.write('--ALTER TABLE %s OWNER TO admin;\n' % full_table_name)
            sql_file.write('GRANT TRIGGER, REFERENCES, TRUNCATE, DELETE, UPDATE, INSERT, SELECT ON %s TO cthomas;\n' % full_table_name)
            sql_file.write('GRANT REFERENCES, TRUNCATE, INSERT, SELECT, DELETE, TRIGGER, UPDATE ON %s TO mara;\n' % full_table_name)
            sql_file.write('COMMIT;\n')
            
            
#testing
if __name__ == '__main__':
    csv_path = 'C:/Users/dell/Desktop/6-11-18/Master_Collected_List_20180611 - OIv2.xlsx'
    destination_schema = 'sandbox'
    destination_table = 'yang_messing_around'
    server = 'premium'
    
    # TODO - dont specify root_path, use default
    sql_creator = CreateRawSql(csv_path, destination_schema, destination_table, server, root_path='D:/bazean/data-collection/magpie/loading_utils/sql_scripts')
    
    sql_creator.run()
        
    
