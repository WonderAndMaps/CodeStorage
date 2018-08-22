# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:15:04 2018

@author: yang
"""
import datetime
import glob
import os
import sys
import luigi
import fnmatch


import local_settings as settings
import statecrawler.postgres.objects_to_sql as objects_to_sql
from Penstock.PenstockBase import PenstockBase
import statecrawler.postgres.pg_utils as pg_utils
from magpie.loading_utils.InsertCSV import InsertCSV
from magpie.loading_utils.CreateRawSql import CreateRawSql



# DataPipe is just glueing DataLoader and PopulateLoadTracking together
# Necessary changes are made in DataLoader.py in order to use @requires
#

class DataPipe(PenstockBase):
    
    #target_date = luigi.DateHourParameter()
    target_date = datetime.datetime.now()
    db_type = luigi.Parameter(default='postgres')
    which_server = luigi.Parameter(default='nightheron')
    pager_alerts = luigi.BoolParameter(default=False)
    slack_alert_channel = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(DataPipe, self).__init__(*args, **kwargs)

        server_links = {"magpie": settings.HORIZON_STRING, #Yang has no password for these two
                        "horizon":settings.HORIZON_STRING,
                        "nightheron": settings.NIGHTHERON_STRING,
                        "premium": settings.POSTGRES_STRING}

        conn_string = server_links[self.which_server]
        self.conn = pg_utils.open_pg(conn_string, True)
    
    
    def run(self):       
        # Init job tables using generic sql script
        os.chdir(self.code_root)
                
        try:
            with open("magpie/loading_utils/sql_scripts/load_job_init.sql", "r") as f:
                load_sql = f.read()
            
            load_sql = load_sql.replace('SCHEMA','tracking')
            cur0 = self.conn.cursor()
            cur0.execute(load_sql)
            cur0.close()
            self.conn.commit()
        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'Creating tracking.load',
                                             description='Failure initializing tables. Msg: %s' % e.message, rows=0)
            raise e     
        
        # Get job details
        try:
            cur1 = self.conn.cursor()
            load_sql = """
                        SELECT 
                            job_name,
                            scan_location,
                            destination_table,
                            schema,
                            transaction_size,
                            schema_tracking_tables,
                            table_name,
                            tracking_table,
                            status_table,
                            action_date,
                            load_interval
                        FROM 
                        (SELECT *,ROW_NUMBER() OVER(PARTITION BY job_name ORDER BY action_date DESC) from SCHEMA.load_job) a
                        WHERE row_number=1 
                            AND ((action!='done' or action is null OR action_date + load_interval <= TIMEZONE('UTC', STATEMENT_TIMESTAMP())) 
                                OR (action!='done' or action is null AND load_interval IS NULL))
                        """
            
            # TODO - parameterize this
            load_sql = load_sql.replace('SCHEMA','tracking')
            cur1.execute(load_sql)
            job_details = cur1.fetchall()
            cur1.close()                        
 
            num_jobs = len(job_details)
            print("%d load jobs to do"% num_jobs)
            self.conn.commit()
             
        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'Grabbing load job details',
                                             description='Failure Grabbing load job details. Msg: %s' % e.message, rows=0)
            raise e

        job_count = 0
        for job_name, scan_location, destination_table, schema, transaction_size, schema_tracking_tables, table_name, tracking_table, status_table, action_date, load_interval in job_details:
            # Populate load tracking for each job
            self.populateLoadTracking(scan_location=scan_location,
                                      destination_table=destination_table,
                                      schema=schema,
                                      transaction_size=transaction_size,
                                      schema_tracking_tables=schema_tracking_tables,
                                      table_name=table_name,
                                      tracking_table=tracking_table,
                                      status_table=status_table)
            
            #print(scan_location)

            
            # Run data loader for each job
            self.dataLoader(scan_location=scan_location,
                            destination_table=destination_table,
                            schema=schema,
                            transaction_size=transaction_size,
                            schema_tracking_tables=schema_tracking_tables,
                            table_name=table_name,
                            tracking_table=tracking_table,
                            status_table=status_table)

            
            
            # Write success
            # TODO - make this pretty
            try:
                cur2 = self.conn.cursor()
                insert_sql = """
                            INSERT INTO SCHEMA.load_job( 
                                job_name,
                                scan_location,
                                destination_table,
                                schema,
                                transaction_size,
                                schema_tracking_tables,
                                table_name,
                                tracking_table,
                                status_table,
                                load_interval,
                                action,
                                created_by)
                            VALUES (
                                %(job_name)s,
                                %(scan_location)s,
                                %(destination_table)s,
                                %(schema)s,
                                %(transaction_size)s,
                                %(schema_tracking_tables)s,
                                %(table_name)s,
                                %(tracking_table)s,
                                %(status_table)s,
                                %(load_interval)s,
                                'done',
                                'bot')
                            """ % {"job_name":self.clean_sql_insert(job_name),
                                "scan_location":self.clean_sql_insert(scan_location),
                                "destination_table":self.clean_sql_insert(destination_table),
                                "schema":self.clean_sql_insert(schema),
                                "transaction_size":self.clean_sql_insert(transaction_size),
                                "schema_tracking_tables":self.clean_sql_insert(schema_tracking_tables),
                                "table_name":self.clean_sql_insert(table_name),
                                "tracking_table":self.clean_sql_insert(tracking_table),
                                "status_table":self.clean_sql_insert(status_table),
                                "load_interval":self.clean_sql_insert(load_interval,interval=True)}
            
                # TODO - parameterize this
                insert_sql = insert_sql.replace('SCHEMA','tracking')
                cur2.execute(insert_sql)
                cur2.close()
             
            except Exception as e:
                sys.stderr.write("Exception: %s" % e)
                if self.pager_alerts:
                    PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                 'Updating load job details',
                                                 description='Failure updating load job details. Msg: %s' % e.message, rows=0)
                raise e
                
            job_count += 1
            print('%d/%d load job done'%(job_count,num_jobs))

            
        self.conn.commit()
        self.conn.close()
        print('All load jobs done!')
        with self.output().open('w') as out_file:
            out_file.write("1")
    
    
    @staticmethod
    def replace_caps(sql_string, **kwargs):
        return sql_string.replace('SCHEMA',
                                  kwargs['schema_tracking_tables']).replace('TRACKING_TABLE',
                                  kwargs['tracking_table']).replace("STATUS_TABLE",
                                  kwargs['status_table'])


    def output(self):
        """ Customized output that includes parameters """
        return self.output_variant("_".join([self.db_type, self.which_server, datetime.datetime.now().strftime('%Y%m%d%H%M%S')]))


    def populateLoadTracking(self, **kwargs):
        """
        A non-luigi version of PopulateLoadTracking.py that shares the same parameters.
        """
        # Check parameters. Some have default value in the sql table.
        if kwargs['destination_table'] is None or kwargs['destination_table']=='':
            raise Exception("No destination to load")
            
        if kwargs['schema'] is None or kwargs['schema']=='':
            print('No schema name specified')
            kwargs['schema'] = None
            
        if kwargs['table_name'] is None or kwargs['table_name']=='':
            kwargs['table_name'] = False        
        
        # Grab files to load
        if kwargs['scan_location'] is not None and kwargs['scan_location']!='':
            #full_path = os.path.join(os.path.expanduser("~"), kwargs['scan_location'])
            full_path = kwargs['scan_location']
            
            upload_list = self.recursiveSearchFiles(full_path,'*.csv') + self.recursiveSearchFiles(full_path,'*.xlsx')
            #upload_list = glob.glob(full_path + "/*.csv") + glob.glob(full_path +"/*.xlsx")

            # TODO - Make more robust file filter options
            upload_list = [x for x in upload_list if self.file_filter(x)]

            # Remove user home directory
            upload_list = [x.split(os.path.expanduser("~") + "/")[-1] for x in upload_list]
                    
        else:
            raise Exception("No folder location was sent to scan")
        
        
        # Init tables using generic sql script
        try:
            with open("magpie/loading_utils/sql_scripts/load_tracking_init.sql", "r") as f:
                load_sql = f.read()
            
            with open("magpie/loading_utils/sql_scripts/load_status_init.sql", "r") as f:
                load_sql += f.read()
                
            with open("magpie/loading_utils/sql_scripts/create_new_files.sql", "r") as f:
                load_sql += f.read()
            
            load_sql = self.replace_caps(load_sql,
                                         schema_tracking_tables=kwargs['schema_tracking_tables'],
                                         tracking_table=kwargs['tracking_table'],
                                         status_table=kwargs['status_table'])
            
            # Change new_files to new_files_jobname
            load_sql = load_sql.replace('new_files','new_files_%s' % kwargs['destination_table'])
            
            cur1 = self.conn.cursor()
            cur1.execute(load_sql)
            cur1.close()
            self.conn.commit()
        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'Creating tracking.load',
                                             description='Failure initializing tables. Msg: %s' % e.message, rows=0)
            raise e

        # Reset new files table
        try:
            
            cur2 = self.conn.cursor()
            load_sql = """DELETE FROM SCHEMA.new_files WHERE date_created < %s """% ("'"+(datetime.datetime.utcnow() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")+"'")
            
            load_sql = self.replace_caps(load_sql,
                                         schema_tracking_tables=kwargs['schema_tracking_tables'],
                                         tracking_table=kwargs['tracking_table'],
                                         status_table=kwargs['status_table'])
            load_sql = load_sql.replace('new_files','new_files_%s'%kwargs['destination_table'])
            
            cur2.execute(load_sql)
            cur2.close()
            self.conn.commit()
        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'Deleting day old rows from new_files table',
                                             description='Failure   Msg: %s' % e.message, rows=0)
            raise e

        try:
            cur2 = self.conn.cursor()
            ots = objects_to_sql.objects_to_sql()
            # don't force to upper case
            ots.upcase = False

            # Setup table writes and upload through loop using ots
            with open("magpie/loading_utils/sql_scripts/create_new_files.sql", "r") as f:
                load_sql = f.read()
            load_sql = self.replace_caps(load_sql,
                                         schema_tracking_tables=kwargs['schema_tracking_tables'],
                                         tracking_table=kwargs['tracking_table'],
                                         status_table=kwargs['status_table'])
            load_sql = load_sql.replace('new_files','new_files_%s'%kwargs['destination_table'])

            
            with open("magpie/loading_utils/sql_scripts/create_new_files.sql", "w") as f:
                f.write(load_sql)
                
                
            ots.open(self.conn, cur2, "%s.new_files_%s"%(kwargs['schema_tracking_tables'],kwargs['destination_table']),
                     'magpie/loading_utils/sql_scripts/create_new_files.sql')

            # Change it back to SCHEMA            
            with open("magpie/loading_utils/sql_scripts/create_new_files.sql", "w") as f:
                f.write(load_sql.replace(kwargs['schema_tracking_tables'],
                                         "SCHEMA").replace('new_files_%s'%kwargs['destination_table'],'new_files'))


            for f in upload_list:
                ots.write({'date_created': datetime.datetime.utcnow(),
                           'file_location': f,
                           'destination_table': kwargs['destination_table'],
                           'destination_schema': kwargs['schema']})

            ots.close()
            self.conn.commit()

        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime, 'Loading new files',
                                             description='Failure. Msg: %s' % e.message, rows=0)
            raise e

        # Do a sql join on tracking.load to add these new files
        # Exclude those with same file_location
        try:
            cur3 = self.conn.cursor()
            load_sql = """INSERT INTO SCHEMA.TRACKING_TABLE (file_location, destination_table, destination_schema)
                (SELECT DISTINCT n.file_location, n.destination_table, n.destination_schema FROM SCHEMA.new_files n 
                LEFT JOIN SCHEMA.TRACKING_TABLE l 
                ON n.file_location = l.file_location
                WHERE l.file_location IS NULL)
                """
            
            load_sql = self.replace_caps(load_sql,
                                         schema_tracking_tables=kwargs['schema_tracking_tables'],
                                         tracking_table=kwargs['tracking_table'],
                                         status_table=kwargs['status_table'])
            load_sql = load_sql.replace('new_files','new_files_%s' % kwargs['destination_table'])
            
            cur3.execute(load_sql)
            cur3.close()
            self.conn.commit()
        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'Inserting new_files to tracking table',
                                             description='Failure joining new_files and tracking table. Msg: %s' % e.message,
                                             rows=0)
            raise e

        # Grab count of new files to load
        try:
            cur4 = self.conn.cursor()
            load_sql = """SELECT COUNT(*) FROM 
                ((SELECT DISTINCT * FROM SCHEMA.new_files) n 
                LEFT JOIN SCHEMA.TRACKING_TABLE l ON n.file_location = l.file_location) t 
                LEFT JOIN 
                (SELECT * FROM SCHEMA.STATUS_TABLE WHERE action_type='loaded') s 
                on t.load_id = s.load_id
                WHERE s.load_id is NULL"""
                            
            load_sql = self.replace_caps(load_sql,
                                         schema_tracking_tables=kwargs['schema_tracking_tables'],
                                         tracking_table=kwargs['tracking_table'],
                                         status_table=kwargs['status_table'])
            load_sql = load_sql.replace('new_files','new_files_%s' % kwargs['destination_table'])
            
            cur4.execute(load_sql)
            count = cur4.fetchone()[0]
            print("%d files to load"%count)
            cur4.close()
        except Exception as e:
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'File count',
                                             description='Failure getting new file count. Msg: %s' % e.message,
                                             rows=0)
            raise e

        if self.slack_alert_channel:
            self.slack_alert(description='Check requested: %d files need to be loaded to %s server' %
                                         (count, self.which_server), channel=self.slack_alert_channel)

        return None

    def dataLoader(self, **kwargs):
        """
        A non-luigi version of DataLoader.py that shares the same parameters.
        """
        
        # Check parameters. Some have default value in the sql table.            
        if kwargs['table_name'] is None or kwargs['table_name']=='':
            kwargs['table_name'] = False
            
            
        # Add additional filtering sql if specified to work on only one table
        if kwargs['table_name']:
            additional_sql = "AND t.destination_table = '%s' " % kwargs['table_name']
        else:
            additional_sql = ""
            
            
            
        os.chdir(self.code_root)

        # Retrieve number of files to load. We do this so we can space out the commits every transaction_size files
        try:
            cur1 = self.conn.cursor()
            sql_count = """
                            SELECT count(1) FROM SCHEMA.TRACKING_TABLE t 
                            LEFT JOIN (SELECT load_id FROM SCHEMA.STATUS_TABLE WHERE action_type = 'loaded') s
                            ON t.load_id = s.load_id
                            WHERE s.load_id IS NULL %(add_sql)s and trim(file_location)!='' and file_location is not null
                        """%{
                              "add_sql": additional_sql
                            }
            
            sql_count = self.replace_caps(sql_count,
                                          schema_tracking_tables=kwargs['schema_tracking_tables'],
                                          tracking_table=kwargs['tracking_table'],
                                          status_table=kwargs['status_table'])
                                
            cur1.execute(sql_count)
            file_count = cur1.fetchone()[0]
            num_files = file_count
            cur1.close()
            print("%d files to load"% num_files)
            self.conn.commit()
        except Exception as e:
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'getting number of new files',
                                             description='Failure getting number of new files. Ask Mara or Chris to fix   Msg: %s' % e.message,
                                             rows=0)
            raise e

        # cursor for insertCSV
        cursor = self.conn.cursor()

        # Upload transaction_size files for each commit
        while file_count > 0:
            try:
                # excel/csvs are entered in to the tracking table, which are populated by a job that scans a target folder
                cur2 = self.conn.cursor()
                load_sql = """SELECT t.load_id, file_location, file_name, sheet_index, destination_schema, destination_table, file_group
                                FROM SCHEMA.TRACKING_TABLE t 
                                LEFT JOIN (SELECT load_id FROM SCHEMA.STATUS_TABLE WHERE action_type = 'loaded') s
                                ON t.load_id = s.load_id
                                WHERE s.load_id IS NULL %(add_sql)s
                                ORDER BY date_created DESC LIMIT %(tran_size)s
                        ---select files that haven't been loaded per the status table order by date if large number of files 
                           """%{
                                 "add_sql": additional_sql,
                                 "tran_size": kwargs['transaction_size']
                               }
                
                load_sql = self.replace_caps(load_sql,
                                         schema_tracking_tables=kwargs['schema_tracking_tables'],
                                         tracking_table=kwargs['tracking_table'],
                                         status_table=kwargs['status_table'])
                cur2.execute(load_sql)
                file_details = cur2.fetchall()
                cur2.close()
            except Exception as e:
                if self.pager_alerts:
                    PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                 'getting a list of new files',
                                                 description='Failure getting list of files - There may have been none to grab, ask Mara to fix.  Msg: %s' % e.message,
                                                 rows=0)
                raise e

            # Initialize dictionary of form {destination_table: ots object}
            ots_dict = {}
            file_loaded_counter = 0
            for load_id, file_location, file_name, sheet_index, destination_schema, destination_table, file_group in file_details:
                
                if not file_location:
                    continue
                # Generate .sql file
                # Check if table already exist
                try:                    
                    cur3 = self.conn.cursor()
                    sql_count = """SELECT count(1) FROM information_schema.tables
                                   WHERE table_schema='%(schema)s' and table_name='%(table)s'
                                   """%{
                                         "schema": destination_schema,
                                         "table": destination_table
                                       }
                    cur3.execute(sql_count)
                    table_exist = cur3.fetchone()[0]
                    cur3.close()
                except Exception as e:
                    if self.pager_alerts:
                        PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                     'Table check',
                                                     description='Failure checking if table exists. Msg: %s' % e.message,
                                                     rows=0)
                    raise e
        
                if not table_exist:
                    print('Creating sql files...')
                    try:
                        sql_creator = CreateRawSql(file_location, destination_schema, destination_table, self.which_server)#, root_path='D:/bazean/data-collection/magpie/loading_utils/sql_scripts')
                        sql_creator.run()
                    except Exception as e:
                        if self.pager_alerts:
                            PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                         'Generate sql file',
                                                         description='Failure generating sql file. Msg: %s' % e.message,
                                                         rows=0)
                
                # Initialize ots if this is a new destination table
                if destination_table not in ots_dict:
                    try:
                        # Initialize postgres load
                        ots_dict[destination_table] = objects_to_sql.objects_to_sql()
                        # don't force to upper case
                        ots_dict[destination_table].upcase = False

                        # Setup table writes
                        # TODO - Have central location with all sql creation scripts
                        create_sql = os.path.join("magpie/loading_utils/sql_scripts", self.which_server, destination_schema, destination_table+".sql")
                        full_table_name = ".".join([destination_schema, destination_table])
                        

                        ots_dict[destination_table].open(self.conn, cursor, full_table_name, create_sql)

                    except Exception as e:
                        sys.stderr.write("Exception: %s" % e)
                        if self.pager_alerts:
                            PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                         'Initializing data load',
                                                         description='Failure Initializing OTS. Likely incorrect table or incorrect sql script for table.  Msg: %s' % e.message,
                                                         rows=0)
                        raise e
                # If no destination table in tracking.load we have encountered an error
                elif not destination_table:
                    with Exception("Tracking.load table has no table name for this row") as e:
                        sys.stderr.write("Exception: %s" % e)
                        if self.pager_alerts:
                            PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                        'Initializing data load',
                                                        description='Failure Initializing OTS. Load tracking table has no destination table  Msg: %s' % e.message,
                                                        rows=0)
                    raise e
                # Otherwise, we don't need to do anything
                else:
                    pass


                # Write a single
                #file_location = os.path.join(os.path.expanduser("~"), file_location)
                csv_inserter = InsertCSV(None)
                try:
                    csv_inserter.insert_csv(file_location, ots_dict[destination_table], load_id, sheet_index)
                except Exception as e:
                    sys.stderr.write("Exception: %s" % e)
                    if self.pager_alerts:
                        PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                     'Inserting a csv',
                                                     description='Failure loading filename %s  Msg: %s' % (
                                                     file_location, e.message),
                                                     rows=0)
                    raise e
                    

                # Write success to load status table here
                try:
                    cur4 = self.conn.cursor()
                    insert_sql = """INSERT INTO SCHEMA.STATUS_TABLE(load_id, schema_title, action_type, action_date)
                                    VALUES('%(load_id)s', '%(des_schema)s', 'loaded', '%(target_date)s')
                                 """ % {
                                         "target_date": datetime.datetime.utcnow(),
                                         "load_id": load_id,
                                         "des_schema": str(destination_schema)
                                       }
                    
                    insert_sql = self.replace_caps(insert_sql,
                                                   schema_tracking_tables=kwargs['schema_tracking_tables'],
                                                   tracking_table=kwargs['tracking_table'],
                                                   status_table=kwargs['status_table'])

                    cur4.execute(insert_sql)
                except Exception as e:
                    sys.stderr.write("Exception: %s" % e)
                    if self.pager_alerts:
                        PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                                     'updating status table',
                                                     description='Failure updating table.  Msg: %s' % e.message,
                                                     rows=0)
                    raise e
                file_loaded_counter += 1
                
                # TODO - Make sure this is correct
                # Slack alerts of files with row counts
                if self.slack_alert_channel:
                    self.slack_alert(description='%(count)s rows loaded into %(schema)s.%(table)s' % 
                                     {
                                       "count": str(csv_inserter.count),
                                       "schema": destination_schema,
                                       "table": destination_table
                                     }, channel=self.slack_alert_channel)


            # commit transaction as you go for each transaction_size files
            for ots in ots_dict:
                ots_dict[ots].close()

            self.conn.commit()
            print("Commited %d files" % file_loaded_counter)
            file_count -= file_loaded_counter

        #self.conn.close()

        if self.slack_alert_channel:
            self.slack_alert(description='Files Loaded %s' % num_files, channel=self.slack_alert_channel)

        return None
    
    @staticmethod
    def file_filter(path):
        # return not ('directional' in path.lower().split("/")[-1].lower() or 'survey' in path.split("/")[-1].lower())
        return True
    
    @staticmethod
    def clean_sql_insert(x,time=False,interval=False):
        if time:
            if x is None:
                return """NULL"""
            else:
                return datetime.datetime.strftime(x,'%Y-%m-%d')
            
        elif interval:
            if x is None:
                return """NULL"""
            else:
                return """interval '%s'"""%str(x)
        
        else:
            if x is None:
                return """''"""
            return """'%s'"""%x
        
    def recursiveSearchFiles(self,dirPath, partFileInfo): 
        fileList = []
        pathList = glob.glob(os.path.join('/', dirPath, '*'))
        for mPath in pathList:
            if fnmatch.fnmatch(mPath, partFileInfo):
                fileList.append(mPath)
            elif os.path.isdir(mPath):
                fileList += self.recursiveSearchFiles(mPath, partFileInfo)
            else:
                pass
        return fileList

# Examples        
if __name__ == '__main__':
    #luigi.run(['DataPipe', '--local-scheduler', 
    #           '--which-server', 'premium'])
    pass

    # Or do this in cmd
    # python -m luigi --module DataPipe DataPipe --local-scheduler --which-server premium
    