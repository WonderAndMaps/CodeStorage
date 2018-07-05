# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:23:31 2018

@author: yang
"""
import datetime
import os
import sys
import luigi
from luigi.util import requires

import local_settings as settings
import statecrawler.postgres.objects_to_sql as objects_to_sql
import statecrawler.postgres.pg_utils as pg_utils
from Penstock.PenstockBase import PenstockBase
from magpie.loading_utils.InsertCSV import InsertCSV
from magpie.loading_utils.CreateRawSql import CreateRawSql
from magpie.loading_utils.PopulateLoadTracking import PopulateLoadTracking


# Data loader to use with load tracking and load_status tables in the load table.  Does not need to
# specify the file location or table - that should be done on the script that populates load tracking in this folder.
# The targeted loading table can be specified so this only loads to one table
#
# The default method is used for whatever you're loading
# Can automatically create the .sql file for creating tables now
#

@requires(PopulateLoadTracking)
class DataLoader(PenstockBase):
    target_date = luigi.DateHourParameter()
    db_type = luigi.Parameter(default='postgres')
    pager_alerts = luigi.BoolParameter(default=False)
    slack_alert_channel = luigi.BoolParameter(default=False)
    scan_location = luigi.Parameter(default=None)
    destination_table = luigi.Parameter()
    schema = luigi.Parameter(default=None)

    # Specify the table_name to target only the data that needs to be loaded to that table
    table_name = luigi.Parameter(default=False)

    # Possibilities are currently magpie, horizon, nightheron, premium
    which_server = luigi.Parameter(default="nightheron")

    # Newly added
    transaction_size = luigi.Parameter(default="25")
    plan_load_date = luigi.Parameter(default=(datetime.datetime.today()).strftime("%Y-%m-%d"))
    
    # Cannot have these if use @requires. Not sure why.
    #def __init__(self, *args, **kwargs):
    #    super(DataLoader, self).__init__(*args, **kwargs)
    #
    #    server_links = {#"magpie": settings.HORIZON_STRING, # Yang has no password for these
    #                    #"horizon":settings.HORIZON_STRING,
    #                    "nightheron": settings.NIGHTHERON_STRING,
    #                    "premium": settings.POSTGRES_STRING}
    #
    #    conn_string = server_links[self.which_server]
    #    self.conn = pg_utils.open_pg(conn_string, True)
    #
    #    # Add additional filtering sql if specified to work on only one table
    #    if self.table_name:
    #        self.additional_sql = "AND t.destination_table = '%s' "%self.table_name
    #    else:
    #        self.additional_sql = ""

    def output(self):
        """ Customized output that includes parameters """
        return self.output_variant("_".join([self.db_type, self.which_server]))

    def run(self):
        server_links = {#"magpie": settings.HORIZON_STRING, # Yang has no password for these
                        #"horizon":settings.HORIZON_STRING,
                        "nightheron": settings.NIGHTHERON_STRING,
                        "premium": settings.POSTGRES_STRING}

        conn_string = server_links[self.which_server]
        self.conn = pg_utils.open_pg(conn_string, True)

        # Add additional filtering sql if specified to work on only one table
        if self.table_name:
            self.additional_sql = "AND t.destination_table = '%s' "%self.table_name
        else:
            self.additional_sql = ""
            
            
            
        os.chdir(self.code_root)

        # Retrieve number of files to load. We do this so we can space out the commits every transaction_size files
        try:
            cur1 = self.conn.cursor()
            sql_count = """
            SELECT count(1) FROM tracking.load t 
                                LEFT JOIN (SELECT load_id FROM tracking.load_status WHERE action_type = 'loaded') s
                                ON t.load_id = s.load_id
                                WHERE s.load_id IS NULL %s AND t.plan_load_date <= '%s'"""%(self.additional_sql,datetime.datetime.today().strftime("%Y-%m-%d"))
            
            # TODO: delete this after testing
            #testing
            sql_count = sql_count.replace('tracking.load','sandbox.load')
            sql_count = sql_count.replace('tracking.new_files','sandbox.new_files')
            sql_count = sql_count.replace('tracking.load_status','sandbox.load_status')
                                
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
                                FROM tracking.load t 
                                LEFT JOIN (SELECT load_id FROM tracking.load_status WHERE action_type = 'loaded') s
                                ON t.load_id = s.load_id
                                WHERE s.load_id IS NULL %s AND t.plan_load_date <= '%s'
                                ORDER BY date_created DESC LIMIT %s
                        ---select files that haven't been loaded per the status table order by date if large number of files """%(self.additional_sql,datetime.datetime.today().strftime("%Y-%m-%d"),self.transaction_size)
                
                # TODO: delete this after testing
                #testing
                load_sql = load_sql.replace('tracking.load','sandbox.load')
                load_sql = load_sql.replace('tracking.new_files','sandbox.new_files')
                load_sql = load_sql.replace('tracking.load_status','sandbox.load_status')
                
                
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
                
                # Generate .sql file
                # Check if table already exist
                try:
                    cur3 = self.conn.cursor()
                    sql_count = """SELECT count(1) FROM information_schema.tables
                                   WHERE table_schema='%s' and table_name='%s'"""%(destination_schema,destination_table)
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
                file_location = os.path.join(os.path.expanduser("~"), file_location)
                csv_inserter = InsertCSV()
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

                # Write success to load status table here
                try:
                    cur4 = self.conn.cursor()
                    insert_sql = """INSERT INTO tracking.load_status(load_id, schema_title, action_type, action_date)
                          VALUES('%(load_id)s', 'SCHEMA', 'loaded', '%(target_date)s')""" % {
                        "target_date": datetime.datetime.utcnow(),
                        "load_id": load_id}
                    
                    # TODO: delete this after testing
                    #testing
                    insert_sql = insert_sql.replace('tracking.load','sandbox.load')
                    insert_sql = insert_sql.replace('tracking.new_files','sandbox.new_files')
                    insert_sql = insert_sql.replace('tracking.load_status','sandbox.load_status')
                          
                    insert_sql = insert_sql.replace("SCHEMA", str(destination_schema))
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

            # commit transaction as you go for each transaction_size files
            for ots in ots_dict:
                ots_dict[ots].close()

            self.conn.commit()
            print("Commited %d files" % file_loaded_counter)
            file_count -= file_loaded_counter

        self.conn.close()

        if self.slack_alert_channel:
            self.slack_alert(description='Files Loaded %s' % num_files, channel=self.slack_alert_channel)

        # TODO - Can someone let me know if this is how luigi works?
        # write sentinel file to show success to luigi
        with self.output().open('w') as out_file:
            out_file.write("1")


# Example
if __name__ == '__main__':
    luigi.run(['DataLoader', '--local-scheduler', 
              '--table-name', 'yang_messing_around',
              '--which-server', 'premium',
              '--target-date', '2017-11-08T07',
              '--transaction-size', '25'])