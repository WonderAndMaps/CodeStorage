# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:13:01 2018

@author: yang
"""

import datetime
import glob
import os
import sys
import luigi
import local_settings as settings

import statecrawler.postgres.objects_to_sql as objects_to_sql
import statecrawler.postgres.pg_utils as pg_utils
from Penstock.PenstockBase import PenstockBase


# Populates the load tracking and load_status table. The code scans a target folder, adds those files to a table called "new_files",
# then adds those files to load_tracking which aren't in the load_tracking table yet.
#
# Use tracking. schema instead of load. Table names are now:
# tracking.load
# tracking.load_status
# tracking.new_files
#

class PopulateLoadTracking(PenstockBase):
    target_date = luigi.DateHourParameter()
    pager_alerts = luigi.BoolParameter(default=False)
    slack_alert_channel = luigi.BoolParameter(default=False)

    # Possibilities are currently magpie, horizon, nightheron, premium
    which_server = luigi.Parameter(default="nightheron")
    db_type = luigi.Parameter(default='postgres')
    scan_location = luigi.Parameter(default=None)
    destination_table = luigi.Parameter()
    schema = luigi.Parameter(default=None)
    
    # Newly added parameters goes into new_files and load table
    plan_load_date = luigi.Parameter(default=(datetime.datetime.today()).strftime("%Y-%m-%d")) 

    def __init__(self, *args, **kwargs):
        super(PopulateLoadTracking, self).__init__(*args, **kwargs)

        server_links = {#"magpie": settings.HORIZON_STRING, #Yang has no password for these two
                        #'horizon':settings.HORIZON_STRING,
                        "nightheron": settings.NIGHTHERON_STRING,
                        "premium": settings.POSTGRES_STRING}

        conn_string = server_links[self.which_server]
        self.conn = pg_utils.open_pg(conn_string, True)
        

        if 'scan_location' in kwargs:
            full_path = os.path.join(os.path.expanduser("~"), kwargs['scan_location'])
            self.upload_list = glob.glob(full_path + "/*.csv") + glob.glob(full_path +"/*.xlsx")

            # TODO - Make more robust file filter options
            self.upload_list = [x for x in self.upload_list if self.file_filter(x)]

            # Remove user home directory
            self.upload_list = [x.split(os.path.expanduser("~") + "/")[-1] for x in self.upload_list]
        else:
            raise Exception("No folder location was sent to scan")
            
        # If plan_load_date is in the past, make it today
        if 'plan_load_date' in kwargs:
            try:
                timedelta = datetime.datetime.strptime(self.plan_load_date,"%Y-%m-%d") - datetime.datetime.today()
                if timedelta.days < 0:
                    self.plan_load_date = (datetime.datetime.today()).strftime("%Y-%m-%d")
            except Exception as e:
                print('%s may be a wrong date format.' % self.plan_load_date)
                raise e

    # Filter for files to be uploaded. WIP on implementing robust way
    @staticmethod
    def file_filter(path):
        # return not ('directional' in path.lower().split("/")[-1].lower() or 'survey' in path.split("/")[-1].lower())
        return True

    def run(self):
        os.chdir(self.code_root)

        # Init tables using generic sql script
        try:
            # Some changes are made in load_tracking_init.sql to use tracking schema
            with open("magpie/loading_utils/sql_scripts/load_tracking_init.sql", "r") as f:
                load_sql = f.read()
                
            # TODO: delete this after testing
            #testing
            load_sql = load_sql.replace('tracking.load','sandbox.load')
            load_sql = load_sql.replace('tracking.new_files','sandbox.new_files')
            load_sql = load_sql.replace('tracking.load_status','sandbox.load_status')
            
            cur1 = self.conn.cursor()
            cur1.execute(load_sql)
            cur1.close()
            self.conn.commit()
        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime,
                                             'Creating load_tracking',
                                             description='Failure initializing tables. Msg: %s' % e.message, rows=0)
            raise e

        # Reset new files table
        try:
            
            # TODO - Is this still needed? Taking plan_load_date into account.
            cur2 = self.conn.cursor()
            load_sql = """DELETE FROM tracking.new_files WHERE date_created < %s """% ("'"+(datetime.datetime.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")+"'")
            
            # TODO: delete this after testing
            #testing
            load_sql = load_sql.replace('tracking.load','sandbox.load')
            load_sql = load_sql.replace('tracking.new_files','sandbox.new_files')
            load_sql = load_sql.replace('tracking.load_status','sandbox.load_status')
            
            
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
            # TODO: change the 'sandbox' to 'tracking' after testing
            ots.open(self.conn, cur2, "sandbox.new_files",
                     'magpie/loading_utils/sql_scripts/create_new_files.sql')
            

            for f in self.upload_list:
                ots.write({'date_created': datetime.datetime.today(),
                           'file_location': f,
                           'destination_table': self.destination_table,
                           'destination_schema': self.schema,
                           'plan_load_date': self.plan_load_date})

            ots.close()
            self.conn.commit()

        except Exception as e:
            sys.stderr.write("Exception: %s" % e)
            if self.pager_alerts:
                PenstockBase.pagerduty_alert('%s Loads' % self.which_server, datetime, datetime, 'Loading new files',
                                             description='Failure. Msg: %s' % e.message, rows=0)
            raise e

        # Do a sql join on tracking.load to add these new files
        # Exclude those with same file_location and same plan_load_date
        try:
            cur3 = self.conn.cursor()
            load_sql = """INSERT INTO tracking.load (file_location, destination_table, destination_schema, plan_load_date)
                (SELECT DISTINCT n.file_location, n.destination_table, n.destination_schema, n.plan_load_date FROM tracking.new_files n 
                LEFT JOIN tracking.load l 
                ON n.file_location = l.file_location
                WHERE l.file_location IS NULL)
                """
                            
            # TODO: delete this after testing
            #testing
            load_sql = load_sql.replace('tracking.load','sandbox.load')
            load_sql = load_sql.replace('tracking.new_files','sandbox.new_files')
            load_sql = load_sql.replace('tracking.load_status','sandbox.load_status')
            
            
            
            
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
                ((SELECT DISTINCT * FROM tracking.new_files WHERE plan_load_date <= '%s') n 
                LEFT JOIN tracking.load l ON n.file_location = l.file_location) t 
                LEFT JOIN 
                (SELECT * FROM tracking.load_status WHERE action_type='loaded') s 
                on t.load_id = s.load_id
                WHERE s.load_id is NULL"""%datetime.datetime.today().strftime("%Y-%m-%d")
                            
            # TODO: delete this after testing            
            #testing
            load_sql = load_sql.replace('tracking.load','sandbox.load')
            load_sql = load_sql.replace('tracking.new_files','sandbox.new_files')
            load_sql = load_sql.replace('tracking.load_status','sandbox.load_status')
            
            
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

        # TODO - Can someone let me know if this is how luigi works?
        # write sentinel file to show success to luigi
        with self.output().open('w') as out_file:
            out_file.write("1")

# Example
if __name__ == '__main__':    
    luigi.run(['PopulateLoadTracking', 
               '--destination-table', 'yang_messing_around',
               '--schema', 'sandbox',
               '--scan-location','C:/Users/dell/Desktop/6-18-18/',
               '--which-server', 'premium',
               '--target-date', '2017-11-08T10',
               '--plan-load-date', '2018-07-03',
               '--local-scheduler'])
    
