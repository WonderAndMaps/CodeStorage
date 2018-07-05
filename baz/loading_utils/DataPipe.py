# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:57:59 2018

@author: yang
"""
import datetime
import luigi
from luigi.util import requires

from magpie.loading_utils.DataLoader import DataLoader
from Penstock.PenstockBase import PenstockBase

import argparse

# DataPipe is just glueing DataLoader and PopulateLoadTracking together
# Necessary changes are made in DataLoader.py in order to use @requires
#

@requires(DataLoader)
class DataPipe(PenstockBase):
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
    
    def run(self):
        print('All tasks done!')
        with self.output().open('w') as out_file:
            out_file.write("1")
    


# Examples        
if __name__ == '__main__':
    #luigi.run(['DataPipe', '--local-scheduler', 
    #           '--target-date', '2018-07-05T01',
    #           '--table-name', 'yang_messing_around',
    #           '--which-server', 'premium',
    #           '--destination-table', 'yang_messing_around',
    #           '--schema', 'sandbox',
    #           '--scan-location','C:/Users/dell/Desktop/6-18-18/',
    #           '--plan-load-date', '2018-07-03',
    #           '--transaction-size', '25'])
    pass

    # Or do this in cmd
    # python -m luigi --module DataPipe DataPipe --target-date 2018-07-05T02 --which-server premium --destination-table yang_messing_around --schema sandbox --scan-location C:/Users/dell/Desktop/6-18-18/ --local-scheduler
    