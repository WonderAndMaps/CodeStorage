CREATE TABLE IF NOT EXISTS sandbox.new_files
(
   file_location  text,
   destination_table text,
   destination_schema text,
   plan_load_date	  timestamp,
   date_created DATE
);
