CREATE TABLE IF NOT EXISTS sandbox.load
(
   date_created       timestamp   DEFAULT now(),
   load_id            uuid        DEFAULT gen_random_uuid(),
   file_location      text,
   file_name          text,
   sheet_index        integer	 DEFAULT 0,
   section_name       text,
   destination_table  text,
   destination_schema text,
   details_notes      text,
   approved_customer  boolean,
   total_rows         integer,
   file_group         text,
   plan_load_date	  timestamp,
   data_load_id       integer
);


CREATE TABLE IF NOT EXISTS sandbox.load_status
(
   load_id       uuid,
   schema_title  text,
   action_type   text,
   action_date   timestamp
);

-- this part is also in create_new_files.sql
CREATE TABLE IF NOT EXISTS sandbox.new_files
(
   file_location  text,
   destination_table text,
   destination_schema text,
   plan_load_date	  timestamp,
   date_created DATE
);
