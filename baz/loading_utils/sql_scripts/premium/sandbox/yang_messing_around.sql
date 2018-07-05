CREATE TABLE IF NOT EXISTS sandbox.yang_messing_around (
load_id uuid, 
pad_number float, 
previous_week text, 
api float, 
ticker text, 
latitude float, 
longitude float, 
first_rig timestamp, 
last_rig timestamp, 
first_pad_rig_date timestamp, 
last_pad_rig_date timestamp, 
planet_url text, 
data_layer_url text, 
wells_pad float, 
pad_grouping_check text, 
pad_grouping_confirmed_0_or_1 float, 
activity_0_or_1 float, 
possible_stage text, 
submitted_for_hd_0_or_1 float, 
hd_photo_stage text, 
hd_photo_date timestamp, 
week_of timestamp, 
date_collected timestamp, 
completed_by text, 
revisit_0_or_1 float, 
target_list_mismatch_w_gmaps_0_or_1 float, 
notes text, 
checked_by text, 
date_checked text, 
error text
); 
--ALTER TABLE sandbox.yang_messing_around OWNER TO admin;
GRANT TRIGGER, REFERENCES, TRUNCATE, DELETE, UPDATE, INSERT, SELECT ON sandbox.yang_messing_around TO cthomas;
GRANT REFERENCES, TRUNCATE, INSERT, SELECT, DELETE, TRIGGER, UPDATE ON sandbox.yang_messing_around TO mara;
COMMIT;
