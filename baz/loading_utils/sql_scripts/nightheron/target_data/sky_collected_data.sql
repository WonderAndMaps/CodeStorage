CREATE TABLE IF NOT EXISTS target_data.sky_collected_data
(
    load_id uuid,
    pad_number int,
    previous_week_activity	text,
    api	VARCHAR(20),
    multiwell_pad_apis  text,
    activity  text,
    possible_stage  text,
    submitted_for_HD  text,
    HD_photo_stage	text,
    week_of	   date,
    date_collected	date,
    completed_by  TEXT,
    daily_coverage  int,
    confidence  int,
    revisit	text,
    notes text,
    checked_by	text,
    date_checked	date,
    ticker	text,
    error   int
    );
