-- Query

select sum(DATEDIFF(stop_time,start_time)) as total_uptime_days 
from
(select server_id,status_time as stop_time,session_status, lead(status_time,1) over(PARTITION BY server_id ORDER BY status_time desc) as start_time
from server_utilization) start_stop_time
where start_time is not null and session_status like 'stop';



-- Environment

CREATE TABLE server_utilization(
    server_id integer,
status_time timestamp,
session_status VARCHAR(50)
);

INSERT INTO server_utilization VALUES (1, "2022/08/02 10:00:00", "start"),
(1, "2022/08/04 10:00:00", "stop"),
(2, "2022/08/17 10:00:00", "start"),
(2, "2022/08/24 10:00:00", "stop");