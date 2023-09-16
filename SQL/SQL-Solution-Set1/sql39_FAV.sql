-- Query

select left(grp_key,1) as person1, right(grp_key,1) as person2, call_count, total_duration
from 
(select grp_key, count(*) as call_count, sum(duration) as total_duration 
from (select *,
    CASE 
        WHEN from_id <  to_id THEN  concat(CONVERT(from_id,CHAR),"-",CONVERT(to_id,CHAR))
        ELSE  concat(CONVERT(to_id,CHAR),"-",CONVERT(from_id,CHAR))
    END grp_key
from Calls ) tmp
GROUP BY grp_key ) final;




-- Environment

create table Calls(
    from_id int,
to_id int,
duration int
);


insert into Calls values (1,2,59),
(2,1,11),
(1,3,20),
(3,4,100),
(3,4,200),
(3,4,200),
(4,3,499);