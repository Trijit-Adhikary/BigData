-- Query

select round(count(*) / (select count(distinct player_id) from Activity),2) as fraction
from
(select *, lead(event_date,1) over(PARTITION BY player_id order by event_date) as next_log_in
from Activity ) a
where DATEDIFF(next_log_in,event_date) = 1;

-- alternate approach
select round(count(distinct player_id) / (select count(distinct player_id) from Activity),2) as fraction
from
(select *,
        count(*) over(partition by player_id order by event_date 
                        range BETWEEN interval '1' day preceding 
                        and current row) as day_count
    from Activity ) a
where day_count = 2;




-- Environment

create table Activity(
    player_id int,
device_id int,
event_date date,
games_played int
);

INSERT into Activity VALUES (1,2,"2016-03-01",5),
(1,2,"2016-03-02",6),
(2,3,"2017-06-25",1),
(3,1,"2016-03-02",0),
(3,4,"2018-07-03",5);