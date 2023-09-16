-- Query

select player_id, min(event_date) as first_login from Activity GROUP BY player_id;


-- Environment

create table Activity(
    player_id int,
device_id int,
event_date date,
games_played int
);

insert into Activity VALUES (1,2,"2016-03-01",5),
(1,2,"2016-05-02",6),
(2,3,"2017-06-25",1),
(3,1,"2016-03-02",0),
(3,4,"2018-07-03",5);