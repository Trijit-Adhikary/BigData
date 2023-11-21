--Query

select gender, day, sum(a.total) over(Partition by gender ORDER BY gender, day) as total
from (select gender, day, sum(score_points) as total from Scores
GROUP BY gender, day
ORDER BY gender, day ) a;



--Environment

create table Scores(
    player_name varchar(100),
gender varchar(3),
day date,
score_points int
);

insert into Scores values ("Aron","F","2020-01-01",17),
("Alice","F","2020-01-07",23),
("Bajrang","M","2020-01-07",7),
("Khali","M","2019-12-25",11),
("Slaman","M","2019-12-30",13),
("Joe","M","2019-12-31",3),
("Jose","M","2019-12-18",2),
("Priya","F","2019-12-31",23),
("Priyanka","F","2019-12-30",17);