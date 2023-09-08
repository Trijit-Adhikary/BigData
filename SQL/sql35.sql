--QUERY

select results from
(select u.name as results from Users u
join
(select user_id, count(*) as review_count from MovieRating GROUP BY user_id ) a
on u.user_id = a.user_id
order by a.review_count desc, u.name limit 1 ) x
Union
select results from
(select m.title as results 
from Movies m
join
(select movie_id, avg(rating) as avg_raiting from MovieRating GROUP BY movie_id ) b
on m.movie_id = b.movie_id
order by b.avg_raiting desc, m.title asc limit 1 ) y;



--Environment

create table Movies(
    movie_id int,
title varchar(100)
);

create table Users(
    user_id int,
name varchar(100)
);

create table MovieRating(
    movie_id int,
user_id int,
rating int,
created_at date
);

insert into Movies VALUES(1,"Avengers"),
(2,"Frozen 2"),
(3,"Joker");


insert into Users VALUES (1,"Daniel"),
(2,"Monica"),
(3,"Maria"),
(4,"James");

insert into MovieRating VALUES(1,1,3,"2020-01-12"),
(1,2,4,"2020-02-11"),
(1,3,2,"2020-02-12"),
(1,4,1,"2020-01-01"),
(2,1,5,"2020-02-17"),
(2,2,2,"2020-02-01"),
(2,3,2,"2020-03-01"),
(3,1,3,"2020-02-22"),
(3,2,4,"2020-02-25");