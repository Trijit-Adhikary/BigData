--Query

select u.name, COALESCE(r.total_dist,0) from Users u
left join
(select user_id, sum(distance) as total_dist from Rides group by user_id ) r
on u.id = r.user_id
order by r.total_dist desc,u.name;



--Environment

create table Users(
    id int,
name varchar(100)
);

create table Rides(
    id int,
user_id int,
distance int
);

insert into Users VALUES (1,"Alice"),
(2,"Bob"),
(3,"Alex"),
(4,"Donald"),
(7,"Lee"),
(13,"Jonathan"),
(19,"Elvis");

insert into Rides VALUES (1,1,120),
(2,2,317),
(3,3,222),
(4,7,100),
(5,13,312),
(6,19,50),
(7,7,120),
(8,19,400),
(9,7,230);