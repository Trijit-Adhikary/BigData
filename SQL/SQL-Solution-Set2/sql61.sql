-- Query

select min(dist) as shortest from 
(select ABS((a.x - b.x)) as dist from Point as a,Point as b where ABS((a.x - b.x)) <> 0 ) sub_qr;

-- if we arrange the points in ascending order then, we only need to check the difference between concecutive numbers. This will decrease the number of comparison we need to check


-- Environment

create table Point(
    x int
);

insert into Point VALUES (-1),
(0),
(2);