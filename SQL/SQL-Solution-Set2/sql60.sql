-- Query

select *,
    Case
        When (x+y) > z and (y+z) > x and (x+z) > y THEN 'Yes'
        Else 'No'
    End as triangle
from Triangle;



-- Environment

create table Triangle(
    x int,
y int,
z int
);

insert into Triangle values (13,15,30),
(10,20,15);