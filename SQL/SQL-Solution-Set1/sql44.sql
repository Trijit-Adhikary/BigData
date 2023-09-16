-- Query

select name from Employee
where id IN
(select distinct managerId from 
(select *, count(managerId) over(PARTITION BY managerId) as prt
from Employee ) a 
where prt >= 5 );




-- Environment

create table Employee(
    id int,
name varchar(100),
department varchar(100),
managerId int
);

insert into  Employee values (101,"John",'A',NULL),
(102,"Dan","A",101),
(103,"James","A",101),
(104,"Amy","A",101),
(105,"Anne","A",101),
(106,"Ron","B",101);