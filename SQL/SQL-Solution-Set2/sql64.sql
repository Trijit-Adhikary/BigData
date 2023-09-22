-- Query

select p.project_id, round(avg(e.experience_years),2) as average_years
from Employee e join Project p
on e.employee_id = p.employee_id
group by p.project_id;


-- Environment

create table Project(
    project_id int,
employee_id int
);

create table Employee(
    employee_id int,
name varchar(100),
experience_years int
);

insert into Project values (1,1),
(1,2),
(1,3),
(2,1),
(2,4);

insert into Employee values (1,"Khaled",3),
(2,"Ali",2),
(3,"John",1),
(4,"Doe",2);