-- Query

select project_id, employee_id from 
(select p.project_id, e.employee_id, DENSE_RANK() over(PARTITION BY p.project_id order by e.experience_years desc) as dns_rnk from Project p
join Employee e on p.employee_id = e.employee_id ) a
where a.dns_rnk = 1;



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

insert into Project VALUES (1,1),
(1,2),
(1,3),
(2,1),
(2,4);

insert into Employee VALUES (1,"Khaled",3),
(2,"Ali",2),
(3,"John",3),
(4,"Doe",2);