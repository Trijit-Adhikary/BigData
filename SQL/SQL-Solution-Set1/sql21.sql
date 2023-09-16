--Query

select employee_id, count(*) over(partition by team_id) as team_size from Employee order by employee_id;


--Environment

create table Employee(
    employee_id int,
team_id int
);

insert into Employee values (1,8),
(2,8),
(3,8),
(4,7),
(5,9),
(6,9);