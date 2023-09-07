--Query

select u.unique_id, e.name from Employees e
left join EmployeeUNI u
on e.id = u.id;



--Environment

create table Employees(
    id int,
name varchar(100)
);

create table EmployeeUNI(
    id int,
unique_id int
);

insert into Employees VALUES (1,"Alice"),
(7,"Bob"),
(11,"Meir"),
(90,"Winston"),
(3,"Jonathan");

insert into EmployeeUNI VALUES (3,1),
(11,2),
(90,3);