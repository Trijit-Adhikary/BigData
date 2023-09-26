-- Query

with RECURSIVE emp as
(
    select employee_id from Employees where employee_id =1
    union
    select empm.employee_id from emp e join Employees empm on e.employee_id = empm.manager_id
)
select * from emp where employee_id <> 1;




-- Environment

create table Employees(
    employee_id int,
employee_name varchar(100),
manager_id int
);

insert into Employees VALUES (1,"Boss",1),
(3,"Alice",3),
(2,"Bob",1),
(4,"Daniel",2),
(7,"Luis",4),
(8,"Jhon",3),
(9,"Angela",8),
(77,"Robert",1);