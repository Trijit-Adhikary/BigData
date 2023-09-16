-- Query

select id,name from Students where department_id not in (select distinct id from Departments);




-- Environment

create table Departments(
    id int,
name varchar(100)
);


create table Students(
    id int,
name varchar(100),
department_id int
);

insert into Departments VALUES (1,"Electrical Engineering"),
(7,"Computer Engineering"),
(13,"Business Administration");

insert into Students VALUES (23,"Alice",1),
(1,"Bob",7),
(5,"Jennifer",13),
(2,"John",14),
(4,"Jasmine",77),
(3,"Steve",74),
(6,"Luis",1),
(8,"Jonathan",7),
(7,"Daiana",33),
(11,"Madelynn",1);