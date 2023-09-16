-- Query

select d.dept_name, COALESCE(s.cnt, 0) as student_number
from Department d
left join 
(select dept_id, count(*) as cnt from Student
group by dept_id ) s
on d.dept_id = s.dept_id;




-- Environment

create table Student(
    student_id int,
student_name varchar(100),
gender varchar(100),
dept_id int
);

create table Department(
    dept_id int,
dept_name varchar(100)
);


insert into Student VALUES (1,"Jack","M",1),
(2,"Jane","F",1),
(3,"Mark","M",2);

insert into Department VALUES (1,"Engineering"),
(2,"Science"),
(3,"Law");