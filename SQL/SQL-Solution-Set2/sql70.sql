-- Query

select stu_sub.student_id, stu_sub.student_name, stu_sub.subject_name, COALESCE(cnt,0) as attended_exams 
from 
(select * from Students stu, Subjects sub ) stu_sub
left join
(select *, count(*) as cnt from Examinations group by student_id, subject_name ) cnt_exm
on stu_sub.student_id = cnt_exm.student_id and stu_sub.subject_name = cnt_exm.subject_name
order by stu_sub.student_id,COALESCE(cnt,0) desc;




-- Environment

create table Students(
    student_id int,
student_name varchar(100)
);

create table Subjects(
    subject_name varchar(100)
);

create table Examinations(
    student_id int,
subject_name varchar(100)
);

insert into Students values (1,"Alice"),
(2,"Bob"),
(13,"John"),
(6,"Alex");

insert into Subjects values ("Math"),
("Physics"),
("Programming");

insert into Examinations values (1,"Math"),
(1,"Physics"),
(1,"Programming"),
(2,"Programming"),
(1,"Physics"),
(1,"Math"),
(13,"Math"),
(13,"Programming"),
(13,"Physics"),
(2,"Math"),
(1,"Math");