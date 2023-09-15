-- Query

select student_id,course_id,grade
from 
(select *, DENSE_RANK() over(partition by student_id order by grade desc, course_id) as dns_rnk from Enrollments ) tmp
where dns_rnk = 1
order by student_id;



-- Environment

create table Enrollments(
    student_id int,
course_id int,
grade int
);

insert into Enrollments VALUES (2,2,95),
(2,3,95),
(1,1,90),
(1,2,99),
(3,1,80),
(3,2,75),
(3,3,82);
