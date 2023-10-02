-- Query

select company_id, employee_id, employee_name, 
round(Case
    when max_sal_com < 1000 then salary - ((0/100)*salary)
    when max_sal_com between 1000 and 10000 then salary - ((24/100)*salary)
    when max_sal_com > 10000 then salary - ((49/100)*salary)
end) as salary
from
(select *, max(salary) over(PARTITION BY company_id) as max_sal_com from Salaries ) as tmp;



-- Environment

create table Salaries(
    company_id int,
employee_id int,
employee_name varchar(100),
salary int
);

insert into Salaries VALUES (1,1,"Tony",2000),
(1,2,"Pronub",21300),
(1,3,"Tyrrox",10800),
(2,1,"Pam",300),
(2,7,"Bassem",450),
(2,9,"Hermione",700),
(3,7,"Bocaben",100),
(3,2,"Ognjen",2200),
(3,13,"Nyan,Cat",3300),
(3,15,"Morning,Cat",7777);