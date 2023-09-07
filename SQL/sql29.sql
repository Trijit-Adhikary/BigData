--Query

select kd_con.title from
(select content_id, title from Content Where Kids_content = 'Y' ) kd_con
join TVProgram tv using(content_id)
where tv.program_date like '2020-06%';



--Environment

create table TVProgram(
    program_date date,
content_id int,
channel varchar(100)
);

create table Content(
    content_id varchar(100),
title varchar(100),
Kids_content enum('Y', 'N'),
content_type varchar(100)
);


insert into TVProgram VALUES ("2020-06-10 08:00", 1, "LC-Channel"),
("2020-05-11 12:00", 2, "LC-Channel"),
("2020-05-12 12:00", 3, "LC-Channel"),
("2020-05-13 14:00", 4, "Disney Ch"),
("2020-06-18 14:00", 4, "Disney Ch"),
("2020-07-15 16:00", 5, "Disney Ch");


insert into Content VALUES (1,"Leetcode Movie", "N", "Movies"),
(2,"Alg. for Kids", "Y", "Series"),
(3,"Database Sols", "N", "Series"),
(4,"Aladdin", "Y", "Movies"),
(5,"Cinderella", "Y", "Movies");