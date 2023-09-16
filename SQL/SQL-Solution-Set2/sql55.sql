-- Query

select * from
(select cl2.caller_id, cl2.caller_duration, cl2.caller_cc, cl2.callee_id, cl2.callee_duration, LEFT(p2.phone_number,3) as callee_cc from
(select cl.caller_id, cl.caller_duration, LEFT(p.phone_number,3) as caller_cc, cl.callee_id, cl.callee_duration from 
(select caller_id, duration as caller_duration, callee_id, duration as callee_duration from Calls ) cl
join Person p on p.id = cl.caller_id ) cl2
join Person p2 on p2.id = cl2.callee_id order by caller_id) final
;




-- Environment

create table Person(
    id int,
name varchar(100),
phone_number varchar(100)
);

create table Country(
    name varchar(100),
country_code varchar(100)
);

create table Calls(
    caller_id int,
callee_id int,
duration int
);

insert into Person VALUES (3,"Jonathan","051-1234567"),
(12,"Elvis","051-7654321"),
(1,"Moncef","212-1234567"),
(2,"Maroua","212-6523651"),
(7,"Meir","972-1234567"),
(9,"Rachel","972-0011100");

insert into Country VALUES ("Peru",51),
("Israel",972),
("Morocco",212),
("Germany",49),
("Ethiopia",251);

insert into Calls VALUES (1,9,33),
(2,9,4),
(1,2,59),
(3,12,102),
(3,12,330),
(12,3,5),
(7,9,13),
(7,1,3),
(9,7,1),
(1,7,7);