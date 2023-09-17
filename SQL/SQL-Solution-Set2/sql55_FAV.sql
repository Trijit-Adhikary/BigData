-- Query

with test as (select * from
(select caller_id, duration,LEFT(p.phone_number,3) as caller_cc from 
( select caller_id, duration from Calls
union ALL
select callee_id, duration from Calls ) a
join Person p
on a.caller_id = p.id
order by a.caller_id ) b
join Country c on cast(b.caller_cc as UNSIGNED) = cast(c.country_code as UNSIGNED))

select name from (select name,country_code, avg(duration) as country_avg_duration 
from test
group by country_code,name
having country_avg_duration > (select avg(duration) from test) ) final;




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