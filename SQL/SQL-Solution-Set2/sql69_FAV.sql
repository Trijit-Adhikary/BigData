-- Query

select min(log_Id), max(log_Id) from (select log_Id,log_id-row_number() over(order by log_id) as gp
from Logs ) as a
group by gp;


-- Environment

create table Logs(
    log_id int
);

insert into Logs VALUES (1),
(2),
(3),
(7),
(8),
(10);