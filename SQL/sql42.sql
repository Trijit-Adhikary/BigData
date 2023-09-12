-- Query

select distinct sale_date, (sold_num - orange_sold_num) as diff
from
(select *, lead(sold_num,1) over(PARTITION BY sale_date order by sale_date) as orange_sold_num
from Sales ) a
where fruit = 'apples';



-- Environment

create table Sales(
    sale_date date,
fruit enum("apples", "oranges"),
sold_num int
);

insert into Sales VALUES ("2020-05-01", "apples" ,10),
("2020-05-01", "oranges", 8),
("2020-05-02", "apples" ,15),
("2020-05-02", "oranges", 15),
("2020-05-03", "apples" ,20),
("2020-05-03", "oranges", 0),
("2020-05-04", "apples" ,15),
("2020-05-04", "oranges", 16);