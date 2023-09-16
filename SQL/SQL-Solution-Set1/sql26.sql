-- Query

select p.product_name, unit_temp.total_unit as unit
from
(select product_id,sum(unit) as total_unit from Orders where order_date like '2020-02-%'
group by product_id
having sum(unit) >= 100 ) unit_temp
join Products p
on p.product_id = unit_temp.product_id;


-- Environment

create table Products(
    product_id int,
product_name varchar(100),
product_category varchar(100)
);

create table Orders(
    product_id int,
order_date date,
unit int
);

insert into Products VALUES (1, "Leetcode Solutions", "Book"),
(2, "Jewels of Stringology", "Book"),
(3, "HP", "Laptop"),
(4, "Lenovo", "Laptop"),
(5, "Leetcode Kit", "T-shirt");

insert into Orders VALUES (1,"2020-02-05",60),
(1,"2020-02-10",70),
(2,"2020-01-18",30),
(2,"2020-02-11",80),
(3,"2020-02-17",2),
(3,"2020-02-24",3),
(4,"2020-03-01",20),
(4,"2020-03-04",30),
(4,"2020-03-04",60),
(5,"2020-02-25",50),
(5,"2020-02-27",50),
(5,"2020-03-01",50);