-- Query

select p.product_name,s.year, s.price
from Sales s 
join Product p on s.product_id = p.product_id;



-- Environment

create table Sales(
    sale_id int,
product_id int,
year int,
quantity int,
price int
);

CREATE table Product(
    product_id int,
product_name varchar(100)
);

insert into Sales VALUES (1,100,2008,10,5000),
(2,100,2009,12,5000),
(7,200,2011,15,9000);

insert into Product VALUES (100,"Nokia"),
(200,"Apple"),
(300,"Samsung");