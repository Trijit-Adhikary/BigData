-- Query

select customer_id
from Customer
group by customer_id
having 
count(DISTINCT product_key) = (select count(DISTINCT product_key) from Product );




-- Environment

create table Customer(
    customer_id int,
product_key int
);

create table Product(
    product_key int
);

INSERT into Customer values (1,5),
(2,6),
(3,5),
(3,6),
(1,6);

insert into Product VALUES (5),
(6);
