-- Query

select customer_number from (select customer_number, count(order_number) as cnt_order
from Orders
group by customer_number
order by cnt_order desc ) a
limit 1;
Test




-- Environment

create table Orders(
    order_number int,
customer_number int
);

insert into Orders VALUES (1,1),
(2,2),
(3,3),
(4,3);
