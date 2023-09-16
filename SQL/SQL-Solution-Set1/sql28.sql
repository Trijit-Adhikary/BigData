-- Query

select cus.customer_id, cus.name from (select * from
(select *, lead(total_price_sum,1) over() as nxt from
(select customer_id,dt, sum(total_price) as total_price_sum from 
(select ord.customer_id, ord.product_id, left(ord.order_date,7) as dt,(ord.quantity * p.price) as total_price from
(select * from Orders where order_date like '2020-06%' or order_date like '2020-07%' ) ord
join Product p
on p.product_id = ord.product_id ) a
group by customer_id,dt order by customer_id,dt ) b ) c
where dt = '2020-06' and total_price_sum >= 100 and nxt >= 100 ) c
join Customers cus
on c.customer_id = cus.customer_id;



-- Environment

create table Customers(
    customer_id int,
name varchar(100),
country varchar(100)
);

create table Product(
    product_id int,
    description varchar(150),
    price int
);

create table Orders(
    order_id int,
customer_id int,
product_id int,
order_date date,
quantity int
);

insert into Customers values (1,"Winston","USA"),
(2,"Jonathan","Peru"),
(3,"Moustafa","Egypt");

insert into Product values (10, "LC Phone", 300),
(20, "LC T-Shirt", 10),
(30, "LC Book", 45),
(40, "LC Keychain", 2);

insert into Orders values (1,1,10,"2020-06-10",1),
(2,1,20,"2020-07-01",1),
(3,1,30,"2020-07-08",2),
(4,2,10,"2020-06-15",2),
(5,2,40,"2020-07-01",10),
(6,3,20,"2020-06-24",2),
(7,3,30,"2020-06-25",2),
(9,3,30,"2020-05-08",3);