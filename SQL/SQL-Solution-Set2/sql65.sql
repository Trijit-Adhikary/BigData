-- Query

select b.seller_id from 
(select seller_id, DENSE_RANK() over(order by total_sale desc) as dns_rnk from
(select seller_id, sum(price) as total_sale from Sales
GROUP BY seller_id ) a ) b
where dns_rnk = 1;



-- Environment

create table Product(
    product_id int,
product_name varchar(100),
unit_price int
);

create table Sales(
    seller_id int,
product_id int,
buyer_id int,
sale_date date,
quantity int,
price int
);

insert into Product VALUES(1,"S8",1000),
(2,"G4",800),
(3,"iPhone",1400);

insert into Sales VALUES (1,1,1,"2019-01-21",2,2000),
(1,2,2,"2019-02-17",1,800),
(2,2,3,"2019-06-02",1,800),
(3,3,4,"2019-05-13",2,2800);