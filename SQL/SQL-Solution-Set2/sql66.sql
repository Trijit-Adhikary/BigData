-- Query

select buyer_id from Sales
where (buyer_id, (select product_id from Product WHERE lower(product_name) = 's8')) in (select buyer_id, product_id from Sales)
and (buyer_id, (select product_id from Product WHERE lower(product_name) = 'iphone')) not in (select buyer_id, product_id from Sales);



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