-- Solution
select p.product_id, p.product_name from 
(select product_id from Sales where product_id not in (select distinct product_id from Sales where sale_date < "2019-01-01" or sale_date > " 2019-03-31" ) ) tmp
join Product p
on p.product_id = tmp.product_id;


-- Environment setup
create table Product (
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

insert into Product values (1,"S8",1000),
(2,"G4",800),
(3,"iPhone",1400);

select * from Product;

create table Sales(
    seller_id int,
product_id int,
buyer_id int,
sale_date date,
quantity int,
price int
);

insert into Sales values (1,1,1,"2019-01-21",2,2000),
(1,2,2,"2019-02-17",1,800),
(2,2,3,"2019-06-02",1,800),
(3,3,4,"2019-05-13",2,2800);
