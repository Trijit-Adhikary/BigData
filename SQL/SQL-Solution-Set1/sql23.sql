--Query

select p.product_id, round(sum((p.price * u.units)) / sum(u.units),2) as average_price 
from Prices p
join UnitsSold u 
on p.product_id = u.product_id 
and u.purchase_date BETWEEN p.start_date and p.end_date
group by p.product_id;


--Environment

Create table Prices(
    product_id int,
start_date date,
end_date date,
price int
);

Create table UnitsSold(
    product_id int,
purchase_date date,
units int
);

insert into Prices VALUES (1,"2019-02-17","2019-02-28",5),
(1,"2019-03-01","2019-03-22",20),
(2,"2019-02-01","2019-02-20",15),
(2,"2019-02-21","2019-03-31",30);

insert into UnitsSold VALUES (1,"2019-02-25",100),
(1,"2019-03-01",15),
(2,"2019-02-10",200),
(2,"2019-03-22",30);
