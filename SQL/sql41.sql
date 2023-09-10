-- Query

select w.name, sum((p.vol * w.units)) as volume from
(select product_id, product_name, (Width * Length * Height) as vol from Products ) p
join Warehouse w
on p.product_id = w.product_id
group by w.name;



-- Environment

create table Warehouse(
    name varchar(100),
product_id int,
units int
);

create table Products(
    product_id int,
product_name varchar(100),
Width int,
Length int,
Height int
);


insert into Warehouse VALUES ("LCHouse1",1,1),
("LCHouse1",2,10),
("LCHouse1",3,5),
("LCHouse2",1,2),
("LCHouse2",2,2),
("LCHouse3",4,1);


insert into Products VALUES (1,"LC-TV",5,50,40),
(2,"LC-KeyChain",5,5,5),
(3,"LC-Phone",2,10,10),
(4,"LC-T-Shirt",4,10,20);