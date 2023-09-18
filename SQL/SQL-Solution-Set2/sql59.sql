-- Query

select sp.name from Orders o
Right join SalesPerson sp
on sp.sales_id = o.sales_id
where ((select com_id from Company where name like "RED"), sp.sales_id) NOT IN (select com_id, sales_id from Orders);


-- Environment

create table SalesPerson(
    sales_id int,
name varchar(100),
salary int,
commission_rate int,
hire_date date
);

create table Company(
    com_id int,
name varchar(100),
city varchar(100)
);

create table Orders(
    order_id int,
order_date date,
com_id int,
sales_id int,
amount int
);

insert into SalesPerson VALUES (1,"John",100000,6,"2006-4-1"),
(2,"Amy",12000,5,"2010-5-1"),
(3,"Mark",65000,12,"2008-12-25"),
(4,"Pam",25000,25,"2005-1-1"),
(5,"Alex",5000,10,"2007-2-3");


insert into Company values (1,"RED","Boston"),
(2,"ORANGE","New,York"),
(3,"YELLOW","Boston"),
(4,"GREEN","Austin");

insert into Orders VALUES (1,"2014-1-1",3,4,10000),
(2,"2014-1-2",4,5,5000),
(3,"2014-1-3",1,1,50000),
(4,"2014-1-4",1,4,25000);