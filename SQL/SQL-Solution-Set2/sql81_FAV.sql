-- Query

with data_stat as (SELECT item_type,
      COUNT(item_type) AS total_item_type,
      SUM(square_footage) AS total_square_footage
FROM inventory
GROUP BY item_type),
prime_area as 
(select item_type, floor(500000/total_square_footage) * total_square_footage as sqft,(floor(500000/total_square_footage) * total_item_type) as item_count from data_stat where item_type like 'prime_eligible')
select item_type, item_count from prime_area
UNION
select 'not_prime' as item_type, floor((500000 - sqft)/(select total_square_footage from data_stat where item_type like 'not_prime')) * (select total_item_type from data_stat where item_type like 'not_prime') from prime_area;



-- Environment

create table inventory(
    item_id int,
item_type varchar(100),
item_category varchar(100),
square_footage float
);

insert into inventory values (1374,"prime_eligible","mini refrigerator",68.00),
(4245,"not_prime","standing lamp",26.40),
(2452,"prime_eligible","television",85.00),
(3255,"not_prime","side table",22.60),
(1672,"prime_eligible","laptop",8.50);