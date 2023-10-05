-- Query

select *, 
    case when prev_year_spend <> '' then round(IFNULL((((curr_year_spend - prev_year_spend)/prev_year_spend) * 100),''),2) else ''
    end as yoy_rate
from 
(select YEAR(transaction_date) as Y, 
    product_id, 
    spend as curr_year_spend, 
    lag(spend,1,'') over(PARTITION BY product_id order by YEAR(transaction_date)) as prev_year_spend
from user_transaction ) primer;



-- Environment

create table user_transaction(
transaction_id int,
product_id int,
spend float(6,2),
transaction_date datetime);

drop table user_transaction;

insert into user_transaction values (1341,123424,1500.60,"2019/12/31 12:00:00"),
(1423,123424,1000.20,"2020/12/31 12:00:00"),
(1623,123424,1246.44,"2021/12/31 12:00:00"),
(1322,123424,2145.32,"2022/12/31 12:00:00");