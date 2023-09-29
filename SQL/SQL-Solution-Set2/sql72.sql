-- Query

select left(trans_date,7) as month, country,
    count(id) as trans_count,
    sum(Case when state like 'approved' then 1 else 0 end ) as trans_count,
    sum(amount) as trans_total_amount,
    sum(Case when state like 'approved' then amount else 0 end ) as approved_total_amount
from Transactions
GROUP BY left(trans_date,7), country;



-- Environment

Create table Transactions(
    id int,
country varchar(100),
state enum("approved", "declined"),
amount int,
trans_date date
);

insert into Transactions values (121,"US","approved",1000,"2018-12-18"),
(122,"US","declined",2000,"2018-12-19"),
(123,"US","approved",2000,"2019-01-01"),
(124,"DE","approved",2000,"2019-01-07");