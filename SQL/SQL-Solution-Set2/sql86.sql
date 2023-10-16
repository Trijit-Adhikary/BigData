-- Query

select count(*) as payment_count
from transactions t1
join transactions t2
on t1.merchant_id = t2.merchant_id
and t1.credit_card_id = t2.credit_card_id
and t1.amount = t2.amount
and t1.transaction_timestamp <> t2.transaction_timestamp
and ABS(EXTRACT(HOUR from TIMEDIFF(t1.transaction_timestamp, t2.transaction_timestamp))) = 0
and (EXTRACT(MINUTE from TIMEDIFF(t1.transaction_timestamp, t2.transaction_timestamp))) BETWEEN 0 and 9;



-- Environment

CREATE TABLE transactions(
    transaction_id integer,
merchant_id integer,
credit_card_id integer,
amount integer,
transaction_timestamp TIMESTAMP
);
DROP TABLE transactions;
insert into transactions VALUES (1, 101, 1, 100, "2022/09/25 12:00:00"),
(2, 101, 1, 100, "2022/09/25 12:08:00"),
(3, 101, 1, 100, "2022/09/25 12:28:00"),
(7, 101, 1, 100, "2022/09/25 12:30:00"),
(4, 102, 2, 300, "2022/09/25 12:00:00"),
(6, 102, 2, 400, "2022/09/25 14:00:00"),
(8, 102, 2, 400, "2022/09/26 14:00:00"),
(9, 102, 2, 400, "2022/09/26 14:05:00");