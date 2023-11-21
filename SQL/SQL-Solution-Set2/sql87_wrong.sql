-- Query

select 100 * avg(o.status <> 'completed successfully') as bad_experience_pct
from (select customer_id, signup_timestamp from customer where month(signup_timestamp) = 6) t
join orders o on
t.customer_id = o.customer_id and TIMESTAMPDIFF(day,t.signup_timestamp,o.order_timestamp) <= 13;


select round(avg(t1.bad), 2) as bad_experience_pct
from
(select t.customer_id, 100 * sum(case when o.status <> 'completed successfully' then 1 else 0 end)/count(*) as bad
from (select customer_id, signup_timestamp from customer where month(signup_timestamp) = 6) t
join orders o on
t.customer_id = o.customer_id
where TIMESTAMPDIFF(day,t.signup_timestamp,o.order_timestamp) <= 13
GROUP BY t.customer_id) t1;



-- Environment

create table orders(
order_id int,
customer_id int,
trip_id int,
status enum('completed successfully', 'completed incorrectly', 'never received'),
order_timestamp timestamp);

create table trips(
dasher_id int,
trip_id int,
estimated_delivery_timestamp timestamp,
actual_delivery_timestamp timestamp);

create table customer(
customer_id int,
signup_timestamp timestamp);

insert into orders VALUES (727424, 8472, 100463, "completed successfully", "2022/06/05 09:12:00"),
(242513, 2341, 100482, "completed incorrectly", "2022/06/05 14:40:00"),
(141367, 1314, 100362, "completed incorrectly", "2022/06/07 15:03:00"),
(582193, 5421, 100657, "never received", "2022/07/07 15:22:00"),
(253613, 1314, 100213, "completed successfully", "2022/06/12 13:43:00");



insert into customer VALUES (8472, "2022/05/30 00:00:00"),
(2341, "2022/06/01 00:00:00"),
(1314, "2022/06/03 00:00:00"),
(1435, "2022/06/05 00:00:00"),
(5421, "2022/06/07 00:00:00");