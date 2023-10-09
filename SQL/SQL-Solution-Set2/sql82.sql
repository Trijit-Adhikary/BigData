-- Query

select month(ua.event_date) as `month`, count(user_sgn.user_id) as monthly_active_users 
from (select * from user_actions where month(event_date) = '6' and event_type like 'sign-in' ) user_sgn
join user_actions ua 
on user_sgn.user_id = ua.user_id
and ua.event_type in ("like","comment")
GROUP BY month(ua.event_date);



-- Environment

create table user_actions(
    user_id int,
event_id int,
event_type ENUM("sign-in", "like", "comment"),
event_date datetime
);

insert into user_actions VALUES (445, 7765, "sign-in", "2022/05/31 12:00:00"),
(742, 6458, "sign-in", "2022/06/03 12:00:00"),
(445, 3634, "like", "2022/06/05 12:00:00"),
(742, 1374, "comment", "2022/06/05 12:00:00"),
(648, 3124, "like", "2022/06/18 12:00:00"); 