-- Query

select distinct a.ad_id, COALESCE(temp2.ctr, 0) as ctr from
(select ad_id, round((avg(action = "Clicked") * 100),2) as ctr
from
(select * from Ads
where action in ("Clicked","Viewed") ) temp
GROUP BY ad_id ) temp2
right join Ads a
on a.ad_id = temp2.ad_id
order by ctr desc;


-- Environment

create table Ads(
    ad_id int,
user_id int,
action enum('Clicked', 'Viewed', 'Ignored')
);

insert into Ads VALUES (1,1,"Clicked"),
(2,2,"Clicked"),
(3,3,"Viewed"),
(5,5,"Ignored"),
(1,7,"Ignored"),
(2,7,"Viewed"),
(3,5,"Clicked"),
(1,4,"Viewed"),
(2,11,"Viewed"),
(1,2,"Clicked");