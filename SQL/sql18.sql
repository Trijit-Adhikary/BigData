-- Query

select distinct v1.author_id from Views v1
join Views v2
on v1.viewer_id = v2.author_id
order by v1.author_id;


-- Environment

create table Views(
    article_id int,
author_id int,
viewer_id int,
view_date date
);

insert into Views VALUES (1,3,5,"2019-08-01"),
(1,3,6,"2019-08-02"),
(2,7,7,"2019-08-01"),
(2,7,6,"2019-08-02"),
(4,7,1,"2019-07-22"),
(3,4,4,"2019-07-21"),
(3,4,4,"2019-07-21");