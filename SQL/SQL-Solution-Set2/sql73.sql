-- Query

select avg(avaraege_percentage) as average_daily_percent from 
(select action_date, (avg(rpost_id is not null) * 100) as avaraege_percentage from 
(select spm.post_id as spmpost_id, spm.action_date, r.post_id as rpost_id, r.remove_date from
(select post_id, action_date, action, extra from Actions where action like 'report' and extra like 'spam' ) spm
left join Removals r
on r.post_id = spm.post_id ) tmp
group by action_date ) av_per;



-- Environment
Create table Actions(
    user_id int,
post_id int,
action_date date,
action enum('view', 'like', 'reaction', 'comment', 'report', 'share'),
extra varchar(100)
);

create table Removals(
    post_id int,
remove_date date
);

insert into Actions values (1,1,"2019-07-01","view",null),
(1,1,"2019-07-01","like",null),
(1,1,"2019-07-01","share",null),
(2,2,"2019-07-04","view",null),
(2,2,"2019-07-04","report","spam"),
(3,4,"2019-07-04","view",null),
(3,4,"2019-07-04","report","spam"),
(4,3,"2019-07-02","view",null),
(4,3,"2019-07-02","report","spam"),
(5,2,"2019-07-03","view",null),
(5,2,"2019-07-03","report","racism"),
(5,5,"2019-07-03","view",null),
(5,5,"2019-07-03","report","racism");

insert into Removals values (2,"2019-07-20"),
(3,"2019-07-18");