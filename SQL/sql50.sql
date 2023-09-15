-- Query

select group_id, first_player as player_id from
(select *, DENSE_RANK() over(partition by group_id order by score desc,first_player) as dns_rnk
from
(select * from
(select first_player, sum(first_score) as score
from
(select match_id,first_player,second_player,first_score,second_score
from Matches
union all
select match_id,second_player,first_player,second_score,first_score
from Matches ) un
group by first_player ) tmp
join Players p on tmp.first_player = p.player_id ) a ) final
where dns_rnk = 1;



-- Environment

create table Players(
    player_id int,
    group_id int
);

create table Matches(
    match_id int,
    first_player int,
    second_player int,
    first_score int,
    second_score int
);

insert into Players values (15,1),
(25,1),
(30,1),
(45,1),
(10,2),
(35,2),
(50,2),
(20,3),
(40,3);

insert into Matches VALUES (1,15,45,3,0),
(2,30,25,1,2),
(3,30,15,2,0),
(4,40,20,5,2),
(5,35,50,1,1);