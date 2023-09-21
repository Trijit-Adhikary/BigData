-- Query

select actor_id,director_id, count(*) as cnt from ActorDirector group by actor_id,director_id having count(*) >=3;



-- Environment

create table ActorDirector(
    actor_id int,
director_id int,
timestamp int
);

insert into ActorDirector VALUES (1,1,0),
(1,1,1),
(1,1,2),
(1,2,3),
(1,2,4),
(2,1,5),
(2,1,6);