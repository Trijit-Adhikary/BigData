-- Query

select seat_id from Cinema
where free =1 and 
((seat_id+1,1) in (select * from Cinema) or  
(seat_id-1,1) in (select * from Cinema) );



-- Environment

create table Cinema(
    seat_id int,
free bool
);

insert into Cinema values (1,1),
(2,0),
(3,1),
(4,1),
(5,1);