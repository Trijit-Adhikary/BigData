-- Query

select left_operand, operator, right_operand, if(bool_val = 1, 'true','false') as value 
from
(select e.left_operand, e.operator, e.right_operand,
    Case
        When e.operator = '<' then (v1.value < v2.value)
        When e.operator = '>' then (v1.value > v2.value)
        When e.operator = '=' then (v1.value = v2.value)
    end as bool_val
from Expressions e
join `Variables` v1
on e.left_operand = v1.name
join `Variables` v2
on e.right_operand = v2.name ) tmp ;


-- Environment

CREATE Table `Variables`(
    name varchar(100),
value int
);

CREATE TABLE Expressions(
    left_operand varchar(100),
operator enum ('<', '>', '='),
right_operand varchar(100)
);

insert into `Variables` values ("x",66),
("y",77);

insert into Expressions values ("x",">","y"),
("x","<","y"),
("x","=","y"),
("y",">","x"),
("y","<","x"),
("x","=","x");