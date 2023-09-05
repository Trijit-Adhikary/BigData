-- Query

SELECT * FROM Users WHERE mail REGEXP '^[a-z][a-zA-Z0-9\._-]*@leetcode.com';



-- Environment

create table Users(
    user_id int,
name varchar(100),
mail varchar(100)
);

insert into Users values(1,"Winston","winston@leetcode.com"),
(2,"Jonathan","jonathanisgreat"),
(3,"Annabelle", "bella-@leetcode.com"),
(4,"Sally","sally.come@leetcode.com"),
(5,"Marwan","quarz#2020@leetcode.com"),
(6,"David","david69@gmail.com"),
(7,"Shapiro",".shapo@leetcode.com");