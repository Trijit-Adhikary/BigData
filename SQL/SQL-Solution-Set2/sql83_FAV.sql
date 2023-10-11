-- Query

select
     case
        when LENGTH(lst) % 2 = 0 Then round(((cast(SUBSTRING(SUBSTRING(lst,floor(LENGTH(lst)/2),2),1,1) as UNSIGNED) + cast(SUBSTRING(SUBSTRING(lst,floor(LENGTH(lst)/2),2),2,1) as UNSIGNED))/2),1) else SUBSTRING(lst,floor(LENGTH(lst)/2)+1,1) end as median
from
(select REPLACE(GROUP_CONCAT(test,''),',','') as lst from 
(select REPEAT(CAST(searches as char),CAST(num_users as char)) as test, 'a' as grp_key from search_frequency ) a
GROUP BY grp_key) a;




-- Environment

CREATE TABLE search_frequency(
    searches int,
num_users int
);

REPLACE into search_frequency values (1,2),
(2,2),
(3,3),
(4,1);