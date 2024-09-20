# 176. Second Highest Salary

Table: `Employee`

| Column Name | Type |
|-------------|------|
| id          | int  |
| salary      | int  |

<br>

id is the primary key (column with unique values) for this table.
Each row of this table contains information about the salary of an employee.

<br>
<br>

Write a solution to find the second highest distinct salary from the Employee table. If there is no second highest salary, return null (return None in Pandas).

The result format is in the following example.

<br>

Example 1:

Input: 
Employee table:

| id | salary |
|----|--------|
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |


Output:

| SecondHighestSalary |
|---------------------|
| 200                 |

<br>

Example 2:

Input: 
Employee table:

| id | salary |
|----|--------|
| 1  | 100    |


Output: 

| SecondHighestSalary |
|---------------------|
| null                |


<br>

## Solution -

```python
highest_salary = employee.select(max("salary")).collect()[0][0]
second_highest_salary = employee.filter(employee.salary != highest_salary) \
                                .select(max("salary").alias("SecondHighestSalary"))
second_highest_salary.show()
```

<br>

```sql
select 
    max(salary) as SecondHighestSalary
from employee_tbl 
    where salary <> (select max(salary) from employee_tbl);
```

---

<br>
<br>
<br>




# 177. Nth Highest Salary

Table: `Employee`

| Column Name | Type |
|-------------|------|
| id          | int  |
| salary      | int  |

<br>

id is the primary key (column with unique values) for this table.
Each row of this table contains information about the salary of an employee.
 
<br>
<br>

Write a solution to find the nth highest salary from the Employee table. If there is no nth highest salary, return null.

The result format is in the following example.

 
<br>

Example 1:

Input: 
Employee table:

| id | salary |
|----|--------|
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |


n = 2
Output: 

| getNthHighestSalary(2) |
|------------------------|
| 200                    |


<br>

Example 2:

Input: 
Employee table:

| id | salary |
|----|--------|
| 1  | 100    |


n = 2
Output: 

| getNthHighestSalary(2) |
|------------------------|
| null                   |

<br>

## Solution -

```python
from pyspark.sql import *

N = 2
salary_window = Window.orderBy(desc("salary"))

salary_ranked_df = employee.withColumn(f"salary_rank", dense_rank().over(salary_window) )

salary_ranked_df.filter(salary_ranked_df.salary_rank == N).select(max(salary_ranked_df.salary).alias(f"{N}_rank")).show()
```

<br>

```sql
select max(salary) as {N}_salary
    from
        (select salary, dense_rank() over(order by salary desc) as sal_rank
        from employee_tbl)
    where sal_rank = {N}
```

---

<br>
<br>
<br>

# 178. Rank Scores

Table: `Scores`

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| score       | decimal |


id is the primary key (column with unique values) for this table.
Each row of this table contains the score of a game. Score is a floating point value with two decimal places.

<br>
<br>

Write a solution to find the rank of the scores. The ranking should be calculated according to the following rules:

The scores should be ranked from the highest to the lowest.
If there is a tie between two scores, both should have the same ranking.
After a tie, the next ranking number should be the next consecutive integer value. In other words, there should be no holes between ranks.
Return the result table ordered by score in descending order.

The result format is in the following example.

<br>

Example 1:

Input: 
Scores table:

| id | score |
|----|-------|
| 1  | 3.50  |
| 2  | 3.65  |
| 3  | 4.00  |
| 4  | 3.85  |
| 5  | 4.00  |
| 6  | 3.65  |


Output: 

| score | rank |
|-------|------|
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 2    |
| 3.65  | 3    |
| 3.65  | 3    |
| 3.50  | 4    |

<br>

## Solution -

```python
from pyspark.sql import *

score_window = Window.orderBy(desc("score"))

ranked_scores_df = scores.withColumn("rank", dense_rank().over(score_window)).select("score","rank")
ranked_scores_df.show()
```

```sql
select score, 
       dense_rank() over(order by score desc) as rank
from scores_tbl;
```


<br>
<br>
<br>

---

# 180. Consecutive Numbers

Table: `Logs`


| Column Name | Type    |
|-------------|---------|
| id          | int     |
| num         | varchar |


In SQL, id is the primary key for this table.
id is an autoincrement column starting from 1.
 
<br>
<br>

Find all numbers that appear at least three times consecutively.

Return the result table in any order.

The result format is in the following example.

 <br>

Example 1:

Input: 
Logs table:

| id | num |
|----|-----|
| 1  | 1   |
| 2  | 1   |
| 3  | 1   |
| 4  | 2   |
| 5  | 1   |
| 6  | 2   |
| 7  | 2   |


Output: 

| ConsecutiveNums |
|-----------------|
| 1               |

Explanation: 1 is the only number that appears consecutively for at least three times.

<br>

## Solution -

### Approach 1 -

```python
from pyspark.sql import *

num_window = Window.orderBy("id")

consecutive_nums = logs.withColumn("second_num", lead(logs.num,1).over(num_window)) \
                       .withColumn("third_num", lead(logs.num,2).over(num_window))
consecutive_nums.filter( (consecutive_nums.num == consecutive_nums.second_num) 
                            & (consecutive_nums.num == consecutive_nums.third_num) ) \
                .select( consecutive_nums.num.alias("ConsecutiveNums") ) \
                .distinct() \
                .show()
```

```sql
select distinct num as ConsecutiveNums  from
(select *, 
        lead(num,1) over(order by id) as secound_num,
        lead(num,2) over(order by id) as third_num
from logs_tbl ) abc
where num = secound_num and num = third_num
```

<br>

### Approach 2 -

```python
logs2 = logs.withColumnRenamed("id","id2").withColumnRenamed("num","num2")
logs3 = logs.withColumnRenamed("id","id3").withColumnRenamed("num","num3")

logs.join(logs2, (logs2.id2 == (logs.id + 1)) & (logs2.num2 == logs.num), "inner" ) \
    .join(logs3, (logs3.id3 == (logs.id + 2)) & (logs3.num3 == logs.num), "inner" ) \
    .selectExpr("num as ConsecutiveNums") \
    .distinct() \
    .show()
```

```sql
select distinct l1.num as ConsecutiveNums  
from logs_tbl l1
join logs_tbl l2
    on l2.id = (l1.id + 1)
    and l1.num = l2.num
join logs_tbl l3
    on l3.id = (l1.id + 2)
    and l3.num = l1.num
```

---

<br>
<br>
<br>


# 184. Department Highest Salary

Table: `Employee`


| Column Name  | Type    |
|--------------|---------|
| id           | int     |
| name         | varchar |
| salary       | int     |
| departmentId | int     |

id is the primary key (column with unique values) for this table.
departmentId is a foreign key (reference columns) of the ID from the Department table.
Each row of this table indicates the ID, name, and salary of an employee. It also contains the ID of their department.
 
<br>

Table: `Department`


| Column Name | Type    |
|-------------|---------|
| id          | int     |
| name        | varchar |

id is the primary key (column with unique values) for this table. It is guaranteed that department name is not NULL.
Each row of this table indicates the ID of a department and its name.

<br>
<br>

Write a solution to find employees who have the highest salary in each of the departments.

Return the result table in any order.

The result format is in the following example.

<br>
<br>

Example 1:

Input: 
`Employee` table:

| id | name  | salary | departmentId |
|----|-------|--------|--------------|
| 1  | Joe   | 70000  | 1            |
| 2  | Jim   | 90000  | 1            |
| 3  | Henry | 80000  | 2            |
| 4  | Sam   | 60000  | 2            |
| 5  | Max   | 90000  | 1            |

<br>

`Department` table:

| id | name  |
|----|-------|
| 1  | IT    |
| 2  | Sales |

<br>

Output: 

| Department | Employee | Salary |
|------------|----------|--------|
| IT         | Jim      | 90000  |
| Sales      | Henry    | 80000  |
| IT         | Max      | 90000  |

Explanation: Max and Jim both have the highest salary in the IT department and Henry has the highest salary in the Sales department.

<br>

## Solution -

### Approach 1 -

```python
from pyspark.sql import *

sal_window = Window.partitionBy("departmentId").orderBy(desc("Salary"))

ranked_sal = employee.withColumn("sal_rank", rank().over(sal_window)) \
                        .filter("sal_rank = 1")

ranked_sal.join(department, ranked_sal.departmentid == department.id, 'left') \
            .select(department.name.alias("Department"), 
                        ranked_sal.name.alias("Employee"), 
                        ranked_sal.salary) \
            .show()
```

```sql
select dept.name as Department, 
        emp.name as Employee,
        emp.salary as Salary
from
(select * 
from
    (select *,
            rank() over(partition by departmentId order by salary desc) as sal_rank
    from employee_tbl ) ranked_sal
where sal_rank = 1 ) emp
left join department_tbl dept
    on emp.departmentId = dept.id
````

<br>

### Approach 2 -

```python
dept_max_sal = employee.groupBy("departmentId").agg(max("salary").alias("dept_max_sal"))

sal_with_dept = employee.join(department, employee.departmentid == department.id, 'left')

sal_with_dept.join(dept_max_sal, (sal_with_dept.departmentid == dept_max_sal.departmentId) & (sal_with_dept.salary == dept_max_sal.dept_max_sal),'inner') \
            .show()
```

```sql
with dept_max_sal as (
    select departmentId, max(salary) as dept_max_sal
    from employee_tbl
    group by departmentId
)
select dept.name as Department, 
        emp.name as Employee,
        emp.salary as Salary
from employee_tbl emp
left join department_tbl dept
on emp.departmentId = dept.id
where (emp.departmentId, emp.salary) IN (select * from dept_max_sal)
```

---

<br>
<br>
<br>



# 534. Game Play Analysis III

Table: `Activity`

| Column Name  | Type    |
|--------------|---------|
| player_id    | int     |
| device_id    | int     |
| event_date   | date    |
| games_played | int     |

(player_id, event_date) is the primary key of this table.
This table shows the activity of players of some game.
Each row is a record of a player who logged in and played a number of games (possibly 0) before logging out on some day using some device.

<br>
<br>

Write an SQL query that reports for each player and date, how many games played so far by the player. That is, the total number of games played by the player until that date. Check the example for clarity.

<br>

The query result format is in the following example:
`Activity` table:

| player_id | device_id | event_date | games_played |
|-----------|-----------|------------|--------------|
| 1         | 2         | 2016-03-01 | 5            |
| 1         | 2         | 2016-05-02 | 6            |
| 1         | 3         | 2017-06-25 | 1            |
| 3         | 1         | 2016-03-02 | 0            |
| 3         | 4         | 2018-07-03 | 5            |


Result table:

| player_id | event_date | games_played_so_far |
|-----------|------------|---------------------|
| 1         | 2016-03-01 | 5                   |
| 1         | 2016-05-02 | 11                  |
| 1         | 2017-06-25 | 12                  |
| 3         | 2016-03-02 | 0                   |
| 3         | 2018-07-03 | 5                   |

For the player with id 1, 5 + 6 = 11 games played by 2016-05-02, and 5 + 6 + 1 = 12 games played by 2017-06-25.
For the player with id 3, 0 + 5 = 5 games played by 2018-07-03.
Note that for each player we only care about the days when the player logged in.

<br>

## Solution -

```python
from pyspark.sql import *

game_play_window = Window.partitionBy("player_id").orderBy("event_date")
activity.withColumn("games_played_so_far", sum("games_played").over(game_play_window)) \
    .select("player_id","event_date","games_played_so_far") \
    .show()
```

```sql
select player_id, event_date, games_played_so_far
from
(select *, sum(games_played) over(partition by player_id order by event_date) as games_played_so_far
from activity_tbl )

```

---

<br>
<br>
<br>


# 550. Game Play Analysis IV

Table: `Activity`


| Column Name  | Type    |
|--------------|---------
| player_id    | int     |
| device_id    | int     |
| event_date   | date    |
| games_played | int     |

(player_id, event_date) is the primary key (combination of columns with unique values) of this table.
This table shows the activity of players of some games.
Each row is a record of a player who logged in and played a number of games (possibly 0) before logging out on someday using some device.
 
<br>
<br>

Write a solution to report the fraction of players that logged in again on the day after the day they first logged in, rounded to 2 decimal places. In other words, you need to count the number of players that logged in for at least two consecutive days starting from their first login date, then divide that number by the total number of players.

The result format is in the following example.


<br>

Example 1:

Input: 
Activity table:

| player_id | device_id | event_date | games_played |
|-----------|-----------|------------|--------------|
| 1         | 2         | 2016-03-01 | 5            |
| 1         | 2         | 2016-03-02 | 6            |
| 2         | 3         | 2017-06-25 | 1            |
| 3         | 1         | 2016-03-02 | 0            |
| 3         | 4         | 2018-07-03 | 5            |


<br>

Output: 

| fraction  |
|-----------|
| 0.33      |

Explanation: 
Only the player with id 1 logged back in after the first day he had logged in so the answer is 1/3 = 0.33

<br>
<br>

```python
first_logged_in = activity.groupBy("player_id") \
                            .agg( min(activity.event_date).alias("first_login_date") )

all_players_cnt = activity.select("player_id").distinct().count()

next_day_logged_cnt = activity.join(first_logged_in, (activity.event_date == date_add(first_logged_in.first_login_date, 1) ) & 
                                                        (activity.player_id == first_logged_in.player_id), 'inner') \
                                .count()

print(__builtin__.round((next_day_logged_cnt/all_players_cnt), 2 ) )

```

```sql
WITH first_logged_in AS (
    select player_id, min(event_date) as first_login_date
    from activity_tbl
    group by player_id
)
select round((count(*) / (select count(distinct player_id) from activity_tbl) ), 2) as fraction
from activity_tbl a
join first_logged_in fli
on a.player_id = fli.player_id
and a.event_date = date_add(fli.first_login_date, 1)
```

---

<br>
<br>
<br>


# 570. Managers with at Least 5 Direct Reports

Table: `Employee`


| Column Name | Type    |
|-------------|---------|
| id          | int     |
| name        | varchar |
| department  | varchar |
| managerId   | int     |

id is the primary key (column with unique values) for this table.
Each row of this table indicates the name of an employee, their department, and the id of their manager.
If managerId is null, then the employee does not have a manager.
No employee will be the manager of themself.

<br>
<br>

Write a solution to find managers with at least five direct reports.

Return the result table in any order.

The result format is in the following example.

<br>

Example 1:

Input: 
`Employee` table:

| id  | name  | department | managerId |
|-----|-------|------------|-----------|
| 101 | John  | A          | null      |
| 102 | Dan   | A          | 101       |
| 103 | James | A          | 101       |
| 104 | Amy   | A          | 101       |
| 105 | Anne  | A          | 101       |
| 106 | Ron   | B          | 101       |

<br>

Output: 

| name |
|------|
| John |


```python
more_thn_5_reports = employee.groupBy("managerid").count().filter("count >= 5")
employee.select("id","name") \ # To avoid ambiguity
        .join(more_thn_5_reports, employee.id == more_thn_5_reports.managerid, 'inner') \
        .select("name") \
        .show()
```

```sql
select e.name
    from employee_tbl e
join
    (select managerId, count(managerId) as reports_count
        from employee_tbl
    group by managerId
    having count(managerId) >= 5 ) mt5r
on e.id = mt5r.managerId
```

---

<br>
<br>
<br>


