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

Write a solution to find employees who have the highest salary in each of the departments.

Return the result table in any order.

The result format is in the following example.

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

