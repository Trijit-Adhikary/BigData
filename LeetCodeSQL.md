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



<br>
<br>
<br>

---
