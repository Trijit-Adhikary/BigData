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


