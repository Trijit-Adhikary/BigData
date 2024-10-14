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

logs.join(logs2, (logs2.id2 == (logs.id | 1)) & (logs2.num2 == logs.num), "inner" ) \
    .join(logs3, (logs3.id3 == (logs.id | 2)) & (logs3.num3 == logs.num), "inner" ) \
    .selectExpr("num as ConsecutiveNums") \
    .distinct() \
    .show()
```

```sql
select distinct l1.num as ConsecutiveNums  
from logs_tbl l1
join logs_tbl l2
    on l2.id = (l1.id | 1)
    and l1.num = l2.num
join logs_tbl l3
    on l3.id = (l1.id | 2)
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

For the player with id 1, 5 | 6 = 11 games played by 2016-05-02, and 5 | 6 | 1 = 12 games played by 2017-06-25.
For the player with id 3, 0 | 5 = 5 games played by 2018-07-03.
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

<br>

## Solution -

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


# 574. Winning Candidate -> Simple

Table: `Candidate`


| id  | Name    |
|-----|---------|
| 1   | A       |
| 2   | B       |
| 3   | C       |
| 4   | D       |
| 5   | E       |

<br>

Table: `Vote`


| id  | CandidateId  |
|-----|--------------|
| 1   |     2        |
| 2   |     4        |
| 3   |     3        |
| 4   |     2        |
| 5   |     5        |

id is the auto-increment primary key,
CandidateId is the id appeared in Candidate table.

<br>
<br>

Write a  sql to find the name of the winning candidate, the above example will return the winner B.

<br>


| Name |
|------|
| B    |

<br>

**Notes:**
You may assume there is no tie, in other words there will be at most one winning candidate.

---

<br>
<br>
<br>

**Review**

# 578. Get Highest Answer Rate Question

The **answer rate** for a question is the number of times a user answered the question by the number of times a user showed the question.

Write a sql query to report the question which has the highest **answer rate**. If multiple questions have the same maximum **answer rate**, report the question with the smallest `question_id`.

<br>
<br>

Table: `survey_log `


| uid  | action    | question_id  | answer_id  | q_num     | timestamp  |
|------|-----------|--------------|------------|-----------|------------|
| 5    | show      | 285          | null       | 1         | 123        |
| 5    | answer    | 285          | 124124     | 1         | 124        |
| 5    | show      | 369          | null       | 2         | 125        |
| 5    | skip      | 369          | null       | 2         | 126        |


<br>

Output: 


| survey_log  |
|-------------|
|    285      |


<br>
<br>

## Solution -

### Approach 1 -

```sql
with answered_questions as (
    select question_id, action, count(*) as ans_cnt
    from survey_log_tbl
    group by question_id, action
    having action = 'answer'
),
showed_questions as (
    select question_id, action, count(*) as show_cnt
    from survey_log_tbl
    group by question_id, action
    having action = 'show'
)
select question_id as survey_log from
(select sq.question_id, (ans_cnt / show_cnt) as answer_rate 
    from showed_questions sq
    join answered_questions aq
on sq.question_id = aq.question_id
order by answer_rate desc, sq.question_id asc)
```

### Approach 2 -

```python
survey_log.withColumn("ans", when(survey_log.action == 'answer', 1 ).otherwise(0) ) \
        .withColumn("show", when(survey_log.action == 'show', 1).otherwise(0) ) \
        .groupBy("question_id").agg( (sum("ans")/sum("show")).alias("answer_rate") ) \
        .orderBy(desc("answer_rate"), "question_id") \
        .select(col("question_id").alias("survey_log")) \
        .limit(1) \
        .show()
```

```sql
select question_id as survey_log
from
(select question_id, sum(if(action = 'answer', 1, 0)) / sum(if(action = 'show', 1, 0)) as answer_rate
from survey_log_tbl
group by question_id
order by answer_rate desc, question_id asc
limit 1 )
```

---

<br>
<br>
<br>


# 580. Count Student Number in Departments

A university uses 2 data tables, student and department, to store data about its students and the departments associated with each major.

Write a query to print the respective department name and number of students majoring in each department for all departments in the department table (even ones with no current students).

Sort your results by descending number of students; if two or more departments have the same number of students, then sort those departments alphabetically by department name.

The `student` is described as follow:

| Column Name  | Type      |
|--------------|-----------|
| student_id   | Integer   |
| student_name | String    |
| gender       | Character |
| dept_id      | Integer   |


where student_id is the student's ID number, student_name is the student's name, gender is their gender, and dept_id is the department ID associated with their declared major.

<br>

And the `department` table is described as below:

| Column Name | Type    |
|-------------|---------|
| dept_id     | Integer |
| dept_name   | String  |

where dept_id is the department's ID number and dept_name is the department name.

Here is an example input:

<br>

`student` table:

| student_id | student_name | gender | dept_id |
|------------|--------------|--------|---------|
| 1          | Jack         | M      | 1       |
| 2          | Jane         | F      | 1       |
| 3          | Mark         | M      | 2       |

`department` table:

| dept_id | dept_name   |
|---------|-------------|
| 1       | Engineering |
| 2       | Science     |
| 3       | Law         |

<br>

The Output should be:

| dept_name   | student_number |
|-------------|----------------|
| Engineering | 2              |
| Science     | 1              |
| Law         | 0              |

<br>

## Solution -

```python
department.join(student, department.dept_id == student.dept_id, 'left') \
            .groupBy("dept_name") \
                .agg(count(col("student_id")).alias("student_number") ) \
            .orderBy(desc("student_number"), "dept_name") \
            .show()
```

```sql
select dept_name, count(student_id) as student_number
from department_tbl d
left join student_tbl s
on d.dept_id = s.dept_id
group by dept_name
order by student_number desc, dept_name
```

---

<br>
<br>
<br>

# 585. Investments in 2016

Table: `Insurance`


| Column Name | Type  |
|-------------|-------|
| pid         | int   |
| tiv_2015    | float |
| tiv_2016    | float |
| lat         | float |
| lon         | float |

pid is the primary key (column with unique values) for this table.
Each row of this table contains information about one policy where:
pid is the policyholder's policy ID.
tiv_2015 is the total investment value in 2015 and tiv_2016 is the total investment value in 2016.
lat is the latitude of the policy holder's city. It's guaranteed that lat is not NULL.
lon is the longitude of the policy holder's city. It's guaranteed that lon is not NULL.
 
<br>
<br>

Write a solution to report the sum of all total investment values in 2016 `tiv_2016`, for all policyholders who:

have the same tiv_2015 value as one or more other policyholders, and
are not located in the same city as any other policyholder (i.e., the (`lat`, `lon`) attribute pairs must be unique).
Round `tiv_2016` to **two decimal places**.

The result format is in the following example.

<br>

Example 1:

Input: 
Insurance table:

| pid | tiv_2015 | tiv_2016 | lat | lon |
|-----|----------|----------|-----|-----|
| 1   | 10       | 5        | 10  | 10  |
| 2   | 20       | 20       | 20  | 20  |
| 3   | 10       | 30       | 20  | 20  |
| 4   | 10       | 40       | 40  | 40  |


<br>

Output: 

| tiv_2016 |
|----------|
| 45.00    |


<br>

Explanation: 
The first record in the table, like the last record, meets both of the two criteria.
The tiv_2015 value 10 is the same as the third and fourth records, and its location is unique.

The second record does not meet any of the two criteria. Its tiv_2015 is not like any other policyholders and its location is the same as the third record, which makes the third record fail, too.
So, the result is the sum of tiv_2016 of the first and last record, which is 45.

<br>

## Solution -

```python
unique_location = insurance.groupBy(insurance.lat, insurance.lon).agg( count(insurance.lat).alias("location_cnt")).filter(col("location_cnt") == 1)
unique_location.show()

insurance_columns = insurance.columns
insurance1 = insurance.selectExpr([f"{clm} as {clm}_1" for clm in insurance_columns])
insurance2 = insurance.selectExpr([f"{clm} as {clm}_2" for clm in insurance_columns])

insurance1.join(insurance2, (col("pid_1") != col("pid_2")) & (col("tiv_2015_1") == col("tiv_2015_2") ), 'inner') \
.join(unique_location, (col("lat_1") == col("lat")) & (col("lon_1") == col("lon") ),'inner') \
.select("pid_1","tiv_2015_1","tiv_2016_1","lat_1","lon_1").distinct() \
.select(round(sum(col("tiv_2016_1")),2).alias("tiv_2016")).show()
```

```sql
with unique_location as (
    select lat,lon from insurance_tbl group by lat,lon having count(*) = 1
)
select round(sum(tiv_2016), 2) as tiv_2016 
from
(select distinct in1.pid, in1.tiv_2015, in1.tiv_2016, in1.lat, in1.lon
from insurance_tbl in1
join insurance_tbl in2
on in1.pid <> in2.pid
and in1.tiv_2015 = in2.tiv_2015
and (in1.lat, in1.lon) in (select * from unique_location) );
```


---

<br>
<br>
<br>


# 602. Friend Requests II: Who Has the Most Friends

Table: `RequestAccepted`


| Column Name    | Type    |
|----------------|---------|
| requester_id   | int     |
| accepter_id    | int     |
| accept_date    | date    |

(requester_id, accepter_id) is the primary key (combination of columns with unique values) for this table.
This table contains the ID of the user who sent the request, the ID of the user who received the request, and the date when the request was accepted.
 
<br>
<br>

Write a solution to find the people who have the most friends and the most friends number.

The test cases are generated so that only one person has the most friends.

The result format is in the following example.

 <br>

Example 1:

Input: 
RequestAccepted table:

| requester_id | accepter_id | accept_date |
|--------------|-------------|-------------|
| 1            | 2           | 2016/06/03  |
| 1            | 3           | 2016/06/08  |
| 2            | 3           | 2016/06/08  |
| 3            | 4           | 2016/06/09  |


<br>

Output: 

| id | num |
|----|-----|
| 3  | 3   |


<br>

Explanation: 
The person with id 3 is a friend of people 1, 2, and 4, so he has three friends in total, which is the most number than any others.

<br>

**Follow up:** In the real world, multiple people could have the same most number of friends. Could you find all these people in this case?

<br>

## Solution -

```python
requestaccepted.select("requester_id") \
    .union(requestaccepted.select("accepter_id")) \
    .groupBy("requester_id") \
    .agg( count(col("requester_id")).alias("num") ) \
    .orderBy( desc(col("num")) ) \
    .limit(1) \
    .selectExpr("requester_id as id ", "num") \
    .show()
```

```sql
with complete_seq as
(select requester_id
from requestaccepted_tbl
union all
select accepter_id
from requestaccepted_tbl )
select requester_id as id, count(*) as num from complete_seq
group by requester_id
order by num desc
limit 1;
```


---


<br>
<br>
<br>


# 608. Tree Node

Table: `Tree`


| Column Name | Type |
|-------------|------|
| id          | int  |
| p_id        | int  |

id is the column with unique values for this table.
Each row of this table contains information about the id of a node and the id of its parent node in a tree.
The given structure is always a valid tree.

 <br>
 <br>

Each node in the tree can be one of three types:

**"Leaf":** if the node is a leaf node. <br>
**"Root":** if the node is the root of the tree. <br>
**"Inner":** If the node is neither a leaf node nor a root node. 

<br>
<br>

Write a solution to report the type of each node in the tree.

Return the result table in any order.

<br>

The result format is in the following example.

<br>

Example 1:

Input: 
Tree table:

| id | p_id |
|----|------|
| 1  | null |
| 2  | 1    |
| 3  | 1    |
| 4  | 2    |
| 5  | 2    |


<br>

Output: 

| id | type  |
|----|-------|
| 1  | Root  |
| 2  | Inner |
| 3  | Leaf  |
| 4  | Leaf  |
| 5  | Leaf  |

Explanation: 
Node 1 is the root node because its parent node is null and it has child nodes 2 and 3.
Node 2 is an inner node because it has parent node 1 and child node 4 and 5.
Nodes 3, 4, and 5 are leaf nodes because they have parent nodes and they do not have child nodes.
Example 2:

<br>
<br>

Input: 
`Tree` table:

| id | p_id |
|----|------|
| 1  | null |


Output: 

| id | type  |
|----|-------|
| 1  | Root  |

Explanation: If there is only one node on the tree, you only need to output its root attributes.

<br>

## Solution -

```python
parents_collect = tree.select("p_id").where("p_id is not null").distinct().collect()
if len(parents_collect) > 0:
    parents = tuple([parent[0] for parent in parents_collect])
else:
    parents = tuple()

root_node = tree.filter("p_id is null").select("id").withColumn("type",lit("Root"))
if len(parents_collect) > 0:
    inner_nodes = tree.filter(f"id in {parents} and  p_id is not null").select("id").withColumn("type",lit("Inner"))
    leaf_nodes = tree.filter(f"id not in {parents}").select("id").withColumn("type",lit("Leaf"))

    root_node.union(inner_nodes).union(leaf_nodes).show()
else:
    root_node.show()
```

```sql
with parents as(
    select distinct p_id
        from tree_tbl
    where p_id is not NULL
),
inner_nodes as (
    select distinct id, 'Inner' as type
        from tree_tbl
    where id in (select * from parents) and p_id is not null
),
leaf_nodes as (
    select distinct id, 'Leaf' as type
        from tree_tbl
    where id not in (select * from parents) and p_id is not null
)

select id, 'Root' as type
from tree_tbl where p_id is null
union
select * from inner_nodes
union
select * from leaf_nodes
order by id;
```

---

<br>
<br>
<br>


# 626. Exchange Seats

Table: `Seat`

| Column Name | Type    |
|-------------|---------|
| id          | int     |
| student     | varchar |

id is the primary key (unique value) column for this table.
Each row of this table indicates the name and the ID of a student.
The ID sequence always starts from 1 and increments continuously.
 
<br>
<br>


Write a solution to swap the seat id of every two consecutive students. If the number of students is odd, the id of the last student is not swapped.

Return the result table ordered by `id` in **ascending order**.

The result format is in the following example.

 <br>

Example 1:

Input: 
`Seat` table:

| id | student |
|----|---------|
| 1  | Abbot   |
| 2  | Doris   |
| 3  | Emerson |
| 4  | Green   |
| 5  | Jeames  |


<br>

Output: 

| id | student |
|----|---------|
| 1  | Doris   |
| 2  | Abbot   |
| 3  | Green   |
| 4  | Emerson |
| 5  | Jeames  |

Explanation: 
Note that if the number of students is odd, there is no need to change the last one's seat.

<br>
<br>

## Solution -

```python
max_id = seat.select(max("id").alias("max_id")).collect()[0][0]
seat.withColumn("swaped_id", when( (seat.id == max_id) & (seat.id % 2 != 0), seat.id ) \
                               .when(seat.id % 2 != 0, (seat.id|1) ) \
                               .when(seat.id % 2 == 0, (seat.id-1) ) ) \
    .select(col("swaped_id").alias("id"), col("student") ) \
    .orderBy(col("id")) \
    .show()
```

```sql
select
    CASE
        WHEN id = (select max(id) from seat_tbl) and id % 2 <> 0 THEN id
        WHEN (id % 2) <> 0 THEN (id | 1)
        WHEN (id % 2) = 0 THEN (id - 1)
    END as id,
    student
FROM seat_tbl
order by id
```

---

<br>
<br>
<br>

# 614. Second Degree Follower

In facebook, there is a `follow` table with two columns: **followee**, **follower**.

A **second-degree follower** is a user who:
* follows at least one user, and
* is followed by at least one user

Wrtie a solution to report the second-degree users and the number of their followers.

Return the result table **ordered** by `follower` in **alphabetical order.**

<br>

**For example:**

| followee    | follower   |
|-------------|------------|
|     A       |     B      |
|     B       |     C      |
|     B       |     D      |
|     D       |     E      |

<br>

**Output:**

| follower    | num        |
|-------------|------------|
|     B       |  2         |
|     D       |  1         |

<br>
<br>

## Solution -

```python
distinct_follower = follow.select("follower").distinct().withColumnRenamed("follower","dist_follower")

follower_count = follow.groupBy("followee") \
                    .count() \
                    .join(distinct_follower, distinct_follower.dist_follower == col("followee"), 'inner') \
                    .selectExpr("followee as follower","count as num")
follower_count.orderBy(col("follower")).show()
```

```sql
select followee as follower, count(*) as num
    from follow_tbl
group by followee
    having followee IN (select distinct follower from follow_tbl)
```

---

<br>
<br>
<br>

# 1045. Customers Who Bought All Products

| Column Name | Type    |
|-------------|---------|
| customer_id | int     |
| product_key | int     |

This table may contain duplicates rows. 
customer_id is not NULL.
product_key is a foreign key (reference column) to Product table.
 
<br>

Table: Product


| Column Name | Type    |
|-------------|---------|
| product_key | int     |

product_key is the primary key (column with unique values) for this table.
 
<br>
<br>


Write a solution to report the customer ids from the Customer table that bought all the products in the Product table.

Return the result table in any order.

The result format is in the following example.

<br>

Example 1:

Input: 
`Customer` table:

| customer_id | product_key |
|-------------|-------------|
| 1           | 5           |
| 2           | 6           |
| 3           | 5           |
| 3           | 6           |
| 1           | 6           |


<br>

`Product` table:

| product_key |
|-------------|
| 5           |
| 6           |


<br>

Output: 

| customer_id |
|-------------|
| 1           |
| 3           |

Explanation: 
The customers who bought all the products (5 and 6) are customers with IDs 1 and 3.

<br>
<br>

## Solution -

```python
dist_prod_exists = product.select(countDistinct("product_key")).collect()[0][0]

customer.groupBy("customer_id") \
        .agg( countDistinct("product_key").alias("dist_prod_cnt") ) \
        .filter(col("dist_prod_cnt") == dist_prod_exists) \
        .select("customer_id") \
        .show()
```

```sql
select customer_id
FROM customer_tbl
GROUP BY customer_id
having count(distinct product_key) = (select count(distinct product_key) from product_tbl)
```

---

<br>
<br>
<br>

# 1070. Product Sales Analysis III

Table: `Sales`


| Column Name | Type  |
|-------------|-------|
| sale_id     | int   |
| product_id  | int   |
| year        | int   |
| quantity    | int   |
| price       | int   |

(sale_id, year) is the primary key (combination of columns with unique values) of this table.
product_id is a foreign key (reference column) to `Product` table.
Each row of this table shows a sale on the product product_id in a certain year.
Note that the price is per unit.
 
<br>

Table: `Product`


| Column Name  | Type    |
|--------------|---------|
| product_id   | int     |
| product_name | varchar |

product_id is the primary key (column with unique values) of this table.
Each row of this table indicates the product name of each product.

<br>
<br>

Write a solution to select the **product id, year, quantity,** and **price** for the first year of every product sold.

Return the resulting table in **any order**.

The result format is in the following example.

<br>

Example 1:

Input: 
`Sales` table:

| sale_id | product_id | year | quantity | price |
|---------|------------|------|----------|-------|
| 1       | 100        | 2008 | 10       | 5000  |
| 2       | 100        | 2009 | 12       | 5000  |
| 7       | 200        | 2011 | 15       | 9000  |

<br>

`Product` table:

| product_id | product_name |
|------------|--------------|
| 100        | Nokia        |
| 200        | Apple        |
| 300        | Samsung      |

<br>
<br>

Output: 

| product_id | first_year | quantity | price |
|------------|------------|----------|-------| 
| 100        | 2008       | 10       | 5000  |
| 200        | 2011       | 15       | 9000  |

<br>
<br>

## Solution -

```python
first_sale = sales.groupBy("product_id").agg(min("year").alias("year"))
first_sale_renamed = first_sale.withColumnRenamed("product_id","fproduct_id").withColumnRenamed("year","fyear")

sales.join(first_sale_renamed, (sales.product_id == first_sale_renamed.fproduct_id) & (sales.year == first_sale_renamed.fyear), 'inner') \
    .selectExpr("product_id", "year as first_year", "quantity", "price") \
    .show()
```

```sql
select product_id, year as first_year, quantity, price
from sales_tbl
where (product_id, year) IN
-- First Sale
(select product_id, min(year) as year
from sales_tbl
group by product_id)
```

---




