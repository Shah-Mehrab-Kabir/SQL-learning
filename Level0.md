# **SQL Basics - Detailed Oneshot**

## **1. SQL OVERVIEW**
**SQL (Structured Query Language)** - Standard language for managing relational databases
- **DDL (Data Definition Language)** - Define/modify structure
- **DML (Data Manipulation Language)** - Manipulate data
- **DQL (Data Query Language)** - Query data (primarily SELECT)
- **DCL (Data Control Language)** - Control access
- **TCL (Transaction Control Language)** - Manage transactions

## **2. BASIC COMMANDS**

### **DDL Commands**
```sql
-- CREATE DATABASE
CREATE DATABASE database_name;

-- CREATE TABLE
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    age INT,
    department VARCHAR(50),
    salary DECIMAL(10, 2),
    hire_date DATE DEFAULT CURRENT_DATE
);

-- ALTER TABLE
ALTER TABLE employees ADD email VARCHAR(100);
ALTER TABLE employees MODIFY COLUMN name VARCHAR(100);
ALTER TABLE employees DROP COLUMN age;

-- DROP TABLE
DROP TABLE employees;

-- TRUNCATE TABLE (removes all data but keeps structure)
TRUNCATE TABLE employees;
```

### **DML Commands**
```sql
-- INSERT
INSERT INTO employees (id, name, department, salary)
VALUES (1, 'John Doe', 'IT', 50000);

-- Multiple rows
INSERT INTO employees VALUES
(2, 'Jane Smith', 'HR', 45000),
(3, 'Bob Johnson', 'IT', 55000);

-- UPDATE
UPDATE employees 
SET salary = salary * 1.10 
WHERE department = 'IT';

-- DELETE
DELETE FROM employees WHERE id = 3;
DELETE FROM employees WHERE salary < 40000;
```

## **3. DATA TYPES**

### **Numeric Types**
- `INT` / `INTEGER` - Whole numbers
- `DECIMAL(p,s)` / `NUMERIC(p,s)` - Exact numbers (p=precision, s=scale)
- `FLOAT` / `REAL` - Approximate numbers
- `SMALLINT`, `BIGINT` - Smaller/larger integers

### **String Types**
- `CHAR(n)` - Fixed length string (padded with spaces)
- `VARCHAR(n)` - Variable length string
- `TEXT` - Large text data

### **Date/Time Types**
- `DATE` - YYYY-MM-DD
- `TIME` - HH:MM:SS
- `DATETIME` / `TIMESTAMP` - Date and time
- `YEAR`

### **Other Types**
- `BOOLEAN` / `BOOL` - TRUE/FALSE
- `BLOB` - Binary large objects

## **4. CONSTRAINTS**
```sql
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10, 2) CHECK (price > 0),
    stock_quantity INT DEFAULT 0,
    supplier_id INT,
    
    -- Table-level constraints
    UNIQUE (product_name, category),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
```

**Constraint Types:**
- `PRIMARY KEY` - Unique identifier
- `FOREIGN KEY` - References another table
- `NOT NULL` - Cannot be empty
- `UNIQUE` - All values must be different
- `CHECK` - Validates condition
- `DEFAULT` - Default value if not specified

## **5. SELECT QUERY - CORE SYNTAX**
```sql
-- Basic structure
SELECT column1, column2, ...
FROM table_name
WHERE condition
GROUP BY column
HAVING group_condition
ORDER BY column [ASC|DESC]
LIMIT number;

-- Examples
SELECT * FROM employees;

SELECT name, department, salary 
FROM employees 
WHERE department = 'IT' 
ORDER BY salary DESC;

SELECT DISTINCT department FROM employees;

-- Aliases
SELECT 
    name AS employee_name,
    salary * 12 AS annual_salary
FROM employees;
```

## **6. FILTERING WITH WHERE**
```sql
-- Comparison operators
SELECT * FROM employees WHERE salary > 50000;
SELECT * FROM employees WHERE hire_date >= '2023-01-01';

-- Logical operators
SELECT * FROM employees 
WHERE department = 'IT' AND salary > 40000;

SELECT * FROM employees 
WHERE department = 'HR' OR department = 'Finance';

SELECT * FROM employees 
WHERE NOT department = 'IT';

-- IN operator
SELECT * FROM employees 
WHERE department IN ('IT', 'HR', 'Finance');

-- BETWEEN
SELECT * FROM employees 
WHERE salary BETWEEN 40000 AND 60000;

-- LIKE (pattern matching)
SELECT * FROM employees 
WHERE name LIKE 'J%';  -- Starts with J

SELECT * FROM employees 
WHERE name LIKE '%son%';  -- Contains 'son'

SELECT * FROM employees 
WHERE name LIKE '_o%';  -- Second letter is 'o'

-- IS NULL / IS NOT NULL
SELECT * FROM employees 
WHERE department IS NULL;
```

## **7. FUNCTIONS**

### **Aggregate Functions**
```sql
SELECT 
    COUNT(*) AS total_employees,
    AVG(salary) AS avg_salary,
    SUM(salary) AS total_salary,
    MAX(salary) AS highest_salary,
    MIN(salary) AS lowest_salary,
    COUNT(DISTINCT department) AS unique_depts
FROM employees;
```

### **String Functions**
```sql
SELECT 
    UPPER(name) AS uppercase_name,
    LOWER(department) AS lowercase_dept,
    CONCAT(name, ' - ', department) AS full_info,
    SUBSTRING(name, 1, 3) AS name_prefix,
    LENGTH(name) AS name_length,
    TRIM('  ' FROM name) AS trimmed_name,
    REPLACE(department, 'IT', 'Information Technology') AS dept_full
FROM employees;
```

### **Date Functions**
```sql
SELECT 
    CURRENT_DATE AS today,
    CURRENT_TIME AS now_time,
    CURRENT_TIMESTAMP AS current_ts,
    YEAR(hire_date) AS hire_year,
    MONTH(hire_date) AS hire_month,
    DAY(hire_date) AS hire_day,
    DATEDIFF(CURRENT_DATE, hire_date) AS days_employed,
    DATE_ADD(hire_date, INTERVAL 1 YEAR) AS anniversary
FROM employees;
```

### **Mathematical Functions**
```sql
SELECT 
    ROUND(salary, 0) AS rounded_salary,
    CEIL(salary) AS ceiling_salary,
    FLOOR(salary) AS floor_salary,
    ABS(salary_difference) AS absolute_difference,
    POWER(salary, 2) AS salary_squared,
    SQRT(salary) AS salary_sqrt
FROM employees;
```

## **8. GROUP BY & HAVING**
```sql
-- Group by department
SELECT 
    department,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    SUM(salary) AS total_salary
FROM employees
GROUP BY department;

-- Multiple grouping columns
SELECT 
    department,
    YEAR(hire_date) AS hire_year,
    COUNT(*) AS hires_count
FROM employees
GROUP BY department, YEAR(hire_date);

-- HAVING (filter groups)
SELECT 
    department,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 45000;

-- WHERE vs HAVING
SELECT department, AVG(salary)
FROM employees
WHERE hire_date > '2020-01-01'  -- Filter rows BEFORE grouping
GROUP BY department
HAVING COUNT(*) > 5;  -- Filter groups AFTER grouping
```

## **9. JOINS**
```sql
-- Sample tables for joins
CREATE TABLE departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    emp_name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

-- INNER JOIN (only matching records)
SELECT e.emp_name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- LEFT JOIN (all from left + matching from right)
SELECT e.emp_name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- RIGHT JOIN (all from right + matching from left)
SELECT e.emp_name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- FULL OUTER JOIN (all records from both)
SELECT e.emp_name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- CROSS JOIN (Cartesian product)
SELECT e.emp_name, d.dept_name
FROM employees e
CROSS JOIN departments d;

-- SELF JOIN
SELECT e1.emp_name AS employee, e2.emp_name AS manager
FROM employees e1
JOIN employees e2 ON e1.manager_id = e2.emp_id;
```

## **10. SUBQUERIES**
```sql
-- Single-value subquery
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Multi-value subquery
SELECT name, department
FROM employees
WHERE department IN (
    SELECT dept_name 
    FROM departments 
    WHERE location = 'New York'
);

-- Correlated subquery
SELECT e.name, e.salary, e.department
FROM employees e
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department = e.department
);

-- EXISTS
SELECT dept_name
FROM departments d
WHERE EXISTS (
    SELECT 1
    FROM employees e
    WHERE e.dept_id = d.dept_id
);

-- Subquery in SELECT
SELECT 
    name,
    salary,
    (SELECT AVG(salary) FROM employees) AS company_avg,
    salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees;
```

## **11. SET OPERATIONS**
```sql
-- UNION (distinct rows from both)
SELECT product_name FROM current_products
UNION
SELECT product_name FROM discontinued_products;

-- UNION ALL (all rows including duplicates)
SELECT city FROM suppliers
UNION ALL
SELECT city FROM customers;

-- INTERSECT (common rows)
SELECT customer_id FROM online_orders
INTERSECT
SELECT customer_id FROM in_store_orders;

-- EXCEPT / MINUS (rows in first but not second)
SELECT product_id FROM all_products
EXCEPT
SELECT product_id FROM discontinued_products;
```

## **12. VIEWS**
```sql
-- Create view
CREATE VIEW employee_summary AS
SELECT 
    department,
    COUNT(*) AS emp_count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department;

-- Use view
SELECT * FROM employee_summary 
WHERE avg_salary > 50000;

-- Update view
CREATE OR REPLACE VIEW high_earners AS
SELECT name, salary, department
FROM employees
WHERE salary > 70000;

-- Drop view
DROP VIEW employee_summary;
```

## **13. INDEXES**
```sql
-- Create index
CREATE INDEX idx_department ON employees(department);

-- Create composite index
CREATE INDEX idx_dept_salary ON employees(department, salary);

-- Create unique index
CREATE UNIQUE INDEX idx_email ON employees(email);

-- Drop index
DROP INDEX idx_department ON employees;
```

## **14. TRANSACTIONS**
```sql
START TRANSACTION;

INSERT INTO orders (order_id, customer_id, total) 
VALUES (1001, 501, 250.00);

INSERT INTO order_items (order_id, product_id, quantity) 
VALUES (1001, 123, 2);

-- Commit or rollback based on conditions
IF (no_errors) THEN
    COMMIT;  -- Save changes permanently
ELSE
    ROLLBACK;  -- Undo all changes in transaction
END IF;

-- Auto-commit settings
SET autocommit = 0;  -- Manual commit mode
SET autocommit = 1;  -- Auto-commit mode (default)
```

## **15. PRACTICE EXAMPLES**

### **Complex Query Example**
```sql
-- Find departments with above-average salaries
SELECT 
    d.dept_name,
    COUNT(e.emp_id) AS employee_count,
    AVG(e.salary) AS avg_salary,
    MAX(e.salary) AS max_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
WHERE e.hire_date >= '2020-01-01'
GROUP BY d.dept_id, d.dept_name
HAVING AVG(e.salary) > (
    SELECT AVG(salary) 
    FROM employees
)
ORDER BY avg_salary DESC
LIMIT 10;
```

### **Case Statement**
```sql
SELECT 
    name,
    salary,
    CASE
        WHEN salary < 40000 THEN 'Junior'
        WHEN salary BETWEEN 40000 AND 70000 THEN 'Mid'
        WHEN salary > 70000 THEN 'Senior'
        ELSE 'Unknown'
    END AS level,
    CASE department
        WHEN 'IT' THEN 'Technology'
        WHEN 'HR' THEN 'Human Resources'
        ELSE 'Other'
    END AS dept_category
FROM employees;
```

## **16. BEST PRACTICES**

1. **Use meaningful table/column names**
2. **Always use WHERE clause with UPDATE/DELETE**
3. **Use transactions for multiple related operations**
4. **Create indexes on frequently searched columns**
5. **Normalize data to reduce redundancy**
6. **Use JOINs instead of multiple queries**
7. **Avoid SELECT * in production**
8. **Use LIMIT with large datasets**
9. **Regularly backup databases**
10. **Use parameterized queries to prevent SQL injection**

## **17. COMMON PITFALLS TO AVOID**

1. **Missing indexes on foreign keys**
2. **Using functions on indexed columns in WHERE**
3. **Not using transactions for related operations**
4. **Over-normalizing data**
5. **Ignoring NULL values in calculations**
6. **Forgetting to COMMIT or ROLLBACK**
7. **Not backing up before major changes**

## **QUICK REFERENCE**

| Operation | Syntax |
|-----------|--------|
| Create DB | `CREATE DATABASE db_name;` |
| Create Table | `CREATE TABLE tname (col TYPE, ...);` |
| Insert | `INSERT INTO table VALUES (...);` |
| Select | `SELECT cols FROM table WHERE cond;` |
| Update | `UPDATE table SET col=val WHERE cond;` |
| Delete | `DELETE FROM table WHERE cond;` |
| Join | `SELECT * FROM t1 JOIN t2 ON condition;` |
| Group By | `SELECT col, COUNT(*) FROM table GROUP BY col;` |

---

**Next Steps:** Practice with real datasets, learn about window functions, stored procedures, triggers, and database optimization techniques.
