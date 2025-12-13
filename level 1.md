# **SQL NEXT LEVEL - Advanced Oneshot**

## **1. WINDOW FUNCTIONS**
### **Core Concepts**
```sql
-- Basic window function syntax
SELECT 
    column1,
    column2,
    window_function() OVER (
        [PARTITION BY partition_expression]
        [ORDER BY sort_expression]
        [ROWS/RANGE frame_clause]
    ) AS result_column
FROM table_name;

-- ROW_NUMBER(): Unique sequential integer
SELECT 
    employee_id,
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;

-- RANK(): Rank with gaps for ties
SELECT 
    employee_id,
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS salary_rank
FROM employees;

-- DENSE_RANK(): Rank without gaps
SELECT 
    employee_id,
    name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_salary_rank
FROM employees;

-- NTILE(): Divide into buckets
SELECT 
    employee_id,
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary DESC) AS salary_quartile
FROM employees;
```

### **Analytic Functions**
```sql
-- LAG/LEAD: Access previous/next rows
SELECT 
    order_date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY order_date) AS prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY order_date) AS next_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY order_date) AS daily_growth
FROM daily_sales;

-- FIRST_VALUE/LAST_VALUE
SELECT 
    department,
    employee_id,
    name,
    salary,
    FIRST_VALUE(name) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) AS highest_paid_in_dept,
    LAST_VALUE(name) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid_in_dept
FROM employees;

-- Running totals and moving averages
SELECT 
    order_date,
    revenue,
    SUM(revenue) OVER (ORDER BY order_date) AS running_total,
    AVG(revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3day,
    SUM(revenue) OVER (
        PARTITION BY YEAR(order_date), MONTH(order_date)
        ORDER BY order_date
    ) AS monthly_running_total
FROM daily_sales;
```

### **Frame Clauses**
```sql
-- Different frame specifications
SELECT 
    order_date,
    revenue,
    -- Current row and previous 2 rows
    SUM(revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS last_3_days,
    
    -- All previous rows
    SUM(revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_total,
    
    -- 1 row before and 1 row after
    AVG(revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS centered_avg,
    
    -- Percentage of month
    revenue * 100.0 / SUM(revenue) OVER (
        PARTITION BY YEAR(order_date), MONTH(order_date)
    ) AS pct_of_monthly_total
FROM daily_sales;
```

## **2. ADVANCED JOINS**
### **Lateral Joins (PostgreSQL, Oracle)**
```sql
-- LATERAL JOIN: Correlated subquery that can reference previous tables
SELECT 
    d.department_name,
    e.employee_name,
    e.salary
FROM departments d
CROSS JOIN LATERAL (
    SELECT employee_name, salary
    FROM employees e
    WHERE e.department_id = d.department_id
    ORDER BY salary DESC
    LIMIT 3
) e;

-- With function calls
SELECT 
    u.user_id,
    u.username,
    recent_orders.*
FROM users u
CROSS JOIN LATERAL (
    SELECT order_id, order_date, total_amount
    FROM orders o
    WHERE o.user_id = u.user_id
    ORDER BY order_date DESC
    LIMIT 5
) recent_orders;
```

### **Natural & Using Joins**
```sql
-- NATURAL JOIN (joins on columns with same name - use cautiously)
SELECT * FROM employees NATURAL JOIN departments;

-- JOIN USING (explicit common column)
SELECT 
    e.employee_name,
    d.department_name
FROM employees e 
JOIN departments d USING (department_id);

-- Multi-table joins with USING
SELECT 
    e.name,
    d.department_name,
    p.project_name
FROM employees e
JOIN departments d USING (department_id)
JOIN projects p USING (department_id, location_id);
```

### **Recursive CTEs (Hierarchical Data)**
```sql
-- Employee hierarchy
WITH RECURSIVE employee_hierarchy AS (
    -- Anchor member: top-level employees (no manager)
    SELECT 
        employee_id,
        employee_name,
        manager_id,
        1 AS level,
        CAST(employee_name AS VARCHAR(1000)) AS hierarchy_path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive member: subordinates
    SELECT 
        e.employee_id,
        e.employee_name,
        e.manager_id,
        eh.level + 1,
        CAST(eh.hierarchy_path || ' -> ' || e.employee_name AS VARCHAR(1000))
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT 
    employee_id,
    employee_name,
    level,
    hierarchy_path
FROM employee_hierarchy
ORDER BY level, employee_name;

-- Bill of Materials (BOM) explosion
WITH RECURSIVE bom_explosion AS (
    SELECT 
        component_id,
        parent_component_id,
        quantity,
        1 AS level,
        component_name
    FROM components
    WHERE parent_component_id IS NULL
    
    UNION ALL
    
    SELECT 
        c.component_id,
        c.parent_component_id,
        c.quantity * be.quantity,
        be.level + 1,
        c.component_name
    FROM components c
    INNER JOIN bom_explosion be ON c.parent_component_id = be.component_id
)
SELECT 
    level,
    LPAD('', (level - 1) * 4, ' ') || component_name AS indented_name,
    quantity
FROM bom_explosion
ORDER BY level, component_name;
```

## **3. COMMON TABLE EXPRESSIONS (CTEs)**
### **Advanced CTE Patterns**
```sql
-- Multiple CTEs
WITH 
department_summary AS (
    SELECT 
        department_id,
        COUNT(*) AS emp_count,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
high_performing_depts AS (
    SELECT 
        department_id,
        emp_count
    FROM department_summary
    WHERE avg_salary > 50000
    AND emp_count > 10
),
employee_details AS (
    SELECT 
        e.*,
        d.department_name
    FROM employees e
    JOIN departments d ON e.department_id = d.department_id
    WHERE e.department_id IN (SELECT department_id FROM high_performing_depts)
)
SELECT * FROM employee_details;

-- Recursive CTE with cycle detection (PostgreSQL)
WITH RECURSIVE org_chart AS (
    SELECT 
        employee_id,
        manager_id,
        employee_name,
        1 AS depth,
        ARRAY[employee_id] AS path,
        FALSE AS cycle
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    SELECT 
        e.employee_id,
        e.manager_id,
        e.employee_name,
        oc.depth + 1,
        oc.path || e.employee_id,
        e.employee_id = ANY(oc.path)
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.employee_id
    WHERE NOT cycle
)
SELECT * FROM org_chart;

-- Updatable CTEs
WITH updated_salaries AS (
    UPDATE employees
    SET salary = salary * 1.05
    WHERE department_id = 1
    RETURNING employee_id, salary AS new_salary
)
SELECT 
    e.employee_name,
    e.salary AS old_salary,
    us.new_salary,
    (us.new_salary - e.salary) AS increase
FROM employees e
JOIN updated_salaries us ON e.employee_id = us.employee_id;
```

## **4. PIVOT & UNPIVOT**
### **PIVOT (Row to Column)**
```sql
-- Basic PIVOT (SQL Server, Oracle)
SELECT *
FROM (
    SELECT 
        department,
        EXTRACT(YEAR FROM hire_date) AS hire_year,
        employee_id
    FROM employees
) AS source
PIVOT (
    COUNT(employee_id)
    FOR hire_year IN ([2020], [2021], [2022], [2023], [2024])
) AS pivoted;

-- Dynamic PIVOT (SQL Server)
DECLARE @columns NVARCHAR(MAX), @sql NVARCHAR(MAX);

SELECT @columns = STRING_AGG(QUOTENAME(year), ',')
FROM (
    SELECT DISTINCT YEAR(hire_date) AS year
    FROM employees
) AS years;

SET @sql = '
SELECT department, ' + @columns + '
FROM (
    SELECT department, YEAR(hire_date) AS year, employee_id
    FROM employees
) AS source
PIVOT (
    COUNT(employee_id)
    FOR year IN (' + @columns + ')
) AS pivoted';

EXEC sp_executesql @sql;

-- Manual PIVOT with CASE (Cross-database)
SELECT 
    department,
    COUNT(CASE WHEN YEAR(hire_date) = 2020 THEN 1 END) AS hires_2020,
    COUNT(CASE WHEN YEAR(hire_date) = 2021 THEN 1 END) AS hires_2021,
    COUNT(CASE WHEN YEAR(hire_date) = 2022 THEN 1 END) AS hires_2022,
    AVG(CASE WHEN department = 'Sales' THEN salary END) AS avg_sales_salary,
    AVG(CASE WHEN department = 'IT' THEN salary END) AS avg_it_salary
FROM employees
GROUP BY department;
```

### **UNPIVOT (Column to Row)**
```sql
-- UNPIVOT
SELECT 
    department,
    metric_type,
    metric_value
FROM (
    SELECT 
        department,
        COUNT(*) AS employee_count,
        AVG(salary) AS avg_salary,
        SUM(salary) AS total_salary
    FROM employees
    GROUP BY department
) AS source
UNPIVOT (
    metric_value FOR metric_type IN (
        employee_count, 
        avg_salary, 
        total_salary
    )
) AS unpivoted;

-- Manual UNPIVOT with UNION ALL
SELECT department, 'employee_count' AS metric_type, employee_count AS metric_value
FROM department_stats
UNION ALL
SELECT department, 'avg_salary', avg_salary
FROM department_stats
UNION ALL
SELECT department, 'total_salary', total_salary
FROM department_stats
ORDER BY department, metric_type;
```

## **5. ADVANCED AGGREGATION**
### **GROUPING SETS, ROLLUP, CUBE**
```sql
-- GROUPING SETS: Multiple grouping levels in one query
SELECT 
    department,
    job_title,
    EXTRACT(YEAR FROM hire_date) AS hire_year,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY GROUPING SETS (
    (department, job_title),
    (department, EXTRACT(YEAR FROM hire_date)),
    (department),
    (job_title),
    ()
);

-- ROLLUP: Hierarchical aggregation
SELECT 
    department,
    job_title,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY ROLLUP (department, job_title);

-- CUBE: All possible combinations
SELECT 
    department,
    job_title,
    EXTRACT(YEAR FROM hire_date) AS hire_year,
    COUNT(*) AS employee_count
FROM employees
GROUP BY CUBE (department, job_title, EXTRACT(YEAR FROM hire_date));

-- GROUPING() function to identify summary rows
SELECT 
    department,
    job_title,
    COUNT(*) AS employee_count,
    GROUPING(department) AS dept_grouping,
    GROUPING(job_title) AS job_grouping,
    CASE 
        WHEN GROUPING(department) = 1 AND GROUPING(job_title) = 1 THEN 'Grand Total'
        WHEN GROUPING(job_title) = 1 THEN 'Department Total'
        ELSE 'Detailed'
    END AS row_type
FROM employees
GROUP BY ROLLUP (department, job_title);
```

## **6. JSON & XML FUNCTIONS**
### **JSON Operations (Modern RDBMS)**
```sql
-- Create JSON
SELECT 
    employee_id,
    employee_name,
    JSON_OBJECT(
        'id', employee_id,
        'name', employee_name,
        'department', department_name,
        'salary', salary,
        'projects', JSON_ARRAY(project1, project2, project3)
    ) AS employee_json
FROM employees;

-- Query JSON data
SELECT 
    employee_data->>'name' AS employee_name,
    employee_data->>'department' AS department,
    employee_data->'salary' AS salary,
    JSON_EXTRACT(employee_data, '$.projects[0]') AS first_project
FROM employee_json_table;

-- JSON functions
SELECT 
    JSON_ARRAY_AGG(employee_name) AS all_employees,
    JSON_OBJECT_AGG(department, employee_count) AS dept_counts,
    JSON_PRETTY(employee_data) AS formatted_json,
    JSON_VALID(employee_data) AS is_valid_json,
    JSON_KEYS(employee_data) AS json_keys
FROM employees;

-- JSON table functions
SELECT *
FROM JSON_TABLE(
    '[{"id":1,"name":"John"},{"id":2,"name":"Jane"}]',
    '$[*]' COLUMNS (
        id INT PATH '$.id',
        name VARCHAR(50) PATH '$.name'
    )
) AS jt;
```

## **7. FULL-TEXT SEARCH**
```sql
-- Create full-text index (SQL Server)
CREATE FULLTEXT CATALOG ft_catalog AS DEFAULT;
CREATE FULLTEXT INDEX ON products(description)
KEY INDEX pk_products
WITH STOPLIST = SYSTEM;

-- Full-text search queries
SELECT 
    product_id,
    product_name,
    description,
    RANK() OVER (ORDER BY score DESC) AS relevance_rank
FROM products
WHERE CONTAINS(description, '"database management" OR "data storage"');

-- Using FREETEXT for natural language
SELECT *
FROM documents
WHERE FREETEXT(content, 'machine learning algorithms');

-- Proximity search
SELECT *
FROM articles
WHERE CONTAINS(
    content, 
    'NEAR((data, analysis), 5, TRUE)'  -- Within 5 words, ordered
);

-- Thesaurus and stopwords
SELECT *
FROM products
WHERE CONTAINS(
    description, 
    'FORMSOF(INFLECTIONAL, "run")'  -- Finds run, running, ran
);
```

## **8. ADVANCED SUBQUERIES**
### **Correlated Subquery Optimization**
```sql
-- EXISTS vs IN performance
-- Use EXISTS for correlated subqueries
SELECT d.department_name
FROM departments d
WHERE EXISTS (
    SELECT 1
    FROM employees e
    WHERE e.department_id = d.department_id
    AND e.salary > 100000
);

-- Lateral derived tables
SELECT 
    d.department_name,
    top_emp.employee_name,
    top_emp.salary
FROM departments d
CROSS APPLY (
    SELECT TOP 3 employee_name, salary
    FROM employees e
    WHERE e.department_id = d.department_id
    ORDER BY salary DESC
) AS top_emp;

-- Recursive correlated subquery
WITH RECURSIVE manager_chain AS (
    SELECT 
        employee_id,
        manager_id,
        employee_name,
        1 AS level
    FROM employees
    WHERE employee_id = 1234  -- Start with specific employee
    
    UNION ALL
    
    SELECT 
        e.employee_id,
        e.manager_id,
        e.employee_name,
        mc.level + 1
    FROM employees e
    JOIN manager_chain mc ON e.employee_id = mc.manager_id
)
SELECT * FROM manager_chain;
```

## **9. PERFORMANCE OPTIMIZATION**
### **Query Tuning Techniques**
```sql
-- Index hints
SELECT /*+ INDEX(employees idx_salary) */
    employee_id, employee_name, salary
FROM employees
WHERE salary BETWEEN 50000 AND 70000;

-- Force join order
SELECT /*+ ORDERED */
    e.employee_name,
    d.department_name,
    p.project_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id
JOIN projects p ON d.department_id = p.department_id;

-- Materialized CTEs (PostgreSQL)
WITH MATERIALIZED department_stats AS (
    SELECT 
        department_id,
        COUNT(*) AS emp_count,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT * FROM department_stats;

-- Query plan analysis
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT e.*, d.department_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE e.salary > 50000;

-- Optimizer hints for specific RDBMS
-- MySQL: STRAIGHT_JOIN, SQL_NO_CACHE
-- SQL Server: OPTION (MAXDOP 2, RECOMPILE)
-- Oracle: /*+ PARALLEL(employees, 4) */
```

### **Indexing Strategies**
```sql
-- Composite indexes
CREATE INDEX idx_employee_search 
ON employees(department_id, hire_date DESC, salary);

-- Covering indexes
CREATE INDEX idx_covering 
ON employees(department_id, salary) 
INCLUDE (employee_name, email);

-- Filtered indexes (SQL Server)
CREATE INDEX idx_active_high_salary 
ON employees(salary) 
WHERE is_active = 1 AND salary > 100000;

-- Function-based indexes (Oracle, PostgreSQL)
CREATE INDEX idx_lower_name ON employees(LOWER(employee_name));

-- Partial indexes (PostgreSQL)
CREATE INDEX idx_nyc_customers 
ON customers(email) 
WHERE city = 'New York';

-- Spatial indexes
CREATE SPATIAL INDEX idx_geo_location 
ON properties(location);
```

## **10. TEMPORAL TABLES**
```sql
-- System-versioned temporal tables (SQL Server 2016+, SQL:2011)
CREATE TABLE Employees
(
    EmployeeID INT PRIMARY KEY,
    Name VARCHAR(100),
    Department VARCHAR(50),
    Salary DECIMAL(10,2),
    ValidFrom DATETIME2 GENERATED ALWAYS AS ROW START,
    ValidTo DATETIME2 GENERATED ALWAYS AS ROW END,
    PERIOD FOR SYSTEM_TIME (ValidFrom, ValidTo)
)
WITH (SYSTEM_VERSIONING = ON (HISTORY_TABLE = dbo.EmployeesHistory));

-- Query temporal data
-- Current data
SELECT * FROM Employees;

-- Historical data at specific time
SELECT * FROM Employees
FOR SYSTEM_TIME AS OF '2024-01-01';

-- Data in time range
SELECT * FROM Employees
FOR SYSTEM_TIME BETWEEN '2023-01-01' AND '2024-01-01';

-- All history
SELECT * FROM Employees
FOR SYSTEM_TIME ALL;

-- Temporal joins
SELECT 
    e_current.EmployeeID,
    e_current.Name,
    e_current.Department AS current_dept,
    e_history.Department AS old_dept,
    e_history.ValidFrom,
    e_history.ValidTo
FROM Employees e_current
JOIN Employees FOR SYSTEM_TIME AS OF '2023-06-01' e_history
    ON e_current.EmployeeID = e_history.EmployeeID;
```

## **11. PARTITIONING**
```sql
-- Range partitioning
CREATE TABLE sales (
    sale_id INT,
    sale_date DATE,
    amount DECIMAL(10,2),
    region VARCHAR(50)
)
PARTITION BY RANGE (YEAR(sale_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- List partitioning
CREATE TABLE customers (
    customer_id INT,
    name VARCHAR(100),
    country VARCHAR(50)
)
PARTITION BY LIST (country) (
    PARTITION p_na VALUES IN ('USA', 'Canada', 'Mexico'),
    PARTITION p_eu VALUES IN ('UK', 'Germany', 'France', 'Italy'),
    PARTITION p_asia VALUES IN ('Japan', 'China', 'India'),
    PARTITION p_other VALUES IN (DEFAULT)
);

-- Hash partitioning
CREATE TABLE logs (
    log_id INT,
    log_time TIMESTAMP,
    message TEXT
)
PARTITION BY HASH(log_id)
PARTITIONS 8;

-- Composite partitioning
CREATE TABLE orders (
    order_id INT,
    order_date DATE,
    customer_id INT,
    total_amount DECIMAL(10,2)
)
PARTITION BY RANGE (YEAR(order_date))
SUBPARTITION BY HASH(customer_id) SUBPARTITIONS 4 (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023)
);

-- Partition management
ALTER TABLE sales DROP PARTITION p2020;
ALTER TABLE sales TRUNCATE PARTITION p2021;
ALTER TABLE sales ADD PARTITION p2025 VALUES LESS THAN (2026);
```

## **12. ADVANCED DATABASE OBJECTS**
### **Stored Procedures with Advanced Features**
```sql
-- Dynamic SQL in procedures
CREATE PROCEDURE dynamic_search (
    @table_name NVARCHAR(100),
    @search_column NVARCHAR(100),
    @search_value NVARCHAR(100)
)
AS
BEGIN
    DECLARE @sql NVARCHAR(MAX);
    
    SET @sql = N'SELECT * FROM ' + QUOTENAME(@table_name) +
               N' WHERE ' + QUOTENAME(@search_column) + N' = @value';
    
    EXEC sp_executesql @sql, N'@value NVARCHAR(100)', @search_value;
END;

-- Error handling with transactions
CREATE PROCEDURE transfer_funds (
    @from_account INT,
    @to_account INT,
    @amount DECIMAL(10,2)
)
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    
    BEGIN TRY
        BEGIN TRANSACTION;
        
        -- Check balance
        IF (SELECT balance FROM accounts WHERE account_id = @from_account) < @amount
            THROW 51000, 'Insufficient funds', 1;
        
        -- Deduct from source
        UPDATE accounts 
        SET balance = balance - @amount 
        WHERE account_id = @from_account;
        
        -- Add to destination
        UPDATE accounts 
        SET balance = balance + @amount 
        WHERE account_id = @to_account;
        
        -- Log transaction
        INSERT INTO transactions (from_account, to_account, amount, timestamp)
        VALUES (@from_account, @to_account, @amount, GETDATE());
        
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;
        
        -- Re-throw error
        THROW;
    END CATCH
END;
```

### **User-Defined Functions**
```sql
-- Scalar function
CREATE FUNCTION dbo.CalculateBonus(@salary DECIMAL(10,2), @rating INT)
RETURNS DECIMAL(10,2)
AS
BEGIN
    DECLARE @bonus DECIMAL(10,2);
    
    SET @bonus = CASE @rating
        WHEN 1 THEN @salary * 0.20
        WHEN 2 THEN @salary * 0.15
        WHEN 3 THEN @salary * 0.10
        ELSE @salary * 0.05
    END;
    
    RETURN @bonus;
END;

-- Table-valued function
CREATE FUNCTION dbo.GetDepartmentEmployees(@dept_id INT)
RETURNS TABLE
AS
RETURN (
    SELECT 
        employee_id,
        employee_name,
        salary,
        hire_date
    FROM employees
    WHERE department_id = @dept_id
    AND is_active = 1
);

-- Inline table-valued function
CREATE FUNCTION dbo.GetEmployeeHierarchy(@employee_id INT)
RETURNS @hierarchy TABLE (
    employee_id INT,
    employee_name VARCHAR(100),
    level INT,
    path VARCHAR(1000)
)
AS
BEGIN
    WITH RECURSIVE emp_cte AS (
        SELECT 
            employee_id,
            employee_name,
            manager_id,
            1 AS level,
            CAST(employee_name AS VARCHAR(1000)) AS path
        FROM employees
        WHERE employee_id = @employee_id
        
        UNION ALL
        
        SELECT 
            e.employee_id,
            e.employee_name,
            e.manager_id,
            ec.level + 1,
            CAST(ec.path + ' > ' + e.employee_name AS VARCHAR(1000))
        FROM employees e
        INNER JOIN emp_cte ec ON e.employee_id = ec.manager_id
    )
    INSERT INTO @hierarchy
    SELECT employee_id, employee_name, level, path
    FROM emp_cte;
    
    RETURN;
END;
```

## **13. ADVANCED ANALYTICS**
### **Statistical Functions**
```sql
-- Window statistical functions
SELECT 
    department,
    employee_name,
    salary,
    PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) AS pct_rank,
    CUME_DIST() OVER (PARTITION BY department ORDER BY salary) AS cumulative_dist,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) 
        OVER (PARTITION BY department) AS median_salary,
    PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY salary) 
        OVER (PARTITION BY department) AS third_quartile,
    STDDEV(salary) OVER (PARTITION BY department) AS salary_stddev,
    VARIANCE(salary) OVER (PARTITION BY department) AS salary_variance
FROM employees;

-- Advanced analytics with MODEL clause (Oracle)
SELECT 
    department,
    month,
    sales,
    forecast,
    confidence
FROM sales_data
MODEL
    PARTITION BY (department)
    DIMENSION BY (month)
    MEASURES (sales, 0 AS forecast, 0 AS confidence)
    RULES (
        forecast[FOR month FROM 1 TO 12 INCREMENT 1] = 
            AVG(sales)[CV(month)-3 TO CV(month)-1],
        confidence[ANY] = 0.95
    );
```

## **14. DATABASE ADMINISTRATION SQL**
```sql
-- User and permission management
CREATE ROLE data_analyst;
GRANT SELECT ON SCHEMA::sales TO data_analyst;
GRANT EXECUTE ON dbo.GetSalesReport TO data_analyst;
GRANT data_analyst TO john_doe;

-- Row-level security (SQL Server 2016+)
CREATE SECURITY POLICY EmployeeSecurityPolicy
ADD FILTER PREDICATE dbo.fn_securitypredicate(employee_id) ON dbo.Employees
WITH (STATE = ON);

-- Dynamic data masking
ALTER TABLE customers
ALTER COLUMN email ADD MASKED WITH (FUNCTION = 'email()');

ALTER TABLE employees
ALTER COLUMN salary ADD MASKED WITH (FUNCTION = 'random(50000, 100000)');

-- Audit logging
CREATE TABLE audit_log (
    log_id INT IDENTITY PRIMARY KEY,
    table_name VARCHAR(100),
    operation CHAR(1),
    old_data XML,
    new_data XML,
    changed_by VARCHAR(100),
    changed_at DATETIME DEFAULT GETDATE()
);

CREATE TRIGGER trg_audit_employees
ON employees
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;
    
    INSERT INTO audit_log (table_name, operation, old_data, new_data, changed_by)
    SELECT 
        'employees',
        CASE 
            WHEN EXISTS(SELECT 1 FROM inserted) AND EXISTS(SELECT 1 FROM deleted) THEN 'U'
            WHEN EXISTS(SELECT 1 FROM inserted) THEN 'I'
            ELSE 'D'
        END,
        (SELECT * FROM deleted FOR XML AUTO),
        (SELECT * FROM inserted FOR XML AUTO),
        SYSTEM_USER;
END;
```

## **15. PERFORMANCE PATTERNS**
### **Advanced Optimization Techniques**
```sql
-- Delayed joins
SELECT 
    e.employee_id,
    e.employee_name,
    d.department_name
FROM employees e
JOIN (
    SELECT department_id, department_name
    FROM departments
    WHERE location = 'Headquarters'
) d ON e.department_id = d.department_id
WHERE e.salary > 50000;

-- Batch processing with keyset pagination
DECLARE @last_id INT = 0;
DECLARE @batch_size INT = 1000;

WHILE 1 = 1
BEGIN
    SELECT TOP (@batch_size)
        order_id,
        customer_id,
        order_date,
        total_amount
    FROM orders
    WHERE order_id > @last_id
    ORDER BY order_id;
    
    IF @@ROWCOUNT = 0
        BREAK;
    
    SET @last_id = (SELECT MAX(order_id) FROM #temp);
END;

-- Materialized view for complex aggregations
CREATE MATERIALIZED VIEW mv_department_stats
AS
SELECT 
    department_id,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    SUM(salary) AS total_salary,
    MAX(hire_date) AS latest_hire,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees
GROUP BY department_id
WITH DATA;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_department_stats;
```

## **16. REAL-WORLD COMPLEX QUERIES**
```sql
-- Customer Lifetime Value (CLV) calculation
WITH customer_metrics AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT order_id) AS order_count,
        SUM(total_amount) AS total_spent,
        MIN(order_date) AS first_order_date,
        MAX(order_date) AS last_order_date,
        DATEDIFF(DAY, MIN(order_date), MAX(order_date)) AS customer_lifespan_days
    FROM orders
    GROUP BY customer_id
),
customer_segments AS (
    SELECT 
        customer_id,
        total_spent,
        order_count,
        CASE 
            WHEN total_spent > 10000 THEN 'VIP'
            WHEN total_spent BETWEEN 5000 AND 10000 THEN 'Premium'
            WHEN total_spent BETWEEN 1000 AND 5000 THEN 'Regular'
            ELSE 'Casual'
        END AS segment,
        total_spent / NULLIF(customer_lifespan_days, 0) * 365 AS projected_annual_value
    FROM customer_metrics
)
SELECT 
    segment,
    COUNT(*) AS customer_count,
    AVG(total_spent) AS avg_total_spent,
    AVG(order_count) AS avg_orders,
    SUM(projected_annual_value) AS total_projected_annual_value
FROM customer_segments
GROUP BY segment
ORDER BY total_projected_annual_value DESC;

-- Churn analysis with window functions
WITH user_sessions AS (
    SELECT 
        user_id,
        session_date,
        LAG(session_date) OVER (PARTITION BY user_id ORDER BY session_date) AS prev_session_date,
        LEAD(session_date) OVER (PARTITION BY user_id ORDER BY session_date) AS next_session_date
    FROM user_activity
),
churn_analysis AS (
    SELECT 
        user_id,
        session_date,
        DATEDIFF(DAY, prev_session_date, session_date) AS days_since_last_session,
        DATEDIFF(DAY, session_date, next_session_date) AS days_until_next_session,
        CASE 
            WHEN DATEDIFF(DAY, session_date, COALESCE(next_session_date, GETDATE())) > 30 
            THEN 1 
            ELSE 0 
        END AS churned
    FROM user_sessions
)
SELECT 
    DATEPART(MONTH, session_date) AS month,
    DATEPART(YEAR, session_date) AS year,
    COUNT(DISTINCT user_id) AS active_users,
    SUM(churned) AS churned_users,
    100.0 * SUM(churned) / COUNT(DISTINCT user_id) AS churn_rate_pct,
    AVG(days_since_last_session) AS avg_session_gap
FROM churn_analysis
GROUP BY DATEPART(YEAR, session_date), DATEPART(MONTH, session_date)
ORDER BY year, month;
```

---

## **KEY TAKEAWAYS**

1. **Window Functions** - Essential for analytical queries
2. **CTEs & Recursive Queries** - Break complex problems into steps
3. **Advanced Joins** - LATERAL, CROSS APPLY for complex relationships
4. **Performance Tuning** - Indexing strategies, query hints, partitioning
5. **Temporal Tables** - Built-in history tracking
6. **Full-Text Search** - Efficient text searching
7. **JSON/XML** - Handle semi-structured data
8. **Security** - Row-level security, data masking
9. **Materialized Views** - Pre-compute complex aggregations
10. **Dynamic SQL** - Build flexible queries

---

**Mastering these concepts will make you proficient in handling real-world, complex SQL scenarios in enterprise environments.**
