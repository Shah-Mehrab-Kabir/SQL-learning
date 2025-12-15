-- First, list databases to confirm
SHOW DATABASES;
-- Then delete (choose one method)
DROP DATABASE franchises;
-- DROP DATABASE IF EXISTS database_name;  -- Safer: won't error if doesn't exist

-- Get rows where a number is not equal to a value with WHERE col <> n or WHERE col != n
SELECT franchise, inception_year
FROM franchises
WHERE inception_year <> 1996

-- Get the total number of rows SELECT COUNT(*)
SELECT COUNT(*)
FROM franchises

