SQL (Structured Query Language) is mainly used to **manage and work with data stored in relational databases**. Its usage is usually grouped into the following five categories:

---

## 1️⃣ Data Definition (DDL – Data Definition Language)

**Purpose:**
Used to **define, modify, and delete database structures** such as tables, views, indexes, schemas, etc.

**What it controls:**

* Structure of the database
* Table design (columns, data types, constraints)

**Common commands:**

* `CREATE` → create database objects
* `ALTER` → modify existing objects
* `DROP` → delete objects
* `TRUNCATE` → remove all records from a table (structure remains)

**Example:**

```sql
CREATE TABLE Students (
    ID INT PRIMARY KEY,
    Name VARCHAR(50),
    Age INT
);
```

---

## 2️⃣ Data Retrieval (DQL – Data Query Language)

**Purpose:**
Used to **retrieve data from one or more tables**.

**What it does:**

* Reads data only
* Does **not change** the database

**Main command:**

* `SELECT`

**Example:**

```sql
SELECT Name, Age
FROM Students
WHERE Age > 18;
```

---

## 3️⃣ Data Manipulation (DML – Data Manipulation Language)

**Purpose:**
Used to **insert, update, and delete data** stored in tables.

**What it affects:**

* Actual data inside tables
* Does not change table structure

**Common commands:**

* `INSERT` → add new records
* `UPDATE` → modify existing records
* `DELETE` → remove records

**Examples:**

```sql
INSERT INTO Students VALUES (1, 'Rahim', 20);

UPDATE Students
SET Age = 21
WHERE ID = 1;

DELETE FROM Students
WHERE ID = 1;
```

---

## 4️⃣ Access Control (DCL – Data Control Language)

**Purpose:**
Used to **control access and permissions** on the database.

**What it manages:**

* User privileges
* Security of data

**Common commands:**

* `GRANT` → give permission
* `REVOKE` → remove permission

**Example:**

```sql
GRANT SELECT, INSERT ON Students TO user1;

REVOKE INSERT ON Students FROM user1;
```

---

## 5️⃣ Transaction Control (TCL – Transaction Control Language)

**Purpose:**
Used to **manage database transactions** to ensure data integrity.

**What it controls:**

* Whether changes are saved or undone
* Ensures ACID properties

**Common commands:**

* `COMMIT` → save changes permanently
* `ROLLBACK` → undo changes
* `SAVEPOINT` → set a point to rollback to

**Example:**

```sql
BEGIN;

UPDATE Students SET Age = 22 WHERE ID = 2;

ROLLBACK;   -- changes undone
```

---

## 🔑 Summary Table

| Category            | Purpose                   | Key Commands           |
| ------------------- | ------------------------- | ---------------------- |
| Data Definition     | Define database structure | CREATE, ALTER, DROP    |
| Data Retrieval      | Read data                 | SELECT                 |
| Data Manipulation   | Change data               | INSERT, UPDATE, DELETE |
| Access Control      | Manage permissions        | GRANT, REVOKE          |
| Transaction Control | Manage transactions       | COMMIT, ROLLBACK       |

---





In **Database Management Systems (DBMS)**, **ACID properties** ensure that **transactions are processed reliably and safely**, even in the presence of errors, crashes, or concurrent access.

ACID stands for:

> **A**tomicity
> **C**onsistency
> **I**solation
> **D**urability

---

## 1️⃣ Atomicity

**Meaning:**
A transaction is **all-or-nothing**.

* Either **all operations** in a transaction are completed successfully
* Or **none of them** are applied

If any part fails, the entire transaction is rolled back.

**Example (Bank transfer):**

```sql
BEGIN;
UPDATE Account SET balance = balance - 500 WHERE id = 1;
UPDATE Account SET balance = balance + 500 WHERE id = 2;
```

If the second update fails, the first one is undone.

👉 No partial transfer allowed.

---

## 2️⃣ Consistency

**Meaning:**
A transaction must take the database from **one valid state to another valid state**.

* All **rules, constraints, and integrity conditions** must be satisfied
* Includes: primary key, foreign key, CHECK, NOT NULL, etc.

**Example:**

* Account balance cannot be negative
* Student ID must be unique

If a transaction violates a constraint, it is **rejected**.

👉 Database never becomes inconsistent.

---

## 3️⃣ Isolation

**Meaning:**
Multiple transactions running at the same time should **not interfere** with each other.

* Each transaction behaves as if it is **executed alone**
* Prevents issues like:

  * Dirty reads
  * Non-repeatable reads
  * Phantom reads

**Example:**
Two users updating the same account simultaneously should not see intermediate results.

Isolation levels:

* READ UNCOMMITTED
* READ COMMITTED
* REPEATABLE READ
* SERIALIZABLE

👉 Ensures correctness in concurrent environments.

---

## 4️⃣ Durability

**Meaning:**
Once a transaction is **committed**, its changes are **permanent**.

* Even if:

  * System crashes
  * Power failure occurs

The committed data will not be lost.

**Example:**

```sql
COMMIT;
```

After commit, data is written to **disk / logs**, not just memory.

👉 Data survives failures.

---

## 🔑 Simple Real-Life Example (ATM)

| Step                              | ACID Property |
| --------------------------------- | ------------- |
| Cash deducted & credited together | Atomicity     |
| Balance rules maintained          | Consistency   |
| Two users don’t clash             | Isolation     |
| Transaction saved after success   | Durability    |

---

## 🧠 One-Line Exam-Friendly Definitions

* **Atomicity:** Transaction executes fully or not at all
* **Consistency:** Transaction preserves database rules
* **Isolation:** Concurrent transactions don’t affect each other
* **Durability:** Committed changes are permanent

---

