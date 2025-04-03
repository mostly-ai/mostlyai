# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import assert_read_only_sql


@pytest.mark.parametrize(
    "sql,expected_error",
    [
        # basic read-only operations (keep a few essential ones)
        ("SELECT * FROM table", None),
        ("WITH cte AS (SELECT 1) SELECT * FROM cte", None),
        ("SELECT * FROM table WHERE column = 'value'", None),
        # complex read-only operations (should be allowed)
        # 1. complex joins and subqueries
        (
            "SELECT t1.*, t2.*, t3.* FROM table1 t1 LEFT JOIN table2 t2 ON t1.id = t2.id RIGHT JOIN table3 t3 ON t2.id = t3.id",
            None,
        ),
        ("SELECT * FROM table1 WHERE id IN (SELECT id FROM table2 WHERE value > 100)", None),
        ("SELECT * FROM table1 WHERE EXISTS (SELECT 1 FROM table2 WHERE table2.id = table1.id)", None),
        # 2. window functions and aggregations
        ("SELECT id, value, ROW_NUMBER() OVER (PARTITION BY category ORDER BY value DESC) as rank FROM table", None),
        (
            "SELECT category, COUNT(*) as count, SUM(value) as total, AVG(value) as avg FROM table GROUP BY category",
            None,
        ),
        ("SELECT category, COUNT(*) as count FROM table GROUP BY category HAVING COUNT(*) > 5", None),
        # 3. complex CTEs and recursive queries
        (
            "WITH RECURSIVE numbers AS (SELECT 1 as n UNION ALL SELECT n + 1 FROM numbers WHERE n < 10) SELECT * FROM numbers",
            None,
        ),
        (
            "WITH cte1 AS (SELECT * FROM table1), cte2 AS (SELECT * FROM table2) SELECT * FROM cte1 JOIN cte2 ON cte1.id = cte2.id",
            None,
        ),
        # 4. set operations and complex filtering
        ("SELECT * FROM table1 UNION SELECT * FROM table2 UNION ALL SELECT * FROM table3", None),
        ("SELECT * FROM table WHERE column1 IN (1, 2, 3) AND column2 BETWEEN 10 AND 20", None),
        ("SELECT * FROM table WHERE column LIKE '%pattern%' AND column REGEXP '^[0-9]+$'", None),
        # 5. complex expressions and functions
        (
            "SELECT CASE WHEN value > 100 THEN 'high' WHEN value > 50 THEN 'medium' ELSE 'low' END as category FROM table",
            None,
        ),
        (
            "SELECT DATE_FORMAT(timestamp, '%Y-%m-%d') as date, CONCAT(first_name, ' ', last_name) as full_name FROM table",
            None,
        ),
        ("SELECT JSON_EXTRACT(data, '$.field') as field, JSON_ARRAYAGG(value) as values FROM table", None),
        # 6. complex ordering and limiting
        (
            "SELECT * FROM table ORDER BY CASE WHEN category = 'A' THEN 1 WHEN category = 'B' THEN 2 ELSE 3 END, value DESC LIMIT 10 OFFSET 20",
            None,
        ),
        ("SELECT * FROM table WHERE id IN (SELECT id FROM other_table ORDER BY value DESC LIMIT 5)", None),
        # 7. complex grouping and pivoting
        (
            "SELECT category, MAX(CASE WHEN type = 'A' THEN value END) as type_a, MAX(CASE WHEN type = 'B' THEN value END) as type_b FROM table GROUP BY category",
            None,
        ),
        ("SELECT year, month, SUM(value) as total FROM table GROUP BY year, month WITH ROLLUP", None),
        # 8. complex subqueries in various clauses
        ("SELECT * FROM table WHERE value > (SELECT AVG(value) FROM table)", None),
        ("SELECT * FROM table WHERE value IN (SELECT value FROM other_table WHERE category = 'A')", None),
        (
            "SELECT * FROM table WHERE EXISTS (SELECT 1 FROM other_table WHERE other_table.id = table.id AND other_table.value > 100)",
            None,
        ),
        # edge cases trying to bypass read-only restrictions
        # 1. hidden DML in subqueries
        ("SELECT * FROM (INSERT INTO table VALUES (1))", "Disallowed operation: INSERT"),
        ("SELECT * FROM (UPDATE table SET col = 1)", "Disallowed operation: UPDATE"),
        ("SELECT * FROM (DELETE FROM table)", "Disallowed operation: DELETE"),
        # 2. hidden DDL in subqueries
        ("SELECT * FROM (CREATE TABLE new_table (id INT))", "Disallowed operation: CREATE"),
        ("SELECT * FROM (DROP TABLE table)", "Disallowed operation: DROP"),
        ("SELECT * FROM (ALTER TABLE table ADD col INT)", "Disallowed operation: ALTER"),
        # 3. hidden DCL in subqueries
        ("SELECT * FROM (GRANT SELECT ON table TO user)", "Disallowed operation: GRANT"),
        ("SELECT * FROM (REVOKE SELECT ON table FROM user)", "Disallowed operation: REVOKE"),
        # 4. hidden TCL in subqueries
        ("SELECT * FROM (COMMIT)", "Disallowed operation: COMMIT"),
        ("SELECT * FROM (ROLLBACK)", "Disallowed operation: ROLLBACK"),
        # 5. hidden operations in CTEs
        ("WITH cte AS (INSERT INTO table VALUES (1)) SELECT * FROM cte", "Disallowed operation: INSERT"),
        ("WITH cte AS (UPDATE table SET col = 1) SELECT * FROM cte", "Disallowed operation: UPDATE"),
        ("WITH cte AS (DELETE FROM table) SELECT * FROM cte", "Disallowed operation: DELETE"),
        # 6. hidden operations in derived tables
        ("SELECT * FROM (SELECT * FROM (INSERT INTO table VALUES (1)))", "Disallowed operation: INSERT"),
        ("SELECT * FROM (SELECT * FROM (UPDATE table SET col = 1))", "Disallowed operation: UPDATE"),
        ("SELECT * FROM (SELECT * FROM (DELETE FROM table))", "Disallowed operation: DELETE"),
        # 7. hidden operations in UNION queries
        ("SELECT * FROM table UNION ALL INSERT INTO table VALUES (1)", "Disallowed operation: INSERT"),
        ("SELECT * FROM table UNION ALL UPDATE table SET col = 1", "Disallowed operation: UPDATE"),
        ("SELECT * FROM table UNION ALL DELETE FROM table", "Disallowed operation: DELETE"),
        # 8. hidden operations in JOIN conditions
        ("SELECT * FROM table1 JOIN (INSERT INTO table2 VALUES (1)) ON 1=1", "Disallowed operation: INSERT"),
        ("SELECT * FROM table1 JOIN (UPDATE table2 SET col = 1) ON 1=1", "Disallowed operation: UPDATE"),
        ("SELECT * FROM table1 JOIN (DELETE FROM table2) ON 1=1", "Disallowed operation: DELETE"),
        # 9. hidden operations in WHERE clauses
        ("SELECT * FROM table WHERE EXISTS (INSERT INTO other_table VALUES (1))", "Disallowed operation: INSERT"),
        ("SELECT * FROM table WHERE EXISTS (UPDATE other_table SET col = 1)", "Disallowed operation: UPDATE"),
        ("SELECT * FROM table WHERE EXISTS (DELETE FROM other_table)", "Disallowed operation: DELETE"),
        # 10. hidden operations in HAVING clauses
        (
            "SELECT col, COUNT(*) FROM table GROUP BY col HAVING EXISTS (INSERT INTO other_table VALUES (1))",
            "Disallowed operation: INSERT",
        ),
        (
            "SELECT col, COUNT(*) FROM table GROUP BY col HAVING EXISTS (UPDATE other_table SET col = 1)",
            "Disallowed operation: UPDATE",
        ),
        (
            "SELECT col, COUNT(*) FROM table GROUP BY col HAVING EXISTS (DELETE FROM other_table)",
            "Disallowed operation: DELETE",
        ),
        # 11. hidden operations in CASE expressions
        ("SELECT CASE WHEN 1=1 THEN (INSERT INTO table VALUES (1)) ELSE 1 END", "Disallowed operation: INSERT"),
        ("SELECT CASE WHEN 1=1 THEN (UPDATE table SET col = 1) ELSE 1 END", "Disallowed operation: UPDATE"),
        ("SELECT CASE WHEN 1=1 THEN (DELETE FROM table) ELSE 1 END", "Disallowed operation: DELETE"),
        # 12. hidden operations in function calls
        ("SELECT * FROM table WHERE column = (INSERT INTO other_table VALUES (1))", "Disallowed operation: INSERT"),
        ("SELECT * FROM table WHERE column = (UPDATE other_table SET col = 1)", "Disallowed operation: UPDATE"),
        ("SELECT * FROM table WHERE column = (DELETE FROM other_table)", "Disallowed operation: DELETE"),
        # 13. hidden operations in string literals (should be allowed)
        ("SELECT 'INSERT INTO table VALUES (1)' as text", None),
        ("SELECT 'UPDATE table SET col = 1' as text", None),
        ("SELECT 'DELETE FROM table' as text", None),
        ("SELECT 'CREATE TABLE new_table' as text", None),
        ("SELECT 'DROP TABLE table' as text", None),
        ("SELECT 'ALTER TABLE table' as text", None),
        ("SELECT 'GRANT SELECT' as text", None),
        ("SELECT 'REVOKE SELECT' as text", None),
        ("SELECT 'COMMIT' as text", None),
        ("SELECT 'ROLLBACK' as text", None),
        # 14. hidden operations in column names (should be allowed)
        ("SELECT insert_column, update_column, delete_column FROM table", None),
        ("SELECT create_column, drop_column, alter_column FROM table", None),
        ("SELECT grant_column, revoke_column FROM table", None),
        ("SELECT commit_column, rollback_column FROM table", None),
        # 15. hidden operations in table names (should be allowed)
        ("SELECT * FROM insert_table", None),
        ("SELECT * FROM update_table", None),
        ("SELECT * FROM delete_table", None),
        ("SELECT * FROM create_table", None),
        ("SELECT * FROM drop_table", None),
        ("SELECT * FROM alter_table", None),
        ("SELECT * FROM grant_table", None),
        ("SELECT * FROM revoke_table", None),
        ("SELECT * FROM commit_table", None),
        ("SELECT * FROM rollback_table", None),
    ],
)
def test_assert_read_only_sql(sql: str, expected_error: str | None) -> None:
    if expected_error is None:
        # should not raise any exception
        assert_read_only_sql(sql)
    else:
        # should raise MostlyDataException with expected error message
        with pytest.raises(MostlyDataException) as exc_info:
            assert_read_only_sql(sql)
        assert str(exc_info.value) == expected_error
