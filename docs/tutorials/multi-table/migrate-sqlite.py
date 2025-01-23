# helper script to convert CSV files into a single SQLite database file

import sqlalchemy as sa
import pandas as pd
from pathlib import Path


def create_table_sql(df, tbl_name):
    cols = []
    pks = []
    fks = []
    for c in df.columns:
        if c == f"{tbl_name}_id":
            pks.append(f'PRIMARY KEY("{c}")')
        elif c.endswith("_id"):
            fks.append(f'FOREIGN KEY("{c}") REFERENCES "{c[:-3]}"("{c}")')
        dtype = df[c].dtype
        if pd.api.types.is_integer_dtype(dtype):
            ctype = "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            ctype = "FLOAT"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            ctype = "DATETIME"
        else:
            ctype = "TEXT"
        cols.append(f'"{c}" {ctype}')
    stmt = f'CREATE TABLE "{tbl_name}" ({", ".join(cols + pks + fks)})'
    return stmt


engine = sa.create_engine("sqlite+pysqlite:///berka-sqlite.db", echo=False)

data = {}
for fn in Path(".").glob("*.csv"):
    df = pd.read_csv(fn)
    # convert dtypes
    for col in df.columns:
        if col in ["date", "issued"]:
            df[col] = pd.to_datetime(df[col])
        if col.endswith("_id"):
            df[col] = df[col].astype(str)
    # get filename w/o extension
    tbl_name = fn.stem
    data[tbl_name] = df

with engine.connect() as conn:
    for tbl_name, df in data.items():
        # create table
        stmt = create_table_sql(df, tbl_name)
        conn.execute(sa.text(stmt))
        print(f"created table {tbl_name}")
    conn.commit()
    conn.close()

with engine.connect() as conn:
    for tbl_name, df in data.items():
        # insert records
        df.to_sql(tbl_name, conn, index=False, if_exists="append")
        print(f"loaded data to {tbl_name}")
    conn.commit()
    conn.close()
