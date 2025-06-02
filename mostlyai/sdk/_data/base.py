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

import abc
import functools
import itertools
import logging
from collections.abc import Generator, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd

from mostlyai.sdk._data.dtype import V_DTYPE_ENCODING_TYPE_MAP, DType, VirtualDType
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import (
    DATA_TABLE_METADATA_FIELDS,
    IS_NULL,
    MAX_SCP_SIBLINGS_LIMIT,
    NON_CONTEXT_COLUMN_INFIX,
    TABLE_COLUMN_INFIX,
    Key,
    OrderBy,
    as_list,
    decrypt,
    get_passphrase,
)
from mostlyai.sdk.domain import ModelEncodingType

_LOG = logging.getLogger(__name__)


# itertools recipe py39
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def key_as_list(key: Key) -> list[str]:
    return as_list(key) if key else [""]


def order_df_by(df: pd.DataFrame, orderby: OrderBy) -> pd.DataFrame:
    if isinstance(orderby, (tuple, str)):
        orderby = [orderby]
    orderby = [(e, "asc") if isinstance(e, str) else e for e in orderby]
    return df.sort_values(
        by=[column for column, _ in orderby],
        ascending=[direction == "asc" for _, direction in orderby],
    )


@dataclass
class ForeignKey:
    column: str
    referenced_table: str
    is_context: bool


@dataclass(frozen=True)
class DataIdentifier:
    table: str | None = None
    column: str | None = None

    def __hash__(self):
        return hash(self.ref_name())

    def __repr__(self):
        return self.ref_name()

    def ref_name(self, prefixed: bool = True, sep: str = TABLE_COLUMN_INFIX) -> str:
        table = self.table or ""
        ref_name = self.column or ""
        prefix = f"{table}{sep}"
        if prefixed:
            ref_name = ref_name if ref_name.startswith(prefix) else f"{prefix}{ref_name}"
        else:
            ref_name = ref_name.removeprefix(prefix)
        return ref_name

    def __rshift__(self, other: "DataIdentifier") -> "DataRelation":
        return DataRelation(parent=self, child=other)

    def __lshift__(self, other: "DataIdentifier") -> "DataRelation":
        return DataRelation(parent=other, child=self)


@dataclass(frozen=True)
class DataRelation:
    parent: DataIdentifier
    child: DataIdentifier

    @property
    def data_identifiers(self):
        return [self.parent, self.child]


@dataclass(frozen=True)
class ContextRelation(DataRelation):
    pass


@dataclass(frozen=True)
class NonContextRelation(DataRelation):
    @property
    def src_table_name(self):
        return self.parent.table

    def get_is_null_column(self, is_target: bool = True):
        original_fk_name = self.child.ref_name(prefixed=not is_target)
        return NON_CONTEXT_COLUMN_INFIX.join([original_fk_name, self.src_table_name, IS_NULL])


@dataclass
class Schema:
    tables: dict[str, "DataTable"] = field(default_factory=dict)

    def __hash__(self):
        return hash(
            frozenset(self.tables.keys()),
        )

    def __eq__(self, other: "Schema"):
        return isinstance(other, Schema) and hash(self) == hash(other)

    @property
    def relations(self) -> list[DataRelation]:
        relations = []
        for table_name, table in self.tables.items():
            for fk in table.foreign_keys:
                if fk.referenced_table not in self.tables:
                    continue  # the referenced table is outside of schema
                parent_identifier = DataIdentifier(
                    table=fk.referenced_table,
                    column=self.tables[fk.referenced_table].primary_key,
                )
                child_identifier = DataIdentifier(table=table_name, column=fk.column)
                relation_class = ContextRelation if fk.is_context else NonContextRelation
                relations.append(relation_class(parent=parent_identifier, child=child_identifier))
        return relations

    @property
    def context_relations(self) -> list[ContextRelation]:
        return [rel for rel in self.relations if isinstance(rel, ContextRelation)]

    @property
    def non_context_relations(self) -> list[NonContextRelation]:
        return [rel for rel in self.relations if isinstance(rel, NonContextRelation)]

    def get_relations_from_table(self, table_name: str) -> list[DataRelation]:
        return [rel for rel in self.relations if rel.parent.table == table_name]

    def get_relations_to_table(self, table_name: str) -> list[DataRelation]:
        return [rel for rel in self.relations if rel.child.table == table_name]

    def get_relations_from_to_table(self, parent: str, child: str) -> list[DataRelation]:
        return [rel for rel in self.relations if rel.parent.table == parent and rel.child.table == child]

    def get_parent_context_relation(self, table_name: str) -> DataRelation | None:
        for rel in self.relations:
            if rel.child.table == table_name and isinstance(rel, ContextRelation):
                return rel

        return None

    def get_parent(self, table_name: str) -> str:
        ctx_rel = self.get_parent_context_relation(table_name)
        if ctx_rel:
            return ctx_rel.parent.table

    def get_context_key(self, table_name: str) -> DataIdentifier | None:
        ctx_rel = self.get_parent_context_relation(table_name)
        if ctx_rel:
            return ctx_rel.child

    def get_primary_key(self, table_name: str) -> DataIdentifier | None:
        context_relations = self.get_child_context_relations(table_name)
        # first, check primary key on table
        if self.tables[table_name].primary_key is not None:
            return DataIdentifier(table_name, self.tables[table_name].primary_key)
        # fall back to checking primary key on schema
        if context_relations:
            return context_relations[0].parent

    def get_child_context_relations(self, parent_table: str) -> list[DataRelation]:
        return [rel for rel in self.relations if rel.parent.table == parent_table and isinstance(rel, ContextRelation)]

    def get_context_children(self, parent_table: str) -> list[str]:
        ctx_rels = self.get_child_context_relations(parent_table)
        return [rel.child.table for rel in ctx_rels]

    def get_ancestors(self, table_name: str) -> list[str]:
        ancestors = []
        while table_name := self.get_parent(table_name):
            if table_name in ancestors:
                raise MostlyDataException("Recursive context relations defined.")
            ancestors += [table_name]
        return ancestors

    def get_context_tables(self, table: str) -> list[str]:
        """
        To identify all tables that can serve as a context for generating a given table,
        we examine both vertical and horizontal contexts.

        The vertical context (a.k.a. GPC) comprises all the tables extending upwards from the parent
        of the given table to the root. This forms a path to the root which we also use
        to establish the horizontal context.

        To delineate the horizontal context (a.k.a. SCP), we arrange the schema alphabetically at each level.
        Tables that are one step to the left of this path (the cut) are then considered
        as part of the horizontal context.
        """

        vertical_context = []
        horizontal_context = []
        while parent_table := self.get_parent(table):
            if parent_table in vertical_context:
                raise MostlyDataException("Recursive context relations defined.")
            vertical_context.append(parent_table)
            older_siblings = self.get_older_sibling_tables(parent=parent_table, child=table)
            horizontal_context.extend(older_siblings)
            table = parent_table
        horizontal_context = horizontal_context[:MAX_SCP_SIBLINGS_LIMIT]
        return vertical_context + horizontal_context

    def get_older_sibling_tables(self, parent: str, child: str) -> list[str]:
        sibling_tables = reversed(sorted(self.get_context_children(parent)))
        # only return siblings that are "older", i.e. that are alphabetically sorted before the child table
        sibling_tables = [t for t in sibling_tables if t < child]
        return sibling_tables

    def get_scp_relations(self, table: str) -> list[ContextRelation]:
        relations = []
        while parent_table := self.get_parent(table):
            sibling_tables = sorted(self.get_context_children(parent_table))
            for s in sibling_tables:
                if s < table:
                    relations += self.get_relations_from_to_table(parent_table, s)
            table = parent_table
        relations = [rel for rel in relations if isinstance(rel, ContextRelation)]
        return relations

    def update_key_encoding_types(self) -> None:
        for tbl_name, tbl_table in self.tables.items():
            if tbl_table.primary_key is not None:
                if tbl_table.primary_key in tbl_table.encoding_types:
                    del tbl_table.encoding_types[tbl_table.primary_key]
            for rel in self.get_relations_to_table(tbl_name):
                if rel.child.column in tbl_table.encoding_types:
                    del tbl_table.encoding_types[rel.child.column]
            self.tables[tbl_name] = tbl_table

    def resolve_auto_encoding_types(self) -> None:
        for tbl_name, tbl_table in self.tables.items():
            for col_name, encoding_type in tbl_table.encoding_types.items():
                if encoding_type == ModelEncodingType.auto:
                    promoted_enctype = V_DTYPE_ENCODING_TYPE_MAP[type(tbl_table.dtypes[col_name].to_virtual())]
                    tbl_table.encoding_types[col_name] = promoted_enctype
                    _LOG.info(
                        f"promoted encoding type {ModelEncodingType.auto.value} to "
                        f"{promoted_enctype.value} for {tbl_name}.{col_name}"
                    )

    def remove_cascading_keys_relations(self):
        tables_with_cascading_keys = set()
        for table_name, table in self.tables.items():
            primary_key = table.primary_key
            cascading_keys = [fk for fk in (table.foreign_keys or []) if fk.column == primary_key]
            if cascading_keys:
                tables_with_cascading_keys.add(table_name)

        # iterate over tables and their foreign keys, and remove those that cascade from identified tables
        for table_name, table in self.tables.items():
            # keep only the foreign keys that are not cascading from the identified tables
            table.foreign_keys = [
                fk for fk in (table.foreign_keys or []) if fk.referenced_table not in tables_with_cascading_keys
            ]

        if tables_with_cascading_keys:
            _LOG.info(f"Removed cascading keys relations for tables: {tables_with_cascading_keys}")

    @property
    @functools.cache
    def graph(self):
        g = nx.MultiDiGraph()
        for table in self.tables.keys():
            g.add_node(table)
        for rel in self.relations:
            g.add_edge(
                rel.parent.table,
                rel.child.table,
                type=type(rel),
                parent_key=rel.parent.column,
                child_key=rel.child.column,
            )
        return g

    def copy_table(self, name: str) -> "DataTable":
        table = self.tables[name]
        init_kwargs = {k: getattr(table, k) for k in ["container", "is_output"]}
        init_kwargs |= {
            table_field: getattr(table, table_field)
            for table_field in table.LAZY_INIT_FIELDS
            if hasattr(table, table_field) and table.__dict__.get(table_field) is not None
        }
        table_copy = type(table)(**init_kwargs)
        return table_copy

    def copy_tables(self):
        tables = {}
        names = list(self.tables.keys())
        for table_name in names:
            tables[table_name] = self.copy_table(table_name)
        return tables

    def subset(
        self,
        relation_type: type[DataRelation] | None = None,
        relations_from: list[str] | str | None = None,
        relations_to: list[str] | str | None = None,
        tables: dict[str, "DataTable"] | list[str] | None = None,
    ) -> "Schema":
        # make sure tables is a dict
        if tables is None:
            tables = self.copy_tables()
        elif isinstance(tables, list):
            tables = {name: self.copy_table(name) for name in tables if name in self.tables}

        if isinstance(relations_from, str):
            relations_from = [relations_from]
        if isinstance(relations_to, str):
            relations_to = [relations_to]

        # filter relations
        relations = [rel for rel in self.relations if rel.parent.table in tables and rel.child.table in tables]
        if relation_type:
            relations = [rel for rel in relations if isinstance(rel, relation_type)]
        if relations_from:
            relations = [rel for rel in relations if rel.parent.table in relations_from]
        if relations_to:
            relations = [rel for rel in relations if rel.child.table in relations_to]

        # update the foreign keys after filtering them based on relations
        for table_name, table in tables.items():
            table.foreign_keys = [
                fk
                for fk in table.foreign_keys
                if fk.referenced_table in tables
                and (ContextRelation if fk.is_context else NonContextRelation)(
                    DataIdentifier(
                        table=fk.referenced_table,
                        column=tables[fk.referenced_table].primary_key,
                    ),
                    DataIdentifier(table=table_name, column=fk.column),
                )
                in relations
            ]

        return Schema(tables=tables)

    @property
    def table_root_map(self) -> dict[str, str]:
        table_root = {}
        ctx_graph = self.subset(relation_type=ContextRelation).graph
        topological_order = list(nx.topological_sort(ctx_graph))
        for table in self.tables.keys():
            ancestors = nx.ancestors(ctx_graph, table)
            table_root[table] = sorted(ancestors, key=lambda x: topological_order.index(x))[0] if ancestors else table

        return table_root

    def path_to(self, table: str) -> tuple[list, list[DataRelation]]:
        nodes = nx.shortest_path(self.graph, source=self.table_root_map[table], target=table)
        path = []
        for parent, child in pairwise(nodes):
            path.append(self.get_relations_from_to_table(parent, child)[0])
        return nodes, path

    def get_referential_integrity_keys(self, table_name: str, is_eager_fetch: bool = True) -> list[str]:
        """
        Get referential integrity keys of a given table using this schema.

        :param table_name: table name
        :param is_eager_fetch: whether to fetch the primary key
        :return: a list of columns, which must be referentially integral
        """
        relations = self.get_relations_from_table(table_name) + self.get_relations_to_table(table_name)
        data_identifiers = list(itertools.chain(*[rel.data_identifiers for rel in relations]))
        if is_eager_fetch:
            data_identifiers.append(DataIdentifier(table=table_name, column=self.tables[table_name].primary_key))

        data_identifiers = [di for di in data_identifiers if di.table == table_name]
        return list(set(itertools.chain(*[key_as_list(di.column) for di in data_identifiers])))


class DataContainer(abc.ABC):
    SCHEMES: list[str] = []
    SECRET_ATTR_NAME: str = ""

    def __init__(self, *args, **kwargs):
        self.schema = Schema()

    def __hash__(self):
        return hash(self.schema)

    def __eq__(self, other):
        return isinstance(other, DataContainer) and hash(self) == hash(other)

    @staticmethod
    def decrypt(secret: str) -> str:
        return decrypt(secret, get_passphrase())

    def decrypt_secret(self, secret_attr_name: str | None = None) -> None:
        secret_attr_name = secret_attr_name or self.SECRET_ATTR_NAME
        if hasattr(self, secret_attr_name):
            secret = getattr(self, secret_attr_name)
            if secret is not None:
                # Note: secrets and ssl certificates are encrypted with the same passphrase
                setattr(self, secret_attr_name, self.decrypt(secret))

    @abc.abstractmethod
    def is_accessible(self) -> bool:
        """
        Check if the data container is accessible.

        :return: True if accessible, False otherwise
        """
        pass

    @abc.abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query against this data source.

        :param sql: SQL query to execute
        :return: DataFrame containing the query results
        """
        pass

    # DEFAULT METHODS (noop, unless applicable)
    def drop_all(self):
        """
        Drop all the relations and tables, if applicable.

        """
        pass

    def fetch_table(self, table_name: str):
        """
        Fetch the single table's schema from the data source.
        Applicable for all tables.
        This should return a structure containing:
        - table
            - their corresponding columns
            - PKs / FKs
        """
        pass

    def fetch_schema(self):
        """
        Fetch the schema of the data source. Applicable for DBs and other sources containing multiple tables.
        This should return a structure containing:
        - tables
            - their corresponding columns
            - PKs / FKs

        Note: when a driver needs to configured: do it lazily or eagerly? applicable to DBs, mostly.
        """
        pass

    def set_location(self, location: str) -> dict:
        return {}


class DataTable(abc.ABC):
    LAZY_INIT_FIELDS: set[str] = frozenset(DATA_TABLE_METADATA_FIELDS)
    DATA_TABLE_TYPE: str | None = None

    def __init__(self, *args, **kwargs):
        self.container: DataContainer | None = kwargs.get("container")
        self.is_output: bool = kwargs.get("is_output")
        self.name: str | None = kwargs.get("name")
        self.columns: list[str] | None = kwargs.get("columns")
        self.dtypes: dict[str, DType] | None = kwargs.get("dtypes")
        self.primary_key: Key | None = kwargs.get("primary_key")
        self.foreign_keys: list[ForeignKey] | None = kwargs.get("foreign_keys") or []
        self.encoding_types: dict[str, ModelEncodingType] | None = kwargs.get("encoding_types")

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __getattribute__(self, item: str):
        # Allow lazy initialization of certain fields (LAZY_INIT_FIELDS)
        # using super() as suggested here: https://stackoverflow.com/a/61413243
        if (
            item in super().__getattribute__("LAZY_INIT_FIELDS")
            and super().__getattribute__(item) is None
            and not self.is_output
        ):
            self._lazy_fetch(item)
        return object.__getattribute__(self, item)

    def __hash__(self):
        return hash((self.container, self.name))

    def __eq__(self, other):
        return isinstance(other, DataTable) and hash(self) == hash(other)

    # PUBLIC METHODS

    @abc.abstractmethod
    def is_accessible(self) -> bool:
        """
        Check if the data source is accessible.

        :return: True if accessible, False otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def row_count(self) -> int:
        """
        Fetch number of rows (if possible)
        """
        pass

    @abc.abstractmethod
    def read_data(
        self,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        shuffle: bool | None = False,
        order_by: OrderBy | None = None,
        do_coerce_dtypes: bool | None = False,
    ) -> pd.DataFrame:
        """
        Read data from this data source.

        :param where: dict of column name and value(s) or a lambda function to filter on
        :param limit: limit the number of rows returned. If None, return all rows
        :param columns: list of columns to include. If None, all columns are included
        :param shuffle: whether to shuffle the returned data rows
        :param order_by: optionally, one or multiple columns (asc/desc) to order by
        :param do_coerce_dtypes: bool on whether dtypes shall be converted corresponding to encoding_types
        """
        pass

    @abc.abstractmethod
    def read_chunks(
        self,
        where: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        do_coerce_dtypes: bool = True,
        fetch_chunk_size: int | None = None,
        yield_chunk_size: int | None = None,
    ) -> Iterable[pd.DataFrame]:
        """
        Read data from this data source in chunks.

        :param where: dict of one column name and values to filter on
        :param columns: list of columns to include. If None, all columns are included
        :param do_coerce_dtypes: bool on whether dtypes shall be converted corresponding to encoding_types
        :param fetch_chunk_size: size of the chunk to fetch from the data source
        :param yield_chunk_size: size of the chunk to yield to the caller
        """
        pass

    @abc.abstractmethod
    def write_data(self, df: pd.DataFrame, **kwargs):
        """
        Write data to this data source.

        :param df: data to write
        """
        pass

    @abc.abstractmethod
    def drop(self):
        """
        Drop the data table.

        """
        pass

    def read_data_prefixed(self, include_table_prefix: bool = True, **kwargs) -> pd.DataFrame:
        """
        Same as read_data, but allow the flexibility of having either prefixed or not column names.
        It will adjust the kwargs, that read_data expects to get, prefixing the column names if needed.

        :param include_table_prefix: whether to include the prefix in the returned df
        :param kwargs: see read_data arguments
        :return: pd.DataFrame representing the read_data result
        """
        prefix = DataIdentifier(table=self.name).ref_name()
        is_prefixed = all(prefix in col for col in self.columns)
        kwargs = self._ensure_kwargs_prefixed(kwargs)
        df = self.read_data(**kwargs)
        return df.add_prefix(prefix) if include_table_prefix and not is_prefixed else df

    def read_chunks_prefixed(self, include_table_prefix: bool = True, **kwargs) -> Iterable[pd.DataFrame]:
        """
        Same as read_chunks, but allow the flexibility of having either prefixed or not column names.
        It will adjust the kwargs, that read_data expects to get, prefixing the column names if needed.

        :param include_table_prefix: whether to include the prefix in the returned df
        :param kwargs: see read_data arguments
        :return: pd.DataFrame representing the read_data result
        """
        prefix = DataIdentifier(table=self.name).ref_name()
        is_prefixed = all(prefix in col for col in self.columns)
        kwargs = self._ensure_kwargs_prefixed(kwargs)
        iterator = self.read_chunks(**kwargs)
        for chunk_df in iterator:
            yield (chunk_df.add_prefix(prefix) if include_table_prefix and not is_prefixed else chunk_df)

    def write_data_partitioned(
        self,
        partitions: Generator[tuple[str, pd.DataFrame], None, None],
        overwrite_tables: bool = False,
        **kwargs,
    ) -> None:
        """
        Write data, which is partitioned.

        :param partitions: a generator, which provides (part_file: str, data: pd.DataFrame)
        """
        for idx, (part_file, data) in enumerate(partitions):
            # drop the table / fail as part of first write_data, then append
            if_exists = "append" if idx > 0 else ("replace" if overwrite_tables else "fail")
            self.write_data(df=data, if_exists=if_exists)

    # PRIVATE METHODS

    def _lazy_fetch(self, item: str) -> None:
        """
        Lazy fetch of an item.

        This method is being called once a field of LAZY_INIT_FIELDS is being accessed while being None. It allows
        lazy initialization of fields - to be implemented by the subclasses, wherever it is needed.

        :param item: name of the field to be fetched (e.g. encoding_types)
        """
        if item == "encoding_types":
            self.encoding_types = self._get_default_encoding_types()

    def _get_default_encoding_types(self) -> dict[str, ModelEncodingType]:
        """
        Fetch for all columns a default ModelEncodingType based on their DType.
        """
        virtual_dtypes = VirtualDType.from_dtypes(self.dtypes or {})
        encoding_types = {
            column: V_DTYPE_ENCODING_TYPE_MAP[type(virtual_dtype)] for column, virtual_dtype in virtual_dtypes.items()
        }
        return encoding_types

    def _ensure_kwargs_prefixed(self, kwargs: dict) -> dict:
        prefix = DataIdentifier(table=self.name).ref_name()
        is_prefixed = all(prefix in col for col in self.columns)
        kwargs = deepcopy(kwargs)

        def _prefix_dict(d):
            if not isinstance(d, dict):
                return

            old_new_k_map = {}
            for k in d.keys():
                if prefix not in k:
                    new_k = DataIdentifier(table=self.name, column=k).ref_name()
                    old_new_k_map[k] = new_k

            for old_k, new_k in old_new_k_map.items():
                d[new_k] = d.pop(old_k)

        def _prefix_list(li):
            if not isinstance(li, list):
                return

            for idx, col in enumerate(li):
                if prefix not in col:
                    new_k = DataIdentifier(table=self.name, column=col).ref_name()
                    li[idx] = new_k

        if is_prefixed:
            if "where" in kwargs:
                _prefix_dict(kwargs["where"])
            _prefix_dict(self.encoding_types)  # hacky and discouraged
            kwargs.setdefault("columns", self.columns)
            _prefix_list(kwargs["columns"])
            if "order_by" in kwargs:
                kwargs["order_by"] = key_as_list(kwargs["order_by"])
                _prefix_list(kwargs["order_by"])

        return kwargs
