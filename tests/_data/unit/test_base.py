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

import functools
from typing import Any
from collections.abc import Iterable
from unittest.mock import patch

import pandas as pd
import pytest
from mostlyai.sdk.domain import ModelEncodingType

from mostlyai.sdk._data.base import (
    ContextRelation,
    DataContainer,
    DataIdentifier,
    DataRelation,
    DataTable,
    Schema,
    ForeignKey,
    NonContextRelation,
)
from mostlyai.sdk._data.util.common import OrderBy
from mostlyai.sdk._data.dtype import VirtualDatetime, VirtualInteger, VirtualVarchar


class UnbiasedDataContainer(DataContainer): ...


class TheSimplestDataContainer(DataContainer):
    def is_accessible(self) -> bool:
        return True

    def drop_all(self):
        pass

    def fetch_table(self, table_name: str):
        if table_name in self.schema.tables.keys():
            return self.schema.tables[table_name]

    def fetch_schema(self):
        pass


class UnbiasedDataTable(DataTable): ...


class TheSimplestDataTable(UnbiasedDataTable):
    df: pd.DataFrame = pd.DataFrame()
    DATA_TABLE_TYPE = "dummy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = kwargs.get("df", self.df)
        self.primary_key = kwargs.get("primary_key")
        self.foreign_keys = kwargs.get("foreign_keys", [])

    def _lazy_fetch(self, item: str) -> None:
        if item == "primary_key":
            self.primary_key = "id"
        else:
            return super()._lazy_fetch(item)

    def is_accessible(self) -> bool:
        return True

    def read_chunks(
        self,
        where: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        do_coerce_dtypes: bool = True,
        fetch_chunk_size: int | None = None,
        yield_chunk_size: int | None = None,
    ) -> Iterable[pd.DataFrame]:
        yield self.df

    def read_data(
        self,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        shuffle: bool | None = False,
        order_by: OrderBy | None = None,
        do_coerce_dtypes: bool | None = False,
    ) -> pd.DataFrame:
        return self.df

    def write_data(self, df: pd.DataFrame, **kwargs):
        self.df = df

    def drop(self):
        self.df = pd.DataFrame()

    def _get_columns(self):
        return list(self.df.columns)

    @functools.cached_property
    def row_count(self) -> int:
        return self.df.shape[0]


class TestDataTable:
    @pytest.fixture
    def dummy_df(self):
        yield pd.DataFrame({"id": [1, 2, 3], "col_a": ["v1", "v2", "v3"]})

    @pytest.fixture
    def dummy_table(self, dummy_df):
        yield TheSimplestDataTable(name="dummy", df=dummy_df, columns=dummy_df.columns)

    @pytest.fixture
    def another_df(self):
        yield pd.DataFrame(
            {
                "id": [1, 2, 3],
                "cat": ["A", "B", "C"],
                "dig": [0, -1, 1],
                "dt": [
                    pd.to_datetime("2020-03-03"),
                    pd.to_datetime("2021-12-12"),
                    pd.to_datetime("2022-02-02"),
                ],
            }
        )

    @pytest.fixture
    def another_table(self, another_df):
        yield TheSimplestDataTable(
            df=another_df,
            primary_key="id",
            name="another",
            columns=["id", "cat", "dig", "dt"],
            dtypes={
                "id": VirtualInteger(),
                "cat": VirtualVarchar(length=1),
                "dig": VirtualInteger(),
                "dt": VirtualDatetime(),
            },
        )

    @pytest.fixture
    def prefixed_table(self):
        yield TheSimplestDataTable(
            df=pd.DataFrame({"prefixed::id": [0, 1], "prefixed::di": [9, 8]}),
            primary_key="prefixed::id",
            name="prefixed",
            columns=["prefixed::id", "prefixed::di"],
            dtypes={
                "prefixed::id": VirtualInteger(),
                "prefixed::di": VirtualInteger(),
            },
        )

    def test_get_encoding_types(self, another_table):
        assert another_table.encoding_types["cat"] == ModelEncodingType.tabular_categorical
        assert another_table.encoding_types["dig"] == ModelEncodingType.tabular_numeric_auto
        assert another_table.encoding_types["dt"] == ModelEncodingType.tabular_datetime

    def test_read_data_prefixed(self, another_table, prefixed_table):
        read_res = another_table.read_data_prefixed(include_table_prefix=True)
        expected_col_names = [f"another::{c}" for c in ["id", "cat", "dig", "dt"]]
        assert list(read_res.columns) == expected_col_names
        read_res = another_table.read_data_prefixed(include_table_prefix=False)
        expected_col_names = ["id", "cat", "dig", "dt"]
        assert list(read_res.columns) == expected_col_names

        with patch.object(prefixed_table, "read_data") as read_data_mock:
            prefixed_table.read_data_prefixed(
                include_table_prefix=True,
                where={"id": [1]},
                columns=["id", "di"],
                order_by="id",
            )
            read_data_mock.assert_called_once()
            assert read_data_mock.call_args[1] == {
                "where": {"prefixed::id": [1]},
                "columns": ["prefixed::id", "prefixed::di"],
                "order_by": ["prefixed::id"],
            }

    def test_count_rows(self, dummy_table, dummy_df):
        assert dummy_table.row_count == dummy_df.shape[0]


class TestDataIdentifier:
    @pytest.fixture
    def simple_key(self):
        yield DataIdentifier("table_name", "primary_key")

    def test_qual_name_simple_key(self, simple_key):
        assert simple_key.ref_name() == "table_name::primary_key"
        assert simple_key.ref_name(sep="+") == "table_name+primary_key"

    def test_qual_name_simple_already_qualified_key(self):
        di = DataIdentifier("table_name", "table_name::primary_key")
        assert di.ref_name() == "table_name::primary_key"
        di = DataIdentifier("table_name", "table_name+primary_key")
        assert di.ref_name(sep="+") == "table_name+primary_key"


class TestSchema:
    @pytest.fixture
    def subject(self):
        df = pd.DataFrame({"id": [1, 2, 3], "sub_param": ["s1", "s2", "s3"]})
        yield TheSimplestDataTable(df=df, primary_key="id", foreign_keys=[])

    @pytest.fixture
    def linked_hop_1(self):
        df = pd.DataFrame({"sub_id": [1, 2, 2], "1_hop_param": ["l1", "l1", "l2"]})
        yield TheSimplestDataTable(
            df=df,
            foreign_keys=[ForeignKey(column="sub_id", referenced_table="subject", is_context=True)],
        )

    @pytest.fixture
    def linked_hop_2(self):
        df = pd.DataFrame(
            {
                "sub_id": [1, 1, 2, 2, 2],
                "1_hop_param": ["l1", "l1", "l1", "l2", "l2"],
                "2_hop_param": ["m1", "m2", "m1", "m3", "m4"],
            }
        )
        yield TheSimplestDataTable(
            df=df,
            foreign_keys=[ForeignKey(column="sub_id", referenced_table="linked_hop_1", is_context=True)],
        )

    @pytest.fixture
    def schema(self, subject, linked_hop_1, linked_hop_2):
        tables = {
            "subject": subject,
            "linked_hop_1": linked_hop_1,
            "linked_hop_2": linked_hop_2,
        }
        yield Schema(tables=tables)

    def test_get_relations_from_table(self, schema):
        relations = schema.get_relations_from_table("subject")
        assert len(relations) == 1
        assert relations[0].child.table == "linked_hop_1"

    def test_get_relations_to_table(self, schema):
        relations = schema.get_relations_to_table("linked_hop_1")
        assert len(relations) == 1
        assert relations[0].parent.table == "subject"

    def test_get_relations_from_to_table(self, schema):
        relations = schema.get_relations_from_to_table("subject", "linked_hop_1")
        assert len(relations) == 1
        assert relations[0].parent.table == "subject"
        assert relations[0].child.table == "linked_hop_1"

    def test_graph(self, schema):
        nodes = schema.graph.nodes._nodes
        assert len(nodes) == 3
        assert all(table in nodes for table in ["subject", "linked_hop_1", "linked_hop_2"])

        expected_adjacency_list = {
            "subject": ["linked_hop_1"],
            "linked_hop_1": ["linked_hop_2"],
        }

        assert len(schema.graph.edges) == 2
        for n1, n2, _ in schema.graph.edges:
            assert n1 in expected_adjacency_list
            assert n2 in expected_adjacency_list[n1]

    @pytest.fixture
    def another_linked_hop_1(self):
        yield pd.DataFrame({"sub_id": [1, 2, 2], "another_1_hop_param": ["l1", "l1", "l2"]})

    def test_rshift_operator(self):
        data_id_a = DataIdentifier("A", "pk")
        data_id_b = DataIdentifier("B", "fk")

        via_operator = data_id_a >> data_id_b
        assert via_operator == DataRelation(parent=data_id_a, child=data_id_b)

    def test_lshift_operator(self):
        data_id_a = DataIdentifier("A", "pk")
        data_id_b = DataIdentifier("B", "fk")

        via_operator = data_id_a << data_id_b
        assert via_operator == DataRelation(parent=data_id_b, child=data_id_a)

    def test_get_context_tables(self):
        schema = Schema(
            tables={
                "A": TheSimplestDataTable(df=pd.DataFrame(columns=["id"]), primary_key="id", foreign_keys=[]),
                "A1": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["a_id"]),
                    primary_key="a_id",
                    foreign_keys=[ForeignKey(column="a_id", referenced_table="A", is_context=True)],
                ),
                "A2": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["a_id"]),
                    primary_key="a_id",
                    foreign_keys=[ForeignKey(column="a_id", referenced_table="A", is_context=True)],
                ),
                "A21": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["a2_id"]),
                    primary_key="a2_id",
                    foreign_keys=[ForeignKey(column="a2_id", referenced_table="A2", is_context=True)],
                ),
                "A22": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["a2_id"]),
                    primary_key="a2_id",
                    foreign_keys=[ForeignKey(column="a2_id", referenced_table="A2", is_context=True)],
                ),
                "A3": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["a_id"]),
                    primary_key="a_id",
                    foreign_keys=[ForeignKey(column="a_id", referenced_table="A", is_context=True)],
                ),
            }
        )
        assert schema.get_context_tables("A") == []
        assert schema.get_context_tables("A1") == ["A"]
        assert schema.get_context_tables("A2") == ["A", "A1"]
        assert schema.get_context_tables("A21") == ["A2", "A", "A1"]
        assert schema.get_context_tables("A22") == ["A2", "A", "A21", "A1"]
        assert schema.get_context_tables("A3") == ["A", "A2", "A1"]

    def test_get_relations_of_siblings(self):
        #   A
        #  / \
        # B   C
        #    / \
        #   D   E
        schema = Schema(
            tables={
                "A": TheSimplestDataTable(df=pd.DataFrame(columns=["id"]), primary_key="id", foreign_keys=[]),
                "B": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["a_id", "id"]),
                    primary_key="id",
                    foreign_keys=[ForeignKey(column="a_id", referenced_table="A", is_context=True)],
                ),
                "C": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["a_id", "id"]),
                    primary_key="id",
                    foreign_keys=[ForeignKey(column="a_id", referenced_table="A", is_context=True)],
                ),
                "D": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["c_id"]),
                    primary_key="c_id",
                    foreign_keys=[ForeignKey(column="c_id", referenced_table="C", is_context=True)],
                ),
                "E": TheSimplestDataTable(
                    df=pd.DataFrame(columns=["c_id"]),
                    primary_key="c_id",
                    foreign_keys=[ForeignKey(column="c_id", referenced_table="C", is_context=True)],
                ),
            }
        )

        assert schema.get_scp_relations("A") == []
        assert schema.get_scp_relations("B") == []
        assert schema.get_scp_relations("C") == [
            ContextRelation(DataIdentifier("A", "id"), DataIdentifier("B", "a_id")),
        ]
        # D will not consider E because of lexical ordering
        assert schema.get_scp_relations("D") == [
            ContextRelation(DataIdentifier("A", "id"), DataIdentifier("B", "a_id")),
        ]
        assert schema.get_scp_relations("E") == [
            ContextRelation(DataIdentifier("C", "id"), DataIdentifier("D", "c_id")),
            ContextRelation(DataIdentifier("A", "id"), DataIdentifier("B", "a_id")),
        ]


class TestSchemaAdvanced:
    @pytest.fixture
    def order_management_schema(self):
        tables = {
            "User": TheSimplestDataTable(
                name="User",
                primary_key="id",
                foreign_keys=[
                    ForeignKey(
                        column="reports_to_user_id",
                        referenced_table="User",
                        is_context=False,
                    )
                ],
            ),
            "BusinessPartner": TheSimplestDataTable(name="BusinessPartner", primary_key="id", foreign_keys=[]),
            "Customer": TheSimplestDataTable(
                name="Customer",
                primary_key="id",
                foreign_keys=[
                    ForeignKey(
                        column="creation_user_id",
                        referenced_table="User",
                        is_context=False,
                    )
                ],
            ),
            "Product": TheSimplestDataTable(
                name="Product",
                primary_key="id",
                foreign_keys=[
                    ForeignKey(
                        column="creation_user_id",
                        referenced_table="User",
                        is_context=False,
                    )
                ],
            ),
            "Order": TheSimplestDataTable(
                name="Order",
                primary_key="id",
                foreign_keys=[
                    ForeignKey(
                        column="customer_id",
                        referenced_table="Customer",
                        is_context=True,
                    ),
                    ForeignKey(
                        column="business_partner_id",
                        referenced_table="BusinessPartner",
                        is_context=False,
                    ),
                    ForeignKey(
                        column="creation_user_id",
                        referenced_table="User",
                        is_context=False,
                    ),
                ],
            ),
            "OrderLineItem": TheSimplestDataTable(
                name="OrderLineItem",
                primary_key="id",
                foreign_keys=[
                    ForeignKey(column="order_id", referenced_table="Order", is_context=True),
                    ForeignKey(
                        column="product_id",
                        referenced_table="Product",
                        is_context=False,
                    ),
                ],
            ),
            "OrderLineItemStatus": TheSimplestDataTable(
                name="OrderLineItemStatus",
                primary_key="id",
                foreign_keys=[
                    ForeignKey(
                        column="order_line_item",
                        referenced_table="OrderLineItem",
                        is_context=True,
                    )
                ],
            ),
        }
        yield Schema(tables=tables)

    @pytest.mark.skip("Get back it to define get_referential_integrity_keys better")
    def test_get_all_keys(self, order_management_schema):
        table_referentially_integral_keys = {
            "Order": ["customer_id", "id"],
            "User": ["id"],
            "BusinessPartner": ["id"],
            "Gender": ["id"],
            "OrderLineItem": ["order_id", "id"],
        }

        for table, expected_keys in table_referentially_integral_keys.items():
            keys = order_management_schema.get_referential_integrity_keys(table)
            assert sorted(keys) == sorted(expected_keys)

    def test_table_root_map(self, order_management_schema):
        # Call the function to be tested
        result = order_management_schema.table_root_map
        # Assert that the result is as expected
        expected_result = {
            "User": "User",
            "BusinessPartner": "BusinessPartner",
            "Customer": "Customer",
            "Product": "Product",
            "Order": "Customer",
            "OrderLineItem": "Customer",
            "OrderLineItemStatus": "Customer",
        }
        assert result == expected_result

    def test_subset(self, order_management_schema):
        from_user = order_management_schema.subset(relations_from="User")
        assert all(rel.parent == DataIdentifier("User", "id") for rel in from_user.relations)
        assert len(from_user.relations) == 4
        to_user = order_management_schema.subset(relations_to="User")
        assert to_user.relations == [
            NonContextRelation(
                DataIdentifier("User", "id"),
                DataIdentifier("User", "reports_to_user_id"),
            )
        ]
        self_user = order_management_schema.subset(relations_from="User", relations_to="User")
        assert self_user == to_user
        to_order_ctx = order_management_schema.subset(relation_type=ContextRelation, relations_to="Order")
        assert to_order_ctx.relations == [
            ContextRelation(DataIdentifier("Customer", "id"), DataIdentifier("Order", "customer_id"))
        ]
        to_order_non_ctx = order_management_schema.subset(relation_type=NonContextRelation, relations_to="Order")
        assert all(
            isinstance(rel, NonContextRelation) and rel.child.table == "Order" for rel in to_order_non_ctx.relations
        )
        to_order = order_management_schema.subset(relations_to="Order")
        assert set(to_order_ctx.relations) | set(to_order_non_ctx.relations) == set(to_order.relations)
