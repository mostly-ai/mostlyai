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
from datetime import datetime

from mostlyai.sdk._data.util.kerberos import is_kerberos_ticket_alive


@pytest.fixture
def mock_datetime_now(monkeypatch):
    class MockDatetime:
        @classmethod
        def now(cls):
            return datetime(2023, 6, 11, 14, 0, 0)

    monkeypatch.setattr("mostlyai.sdk._data.util.kerberos.datetime", MockDatetime)


@pytest.mark.parametrize(
    "klist_result, expected",
    [
        (
            """
            Ticket cache: FILE:/tmp/ba42aff58f0e6d3d6b55a531459a56cc666c4c58ddc53d3b9a34083da6b2739a
            Default principal: hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL

            Valid starting       Expires              Service principal
            06/11/2024 14:47:57  06/11/2024 14:48:51  krbtgt/INTERNAL@INTERNAL
            06/11/2023 11:00:00  06/11/2023 15:00:00  hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL
            """,
            True,
        ),
        (
            """
        Ticket cache: FILE:/tmp/ba42aff58f0e6d3d6b55a531459a56cc666c4c58ddc53d3b9a34083da6b2739a
        Default principal: hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL

        Valid starting       Expires              Service principal
        06/11/2024 14:47:57  06/11/2024 14:48:51  krbtgt/INTERNAL@INTERNAL
        06/11/2023 11:00:00  06/11/2023 14:01:10  hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL
        """,
            True,
        ),
        (
            """
        Ticket cache: FILE:/tmp/ba42aff58f0e6d3d6b55a531459a56cc666c4c58ddc53d3b9a34083da6b2739a
        Default principal: hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL

        Valid starting       Expires              Service principal
        06/11/2024 14:47:57  06/11/2024 14:48:51  krbtgt/INTERNAL@INTERNAL
        06/11/2023 11:00:00  06/11/2023 14:00:50  hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL
        """,
            False,
        ),
        (
            """
        Ticket cache: FILE:/tmp/ba42aff58f0e6d3d6b55a531459a56cc666c4c58ddc53d3b9a34083da6b2739a
        Default principal: hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL

        Valid starting       Expires              Service principal
        06/11/2024 14:47:57  06/11/2024 14:48:51  krbtgt/INTERNAL@INTERNAL
        06/11/2023 11:00:00  06/10/2023 14:01:10  hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL
        """,
            False,
        ),
        (
            """
        Ticket cache: FILE:/tmp/ba42aff58f0e6d3d6b55a531459a56cc666c4c58ddc53d3b9a34083da6b2739a
        Default principal: hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL

        Valid starting       Expires              Service principal
        06/11/2024 14:47:57  06/11/2024 14:48:51  krbtgt/INTERNAL@INTERNAL
        06/11/2023 11:00:00  06/12/2023 14:00:50  hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL
        """,
            True,
        ),
        (
            """
            Ticket cache: FILE:/tmp/krb5cc_1000
            Default principal: user@EXAMPLE.COM
            """,
            False,
        ),
        (
            """
            Credentials cache: FILE:/var/folders/0w/qcxns49j7fn6k52f0q1n41740000gn/T/456688f9f3ee845a58b75ced8c634f9e58d86a86ed42a947855363a3ca7fbc58
            Principal: hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL

            Issued                Expires               Principal
            Jun 14 17:52:18 2024  Jun 15 03:52:18 2024  krbtgt/INTERNAL@INTERNAL
            Jun 14 17:52:18 2024  Jun 15 03:52:18 2024  hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL
            """,
            True,
        ),
        (
            """
            Credentials cache: FILE:/var/folders/0w/qcxns49j7fn6k52f0q1n41740000gn/T/625797d527c72b4062bf2a8c721428e9ae2a5d17e94f923f78a2a78924565f43
            Principal: randomuser/hive-kerberized-ssl.test.mostlylab.com@INTERNAL

            Issued                Expires               Principal
            Jun 14 17:52:18 2024  Jun 15 03:52:18 2024  krbtgt/INTERNAL@INTERNAL
            Jun 14 17:52:18 2024  Jun 15 03:52:18 2024  hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL
            """,
            True,
        ),
    ],
)
def test_is_kerberos_ticket_alive(mock_datetime_now, klist_result, expected):
    service_principal = "hive/hive-kerberized-ssl.test.mostlylab.com@INTERNAL"
    assert is_kerberos_ticket_alive(klist_result, service_principal) == expected
