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

import re
from datetime import datetime
from io import StringIO

import pandas as pd
from dateutil.parser import parse


def is_kerberos_ticket_alive(klist_result: str, service_principal: str):
    # Extract the relevant part of the data that contains the tickets
    tickets_section_match = re.search(
        r"((Issued\s+Expires\s+Principal)|(Valid starting\s+Expires\s+Service principal))\n(.*)",
        klist_result,
        re.DOTALL,
    )
    if not tickets_section_match:
        return False

    tickets_data = tickets_section_match.group(4)

    # Read the tickets data using pandas
    try:
        df = pd.read_csv(
            StringIO(tickets_data),
            sep=r"\s+",
            header=None,
            dtype="str",
        )
        df.columns = [str(c) for c in df.columns]
    except pd.errors.EmptyDataError:
        return False

    # Convert dates to datetime objects using dateutil.parser.parse
    def safe_parse(x):
        try:
            return parse(x, fuzzy=True)
        except (ValueError, TypeError):
            return None

    # equally split the columns into issued and expires except for last column which is principal
    df.columns.values[-1] = "principal"
    n_date_columns = (df.shape[1] - 1) // 2
    issued = df.iloc[:, :n_date_columns].apply(lambda x: " ".join(x.values), axis=1)
    expires = df.iloc[:, n_date_columns:-1].apply(lambda x: " ".join(x.values), axis=1)

    df["issued"] = issued.apply(safe_parse)
    df["expires"] = expires.apply(safe_parse)

    # Drop rows where date parsing failed
    df.dropna(subset=["issued", "expires"], inplace=True)

    # Check if the principal's ticket has not expired
    now = datetime.now()

    # Check if the ticket for a specific service principal is alive
    is_alive = any((df["principal"] == service_principal) & (df["expires"] > now + pd.Timedelta(seconds=60)))

    return is_alive
