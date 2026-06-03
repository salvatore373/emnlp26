import json
import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tools.base_noisy_tool import BaseNoisyTool

COFFEE_DATA_PATH = "/workspace/tools/toolqa_preprocessing/data/coffee/coffee_price.csv"
FLIGHTS_DATA_PATH = "/workspace/tools/toolqa_preprocessing/data/flights/Combined_Flights_2022.csv"
YELP_DATA_PATH = "/workspace/tools/toolqa_preprocessing/data/yelp/yelp_academic_dataset_business.json"


class DatabaseQueryTools:
    """
    A wrapper class that encapsulates all shared helpers and database query tools.
    """

    # Global cache for SQLite connections
    _conns = {}

    @classmethod
    def _get_csv_conn(cls, data_path: str, table_name: str, default_cols: list):
        """
        Compresses the previous _get_coffee_conn and _get_flights_conn into one function.
        Memory efficient implementation using chunking and disk-backed SQLite DBs.
        """
        if table_name in cls._conns:
            return cls._conns[table_name]

        db_path = f"{data_path}.sqlite"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Using a disk-backed sqlite DB to avoid overloading RAM with huge tables
        conn = sqlite3.connect(db_path, check_same_thread=False)

        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone() is not None:
            # Table already built and cached to disk
            cls._conns[table_name] = conn
            return conn

        if os.path.exists(data_path):
            chunksize = 100000
            for chunk in pd.read_csv(data_path, chunksize=chunksize):
                chunk.to_sql(table_name, conn, index=False, if_exists="append")
        else:
            pd.DataFrame(columns=default_cols).to_sql(table_name, conn, index=False)

        cls._conns[table_name] = conn
        return conn

    @classmethod
    def _get_jsonl_conn(cls, data_path: str, table_name: str, default_cols: list):
        """
        Memory efficient implementation for parsing huge JSON lines files using chunks and disk-backed DBs.
        """
        if table_name in cls._conns:
            return cls._conns[table_name]

        db_path = f"{data_path}.sqlite"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)

        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone() is not None:
            cls._conns[table_name] = conn
            return conn

        if os.path.exists(data_path):
            chunk = []
            chunksize = 10000
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk.append(json.loads(line.strip()))
                    if len(chunk) >= chunksize:
                        df = pd.DataFrame(chunk)
                        for col in df.columns:
                            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
                        df.to_sql(table_name, conn, index=False, if_exists="append")
                        chunk = []
                # Process the last chunk
                if chunk:
                    df = pd.DataFrame(chunk)
                    for col in df.columns:
                        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
                    df.to_sql(table_name, conn, index=False, if_exists="append")
        else:
            pd.DataFrame(columns=default_cols).to_sql(table_name, conn, index=False)

        cls._conns[table_name] = conn
        return conn

    @classmethod
    def _execute_sql_query(cls, conn, sql_query: str) -> str:
        try:
            df = pd.read_sql_query(sql_query, conn)
            return df.to_string()
        except Exception as e:
            return f"Error executing query: {str(e)}"

    @classmethod
    def _get_distractor_sql_output(cls, conn, sql_query: str) -> str:
        try:
            df = pd.read_sql_query(sql_query, conn)
            if df.empty:
                # Fallback plausible table output
                return pd.DataFrame({"id": [1, 2], "value": [10.5, 12.0]}).to_string()

            # Modify the DataFrame to create a distractor without obvious markers
            if len(df.columns) > 0 and len(df) > 0:
                num_cols = len(df.columns)

                # For each row, corrupt two random columns (or all if <=2 cols)
                for row_idx in range(len(df)):
                    if num_cols <= 2:
                        cols_to_modify = list(range(num_cols))
                    else:
                        cols_to_modify = random.sample(range(num_cols), 2)

                    for col_idx in cols_to_modify:
                        original_val = df.iat[row_idx, col_idx]
                        col_series = df.iloc[:, col_idx]
                        try:
                            col_dtype = col_series.dtype
                        except Exception:
                            col_dtype = None

                        # Detect datetime-like columns or values
                        is_datetime_col = False
                        try:
                            is_datetime_col = pd.api.types.is_datetime64_any_dtype(col_dtype)
                        except Exception:
                            is_datetime_col = False

                        parsed_dt = None
                        if not is_datetime_col and isinstance(original_val, str):
                            try:
                                parsed_dt = pd.to_datetime(original_val, errors='coerce')
                                if pd.isna(parsed_dt):
                                    parsed_dt = None
                            except Exception:
                                parsed_dt = None

                        # Apply corruption while preserving dtype compatibility when possible
                        if is_datetime_col or isinstance(original_val,
                                                         (pd.Timestamp, datetime)) or parsed_dt is not None:
                            # generate a random date and keep as Timestamp for datetime columns
                            start = datetime(2019, 1, 1)
                            end = datetime(2024, 12, 31)
                            rand_days = random.randint(0, (end - start).days)
                            new_dt = start + timedelta(days=rand_days)
                            if is_datetime_col:
                                df.iat[row_idx, col_idx] = pd.Timestamp(new_dt)
                            else:
                                df.iat[row_idx, col_idx] = new_dt.strftime('%Y-%m-%d')
                        elif pd.api.types.is_numeric_dtype(col_series.dtype):
                            # Use a real-valued multiplier to avoid producing complex numbers
                            scale = random.uniform(0.5, 1.5)
                            flip = random.choice([True, False])
                            try:
                                new_val = float(original_val) * scale
                                if flip:
                                    new_val = -new_val
                            except Exception:
                                # cannot coerce, skip modification for this cell
                                continue

                            # If the column was integer-typed, cast back to int to preserve dtype compatibility
                            if pd.api.types.is_integer_dtype(col_series.dtype):
                                try:
                                    new_val = int(round(new_val))
                                except Exception:
                                    pass

                            df.iat[row_idx, col_idx] = new_val
                        elif isinstance(original_val, str):
                            df.iat[row_idx, col_idx] = original_val + " LLC"
                        else:
                            # Fallback: stringify and append a suffix
                            try:
                                df.iat[row_idx, col_idx] = str(original_val) + "_mod"
                            except Exception:
                                pass

            return df.to_string()
        except Exception as e:
            # Plausible generic table on error
            return pd.DataFrame({"date": ["2022-01-01", "2022-01-02"], "amount": [150.0, 200.0]}).to_string()

    # ==========================================
    # Inner Tool Classes
    # ==========================================

    class QueryCoffeepricehistoryDatabaseTool(BaseNoisyTool):
        name = "query_coffeepricehistory_database"
        description = (
            "Perform the given SQL query on the dataset of Coffee Price History. Return the result table as output. "
            "The database has a single table named 'coffeepricehistory' with the following columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Currency']."
            "Query it using standard SQLite SQL syntax."
        )
        output_type = "string"
        inputs = {
            "sql_query": {"type": "string",
                          "description": "The SQL query to execute on the 'coffeepricehistory' table."}
        }

        def execute_tool(self, sql_query: str) -> Any:
            conn = DatabaseQueryTools._get_csv_conn(COFFEE_DATA_PATH, "coffeepricehistory", ["date", "price"])
            return DatabaseQueryTools._execute_sql_query(conn, sql_query)

    class QueryFlightsDatabaseTool(BaseNoisyTool):
        name = "query_flights_database"
        description = (
            "Perform the given SQL query on the dataset of Flights (Combined_Flights_2022.csv). Return the the result"
            " table as output. The database has a single table named 'flights' with the following columns:"
            " ['FlightDate', 'Airline', 'Origin', 'Dest', 'Cancelled', 'Diverted', 'CRSDepTime', 'DepTime',"
            " 'DepDelayMinutes', 'DepDelay', 'ArrTime', 'ArrDelayMinutes', 'AirTime', 'CRSElapsedTime',"
            " 'ActualElapsedTime', 'Distance', 'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',"
            " 'Marketing_Airline_Network', 'Operated_or_Branded_Code_Share_Partners', 'DOT_ID_Marketing_Airline',"
            " 'IATA_Code_Marketing_Airline', 'Flight_Number_Marketing_Airline', 'Operating_Airline', "
            "'DOT_ID_Operating_Airline', 'IATA_Code_Operating_Airline', 'Tail_Number', "
            "'Flight_Number_Operating_Airline', 'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID',"
            " 'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportID', "
            "'DestAirportSeqID', 'DestCityMarketID', 'DestCityName', 'DestState', 'DestStateFips', 'DestStateName',"
            " 'DestWac', 'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'TaxiOut', 'WheelsOff', 'WheelsOn', "
            "'TaxiIn', 'CRSArrTime', 'ArrDelay', 'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk', 'DistanceGroup', "
            "'DivAirportLandings'].\nQuery it using standard SQLite SQL syntax."
        )
        output_type = "string"
        inputs = {
            "sql_query": {"type": "string", "description": "The SQL query to execute on the 'flights' table."}
        }

        def execute_tool(self, sql_query: str) -> Any:
            conn = DatabaseQueryTools._get_csv_conn(FLIGHTS_DATA_PATH, "flights",
                                                    ["FlightDate", "Airline", "Origin", "Dest"])
            return DatabaseQueryTools._execute_sql_query(conn, sql_query)

    class QueryYelpDatabaseTool(BaseNoisyTool):
        name = "query_yelp_database"
        description = (
            "Perform the given SQL query on the dataset of Yelp business data. Return the the result table as output. "
            "The database has a single table named 'yelp' with the following columns: ['business_id', 'name',"
            " 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open',"
            " 'attributes', 'categories', 'hours'].\nQuery it using standard SQLite SQL syntax."
        )
        output_type = "string"
        inputs = {
            "sql_query": {"type": "string", "description": "The SQL query to execute on the 'yelp' table."}
        }

        def execute_tool(self, sql_query: str) -> Any:
            conn = DatabaseQueryTools._get_jsonl_conn(YELP_DATA_PATH, "yelp", ["business_id", "name", "address"])
            return DatabaseQueryTools._execute_sql_query(conn, sql_query)
