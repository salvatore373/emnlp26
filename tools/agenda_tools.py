import json
import os
import sqlite3
from typing import List, Dict, Any, Optional

import pandas as pd

from tools.base_noisy_tool import BaseNoisyTool

# Determine the absolute path to the data
AGENDA_DATA_PATH = "/workspace/tools/toolqa_preprocessing/data/agenda/agenda_restructured.jsonl"


class AgendaTools:
    """
    A wrapper class that encapsulates all shared helpers and agenda tools.
    """
    _conns = {}
    _dfs = {}

    @classmethod
    def _load_agenda_df(cls, data_path: str) -> pd.DataFrame:
        """Helper to load agenda data into a DataFrame with caching."""
        if data_path in cls._dfs:
            return cls._dfs[data_path]

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Agenda data file not found at {data_path}")

        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        df = pd.DataFrame(data)
        cls._dfs[data_path] = df
        return df

    @classmethod
    def _get_agenda_conn(cls, data_path: str, table_name: str, default_cols: list):
        """Helper to load agenda data into a SQLite connection."""
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

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Agenda data file not found at {data_path}")

        # Memory efficient loading
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
            if chunk:
                df = pd.DataFrame(chunk)
                for col in df.columns:
                    if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
                df.to_sql(table_name, conn, index=False, if_exists="append")

        cls._conns[table_name] = conn
        return conn

    @classmethod
    def format_event(cls, events: List[Dict[str, Any]]) -> str:
        """
        Given the events in structured format, print them in a format suitable for the agent's conversation.
        """
        if not events:
            return "No events found."

        formatted = []
        for ev in events:
            time_start = ev.get('time_start', 'Unknown Time')
            time_end = ev.get('time_end', 'Unknown Time')
            name = ev.get('name', 'Untitled Event')
            participants = ev.get('participants', [])

            invitees_str = (", ".join(participants) if isinstance(participants, list) else str(participants)) or 'None'
            formatted.append(
                f"# Event: {name}\n## Start: {time_start}\n## End: {time_end}\n## Location: {ev.get('location', 'Unknown')}\n## Invitees: {invitees_str}\n## Additional Information: {ev.get('more_info', '')}")

        return ("\n\n" + ('-' * 20) + "\n\n").join(formatted)

    # ==========================================
    # Inner Tool Classes
    # ==========================================

    class GetAgendaEventsTool(BaseNoisyTool):
        name = "get_agenda_events"
        description = (
            "Return all the events from date:hour to date:hour + interval. "
            "If the list of invitees is provided, the list contains only events where at least one invitee "
            "appears among the invitees of each event."
        )
        output_type = "string"
        inputs = {
            "date": {"type": "string", "description": "The date of the event (e.g., '2022-07-29')."},
            "hour": {"type": "integer", "description": "The starting hour of the event (0-23)."},
            "interval": {"type": "integer", "description": "The interval in hours to search for events."},
            "invitees": {"type": "array", "items": {"type": "string"},
                         "description": "Optional list of invitees to filter by.", "nullable": True}
        }

        def execute_tool(self, date: str, hour: int, interval: int, invitees: Optional[List[str]] = None) -> Any:
            try:
                df = AgendaTools._load_agenda_df(AGENDA_DATA_PATH)
            except FileNotFoundError:
                return "[]"

            if df.empty:
                return "[]"

            df_temp = df.copy()
            df_temp['dt_start'] = pd.to_datetime(df_temp['time_start'], format='mixed', errors='coerce')

            try:
                start_dt = pd.to_datetime(f"{date} {hour:02d}:00:00")
                end_dt = start_dt + pd.Timedelta(hours=interval)
            except Exception:
                return "Invalid date or hour format."

            mask = (df_temp['dt_start'] >= start_dt) & (df_temp['dt_start'] <= end_dt)
            # Fill NA values in the mask with False to avoid indexing errors
            filtered_df = df[mask.fillna(False)]

            if invitees and len(invitees) > 0:
                # If a single string is passed instead of a list, wrap it
                if isinstance(invitees, str):
                    invitees = [invitees]

                def check_invitees(event_invitees):
                    if isinstance(event_invitees, str):
                        try:
                            event_invitees = json.loads(event_invitees.replace("'", '"'))
                        except:
                            pass
                    if isinstance(event_invitees, list):
                        return any(inv in event_invitees for inv in invitees)
                    return False

                mask_invitees = filtered_df["participants"].apply(check_invitees)
                filtered_df = filtered_df[mask_invitees]

            events = filtered_df.to_dict(orient="records")
            return AgendaTools.format_event(events)

    class QueryAgendaEventsDatabaseTool(BaseNoisyTool):
        name = "query_agenda_events_database"
        description = (
            "Perform the given SQL query on the dataset of events. Return the un-formatted output. "
            "The database has a single table named 'agenda' with the following columns: ['id', 'name', 'time_start', "
            "'time_end', 'location', 'participants', 'more_info']."
        )
        output_type = "string"
        inputs = {
            "sql_query": {"type": "string", "description": "The SQL query to execute on the 'agenda' table."}
        }

        def execute_tool(self, sql_query: str) -> Any:
            try:
                conn = AgendaTools._get_agenda_conn(AGENDA_DATA_PATH, "agenda",
                                                    ["id", "name", "time_start", "time_end", "location", "participants",
                                                     "more_info"])
                df = pd.read_sql_query(sql_query, conn)
                return df.to_json(orient="records", indent=2)
            except Exception as e:
                return f"Error executing query: {str(e)}"
