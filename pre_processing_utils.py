import re
import ast
import json
import numpy as np
import pandas as pd

from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple, Any

Number = float | int



class Utils:

    @staticmethod
    def format_p(p:float) -> str:
        p_str = str(p).split('.')[-1]
        return f".{p_str[:3]}"

    @staticmethod
    def str_to_dict(cadena):
        dict_str = {}
        c = deepcopy(cadena)
        c = c[1:-2]
        c = re.sub('"', "'", c)
        items = c.split(',')
        for item in items:
            try:
                result = item.split(':')
                key = result[0].strip(" '")
                value = ':'.join(result[1:])
                try:
                    dict_str[key] = ast.literal_eval(value)
                except Exception as e:
                    dict_str[key] = f"Error: {e}\nOriginal value: {value}"
            except:
                print(f"Warning: cannot process {item}")
        return dict_str
    
    def format_time(seconds):
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}:{int(remaining_seconds)}"
    
    @staticmethod
    def parse_dicts_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            first_valid = df[col].dropna().astype(str)

            if len(first_valid) == 0:
                continue

            sample = first_valid.iloc[0]

            if sample.startswith("{") and sample.endswith("}"):
                try:
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if pd.notna(x) else None
                    )
                except Exception:
                    pass
        return df
    
    @staticmethod
    def get_dict_time_steps(dataframe: pd.DataFrame) -> Dict[int, int]:
        dict_nodes = {}
        for network_id, grp in dataframe.groupby('network_id'):
            nodes_ids = grp['id'].unique().tolist()
            nodes_ids.sort()
            nodes_for_network = {
                (network_id, node_id):i+1 for i, node_id in enumerate(nodes_ids)
            }
            dict_nodes.update(nodes_for_network)
        for key, value in dict_nodes.items():
            if pd.isna(key[1]):
                dict_nodes[key] = np.nan
        return dict_nodes
    
    @staticmethod
    def add_time_steps_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:

        def add_group_time_steps(group: pd.DataFrame) -> pd.DataFrame:

            dict_time_steps = Utils.get_dict_time_steps(group)

            group = group.copy()

            group['time_step'] = group.apply(
                lambda row: dict_time_steps.get(
                    (row['network_id'], row['id']),
                    np.nan
                ),
                axis=1
            )

            return group

        return (
            df.groupby('session', group_keys=False)
            .apply(add_group_time_steps)
        )