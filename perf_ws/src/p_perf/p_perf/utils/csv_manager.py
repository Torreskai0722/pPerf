#!/usr/bin/env python3
"""
CSV mapping file manager for experiment tracking.
"""

import csv
import os
import pandas as pd
from typing import List, Dict, Any, Tuple


class ExperimentCSVManager:
    """Manages experiment mapping CSV files with proper handling of OVERWRITE and CONTINUE modes."""
    
    def __init__(self, mapping_file: str, columns: List[str]):
        """
        Initialize CSV manager.
        
        Args:
            mapping_file: Path to the CSV mapping file
            columns: List of column names for the CSV
        """
        self.mapping_file = mapping_file
        self.columns = columns
        
    def create_mapping(
        self,
        combinations: List[Tuple],
        row_formatter: callable,
        overwrite: bool = True,
        continue_mode: bool = False
    ) -> pd.DataFrame:
        """
        Create or update mapping CSV file.
        
        Args:
            combinations: List of parameter combinations
            row_formatter: Function that takes (index, combination) and returns a row list
            overwrite: Whether to overwrite existing file
            continue_mode: Whether to continue from existing file
            
        Returns:
            pd.DataFrame: The resulting dataframe
        """
        if overwrite and not continue_mode:
            self._create_new_mapping(combinations, row_formatter)
        elif continue_mode and os.path.exists(self.mapping_file):
            self._continue_mapping(combinations, row_formatter)
        else:
            # Default behavior - create new file
            self._create_new_mapping(combinations, row_formatter)
        
        return pd.read_csv(self.mapping_file)
    
    def _create_new_mapping(self, combinations: List[Tuple], row_formatter: callable):
        """Create new mapping file from scratch."""
        with open(self.mapping_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.columns)
            for i, combo in enumerate(combinations):
                row = row_formatter(i, combo)
                writer.writerow(row)
        print(f"Created new mapping file with {len(combinations)} combinations")
    
    def _continue_mapping(self, combinations: List[Tuple], row_formatter: callable):
        """Continue from existing mapping, adding only new combinations."""
        existing_df = pd.read_csv(self.mapping_file)
        start_index = len(existing_df)
        new_combinations = []
        
        for i, combo in enumerate(combinations):
            if not self._combination_exists(existing_df, combo, row_formatter(i, combo)):
                row = row_formatter(start_index + len(new_combinations), combo)
                new_combinations.append(row)
        
        # Append new combinations to existing file
        if new_combinations:
            with open(self.mapping_file, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in new_combinations:
                    writer.writerow(row)
            print(f"Added {len(new_combinations)} new combinations to existing mapping file")
        else:
            print("No new combinations to add - all combinations already exist in mapping file")
    
    def _combination_exists(self, df: pd.DataFrame, combo: Tuple, formatted_row: List) -> bool:
        """Check if a combination already exists in the dataframe."""
        # Skip the run_index (first column) and status/timestamps (last columns)
        # Compare the actual parameter columns
        param_columns = self.columns[1:-2]  # Exclude run_index, status, and start_time
        formatted_dict = dict(zip(self.columns, formatted_row))
        
        for _, row in df.iterrows():
            match = True
            for col in param_columns:
                if str(row[col]) != str(formatted_dict[col]):
                    match = False
                    break
            if match:
                return True
        return False
    
    def load_dataframe(self) -> pd.DataFrame:
        """Load the CSV as a pandas DataFrame."""
        return pd.read_csv(self.mapping_file)
    
    def update_status(self, df: pd.DataFrame, index: int, status: str, **kwargs):
        """
        Update experiment status in the dataframe and save to CSV.
        
        Args:
            df: The dataframe to update
            index: Row index to update
            status: New status value
            **kwargs: Additional columns to update (e.g., start_time=123.45)
        """
        df.at[index, "status"] = status
        for key, value in kwargs.items():
            if key in df.columns:
                df.at[index, key] = value
        df.to_csv(self.mapping_file, index=False)


def create_failure_log(output_base: str) -> str:
    """
    Create a failure log file.
    
    Args:
        output_base: Base output directory
        
    Returns:
        str: Path to failure log file
    """
    failure_log = os.path.join(output_base, "failures.log")
    with open(failure_log, "w") as flog:
        flog.write("Failed Runs Log\n")
        flog.write("================\n")
    return failure_log

