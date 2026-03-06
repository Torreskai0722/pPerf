import pandas as pd
import numpy as np
import os
import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
import logging
from datetime import datetime

# Set matplotlib backend to avoid Qt/GUI dependencies
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from p_perf.config.constant import model_name_mappings, image_models, lidar_models, seg_models


class KernelProcessor:
    """
    A processor for analyzing kernel timings and memcpy operations from performance experiments.
    
    This class can:
    1. Combine kernel timings from multiple runs
    2. Filter kernels by specific layers (default: e2e)
    3. Analyze aligned memcpy operations between model pairs
    4. Generate CSV files with memcpy alignment analysis
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the KernelProcessor.
        
        Args:
            output_dir: Directory containing the experiment outputs
        """
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the processor."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _get_model_type_and_color(self, model_name: str) -> Tuple[str, str]:
        """
        Categorize a model and return its type and fixed color.
        
        Args:
            model_name: The model name to categorize
            
        Returns:
            Tuple of (model_type, color) where model_type is 'image', 'lidar', or 'seg'
        """
        # Fixed colors for each model type
        type_colors = {
            'image': '#1f77b4',  # Blue
            'lidar': '#ff7f0e',  # Orange  
            'seg': '#2ca02c'     # Green
        }
        
        # Check if it's an image model
        if model_name in image_models:
            return ('image', type_colors['image'])
        
        # Check if it's a lidar model (handle tuple format)
        for lidar_model_info in lidar_models:
            if isinstance(lidar_model_info, tuple):
                lidar_model_name = lidar_model_info[0]
            else:
                lidar_model_name = lidar_model_info
            
            if model_name == lidar_model_name or model_name in str(lidar_model_info):
                return ('lidar', type_colors['lidar'])
        
        # Check if it's a segmentation model (handle tuple format)
        for seg_model_info in seg_models:
            if isinstance(seg_model_info, tuple):
                seg_model_name = seg_model_info[0]
            else:
                seg_model_name = seg_model_info
                
            if model_name == seg_model_name or model_name in str(seg_model_info):
                return ('seg', type_colors['seg'])
        
        # Default fallback (shouldn't happen with known models)
        self.logger.warning(f"Unknown model type for: {model_name}, defaulting to 'image'")
        return ('image', type_colors['image'])
    
    def load_mapping_csv(self, mapping_file: str) -> pd.DataFrame:
        """
        Load the full stack mapping CSV file.
        
        Args:
            mapping_file: Path to the mapping CSV file
            
        Returns:
            DataFrame containing the mapping information
        """
        mapping_path = self.output_dir / mapping_file
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
            
        mapping_df = pd.read_csv(mapping_path)
        return mapping_df
    
    def load_kernel_timings(self, run_index: int) -> pd.DataFrame:
        """
        Load kernel timings for a specific run index.
        
        Args:
            run_index: The run index to load
            
        Returns:
            DataFrame containing the kernel timings with run_index column added
        """
        timings_file = f"kernel_timings_{run_index}.csv"
        timings_path = self.output_dir / timings_file
        
        if not timings_path.exists():
            raise FileNotFoundError(f"Kernel timings file not found: {timings_path}")
            
        self.logger.info(f"Loading kernel timings: {timings_path}")
        
        # Read in chunks to handle large files
        chunk_size = 100000  # Adjust based on available memory
        chunks = []
        
        for chunk in pd.read_csv(timings_path, chunksize=chunk_size):
            # Add run_index column to identify which run this data belongs to
            chunk['run_index'] = run_index
            chunks.append(chunk)
        
        if chunks:
            timings_df = pd.concat(chunks, ignore_index=True)
            return timings_df
        else:
            self.logger.warning(f"No kernel data found in run {run_index}")
            return pd.DataFrame()

    def load_layer_timings(self, run_index: int) -> pd.DataFrame:
        """
        Load layer timings for a specific run index.

        Args:
            run_index: The run index to load

        Returns:
            DataFrame containing the layer timings with run_index column added
        """
        layer_file = f"layer_timings_{run_index}.csv"
        layer_path = self.output_dir / layer_file

        if not layer_path.exists():
            raise FileNotFoundError(f"Layer timings file not found: {layer_path}")

        self.logger.info(f"Loading layer timings: {layer_path}")
        layer_df = pd.read_csv(layer_path)
        layer_df["run_index"] = run_index
        return layer_df

    def _safe_model_filename(self, model_name: str) -> str:
        """Create a filesystem-safe model filename."""
        return (
            model_name.replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("'", "")
            .replace(",", "_")
            .replace(" ", "_")
        )

    def _model_candidate_names(self, model_name: str) -> Set[str]:
        """Return full-name and display-name candidates for a model."""
        candidates = {model_name, model_name_mappings.get(model_name, model_name)}
        for original_name, display_name in model_name_mappings.items():
            if model_name == display_name:
                candidates.add(original_name)
                candidates.add(display_name)
        return {str(candidate) for candidate in candidates}

    def _mapping_value_matches_model(self, value: Any, model_name: str) -> bool:
        """Return whether a mapping CSV cell refers to the target model."""
        if pd.isna(value):
            return False

        candidates = self._model_candidate_names(model_name)

        if isinstance(value, str):
            text_value = value.strip()
            if text_value in candidates:
                return True

            try:
                parsed = ast.literal_eval(text_value)
            except (ValueError, SyntaxError):
                parsed = None

            if isinstance(parsed, (tuple, list, set)):
                return any(str(item) in candidates for item in parsed)

            return False

        if isinstance(value, (tuple, list, set)):
            return any(str(item) in candidates for item in value)

        return str(value) in candidates

    def find_run_indices_for_model(self, target_model: str, mapping_file: str) -> List[int]:
        """
        Find all run indices in the mapping CSV that contain the target model.

        Args:
            target_model: Full or display model name
            mapping_file: Mapping CSV path relative to output_dir or absolute

        Returns:
            Sorted list of matching run indices
        """
        mapping_df = self.load_mapping_csv(mapping_file)
        model_columns = ["image_model", "lidar_model", "seg_model"]
        available_columns = [col for col in model_columns if col in mapping_df.columns]

        run_indices: Set[int] = set()
        for _, row in mapping_df.iterrows():
            for model_column in available_columns:
                if self._mapping_value_matches_model(row[model_column], target_model):
                    run_indices.add(int(row["run_index"]))
                    break

        return sorted(run_indices)

    def load_kernel_profile(
        self,
        target_model: str,
        profile_dir: Optional[str] = None,
        profile_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a per-model kernel profile CSV.

        Args:
            target_model: Full or display model name
            profile_dir: Directory containing kernel profile CSVs
            profile_csv: Explicit profile CSV path

        Returns:
            Normalized kernel profile DataFrame
        """
        candidate_paths: List[Path] = []

        if profile_csv is not None:
            candidate_paths.append(Path(profile_csv))

        if profile_dir is None:
            base_dir = self.output_dir / "kernel_profiles"
        else:
            base_dir = Path(profile_dir)

        for candidate_name in self._model_candidate_names(target_model):
            candidate_paths.append(base_dir / f"{candidate_name}.csv")
            candidate_paths.append(base_dir / f"{self._safe_model_filename(candidate_name)}.csv")

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                profile_df = pd.read_csv(candidate_path)
                return self._normalize_kernel_profile_df(profile_df, target_model)

        raise FileNotFoundError(
            f"Could not find a kernel profile CSV for '{target_model}' in {base_dir}"
        )

    def _normalize_kernel_profile_df(
        self,
        profile_df: pd.DataFrame,
        target_model: str,
    ) -> pd.DataFrame:
        """Normalize profile columns for downstream kernel ID matching."""
        normalized_df = profile_df.copy()

        column_aliases = {
            "Kernel Name": "kernel_name",
            "Kernel ID": "kernel_id",
            "Input Shape": "input_shape",
            "Engine Config": "engine",
            "Engine": "engine",
            "Block Size": "block_size",
            "Grid Size": "grid_size",
            "Theoretical Occupancy": "theoretical_occupancy",
            "theoretical occupancy": "theoretical_occupancy",
        }
        normalized_df = normalized_df.rename(columns=column_aliases)

        if "kernel_name" not in normalized_df.columns:
            raise ValueError("Kernel profile CSV must contain a kernel_name column")

        if "kernel_id" not in normalized_df.columns:
            normalized_df["kernel_id"] = np.arange(1, len(normalized_df) + 1)

        if "model_name" not in normalized_df.columns:
            normalized_df["model_name"] = target_model

        if "theoretical_occupancy" not in normalized_df.columns:
            normalized_df["theoretical_occupancy"] = np.nan
        for optional_column in ["engine", "input_shape", "block_size", "grid_size"]:
            if optional_column not in normalized_df.columns:
                normalized_df[optional_column] = np.nan

        occupancy_series = pd.to_numeric(
            normalized_df["theoretical_occupancy"], errors="coerce"
        )
        occupancy_mask = occupancy_series.notna() & (occupancy_series <= 1.0)
        occupancy_series.loc[occupancy_mask] = occupancy_series.loc[occupancy_mask] * 100.0
        normalized_df["theoretical_occupancy"] = occupancy_series

        normalized_df["kernel_occurrence_index"] = (
            normalized_df.groupby("kernel_name").cumcount()
        )
        normalized_df["kernel_id"] = pd.to_numeric(
            normalized_df["kernel_id"], errors="coerce"
        ).astype("Int64")

        return normalized_df

    def assign_kernel_ids_to_kernel_timings(
        self,
        kernel_df: pd.DataFrame,
        target_model: str,
        profile_df: Optional[pd.DataFrame] = None,
        profile_dir: Optional[str] = None,
        profile_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Assign execution-order kernel IDs to target-model runtime kernels.

        The mapping is done by occurrence order within each kernel name for every
        `(run_index, Input, Model)` group, then joined back to the execution-order
        `kernel_id` from the profiled kernel sequence.
        """
        if kernel_df.empty:
            return kernel_df.copy()

        if profile_df is None:
            profile_df = self.load_kernel_profile(
                target_model=target_model,
                profile_dir=profile_dir,
                profile_csv=profile_csv,
            )
        else:
            profile_df = self._normalize_kernel_profile_df(profile_df, target_model)

        result_df = kernel_df.copy()
        candidates = self._model_candidate_names(target_model)
        model_series = result_df["Model"].astype(str)
        target_mask = model_series.isin(candidates)

        metadata_columns = [
            "kernel_id",
            "theoretical_occupancy",
            "engine",
            "input_shape",
            "block_size",
            "grid_size",
        ]
        for column in metadata_columns:
            if column not in result_df.columns:
                if column in {"engine", "input_shape"}:
                    result_df[column] = pd.Series(index=result_df.index, dtype="object")
                else:
                    result_df[column] = np.nan

        if not target_mask.any():
            return result_df

        profile_lookup = profile_df[
            [
                "kernel_name",
                "kernel_occurrence_index",
                "kernel_id",
                "theoretical_occupancy",
                "engine",
                "input_shape",
                "block_size",
                "grid_size",
            ]
        ].copy().rename(
            columns={
                "kernel_id": "profile_kernel_id",
                "theoretical_occupancy": "profile_theoretical_occupancy",
                "engine": "profile_engine",
                "input_shape": "profile_input_shape",
                "block_size": "profile_block_size",
                "grid_size": "profile_grid_size",
            }
        )

        group_columns = ["Input", "Model"]
        if "run_index" in result_df.columns:
            group_columns = ["run_index"] + group_columns

        target_df = result_df[target_mask].copy()
        target_df["_original_index"] = target_df.index
        target_df = target_df.sort_values(
            by=["Start Timestamp", "End Timestamp", "Kernel Name"]
        )

        assigned_groups: List[pd.DataFrame] = []
        for _, group in target_df.groupby(group_columns, sort=False):
            sorted_group = group.sort_values(
                by=["Start Timestamp", "End Timestamp", "Kernel Name"]
            ).copy()
            sorted_group["kernel_occurrence_index"] = (
                sorted_group.groupby("Kernel Name").cumcount()
            )

            merged_group = sorted_group.merge(
                profile_lookup,
                left_on=["Kernel Name", "kernel_occurrence_index"],
                right_on=["kernel_name", "kernel_occurrence_index"],
                how="left",
            )

            if (
                merged_group["profile_kernel_id"].isna().any()
                and len(merged_group) == len(profile_df)
            ):
                merged_group["profile_kernel_id"] = profile_df["kernel_id"].to_numpy()
                merged_group["profile_theoretical_occupancy"] = profile_df[
                    "theoretical_occupancy"
                ].to_numpy()
                merged_group["profile_engine"] = profile_df["engine"].to_numpy()
                merged_group["profile_input_shape"] = profile_df["input_shape"].to_numpy()
                merged_group["profile_block_size"] = profile_df["block_size"].to_numpy()
                merged_group["profile_grid_size"] = profile_df["grid_size"].to_numpy()

            assigned_groups.append(
                merged_group[
                    [
                        "_original_index",
                        "profile_kernel_id",
                        "profile_theoretical_occupancy",
                        "profile_engine",
                        "profile_input_shape",
                        "profile_block_size",
                        "profile_grid_size",
                    ]
                ].copy().rename(
                    columns={
                        "profile_kernel_id": "kernel_id",
                        "profile_theoretical_occupancy": "theoretical_occupancy",
                        "profile_engine": "engine",
                        "profile_input_shape": "input_shape",
                        "profile_block_size": "block_size",
                        "profile_grid_size": "grid_size",
                    }
                )
            )

        if assigned_groups:
            assigned_df = pd.concat(assigned_groups, ignore_index=True)
            assigned_df = assigned_df.set_index("_original_index")
            for column in metadata_columns:
                result_df.loc[assigned_df.index, column] = assigned_df[column]

        return result_df

    def get_resource_heavy_kernel_ids(
        self,
        target_model: str,
        occupancy_threshold: float = 20.0,
        profile_dir: Optional[str] = None,
        profile_csv: Optional[str] = None,
    ) -> List[int]:
        """
        Return kernel IDs whose theoretical occupancy is below the threshold.
        """
        profile_df = self.load_kernel_profile(
            target_model=target_model,
            profile_dir=profile_dir,
            profile_csv=profile_csv,
        )
        heavy_kernel_df = profile_df[
            profile_df["theoretical_occupancy"].notna()
            & (profile_df["theoretical_occupancy"] < occupancy_threshold)
        ]
        return sorted(heavy_kernel_df["kernel_id"].dropna().astype(int).unique().tolist())

    @staticmethod
    def _compute_overlap_duration(
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> float:
        """Return the overlap duration between two intervals."""
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        return max(0.0, overlap_end - overlap_start)

    def analyze_resource_heavy_kernel_overlap(
        self,
        target_model: str,
        mapping_file: str = "full_stack_mapping.csv",
        occupancy_threshold: float = 20.0,
        profile_dir: Optional[str] = None,
        profile_csv: Optional[str] = None,
        output_csv: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze how much each resource-heavy kernel overlaps with corunning models.

        For each resource-heavy kernel occurrence, overlap is summed separately per
        competing model and the maximum summed overlap is divided by the target
        kernel duration.

        Returns:
            Tuple of (summary_df, detail_df)
        """
        run_indices = self.find_run_indices_for_model(target_model, mapping_file)
        if not run_indices:
            self.logger.warning(f"No run indices found for target model '{target_model}'")
            return pd.DataFrame(), pd.DataFrame()

        profile_df = self.load_kernel_profile(
            target_model=target_model,
            profile_dir=profile_dir,
            profile_csv=profile_csv,
        )
        resource_heavy_ids = set(
            self.get_resource_heavy_kernel_ids(
                target_model=target_model,
                occupancy_threshold=occupancy_threshold,
                profile_dir=profile_dir,
                profile_csv=profile_csv,
            )
        )

        if not resource_heavy_ids:
            self.logger.warning(f"No resource-heavy kernels found for '{target_model}'")
            return pd.DataFrame(), pd.DataFrame()

        target_candidates = self._model_candidate_names(target_model)
        summary_rows: List[Dict[str, Any]] = []
        detail_rows: List[Dict[str, Any]] = []

        for run_index in run_indices:
            kernel_df = self.load_kernel_timings(run_index)
            kernel_df = self.assign_kernel_ids_to_kernel_timings(
                kernel_df=kernel_df,
                target_model=target_model,
                profile_df=profile_df,
            )
            layer_df = self.load_layer_timings(run_index)

            inference_df = layer_df[
                (layer_df["Layer"] == "inference")
                & (layer_df["Model"].astype(str).isin(target_candidates))
            ].copy()

            for _, inference_row in inference_df.iterrows():
                inference_start = float(inference_row["Start Timestamp"])
                inference_end = float(inference_row["End Timestamp"])
                input_token = inference_row["Input"]
                kernel_id_series = pd.to_numeric(kernel_df["kernel_id"], errors="coerce")

                target_kernel_df = kernel_df[
                    kernel_df["Model"].astype(str).isin(target_candidates)
                    & (kernel_df["Input"] == input_token)
                    & kernel_id_series.notna()
                    & kernel_id_series.isin(resource_heavy_ids)
                    & (kernel_df["Start Timestamp"] >= inference_start)
                    & (kernel_df["End Timestamp"] <= inference_end)
                ].copy()

                other_kernel_df = kernel_df[
                    ~kernel_df["Model"].astype(str).isin(target_candidates)
                    & (kernel_df["Start Timestamp"] < inference_end)
                    & (kernel_df["End Timestamp"] > inference_start)
                ].copy()

                current_detail_rows: List[Dict[str, Any]] = []
                for _, kernel_row in target_kernel_df.iterrows():
                    kernel_start = float(kernel_row["Start Timestamp"])
                    kernel_end = float(kernel_row["End Timestamp"])
                    kernel_duration = max(0.0, kernel_end - kernel_start)
                    if kernel_duration <= 0.0:
                        max_overlap_model = None
                        max_overlap_duration = 0.0
                        max_overlap_ratio = 0.0
                    else:
                        overlapping_df = other_kernel_df[
                            (other_kernel_df["Start Timestamp"] < kernel_end)
                            & (other_kernel_df["End Timestamp"] > kernel_start)
                        ]

                        overlap_by_model: Dict[str, float] = {}
                        for _, other_row in overlapping_df.iterrows():
                            overlap_duration = self._compute_overlap_duration(
                                kernel_start,
                                kernel_end,
                                float(other_row["Start Timestamp"]),
                                float(other_row["End Timestamp"]),
                            )
                            if overlap_duration <= 0.0:
                                continue

                            corunning_model = str(other_row["Model"])
                            overlap_by_model[corunning_model] = (
                                overlap_by_model.get(corunning_model, 0.0)
                                + overlap_duration
                            )

                        if overlap_by_model:
                            max_overlap_model, max_overlap_duration = max(
                                overlap_by_model.items(), key=lambda item: item[1]
                            )
                        else:
                            max_overlap_model, max_overlap_duration = (None, 0.0)

                        max_overlap_ratio = max_overlap_duration / kernel_duration

                    detail_record = {
                        "run_index": run_index,
                        "input": input_token,
                        "model": inference_row["Model"],
                        "kernel_id": int(kernel_row["kernel_id"]),
                        "kernel_name": kernel_row["Kernel Name"],
                        "kernel_duration": kernel_duration,
                        "theoretical_occupancy": kernel_row["theoretical_occupancy"],
                        "max_overlap_model": max_overlap_model,
                        "max_overlap_duration": max_overlap_duration,
                        "max_overlap_ratio": max_overlap_ratio,
                        "overlap_ge_5pct": max_overlap_ratio >= 0.05,
                        "overlap_ge_10pct": max_overlap_ratio >= 0.10,
                        "overlap_ge_50pct": max_overlap_ratio >= 0.50,
                    }
                    current_detail_rows.append(detail_record)
                    detail_rows.append(detail_record)

                current_detail_df = pd.DataFrame(current_detail_rows)
                if current_detail_df.empty:
                    resource_heavy_kernel_count = 0
                    overlap_ge_5pct_count = 0
                    overlap_ge_10pct_count = 0
                    overlap_ge_50pct_count = 0
                    overlap_ge_5pct_kernel_ids: List[int] = []
                    overlap_ge_10pct_kernel_ids = []
                    overlap_ge_50pct_kernel_ids = []
                else:
                    resource_heavy_kernel_count = (
                        current_detail_df["kernel_id"].nunique()
                    )
                    overlap_ge_5pct_kernel_ids = sorted(
                        current_detail_df.loc[
                            current_detail_df["overlap_ge_5pct"], "kernel_id"
                        ].unique().tolist()
                    )
                    overlap_ge_10pct_kernel_ids = sorted(
                        current_detail_df.loc[
                            current_detail_df["overlap_ge_10pct"], "kernel_id"
                        ].unique().tolist()
                    )
                    overlap_ge_50pct_kernel_ids = sorted(
                        current_detail_df.loc[
                            current_detail_df["overlap_ge_50pct"], "kernel_id"
                        ].unique().tolist()
                    )
                    overlap_ge_5pct_count = len(overlap_ge_5pct_kernel_ids)
                    overlap_ge_10pct_count = len(overlap_ge_10pct_kernel_ids)
                    overlap_ge_50pct_count = len(overlap_ge_50pct_kernel_ids)

                summary_rows.append(
                    {
                        "run_index": run_index,
                        "input": input_token,
                        "model": inference_row["Model"],
                        "inference_time": inference_row["Elapsed Time"],
                        "resource_heavy_kernel_count": resource_heavy_kernel_count,
                        "overlap_ge_5pct_count": overlap_ge_5pct_count,
                        "overlap_ge_10pct_count": overlap_ge_10pct_count,
                        "overlap_ge_50pct_count": overlap_ge_50pct_count,
                        "overlap_ge_5pct_kernel_ids": overlap_ge_5pct_kernel_ids,
                        "overlap_ge_10pct_kernel_ids": overlap_ge_10pct_kernel_ids,
                        "overlap_ge_50pct_kernel_ids": overlap_ge_50pct_kernel_ids,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        detail_df = pd.DataFrame(detail_rows)

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_path, index=False)

            detail_path = output_path.with_name(f"{output_path.stem}_details.csv")
            detail_df.to_csv(detail_path, index=False)

        return summary_df, detail_df
    
    def filter_kernels_by_layer(self, kernel_df: pd.DataFrame, target_layer: str = "e2e") -> pd.DataFrame:
        """
        Filter kernel timings by a specific layer.
        
        Args:
            kernel_df: DataFrame containing kernel timings
            target_layer: Layer to filter by (default: "e2e")
            
        Returns:
            Filtered DataFrame containing only kernels from the target layer
        """
        filtered_df = kernel_df[kernel_df['Layer'] == target_layer].copy()
        return filtered_df
    
    def identify_memcpy_operations(self, kernel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and extract memcpy operations from kernel timings.
        
        Args:
            kernel_df: DataFrame containing kernel timings
            
        Returns:
            DataFrame containing only memcpy operations
        """
        # Look for only device2host and host2device kernel names
        memcpy_patterns = ['device2host', 'host2device']
        
        memcpy_mask = kernel_df['Kernel Name'].str.contains('|'.join(memcpy_patterns), 
                                                          case=False, na=False)
        
        memcpy_df = kernel_df[memcpy_mask].copy()
        
        # Add memcpy type classification
        def classify_memcpy(kernel_name):
            kernel_lower = kernel_name.lower()
            if 'device2host' in kernel_lower:
                return 'device2host'
            elif 'host2device' in kernel_lower:
                return 'host2device'
            else:
                return 'unknown'
        
        memcpy_df['memcpy_type'] = memcpy_df['Kernel Name'].apply(classify_memcpy)
        
        return memcpy_df

    
    def find_aligned_memcpy_for_input_optimized(self, memcpy_df: pd.DataFrame,
                                               all_kernels_df: pd.DataFrame,
                                               input_token: str, model: str,
                                               inference_start_us: float,
                                               inference_end_us: float,
                                               alignment_threshold_ms: float = 2.0) -> Dict:
        """
        Optimized version: Calculate GPU utilization score for memcpy operations of a specific input/model.
        The score represents the percentage of time other models are running GPU operations (all kernel types) within the time window.
        
        Args:
            memcpy_df: Pre-processed DataFrame containing memcpy operations with timestamps in microseconds
            all_kernels_df: Pre-processed DataFrame containing all kernel operations with timestamps in microseconds
            input_token: The specific input token to search for
            model: The model name to search in
            inference_start_us: Start time of inference in microseconds to limit search range
            inference_end_us: End time of inference in microseconds to limit search range
            alignment_threshold_ms: Time threshold for alignment in milliseconds
            
        Returns:
            Dictionary containing GPU utilization analysis results
        """
        # Filter memcpy operations for the specific input token and model
        target_memcpy = memcpy_df[
            (memcpy_df['Input'] == input_token) & 
            (memcpy_df['Model'] == model)
        ].copy()
        
        if len(target_memcpy) == 0:
            self.logger.warning(f"No memcpy operations found for {model}/{input_token}")
            return {
                'input_token': input_token,
                'model': model,
                'total_memcpy_count': 0,
                'gpu_utilization_scores': [],
                'A_score': 0.0
            }
        
        # Convert alignment threshold to microseconds
        alignment_threshold_us = alignment_threshold_ms * 1000
        
        # Filter all kernels from other models that could potentially overlap with inference windows
        # Expand the search window by the alignment threshold for efficiency
        search_start = inference_start_us - alignment_threshold_us
        search_end = inference_end_us + alignment_threshold_us
        
        other_kernels = all_kernels_df[
            (all_kernels_df['Model'] != model) & 
            (all_kernels_df['center_us'] >= search_start) & 
            (all_kernels_df['center_us'] <= search_end)
        ].copy()
        
        gpu_utilization_scores = []
        
        # For each memcpy operation in the target input/model, calculate GPU utilization score
        for _, target_row in target_memcpy.iterrows():
            target_center = target_row['center_us']
            
            # Define time window: look for operations that come BEFORE the target (within previous threshold)
            window_start = target_center - alignment_threshold_us
            window_end = target_center
            window_duration = alignment_threshold_us  # Total window duration
            
            # Find all other model operations (any kernel type) that overlap with this time window
            overlapping_operations = other_kernels[
                (other_kernels['start_us'] < window_end) & 
                (other_kernels['end_us'] > window_start)
            ]
            
            # Calculate total GPU time used by other models within the window
            total_gpu_time = 0.0
            
            for _, other_row in overlapping_operations.iterrows():
                # Calculate the overlap between the operation and the time window
                overlap_start = max(other_row['start_us'], window_start)
                overlap_end = min(other_row['end_us'], window_end)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    total_gpu_time += overlap_duration
            
            # Calculate GPU utilization score as percentage (should be <= 1.0)
            gpu_utilization_score = min(1.0, (total_gpu_time / window_duration))

            
            gpu_utilization_scores.append({
                'target_memcpy_start_us': target_row['start_us'],
                'target_memcpy_end_us': target_row['end_us'],
                'target_memcpy_center_us': target_center,
                'window_start_us': window_start,
                'window_end_us': window_end,
                'gpu_utilization_score': gpu_utilization_score,
                'overlapping_operations_count': len(overlapping_operations)
            })
        
        # Calculate average GPU utilization score
        scores = np.array([s['gpu_utilization_score'] for s in gpu_utilization_scores])
        mean_term = np.mean(scores)
        A_score = mean_term

        return {
            'input_token': input_token,
            'model': model,
            'total_memcpy_count': len(target_memcpy),
            'gpu_utilization_scores': gpu_utilization_scores,
            'A_score': A_score
        }


    def get_inference_time(self, run_index: int, input_token: str, model: str) -> Optional[Tuple[float, float, float]]:
        """
        Get the inference time and timing range for a specific input token and model.
        
        Args:
            run_index: The run index
            input_token: The input token
            model: The model name
            
        Returns:
            Tuple of (inference_time_seconds, start_us, end_us) or None if not found
        """
        try:
            layer_df = self.load_layer_timings(run_index)
            inference_data = layer_df[
                (layer_df['Input'] == input_token) & 
                (layer_df['Model'] == model) & 
                (layer_df['Layer'] == 'inference') &
                (layer_df['run_index'] == run_index)
            ]
            
            if len(inference_data) > 0:
                inference_time = inference_data.iloc[0]['Elapsed Time']
                
                # Get timing range from Start/End Timestamp
                if 'Start Timestamp' in inference_data.columns and 'End Timestamp' in inference_data.columns:
                    start_us = inference_data.iloc[0]['Start Timestamp'] / 1000
                    end_us = inference_data.iloc[0]['End Timestamp'] / 1000
                    return (inference_time, start_us, end_us)
            else:
                # No inference layer found
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting inference time: {e}")
            return None

    

    def memcpy_analysis(self, run_index: int, 
                                            alignment_threshold_ms: float = 2.0,
                                            output_dir: Optional[str] = None,
                                            create_plots: bool = True,
                                            max_points_per_model: int = 250) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Optimized version: Generate CSV files for each model with input_token, A_score, and inference_time.
        Preprocesses data once and uses inference time ranges to optimize searches.
        Now includes integrated plotting functionality.
        
        Args:
            run_index: The run index to analyze
            alignment_threshold_ms: Time threshold for alignment in milliseconds
            output_dir: Directory to save the CSV files
            create_plots: Whether to automatically create plots after generating CSVs
            max_points_per_model: Maximum number of points to plot per model (random sampling if exceeded)
            
        Returns:
            Tuple of (dictionary mapping model names to their CSV file paths, 
                     dictionary mapping model names to their correlation coefficients)
        """
        self.logger.info(f"Generating model memcpy CSV files (optimized) for run {run_index}")
        
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)

        # If CSVs for this run/threshold already exist, reuse them and compute correlations directly.
        target_threshold = float(np.round(alignment_threshold_ms, 2))
        generated_files = {}
        correlations = {}

        def _compute_correlation_from_df(model_df: pd.DataFrame, model_name: str) -> float:
            try:
                x_vals = model_df['A_score'].values
                y_vals = model_df['inference_time'].values
                valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)

                if np.sum(valid_mask) <= 1:
                    self.logger.warning(
                        f"Insufficient valid data points for correlation calculation for model {model_name}"
                    )
                    return 0.0

                x_clean = x_vals[valid_mask]
                y_clean = y_vals[valid_mask]

                if np.std(x_clean) <= 1e-10 or np.std(y_clean) <= 1e-10:
                    self.logger.warning(
                        f"Insufficient variance in data for correlation calculation for model {model_name}"
                    )
                    return 0.0

                correlation = np.corrcoef(x_clean, y_clean)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            except Exception as e:
                self.logger.warning(f"Error calculating correlation for model {model_name}: {e}")
                return 0.0

        existing_csv_files = list(output_dir.glob(f"*_run_{run_index}_*.csv"))
        for csv_file in existing_csv_files:
            filename = csv_file.stem
            suffix = f"_run_{run_index}_"
            if suffix not in filename:
                continue

            safe_model_name, threshold_str = filename.rsplit(suffix, 1)
            try:
                csv_threshold = float(threshold_str)
            except ValueError:
                continue

            if not np.isclose(csv_threshold, target_threshold):
                continue

            model_name = self._reverse_safe_filename(safe_model_name)
            try:
                model_df = pd.read_csv(csv_file)
                # Ensure expected columns exist before attempting correlation calculation.
                if {'A_score', 'inference_time'}.issubset(model_df.columns):
                    correlations[model_name] = _compute_correlation_from_df(model_df, model_name)
                    generated_files[model_name] = str(csv_file)
                else:
                    self.logger.warning(
                        f"Skipping malformed CSV (missing required columns): {csv_file}"
                    )
            except Exception as e:
                self.logger.warning(f"Error loading existing CSV {csv_file}: {e}")

        if len(generated_files) > 0:
            self.logger.info(
                f"Reused {len(generated_files)} existing model CSV files for run {run_index} "
                f"at threshold {target_threshold}"
            )
            if create_plots:
                self.logger.info("Creating plots from existing model CSV files...")
                try:
                    self.plot_multi_model_memcpy_vs_inference(
                        run_index=run_index,
                        csv_files=generated_files,
                        output_dir=output_dir,
                        save_plot=True,
                        max_points_per_model=max_points_per_model
                    )
                except Exception as e:
                    self.logger.error(f"Error creating plots: {e}")
            return generated_files, correlations
        
        # Load and preprocess data once
        kernel_df = self.load_kernel_timings(run_index)
        
        if len(kernel_df) == 0:
            raise RuntimeError(f"No kernel data found for run {run_index}")
        
        # Filter by e2e layer and identify memcpy operations once
        kernel_df = self.filter_kernels_by_layer(kernel_df, "e2e")
        memcpy_df = self.identify_memcpy_operations(kernel_df)
        
        if len(memcpy_df) == 0:
            self.logger.warning(f"No memcpy operations found in run {run_index}")
            return {}
        
        # Precompute center timestamps for all memcpy operations
        memcpy_df['start_us'] = memcpy_df['Start Timestamp'] / 1000
        memcpy_df['end_us'] = memcpy_df['End Timestamp'] / 1000
        memcpy_df['center_us'] = (memcpy_df['start_us'] + memcpy_df['end_us']) / 2
        
        # Also precompute center timestamps for all kernel operations (needed for GPU utilization calculation)
        kernel_df['start_us'] = kernel_df['Start Timestamp'] / 1000
        kernel_df['end_us'] = kernel_df['End Timestamp'] / 1000
        kernel_df['center_us'] = (kernel_df['start_us'] + kernel_df['end_us']) / 2
        
        # Get unique models and input tokens
        unique_models = memcpy_df['Model'].unique()
        unique_inputs = memcpy_df['Input'].unique()
        
        generated_files = {}
        correlations = {}
        
        for model in unique_models:
            model_data = []
            
            # For each input token, find aligned memcpy operations for this model
            for input_token in unique_inputs:
                try:
                    # Get inference time using existing method
                    inference_time_tuple = self.get_inference_time(run_index, input_token, model)
                    
                    # Only proceed if we have complete inference timing data
                    if inference_time_tuple is not None:
                        inference_time, inference_start_us, inference_end_us = inference_time_tuple
                        
                        # Check if we have timing ranges for the optimized method
                        if inference_start_us is not None and inference_end_us is not None:
                            # Get GPU utilization scores using optimized method with all kernel types
                            aligned_results = self.find_aligned_memcpy_for_input_optimized(
                                memcpy_df=memcpy_df,
                                all_kernels_df=kernel_df,  # Pass all kernel types for GPU utilization calculation
                                input_token=input_token,
                                model=model,
                                inference_start_us=inference_start_us,
                                inference_end_us=inference_end_us,
                                alignment_threshold_ms=alignment_threshold_ms
                            )
                            
                            A_score = aligned_results.get('A_score', 0.0)

                            model_data.append({
                                'input_token': input_token,
                                'A_score': A_score,
                                'inference_time': inference_time,
                            })
                        
                except Exception as e:
                    self.logger.warning(f"Error processing input {input_token} for model {model}: {e}")
                    continue
            
            if model_data:
                # Create DataFrame and save to CSV
                model_df = pd.DataFrame(model_data)
                
                # Calculate correlation with robust handling
                try:
                    # Check for valid data before correlation calculation
                    x_vals = model_df['A_score'].values
                    y_vals = model_df['inference_time'].values
                    
                    # Remove any NaN or infinite values
                    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    
                    if np.sum(valid_mask) > 1:  # Need at least 2 valid points
                        x_clean = x_vals[valid_mask]
                        y_clean = y_vals[valid_mask]
                        
                        # Check for variance in both variables
                        if np.std(x_clean) > 1e-10 and np.std(y_clean) > 1e-10:
                            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
                            correlations[model] = correlation if not np.isnan(correlation) else 0.0
                        else:
                            self.logger.warning(f"Insufficient variance in data for correlation calculation for model {model}")
                            correlations[model] = 0.0
                    else:
                        self.logger.warning(f"Insufficient valid data points for correlation calculation for model {model}")
                        correlations[model] = 0.0
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating correlation for model {model}: {e}")
                    correlations[model] = 0.0
                
                # Create safe filename
                safe_model_name = model.replace('/', '_').replace('(', '').replace(')', '').replace("'", '').replace(',', '_').replace(' ', '_')
                csv_filename = f"{safe_model_name}_run_{run_index}_{alignment_threshold_ms}.csv"
                csv_filepath = output_dir / csv_filename
                
                model_df.to_csv(csv_filepath, index=False)
                generated_files[model] = str(csv_filepath)

            else:
                self.logger.warning(f"No valid data found for model {model}")
        
        self.logger.info(f"Generated CSV files for {len(generated_files)} models")
        
        # Create plots if requested and we have data
        if create_plots and len(generated_files) > 0:
            self.logger.info("Creating plots for generated models...")
            try:
                self.plot_multi_model_memcpy_vs_inference(
                    run_index=run_index,
                    csv_files=generated_files,
                    output_dir=output_dir,
                    save_plot=True,
                    max_points_per_model=max_points_per_model
                )
            except Exception as e:
                self.logger.error(f"Error creating plots: {e}")
        
        return generated_files, correlations


    def plot_multi_model_memcpy_vs_inference(self, 
                                             run_index: int,
                                             csv_files: Optional[Dict[str, str]] = None,
                                             csv_dir: Optional[str] = None,
                                             output_dir: Optional[str] = None,
                                             save_plot: bool = True,
                                             max_points_per_model: int = 330) -> None:
        """
        Plot the relationship between aligned memcpy count and inference time for multiple models.
        
        Args:
            run_index: The run index for labeling
            csv_files: Dictionary mapping model names to their CSV file paths. If None, will auto-discover CSV files.
            csv_dir: Directory containing CSV files (used when csv_files is None for auto-discovery)
            output_dir: Directory to save the plot
            save_plot: Whether to save the plot to file
            max_points_per_model: Maximum number of points to plot per model (random sampling if exceeded)
        """
        # Auto-discover CSV files if not provided
        if csv_files is None:
            self.logger.info(f"Auto-discovering CSV files for run {run_index}")
            csv_files = self.find_existing_csv_files(run_index, csv_dir)
            
        self.logger.info(f"Creating multi-model memcpy vs inference time plot for {len(csv_files)} models")
        
        if len(csv_files) == 0:
            self.logger.warning("No CSV files provided for plotting")
            return
        
        # Load all CSV files
        model_data = {}
        for model_name, csv_path in csv_files.items():
            try:
                df = pd.read_csv(csv_path)
                
                # Sample data if there are too many points
                if len(df) > max_points_per_model:
                    df_sampled = df.sample(n=max_points_per_model, random_state=42)
                    model_data[model_name] = df_sampled
                else:
                    model_data[model_name] = df
            except Exception as e:
                self.logger.error(f"Error loading CSV for model {model_name}: {e}")
                continue
        
        if not model_data:
            self.logger.error("No valid CSV files could be loaded")
            return
        
        # Define markers for different models within the same type
        markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', '+', 'x']
        
        # Always create a combined plot
        plt.figure(figsize=(12, 8))
        
        # Sort models by type for consistent legend order
        model_order = ['image', 'lidar', 'seg']
        sorted_models = []
        
        # Group models by type first
        models_by_type = {'image': [], 'lidar': [], 'seg': []}
        for model_name in model_data.keys():
            model_type, _ = self._get_model_type_and_color(model_name)
            models_by_type[model_type].append(model_name)
        
        # Create ordered list: image models first, then lidar, then seg
        for model_type in model_order:
            sorted_models.extend(models_by_type[model_type])
        
        self.logger.info(f"Legend order: {[model_name_mappings.get(m, m) for m in sorted_models]}")
        
        # Group models by type to assign consistent markers within type
        type_counters = {'image': 0, 'lidar': 0, 'seg': 0}
        
        for model_name in sorted_models:
            df = model_data[model_name]
            
            # Get model type and fixed color
            model_type, color = self._get_model_type_and_color(model_name)
            
            # Use different markers for models of the same type
            marker = markers[type_counters[model_type] % len(markers)]
            type_counters[model_type] += 1
            
            # Get display name from mapping, fallback to original name if not found
            display_name = model_name_mappings.get(model_name, model_name)
            
            # Calculate slope for legend
            slope = None
            if len(df) > 1:
                try:
                    # Check for valid data before fitting
                    print(df)
                    x_data = df['A_score'].values
                    y_data = df['inference_time'].values
                    
                    # Remove any NaN or infinite values
                    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_clean = x_data[valid_mask]
                    y_clean = y_data[valid_mask]
                    
                    # Check if we have enough valid points and variance in x
                    if len(x_clean) > 1 and np.std(x_clean) > 1e-10:
                        z = np.polyfit(x_clean, y_clean, 1)
                        slope = z[0]  # First coefficient is the slope
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate slope for model {model_name}: {e}")
            
            # Create label with slope information
            if slope is not None:
                label_with_slope = f"{display_name} (slope: {slope:.2f})"
            else:
                label_with_slope = f"{display_name} (slope: N/A)"
            
            # Scatter plot
            plt.scatter(df['A_score'], df['inference_time'], 
                       alpha=0.6, s=75, color=color, label=label_with_slope, marker=marker)
            
            # Add trend line with robust error handling
            if len(df) > 1:
                try:
                    # Check for valid data before fitting
                    x_data = df['A_score'].values
                    y_data = df['inference_time'].values
                    
                    # Remove any NaN or infinite values
                    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_clean = x_data[valid_mask]
                    y_clean = y_data[valid_mask]
                    
                    # Check if we have enough valid points and variance in x
                    if len(x_clean) > 1 and np.std(x_clean) > 1e-10:
                        z = np.polyfit(x_clean, y_clean, 1)
                        p = np.poly1d(z)
                        
                        # Calculate correlation coefficient
                        correlation = np.corrcoef(x_clean, y_clean)[0, 1]
                        
                        # Plot trend line using original data range
                        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                        plt.plot(x_range, p(x_range), 
                                "--", alpha=0.8, linewidth=4, color=color)
                    else:
                        self.logger.warning(f"Insufficient variance in data for trend line fitting for model {model_name}")
                        
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                    self.logger.warning(f"Could not fit trend line for model {model_name}: {e}")
                except Exception as e:
                    self.logger.warning(f"Unexpected error fitting trend line for model {model_name}: {e}")
        
        # plt.xlabel('Alignment Score (A)', fontsize=16)
        # plt.ylabel('Inference Time (ms)', fontsize=16)
        
        # Set tick label font sizes
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)
        
        # Handle legend placement inside the plot but not overlapping data
        try:
            plt.legend(loc='upper right', fontsize=20)
        except Exception as e:
            self.logger.warning(f"Error placing legend, using default: {e}")
            plt.legend(loc='best', fontsize=20)
            
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            try:
                if output_dir is None:
                    output_dir = self.output_dir
                else:
                    output_dir = Path(output_dir)
                    
                output_dir.mkdir(exist_ok=True)
                
                plot_filename = f"memcpy_vs_inference_{run_index}.png"
                plot_path = output_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved plot to: {plot_path}")
                
            except Exception as e:
                self.logger.error(f"Error saving plot: {e}")
                # Try saving with simpler options
                try:
                    plot_filename = f"memcpy_vs_inference_{run_index}_fallback.png"
                    plot_path = output_dir / plot_filename
                    plt.savefig(plot_path, dpi=150)
                    self.logger.info(f"Saved fallback plot to: {plot_path}")
                except Exception as e2:
                    self.logger.error(f"Failed to save even fallback plot: {e2}")
        
        # Always try to show/close the plot properly
        try:
            plt.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not apply tight layout: {e}")
            
        try:
            plt.close()
        except Exception as e:
            self.logger.warning(f"Error closing plot: {e}")


    def plot_multi_model_memcpy_vs_inference_subplots(self, 
                                                     run_index: int,
                                                     csv_files: Optional[Dict[str, str]] = None,
                                                     csv_dir: Optional[str] = None,
                                                     output_dir: Optional[str] = None,
                                                     save_plot: bool = True,
                                                     max_points_per_model: int = 250,
                                                     alignment_threshold_ms: float = 2.0) -> None:
        """
        Plot the relationship between aligned memcpy count and inference time for multiple models 
        in separate subplots, ordered by model type (image, lidar, seg).
        
        Args:
            run_index: The run index for labeling
            csv_files: Dictionary mapping model names to their CSV file paths. If None, will auto-discover CSV files.
            csv_dir: Directory containing CSV files (used when csv_files is None for auto-discovery)
            output_dir: Directory to save the plot
            save_plot: Whether to save the plot to file
            max_points_per_model: Maximum number of points to plot per model (random sampling if exceeded)
        """
        # Auto-discover CSV files if not provided
        if csv_files is None:
            csv_files = self.find_existing_csv_files(run_index, csv_dir, alignment_threshold_ms)
            
        if len(csv_files) == 0:
            self.logger.warning("No CSV files provided for plotting")
            return
        
        # Load all CSV files
        model_data = {}
        for model_name, csv_path in csv_files.items():
            try:
                df = pd.read_csv(csv_path)
                
                # Sample data if there are too many points
                if len(df) > max_points_per_model:
                    df_sampled = df.sample(n=max_points_per_model, random_state=42)
                    model_data[model_name] = df_sampled
                else:
                    model_data[model_name] = df
            except Exception as e:
                self.logger.error(f"Error loading CSV for model {model_name}: {e}")
                continue
        
        if not model_data:
            self.logger.error("No valid CSV files could be loaded")
            return
        
        # Sort models by type for consistent subplot order
        model_order = ['image', 'lidar', 'seg']
        sorted_models = []
        
        # Group models by type first
        models_by_type = {'image': [], 'lidar': [], 'seg': []}
        for model_name in model_data.keys():
            model_type, _ = self._get_model_type_and_color(model_name)
            models_by_type[model_type].append(model_name)
        
        # Create ordered list: image models first, then lidar, then seg
        for model_type in model_order:
            sorted_models.extend(models_by_type[model_type])
        
        # Calculate subplot layout
        n_models = len(sorted_models)
        if n_models == 0:
            self.logger.warning("No models to plot")
            return
        
        # Determine optimal subplot layout
        if n_models <= 3:
            rows, cols = 1, n_models
            figsize = (6*cols, 5)
        elif n_models <= 6:
            rows, cols = 2, 3
            figsize = (18, 10)
        elif n_models <= 9:
            rows, cols = 3, 3
            figsize = (18, 15)
        else:
            rows, cols = (n_models + 2) // 3, 3  # Ceiling division
            figsize = (18, 5*rows)
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Ensure axes is always a list for consistent indexing
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten() if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Define marker for each model type
        type_markers = {'image': 'o', 'lidar': 's', 'seg': '^'}
        
        # Plot each model in its own subplot
        for idx, model_name in enumerate(sorted_models):
            ax = axes[idx]
            df = model_data[model_name]
            
            # Get model type and fixed color
            model_type, color = self._get_model_type_and_color(model_name)
            marker = type_markers[model_type]
            
            # Get display name from mapping, fallback to original name if not found
            display_name = model_name_mappings.get(model_name, model_name)
            
            # Calculate slope for title
            slope = None
            if len(df) > 1:
                try:
                    # Check for valid data before fitting
                    x_data = df['A_score'].values
                    y_data = df['inference_time'].values
                    
                    # Remove any NaN or infinite values
                    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_clean = x_data[valid_mask]
                    y_clean = y_data[valid_mask]
                    
                    # Check if we have enough valid points and variance in x
                    if len(x_clean) > 1 and np.std(x_clean) > 1e-10:
                        z = np.polyfit(x_clean, y_clean, 1)
                        slope = z[0]  # First coefficient is the slope
                        
                        # Plot trend line in red
                        p = np.poly1d(z)
                        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                        ax.plot(x_range, p(x_range), 
                               "--", alpha=0.8, linewidth=5, color='red')
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate slope for model {model_name}: {e}")
            
            # Scatter plot
            ax.scatter(df['A_score'], df['inference_time'], 
                      alpha=0.7, s=60, color=color, marker=marker)
            
            # Set subplot title with slope information
            if slope is not None:
                title = f"{display_name}\n(slope: {slope:.2f})"
            else:
                title = f"{display_name}\n(slope: N/A)"
            
            # ax.set_title(title, fontsize=25, fontweight='bold')
            # ax.set_xlabel('Alignment Score (A)', fontsize=12)
            # ax.set_ylabel('Inference Time (ms)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis range to be 20% larger than max point
            y_data = df['inference_time'].values
            if len(y_data) > 0:
                y_max = np.max(y_data)
                y_min = np.min(y_data)
                y_range = y_max - y_min
                # Extend the upper limit by 20% of the max value
                ax.set_ylim(y_min - 0.05 * y_range, y_max * 1.2)
            
            # Set tick label font sizes
            ax.tick_params(axis='both', labelsize=25)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        # Adjust layout with minimal spacing
        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        
        # Save plot
        if save_plot:
            try:
                if output_dir is None:
                    output_dir = self.output_dir
                else:
                    output_dir = Path(output_dir)
                    
                output_dir.mkdir(exist_ok=True)
                
                plot_filename = f"memcpy_vs_inference_subplots_{run_index}.png"
                plot_path = output_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved subplot plot to: {plot_path}")
                
            except Exception as e:
                self.logger.error(f"Error saving subplot plot: {e}")
                # Try saving with simpler options
                try:
                    plot_filename = f"memcpy_vs_inference_subplots_{run_index}_fallback.png"
                    plot_path = output_dir / plot_filename
                    plt.savefig(plot_path, dpi=150)
                    self.logger.info(f"Saved fallback subplot plot to: {plot_path}")
                except Exception as e2:
                    self.logger.error(f"Failed to save even fallback subplot plot: {e2}")
        
        # Clean up plot
        try:
            plt.close()
        except Exception as e:
            self.logger.warning(f"Error closing subplot plot: {e}")


    def plot_target_model_multi_experiment(self, 
                                          target_model: str,
                                          model_mode: str,
                                          mapping_file: str,
                                          csv_dir: Optional[str] = None,
                                          output_dir: Optional[str] = None,
                                          save_plot: bool = True,
                                          max_points_per_experiment: int = 100,
                                          alignment_threshold_ms: float = 2.0) -> None:
        """
        Plot the relationship between GPU utilization score and inference time for a target model 
        across multiple experiments, with different colors for each experiment and a legend showing 
        the contention models.
        
        Args:
            target_model: The target model name to analyze
            model_mode: The mode of the target model ('image_model', 'lidar_model', 'seg_model')
            mapping_file: Path to the mapping CSV file containing experiment configurations
            csv_dir: Directory containing the CSV files from memcpy_analysis (defaults to output_dir)
            output_dir: Directory to save the plot (defaults to self.output_dir)
            save_plot: Whether to save the plot to file
            max_points_per_experiment: Maximum number of points to plot per experiment (random sampling if exceeded)
        """
        
        # Set default directories
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            
        if csv_dir is None:
            csv_dir = output_dir
        else:
            csv_dir = Path(csv_dir)
            
        # Load mapping file
        mapping_df = self.load_mapping_csv(mapping_file)
        
        # Filter experiments by target model and mode
        if model_mode not in ['image_model', 'lidar_model', 'seg_model']:
            raise ValueError(f"Invalid model_mode: {model_mode}. Must be one of: image_model, lidar_model, seg_model")
            
        # Filter mapping based on the target model and mode
        target_experiments = mapping_df[mapping_df[model_mode] == target_model].copy()
        
        if len(target_experiments) == 0:
            self.logger.warning(f"No experiments found for target model '{target_model}' in mode '{model_mode}'")
            return
            
        # Prepare data for plotting
        experiment_data = {}
        contention_models = {}
        
        for _, experiment in target_experiments.iterrows():
            run_index = experiment['run_index']
            
            # Determine contention model(s) based on the target model mode
            if model_mode == 'image_model':
                # If target is image model, contention is lidar model
                contention_model = experiment['lidar_model']
                # Clean up the lidar model string (remove tuple formatting)
                if isinstance(contention_model, str) and contention_model.startswith("('"):
                    import ast
                    try:
                        parsed = ast.literal_eval(contention_model)
                        if isinstance(parsed, tuple) and len(parsed) > 0:
                            contention_model = parsed[0]  # Use the model name part
                    except (ValueError, SyntaxError):
                        pass  # Keep original if parsing fails
                        
            elif model_mode == 'lidar_model':
                # If target is lidar model, contention is image model
                contention_model = experiment['image_model']
            else:  # seg_model
                # For seg_model, we need to look at both image and lidar as contention
                contention_model = f"{experiment['image_model']} + {experiment['lidar_model']}"
            
            # Load corresponding CSV file
            safe_target_name = target_model.replace('/', '_').replace('(', '').replace(')', '').replace("'", '').replace(',', '_').replace(' ', '_')
            csv_filename = f"{safe_target_name}_run_{run_index}.csv"
            csv_filepath = csv_dir / csv_filename
            
            if not csv_filepath.exists():
                self.logger.warning(f"CSV file not found: {csv_filepath}")
                continue
                
            try:
                df = pd.read_csv(csv_filepath)
                
                # Sample data if there are too many points
                if len(df) > max_points_per_experiment:
                    df_sampled = df.sample(n=max_points_per_experiment, random_state=42)
                    experiment_data[run_index] = df_sampled
                else:
                    experiment_data[run_index] = df
                    
                contention_models[run_index] = contention_model
                
            except Exception as e:
                self.logger.error(f"Error loading CSV for run {run_index}: {e}")
                continue
        
        if not experiment_data:
            self.logger.error("No valid CSV files could be loaded for any experiment")
            return
            
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Define colors and markers for experiments
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiment_data)))
        markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', '+', 'x', '<', '>', '8', 'P', 'X']
        
        # Plot each experiment
        for idx, (run_index, df) in enumerate(experiment_data.items()):
            color = colors[idx]
            marker = markers[idx % len(markers)]
            contention_model = contention_models[run_index]
            
            # Create label with contention model info
            label = f"Run {run_index}: {contention_model}"
            
            # Scatter plot
            plt.scatter(df['A_score'], df['inference_time'], 
                       alpha=0.7, s=120, color=color, label=label, marker=marker)
            
            # Add trend line with robust error handling
            if len(df) > 1:
                try:
                    # Check for valid data before fitting
                    x_data = df['A_score'].values
                    y_data = df['inference_time'].values
                    
                    # Remove any NaN or infinite values
                    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_clean = x_data[valid_mask]
                    y_clean = y_data[valid_mask]
                    
                    # Check if we have enough valid points and variance in x
                    if len(x_clean) > 1 and np.std(x_clean) > 1e-10:
                        z = np.polyfit(x_clean, y_clean, 1)
                        p = np.poly1d(z)
                        
                        # Plot trend line using original data range
                        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                        plt.plot(x_range, p(x_range), 
                                "--", alpha=0.8, linewidth=4, color=color)
                    else:
                        self.logger.warning(f"Insufficient variance in data for trend line fitting for run {run_index}")
                        
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                    self.logger.warning(f"Could not fit trend line for run {run_index}: {e}")
                except Exception as e:
                    self.logger.warning(f"Unexpected error fitting trend line for run {run_index}: {e}")
        
        # Set tick label font sizes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Handle legend placement inside the plot but not overlapping data
        try:
            plt.legend(loc='upper right', fontsize=10)
        except Exception as e:
            self.logger.warning(f"Error placing legend, using default: {e}")
            plt.legend(loc='best', fontsize=10)
            
        plt.grid(True, alpha=0.3)
        
        # Add plot statistics as text
        total_points = sum(len(df) for df in experiment_data.values())
        total_experiments = len(experiment_data)
        plt.figtext(0.02, 0.02, 
                   f'Total experiments: {total_experiments}, Total data points: {total_points}', 
                   fontsize=8, alpha=0.7)
        
        # Save plot
        if save_plot:
            try:
                output_dir.mkdir(exist_ok=True)
                
                # Create safe filename
                safe_target_name = target_model.replace('/', '_').replace('(', '').replace(')', '').replace("'", '').replace(',', '_').replace(' ', '_')
                plot_filename = f"multi_experiment_{safe_target_name}_{model_mode}.png"
                plot_path = output_dir / plot_filename
                
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                
            except Exception as e:
                self.logger.error(f"Error saving plot: {e}")
                # Try saving with simpler options
                try:
                    plot_filename = f"multi_experiment_{safe_target_name}_{model_mode}_fallback.png"
                    plot_path = output_dir / plot_filename
                    plt.savefig(plot_path, dpi=150)
                    self.logger.info(f"Saved fallback plot to: {plot_path}")
                except Exception as e2:
                    self.logger.error(f"Failed to save even fallback plot: {e2}")
        
        # Clean up plot
        try:
            plt.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not apply tight layout: {e}")
            
        try:
            plt.close()
        except Exception as e:
            self.logger.warning(f"Error closing plot: {e}")


    def find_existing_csv_files(self, run_index: int, csv_dir: Optional[str] = None, alignment_threshold_ms: float = 2.0) -> Dict[str, str]:
        """
        Find existing CSV files from previous memcpy_analysis runs.
        
        Args:
            run_index: The run index to find CSV files for
            csv_dir: Directory containing the CSV files (defaults to output_dir/memcpy_analysis)
            
        Returns:
            Dictionary mapping model names to their CSV file paths
        """
        if csv_dir is None:
            csv_dir = self.output_dir / "memcpy_analysis"
        else:
            csv_dir = Path(csv_dir)
            
        if not csv_dir.exists():
            self.logger.warning(f"CSV directory not found: {csv_dir}")
            return {}
            
        # Support both naming conventions:
        # 1) {safe_model_name}_run_{run_index}.csv (legacy)
        # 2) {safe_model_name}_run_{run_index}_{alignment_threshold}.csv (threshold-specific)
        csv_files = {}
        run_suffix = f"_run_{run_index}"
        target_threshold = float(alignment_threshold_ms)
        legacy_candidates = {}

        for csv_file in csv_dir.glob(f"*{run_suffix}*.csv"):
            filename = csv_file.stem  # Remove .csv extension
            if run_suffix not in filename:
                continue

            model_name_safe, tail = filename.rsplit(run_suffix, 1)

            # Exact run match without threshold suffix (legacy format).
            if tail == "":
                original_model_name = self._reverse_safe_filename(model_name_safe)
                legacy_candidates[original_model_name] = str(csv_file)
                continue

            # Threshold-specific format must be: _<float>
            if not tail.startswith("_"):
                continue

            threshold_str = tail[1:]
            try:
                csv_threshold = float(threshold_str)
            except ValueError:
                continue

            if not np.isclose(csv_threshold, target_threshold):
                continue

            original_model_name = self._reverse_safe_filename(model_name_safe)
            csv_files[original_model_name] = str(csv_file)

        # Backward-compatibility fallback when only legacy files exist.
        if not csv_files:
            csv_files = legacy_candidates

        return csv_files
    
    def _reverse_safe_filename(self, safe_name: str) -> str:
        """
        Attempt to reverse the safe filename transformation back to original model name.
        This is a best-effort approach and may not be perfect for all cases.
        
        Args:
            safe_name: The safe filename (without extension)
            
        Returns:
            Best guess at the original model name
        """
        # Try to match against known model names from constants
        from p_perf.config.constant import model_name_mappings
        
        # First, try direct lookup in the mapping values (reverse lookup)
        for original_name, mapped_name in model_name_mappings.items():
            safe_original = original_name.replace('/', '_').replace('(', '').replace(')', '').replace("'", '').replace(',', '_').replace(' ', '_')
            if safe_original == safe_name:
                return original_name
                
        # If no match found, return the safe_name as-is (fallback)
        # Could be enhanced with more sophisticated reverse mapping if needed
        return safe_name
    



def main():
    """Example usage of the KernelProcessor."""
    # Initialize processor
    processor = KernelProcessor("outputs/det_1I1L-1")
    
    # Test the model type and color assignment
    test_models = [
        'faster-rcnn_r50_fpn_1x_coco',  # Should be image/blue
        'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d',  # Should be lidar/orange
        'cascade_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k'  # Should be seg/green
    ]
    
    print("Testing model type and color assignment:")
    for model in test_models:
        model_type, color = processor._get_model_type_and_color(model)
        print(f"  {model} -> {model_type} ({color})")
    
    try:
        # Example: Plot multi-experiment analysis for a target model
        print("\nCreating multi-experiment plot for target model...")
        try:
            processor.plot_target_model_multi_experiment(
                target_model="faster-rcnn_r50_fpn_1x_coco",
                model_mode="image_model",
                mapping_file="full_stack_mapping.csv",
                csv_dir="outputs/det_1I1L-1/memcpy_analysis",  # Directory with CSV files from memcpy_analysis
                output_dir="outputs/det_1I1L-1/multi_experiment_plots",
                save_plot=True,
                max_points_per_experiment=100
            )
            print("Multi-experiment plot created successfully!")
        except Exception as e:
            print(f"Error creating multi-experiment plot: {e}")
        
        # Example for lidar model mode
        print("\nCreating multi-experiment plot for lidar model...")
        try:
            processor.plot_target_model_multi_experiment(
                target_model="pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d",
                model_mode="lidar_model",
                mapping_file="full_stack_mapping.csv",
                csv_dir="outputs/det_1I1L-1/memcpy_analysis",
                output_dir="outputs/det_1I1L-1/multi_experiment_plots",
                save_plot=True,
                max_points_per_experiment=100
            )
            print("Multi-experiment plot for lidar model created successfully!")
        except Exception as e:
            print(f"Error creating multi-experiment plot for lidar model: {e}")

        
        # Generate model-specific CSV files with integrated plotting
        print("\nGenerating model-specific CSV files with integrated plotting...")
        try:
            generated_files, correlations = processor.memcpy_analysis(
                run_index=4,
                alignment_threshold_ms=2,
                output_dir="outputs/det_1I1L-1/memcpy_analysis",
                create_plots=True  # This will automatically create plots for all models
            )
            
            if generated_files:
                print(f"Successfully generated {len(generated_files)} CSV files:")
                for model, filepath in generated_files.items():
                    print(f"  {model}: {filepath}")
                
                print(f"\nCorrelation coefficients:")
                for model, correlation in correlations.items():
                    print(f"  {model}: {correlation:.3f}")
                
                print("\nPlots have been automatically generated and saved.")
            else:
                print("No CSV files were generated")
                
        except Exception as e:
            print(f"Error generating CSV files: {e}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
