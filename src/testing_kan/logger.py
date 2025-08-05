import json
import os
from collections import defaultdict
from typing import Dict, List, Any

import pandas as pd


class Logger:
    """
    Lightweight experiment logger that stores every run in
    memory and dumps the aggregated results to JSON / CSV.
    """

    # prettier dataset names and the desired metric direction
    _DATASET_COLUMN_MAP: Dict[str, tuple[str, bool]] = {
        "adult": ("adult ↑", True),
        "gesture": ("gesture ↑", True),
        "california": ("california ↓", False),
        "churn": ("churn ↑", True),
        "house": ("house ↓", False),
        "fb-comments": ("fb-comments ↓", False),
        "otto": ("otto ↑", True),
        "microsoft": ("microsoft ↓", False),
        "black-friday": ("black-friday ↓", False),
        "higgs-small": ("higgs-small ↑", True),
        "diamond": ("diamond ↓", False),
    }

    def __init__(self, results_dir: str, exp_name: str) -> None:
        self.results_dir = results_dir
        self.exp_name = exp_name
        self._runs: List[Dict[str, Any]] = []
        os.makedirs(self.results_dir, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    
    def log_run(
        self,
        dataset_name: str,
        model_name: str,
        emb_name: str,
        arch_type: str,
        optim_name: str,
        stats: Dict[str, Any],
    ) -> None:
        """Append a single run (after tuning + testing) to memory."""
        self._runs.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "emb": emb_name,
                "arch_type": arch_type,
                "optimizer": optim_name,
                **stats,
            }
        )

    def save(self) -> None:
        """Persist collected runs to disk (JSON + 3 summary CSVs)."""
        if not self._runs:
            print("No data to save.")
            return
        self._save_as_json()
        self._save_as_csv()
        print(f"Experiment '{self.exp_name}' saved to {self.results_dir}")

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Convert NumPy scalars to native Python types for JSON."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: Logger._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Logger._json_safe(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    
    def _save_as_json(self) -> None:
        """Group runs by dataset and dump to a JSON file."""
        out_path = os.path.join(self.results_dir, f"{self.exp_name}_results.json")

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for run in self._runs:
            grouped[run["dataset"]].append(run)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(grouped), f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------ CSV

    def _save_as_csv(self) -> None:
        """Create three pivot tables (performance / train / inference)."""
        df = pd.DataFrame(self._runs)
        if df.empty:
            print("No data for CSV.")
            return

        # unified model identifier
        df["Model"] = (
            df["model"]
            + "_"
            + df["arch_type"]
            + "_"
            + df["emb"]
            + "_"
            + df["optimizer"]
        )

        # formatted strings for ± std columns
        df["performance_str"] = df.apply(
            lambda r: f"{r['metric']:.3f} ± {r['metric_std']:.3f}", axis=1
        )
        df["train_time_str"] = df.apply(
            lambda r: f"{r['full_train_time']:.3f} ± {r['full_train_time_std']:.3f}",
            axis=1,
        )
        df["inference_time_str"] = df.apply(
            lambda r: f"{r['val_epoch_time']:.3f} ± {r['val_epoch_time_std']:.3f}",
            axis=1,
        )
        # pretty dataset names
        df["dataset_pretty"] = df["dataset"].apply(
            lambda x: self._DATASET_COLUMN_MAP.get(x, (x, True))[0]
        )
        ordered_cols = [v[0] for v in self._DATASET_COLUMN_MAP.values()]

        self._pivot_and_save(df, "performance_str", "results.csv", ordered_cols)
        self._pivot_and_save(df, "train_time_str", "train_time.csv", ordered_cols)
        self._pivot_and_save(df, "inference_time_str", "inference_time.csv", ordered_cols)

    def _pivot_and_save(
            self,
            df: pd.DataFrame,
            value_col: str,
            fname: str,
            col_order: List[str],
        ) -> None:
            pivot = (
                df.pivot_table(
                    index="Model",
                    columns="dataset_pretty",
                    values=value_col,
                    aggfunc="first",
                )
                .loc[:, [c for c in col_order if c in df["dataset_pretty"].values]]
                .sort_index()
            )
            pivot.to_csv(os.path.join(self.results_dir, f"{self.exp_name}_{fname}"))