"""Data Agent — CSV, Excel, JSON processing, analysis, and report generation."""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Any

from core.llm_core import TaskType

logger = logging.getLogger(__name__)


class DataAgent:
    """Handles data processing: load, clean, analyze, export, chart."""

    def __init__(self, config: dict, llm_core, ui_layer) -> None:
        self.config = config
        self.llm = llm_core
        self.ui = ui_layer
        self._df = None       # last loaded DataFrame
        self._df_name = ""    # filename of last loaded DataFrame

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle(self, user_input: str, context: dict) -> str:
        """Route natural-language data requests to the right action.

        Args:
            user_input: User's data task description.
            context: Context dict from ContextBuilder.

        Returns:
            Result string.
        """
        low = user_input.lower()

        # Detect file path in input
        path_match = re.search(r'[\w/\\:\-]+\.(?:csv|xlsx?|json|tsv)', user_input, re.I)
        file_path = path_match.group(0) if path_match else None

        if file_path or any(w in low for w in ("load", "open", "read")) and self._df is None:
            if file_path:
                return await self._load_file(file_path, user_input)

        if self._df is None:
            return "No dataset loaded. Say: 'load data from myfile.csv'"

        if any(w in low for w in ("summary", "summarize", "describe", "overview", "info")):
            return self._summary()

        if any(w in low for w in ("clean", "drop null", "remove empty", "fill missing")):
            return self._clean(low)

        if any(w in low for w in ("chart", "plot", "graph", "visualize", "histogram", "bar chart", "scatter")):
            return await self._chart(low, user_input)

        if any(w in low for w in ("export", "save", "write to", "convert to")):
            return self._export(low, user_input)

        if any(w in low for w in ("filter", "where", "rows where", "select")):
            return await self._filter_query(user_input)

        if any(w in low for w in ("group by", "groupby", "aggregate", "count by", "sum by", "mean by")):
            return await self._group_query(user_input)

        if any(w in low for w in ("top", "largest", "smallest", "sort by", "rank")):
            return await self._sort_query(user_input)

        if any(w in low for w in ("report", "generate report", "analysis report")):
            return await self._generate_report(user_input)

        # Fall back to LLM for anything else
        return await self._llm_data_query(user_input)

    # ── File loading ──────────────────────────────────────────────────────────

    async def _load_file(self, path: str, user_input: str) -> str:
        try:
            import pandas as pd
        except ImportError:
            return "pandas is not installed. Run: pip install pandas openpyxl"

        p = Path(path)
        if not p.exists():
            # Try searching
            results = list(Path("C:/Users").rglob(p.name))[:3]
            if results:
                p = results[0]
            else:
                return f"File not found: {path}"

        try:
            ext = p.suffix.lower()
            if ext == ".csv":
                self._df = pd.read_csv(p)
            elif ext in (".xlsx", ".xls"):
                self._df = pd.read_excel(p)
            elif ext == ".json":
                self._df = pd.read_json(p)
            elif ext == ".tsv":
                self._df = pd.read_csv(p, sep="\t")
            else:
                return f"Unsupported file type: {ext}"

            self._df_name = p.name
            rows, cols = self._df.shape
            col_list = ", ".join(self._df.columns[:10].tolist())
            return (
                f"Loaded '{p.name}' — {rows:,} rows x {cols} columns\n"
                f"Columns: {col_list}"
                + (" ..." if cols > 10 else "")
            )
        except Exception as e:
            return f"Error loading file: {e}"

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _summary(self) -> str:
        import pandas as pd
        df = self._df
        buf = io.StringIO()
        rows, cols = df.shape
        lines = [f"Dataset: {self._df_name} — {rows:,} rows x {cols} columns\n"]

        # Numeric columns
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            lines.append("Numeric columns:")
            for col in num_cols[:8]:
                s = df[col]
                lines.append(
                    f"  {col}: min={s.min():.2f}, max={s.max():.2f}, "
                    f"mean={s.mean():.2f}, nulls={s.isna().sum()}"
                )

        # Categorical columns
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if cat_cols:
            lines.append("\nCategorical columns:")
            for col in cat_cols[:5]:
                top = df[col].value_counts().head(3)
                lines.append(f"  {col}: {dict(top)}")

        lines.append(f"\nTotal nulls: {df.isna().sum().sum()}")
        return "\n".join(lines)

    def _clean(self, low: str) -> str:
        df = self._df
        before = len(df)
        if "drop" in low or "remove" in low:
            self._df = df.dropna()
        elif "fill" in low:
            self._df = df.fillna(0)
        else:
            self._df = df.dropna()
        after = len(self._df)
        return f"Cleaned data: removed {before - after} rows with nulls. {after:,} rows remaining."

    # ── Chart ─────────────────────────────────────────────────────────────────

    async def _chart(self, low: str, user_input: str) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return "matplotlib not installed. Run: pip install matplotlib"

        df = self._df
        out_path = Path("data/charts")
        out_path.mkdir(parents=True, exist_ok=True)

        # Guess chart type and columns from input
        col_match = re.findall(r"'([^']+)'|\"([^\"]+)\"|\b(" + "|".join(df.columns) + r")\b", user_input)
        cols = [next(c for c in m if c) for m in col_match if any(c in df.columns for c in m)]

        fig, ax = plt.subplots(figsize=(10, 6))
        chart_name = "chart.png"

        try:
            if "histogram" in low or "hist" in low:
                col = cols[0] if cols else df.select_dtypes(include="number").columns[0]
                df[col].hist(ax=ax, bins=30)
                ax.set_title(f"Histogram of {col}")
                chart_name = f"hist_{col}.png"

            elif "scatter" in low:
                num_cols = df.select_dtypes(include="number").columns
                x = cols[0] if len(cols) >= 1 else num_cols[0]
                y = cols[1] if len(cols) >= 2 else num_cols[1]
                ax.scatter(df[x], df[y], alpha=0.5)
                ax.set_xlabel(x); ax.set_ylabel(y)
                ax.set_title(f"{x} vs {y}")
                chart_name = f"scatter_{x}_{y}.png"

            elif "bar" in low:
                col = cols[0] if cols else df.select_dtypes(exclude="number").columns[0]
                df[col].value_counts().head(15).plot(kind="bar", ax=ax)
                ax.set_title(f"Bar chart of {col}")
                plt.xticks(rotation=45, ha="right")
                chart_name = f"bar_{col}.png"

            elif "line" in low:
                num_cols = df.select_dtypes(include="number").columns
                y = cols[0] if cols else num_cols[0]
                df[y].plot(ax=ax)
                ax.set_title(f"Line chart of {y}")
                chart_name = f"line_{y}.png"

            else:
                # Default: first numeric column histogram
                col = df.select_dtypes(include="number").columns[0]
                df[col].hist(ax=ax, bins=30)
                ax.set_title(f"Distribution of {col}")
                chart_name = "chart.png"

            plt.tight_layout()
            chart_path = out_path / chart_name
            fig.savefig(chart_path, dpi=150)
            plt.close(fig)
            return f"Chart saved to {chart_path}. Open with: start {chart_path}"

        except Exception as e:
            plt.close(fig)
            return f"Chart error: {e}"

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, low: str, user_input: str) -> str:
        out_match = re.search(r'(?:to|as|save as?)\s+([\w/\\:\-]+\.(?:csv|xlsx?|json))', user_input, re.I)
        if out_match:
            out_path = Path(out_match.group(1))
        else:
            stem = Path(self._df_name).stem
            if "excel" in low or "xlsx" in low:
                out_path = Path(f"data/{stem}_export.xlsx")
            elif "json" in low:
                out_path = Path(f"data/{stem}_export.json")
            else:
                out_path = Path(f"data/{stem}_export.csv")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        try:
            if ext == ".csv":
                self._df.to_csv(out_path, index=False)
            elif ext in (".xlsx", ".xls"):
                self._df.to_excel(out_path, index=False)
            elif ext == ".json":
                self._df.to_json(out_path, orient="records", indent=2)
            return f"Exported to {out_path} ({len(self._df):,} rows)"
        except Exception as e:
            return f"Export error: {e}"

    # ── Queries ───────────────────────────────────────────────────────────────

    async def _filter_query(self, user_input: str) -> str:
        # Use LLM to generate pandas filter expression
        prompt = (
            f"DataFrame columns: {list(self._df.columns)}\n"
            f"Data types: {dict(self._df.dtypes.astype(str))}\n"
            f"Task: {user_input}\n\n"
            "Write ONLY a Python pandas boolean expression for df.query() or df[...]. "
            "Example: df[df['age'] > 30]. Return just the expression, no explanation."
        )
        expr = await self.llm.generate(prompt, task_type=TaskType.CODE_WRITE)
        expr = expr.strip().strip("`").replace("```python", "").replace("```", "").strip()
        try:
            result = eval(expr, {"df": self._df, "__builtins__": {}})  # noqa: S307
            if hasattr(result, "shape"):
                rows, _ = result.shape
                return f"Filter result: {rows:,} rows\n{result.head(10).to_string()}"
            return str(result)
        except Exception as e:
            return f"Filter error: {e}\nExpression was: {expr}"

    async def _group_query(self, user_input: str) -> str:
        prompt = (
            f"DataFrame columns: {list(self._df.columns)}\n"
            f"Task: {user_input}\n\n"
            "Write a ONE-LINE Python pandas groupby expression on variable 'df'. "
            "Return only the expression."
        )
        expr = await self.llm.generate(prompt, task_type=TaskType.CODE_WRITE)
        expr = expr.strip().strip("`").replace("```python", "").replace("```", "").strip()
        try:
            result = eval(expr, {"df": self._df, "__builtins__": {}})  # noqa: S307
            return str(result.head(20))
        except Exception as e:
            return f"Group error: {e}\nExpression: {expr}"

    async def _sort_query(self, user_input: str) -> str:
        col_match = re.search(r"by\s+['\"]?(\w+)['\"]?", user_input, re.I)
        n_match = re.search(r"\b(\d+)\b", user_input)
        col = col_match.group(1) if col_match and col_match.group(1) in self._df.columns else self._df.columns[0]
        n = int(n_match.group(1)) if n_match else 10
        asc = "smallest" in user_input.lower() or "asc" in user_input.lower()
        result = self._df.sort_values(col, ascending=asc).head(n)
        return f"Top {n} by {col}:\n{result.to_string()}"

    async def _generate_report(self, user_input: str) -> str:
        summary = self._summary()
        prompt = (
            f"You are a data analyst. Here is a dataset summary:\n\n{summary}\n\n"
            f"User wants: {user_input}\n\n"
            "Write a concise analysis report with key insights, patterns, and recommendations."
        )
        report = await self.llm.generate(prompt, task_type=TaskType.CODE_WRITE)
        # Save report
        out = Path("data/report.txt")
        out.write_text(report, encoding="utf-8")
        return f"Report generated:\n\n{report}\n\n(Saved to {out})"

    async def _llm_data_query(self, user_input: str) -> str:
        summary = self._summary()
        prompt = (
            f"Dataset: {self._df_name}\n{summary}\n\n"
            f"Question: {user_input}\n\n"
            "Answer concisely based on the data summary above."
        )
        return await self.llm.generate(prompt, task_type=TaskType.CODE_WRITE)
