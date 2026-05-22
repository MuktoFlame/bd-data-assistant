"""
setup_databases.py
------------------
Downloads the three HuggingFace datasets and converts them to SQLite databases.

Datasets:
  • Mahadih534/Institutional-Information-of-Bangladesh  → data/institutions.db
  • Mahadih534/all-bangladeshi-hospitals                → data/hospitals.db
  • Mahadih534/Bangladeshi-Restaurant-Data              → data/restaurants.db

Run once before starting the agent:
    python setup_databases.py
"""

import os
import sqlite3
import pandas as pd
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Column-name normaliser ─────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Snake-case column names and drop fully-empty columns."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    df = df.dropna(axis=1, how="all")
    return df


# ── Generic loader ─────────────────────────────────────────────────────────────

def hf_to_sqlite(hf_dataset_id: str, db_path: str, table_name: str,
                 column_map: dict | None = None):
    """
    Load a HuggingFace dataset → pandas DataFrame → SQLite table.

    Parameters
    ----------
    hf_dataset_id : str   HuggingFace dataset identifier
    db_path       : str   Path to the output .db file
    table_name    : str   SQLite table name
    column_map    : dict  Optional rename map applied after normalisation
    """
    print(f"\n{'='*60}")
    print(f"  Loading: {hf_dataset_id}")
    print(f"{'='*60}")

    ds = load_dataset(hf_dataset_id, trust_remote_code=True)

    # Use the first available split (usually 'train')
    split = list(ds.keys())[0]
    df: pd.DataFrame = ds[split].to_pandas()

    print(f"  Raw shape : {df.shape}")
    print(f"  Columns   : {list(df.columns)}")

    df = normalize_columns(df)

    if column_map:
        df = df.rename(columns=column_map)

    # Infer SQLite types
    dtype_map = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            dtype_map[col] = "INTEGER"
        elif pd.api.types.is_float_dtype(df[col]):
            dtype_map[col] = "REAL"
        else:
            dtype_map[col] = "TEXT"
            df[col] = df[col].astype(str).str.strip()

    con = sqlite3.connect(db_path)
    df.to_sql(table_name, con, if_exists="replace", index=False,
              dtype=dtype_map)
    con.close()

    print(f"  ✅ Saved {len(df):,} rows → {db_path}  (table: {table_name})")
    print(f"  Final columns: {list(df.columns)}")
    return df


# ── Dataset-specific loaders ──────────────────────────────────────────────────

def setup_institutions():
    column_map = {
        # adjust if raw column names differ; normalise handles casing/spaces
        "institution_name": "name",
        "institute_name":   "name",
        "institutionname":  "name",
        "district":         "location",
        "division":         "division",
        "type":             "type",
        "established":      "established_year",
    }
    return hf_to_sqlite(
        hf_dataset_id="Mahadih534/Institutional-Information-of-Bangladesh",
        db_path=os.path.join(DATA_DIR, "institutions.db"),
        table_name="institutions",
        column_map=column_map,
    )


def setup_hospitals():
    column_map = {
        "hospital_name":  "name",
        "hospitalname":   "name",
        "district":       "location",
        "division":       "division",
        "beds":           "bed_count",
        "total_beds":     "bed_count",
        "bed":            "bed_count",
        "type":           "hospital_type",
        "contact":        "contact",
        "phone":          "contact",
    }
    return hf_to_sqlite(
        hf_dataset_id="Mahadih534/all-bangladeshi-hospitals",
        db_path=os.path.join(DATA_DIR, "hospitals.db"),
        table_name="hospitals",
        column_map=column_map,
    )


def setup_restaurants():
    column_map = {
        "restaurant_name": "name",
        "restaurantname":  "name",
        "cuisine_type":    "cuisine",
        "cuisinetype":     "cuisine",
        "district":        "location",
        "area":            "area",
        "rating":          "rating",
        "price_range":     "price_range",
    }
    return hf_to_sqlite(
        hf_dataset_id="Mahadih534/Bangladeshi-Restaurant-Data",
        db_path=os.path.join(DATA_DIR, "restaurants.db"),
        table_name="restaurants",
        column_map=column_map,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_institutions()
    setup_hospitals()
    setup_restaurants()
    print("\n🎉 All databases created successfully inside ./data/")
