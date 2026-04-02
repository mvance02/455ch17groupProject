from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Supabase connection (used by ETL / training / inference)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
    "SUPABASE_ANON_KEY", ""
)
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "")

# Local artifacts directory (model files + metadata)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "late_delivery_model.sav"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

