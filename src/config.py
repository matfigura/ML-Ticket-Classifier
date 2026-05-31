from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATABASE_DIR = ROOT_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "tickets.db"

REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

TICKETS_TABLE_NAME = "tickets"

REQUIRED_COLUMNS = [
    "subject",
    "body",
    "type",
    "queue",
    "priority",
    "language",
]

OPTIONAL_COLUMNS = [
    "answer",
    "business_type",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
    "tag_5",
    "tag_6",
    "tag_7",
    "tag_8",
    "tag_9",
    "tag_10",
]