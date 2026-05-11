import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RESUME_DATASET_PATH = BASE_DIR / "data" / "resume_dataset.csv"
JOB_DESCRIPTION_DATASET_PATH = BASE_DIR / "data" / "job_description_dataset.csv"
SKILL_ALIASES_PATH = BASE_DIR / "data" / "skill_aliases.json"
RESULTS_PATH = BASE_DIR / "output" / "results.txt"


with open(SKILL_ALIASES_PATH, "r") as file:
    SKILL_ALIASES = json.load(file)
