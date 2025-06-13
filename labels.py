from typing import Dict

ANOMALY_TYPES: Dict[str, Dict[str, str | int]] = {
    "normal": {"score": 0,  "color": "#00FF00"},
    "depression": {"score": 4, "color": "#FF0000"},
    "cover": {"score": 2, "color": "#FFA500"},
    "cobble road/ traditional road": {"score": 1, "color": "#FFFF00"},
    "transverse grove": {"score": 1, "color": "#008000"},
    "gravel road": {"score": 4, "color": "#FAF2A1"},
    "cracked / irregular pavement and aspahlt": {"score": 2, "color": "#E06D06"},
    "bump": {"score": 1, "color": "#54F2F2"},
    "uneven/repaired asphalt road": {"score": 1, "color": "#A30B37"},
    "Damaged pavemant / asphalt road": {"score": 4, "color": "#2B15AA"},
}
UNKNOWN_ID = 99
UNKNOWN_NAME = "unknown"
UNKNOWN_COLOR = "#808080"

LABEL_IDS = {name: i + 1 for i, name in enumerate(ANOMALY_TYPES)}

