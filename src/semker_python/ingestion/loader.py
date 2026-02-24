from __future__ import annotations

import math
from typing import Any

import pandas as pd

from semker_python import DATA_DIR


def _is_valid(value: Any) -> bool:
  if value is None:
    return False
  if isinstance(value, float) and math.isnan(value):
    return False
  return bool(value)


def _build_description(row: dict[str, Any]) -> str:
  species = row.get("species")
  name = f"{row['name']} {species}" if _is_valid(species) else row["name"]

  diet = row.get("diet")
  intro = f"{name} was a {diet} dinosaur" if _is_valid(diet) else f"{name} was a dinosaur"

  period = row.get("period")
  if _is_valid(period):
    intro += f" that lived during the {period}"
  intro += "."

  facts = [intro]
  optional = [
    ("continent", "It was found in {}."),
    ("type", "It is classified as a {}."),
    ("length_m", "It measured approximately {} meters in length."),
    ("body_mass_kg", "Its estimated body mass was {} kg."),
  ]
  for key, template in optional:
    value = row.get(key)
    if _is_valid(value):
      facts.append(template.format(value))

  year = row.get("year_named")
  if _is_valid(year):
    facts.append(f"It was first named in {int(year)}.")

  return " ".join(facts)


def load_dinosaurs() -> list[dict[str, Any]]:
  path = DATA_DIR / "dinosaurs.csv"
  df = pd.read_csv(path)
  df.columns = df.columns.str.strip().str.lower()
  print(f"  Loaded {len(df)} dinosaurs from dinosaurs.csv")

  records = df.to_dict(orient="records")
  for rec in records:
    rec["description"] = _build_description(rec)
  return records
