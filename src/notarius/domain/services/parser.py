import os
import json
from pathlib import Path
from typing import Any, cast

from notarius.domain.entities.schematism import SchematismPage
from thefuzz import fuzz, process

from structlog import get_logger

from notarius.shared.constants import MAPPINGS_DIR

logger = get_logger(__name__)


class Parser:
    def __init__(
        self,
        mappings_dir: Path | None = None,
        building_material_mapping: dict[str, str] | None = None,
        dedication_mapping: dict[str, str] | None = None,
        deanery_mapping: dict[str, str] | None = None,
        fuzzy_threshold: int = 80,
    ):
        if mappings_dir is None:
            mappings_dir = MAPPINGS_DIR
        missing_env_vars = []

        if building_material_mapping is None:
            building_material_mapping_path = os.getenv("BUILDING_MATERIAL_MAPPINGS")
            if not building_material_mapping_path:
                missing_env_vars.append("BUILDING_MATERIAL_MAPPINGS")
            else:
                with open(
                    mappings_dir / Path(building_material_mapping_path), "r"
                ) as f:
                    building_material_mapping = cast(dict[str, str], json.load(f))

        if dedication_mapping is None:
            dedication_mapping_path = os.getenv("SAINTS_MAPPINGS")
            if not dedication_mapping_path:
                missing_env_vars.append("SAINTS_MAPPINGS")
            else:
                with open(mappings_dir / Path(dedication_mapping_path), "r") as f:
                    dedication_mapping = cast(dict[str, str], json.load(f))

        if deanery_mapping is None:
            deanery_mapping_path = os.getenv("DEANERY_MAPPINGS")
            if not deanery_mapping_path:
                missing_env_vars.append("DEANERY_MAPPINGS")
            else:
                with open(mappings_dir / Path(deanery_mapping_path), "r") as f:
                    deanery_mapping = cast(dict[str, str], json.load(f))

        if missing_env_vars:
            raise ValueError(f"Missing mapping env vars: {', '.join(missing_env_vars)}")

        self.mappings: dict[str, dict[str, str]] = {
            "dedication": cast(dict[str, str], dedication_mapping),
            "building_material": cast(dict[str, str], building_material_mapping),
            "deanery": cast(dict[str, str], deanery_mapping),
        }

        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_scorer = fuzz.ratio

    def fuzzy_match(self, text: str, keys: list[str]) -> tuple[str, int] | None:
        result: Any = process.extractOne(
            text, keys, scorer=self.fuzzy_scorer, score_cutoff=self.fuzzy_threshold
        )
        if not result:
            return None
        # Normalize possible 2- or 3-tuple from thefuzz into (choice, score)
        choice = result[0]
        score = int(result[1])
        return choice, score

    def parse(self, text: str, field_name: str) -> str | None:
        if field_name not in self.mappings:
            raise ValueError(f"Invalid field name: {field_name}")
        else:
            mappings: dict[str, str] = self.mappings[field_name]

        for key, value in mappings.items():
            if key == text:
                return value

        match = self.fuzzy_match(text, list(mappings.keys()))

        if match:
            found_key, score = match
            logger.debug(
                "Fuzzy match used",
                field=field_name,
                input=text,
                match=found_key,
                score=score,
            )
            return mappings[found_key]
        else:
            return None

    def parse_page(self, page_data: SchematismPage) -> SchematismPage:
        """Return a *new* parsed page dictionary, leaving the original untouched.

        A shallow ``dict.copy()`` is not enough because the ``entries`` list (and the
        dictionaries inside it) would still reference the same objects, causing
        in-place mutation of the original *raw* prediction. This resulted in the
        “raw_llm_response” column in the W&B table containing already-parsed
        results. We therefore perform a deep copy so every nested structure is
        duplicated before modification.
        """

        page_data_dump = page_data.model_dump()

        for entry in page_data_dump["entries"]:
            for field, value in entry.items():
                if field in self.mappings and value:
                    entry[field] = self.parse(value, field)

        return SchematismPage(**page_data_dump)
