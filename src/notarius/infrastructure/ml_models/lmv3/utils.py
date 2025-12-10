from notarius.schemas.data.structs import BBox


import random
from typing import Any

import numpy as np
from transformers import LayoutLMv3Processor

from .types import EncodingProtocol


def unnormalize_bbox(bbox: list[int], width: int, height: int) -> BBox:
    return (
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    )


def _resolve_prediction_tie(
    max_predictions: list[int],
    count_dict: dict[int, int],
    box_prediction_dict: dict[tuple[int, ...], list[int]],
    box_keys: list[tuple[int, ...]],
    current_index: int,
    id2label: dict[int, str],
) -> int:
    """
    Resolve ties when multiple predictions have the same count.

    Strategy:
    1. Try removing 'others' label and pick the new winner
    2. Look at neighboring boxes for context clues
    3. Fall back to random choice

    Args:
        max_predictions: List of prediction IDs tied for max count
        count_dict: Mapping of prediction ID to occurrence count
        box_prediction_dict: All predictions grouped by bounding box
        box_keys: Ordered list of bounding box keys
        current_index: Index of current box in box_keys
        id2label: Mapping from prediction IDs to label strings

    Returns:
        The chosen prediction ID
    """
    # Strategy 1: Remove 'others' label and recount
    others_id = next(
        (id_ for id_, label in id2label.items() if label == "others"), None
    )

    if others_id is not None and others_id in count_dict:
        count_dict_filtered = {k: v for k, v in count_dict.items() if k != others_id}

        if count_dict_filtered:
            new_max_count = max(count_dict_filtered.values())
            new_max_preds = [
                k for k, v in count_dict_filtered.items() if v == new_max_count
            ]

            if len(new_max_preds) == 1:
                return new_max_preds[0]

    # Strategy 2: Look at neighboring boxes for context
    neighbor_preds: list[int] = []

    # Check previous box
    if current_index > 0:
        prev_key = box_keys[current_index - 1]
        prev_preds = box_prediction_dict[prev_key]
        prev_counts = {k: prev_preds.count(k) for k in set(prev_preds)}
        prev_max_count = max(prev_counts.values())
        neighbor_preds.extend(
            [k for k, v in prev_counts.items() if v == prev_max_count]
        )

    # Check next box
    if current_index < len(box_keys) - 1:
        next_key = box_keys[current_index + 1]
        next_preds = box_prediction_dict[next_key]
        next_counts = {k: next_preds.count(k) for k in set(next_preds)}
        next_max_count = max(next_counts.values())
        neighbor_preds.extend(
            [k for k, v in next_counts.items() if v == next_max_count]
        )

    # Find predictions that appear in both current candidates and neighbors
    common_preds = list(set(max_predictions) & set(neighbor_preds))

    if common_preds:
        return random.choice(common_preds)

    # Strategy 3: Random fallback
    return random.choice(max_predictions)


def _is_valid_bbox(bbox: list[int]) -> bool:
    """Check if a bounding box is valid (not padding or special token)."""
    arr = np.asarray(bbox)
    if arr.shape != (4,):
        return False

    if isinstance(bbox, list) and bbox == [0, 0, 0, 0]:
        return False

    return True


def sliding(
    processor: LayoutLMv3Processor,
    token_boxes: list[list[list[int]]],
    predictions: list[list[int]],
    encoding: EncodingProtocol,
    width: int,
    height: int,
    id2label: dict[int, str],
    stride: int = 128,
) -> tuple[list[BBox], list[int], list[str]]:
    """
    Aggregate predictions from overlapping sliding windows.

    When a document exceeds the model's max sequence length (512 tokens),
    LayoutLMv3 processes it in overlapping windows. This function:
    1. Collects tokens and predictions from all windows
    2. Skips the overlapping stride region to avoid duplicates
    3. Aggregates predictions per bounding box using majority voting
    4. Resolves ties using neighbor context

    Args:
        processor: LayoutLMv3Processor for decoding tokens
        token_boxes: List of bbox lists for each window, shape [num_windows, seq_len, 4]
        predictions: List of prediction lists for each window, shape [num_windows, seq_len]
        encoding: The encoding dict containing input_ids
        width: Image width in pixels for unnormalizing boxes
        height: Image height in pixels for unnormalizing boxes
        id2label: Mapping from prediction IDs to label strings
        stride: Overlap stride value used during encoding (default: 128)

    Returns:
        boxes: List of unnormalized bounding boxes as (x1, y1, x2, y2) tuples
        preds: List of final prediction IDs (one per box)
        words: List of reconstructed words/tokens (one per box)
    """
    # Step 1: Aggregate tokens by bounding box (unnormalized coordinates)

    tokenizer = getattr(processor, "tokenizer")

    box_token_dict: dict[BBox, list[str]] = {}

    for window_idx in range(len(token_boxes)):
        # Skip overlapping tokens from second window onwards
        # stride + 1 because we want to skip the overlapping region entirely
        start_token_idx = 0 if window_idx == 0 else stride + 1

        for token_idx in range(start_token_idx, len(token_boxes[window_idx])):
            bbox_raw = token_boxes[window_idx][token_idx]

            if not _is_valid_bbox(bbox_raw):
                continue

            # Unnormalize to pixel coordinates for final output
            unnormal_box = unnormalize_bbox(bbox_raw, width, height)
            bbox_key = unnormal_box

            # Decode the token
            input_id = encoding["input_ids"][window_idx][
                token_idx
            ]  # pyright: ignore[reportIndexIssue]
            token = tokenizer.decode(input_id)

            if bbox_key not in box_token_dict:
                box_token_dict[bbox_key] = [token]
            else:
                box_token_dict[bbox_key].append(token)

    # Step 2: Aggregate predictions by bounding box (normalized coordinates as key)
    # We use normalized coords here to match across windows before unnormalizing
    box_prediction_dict: dict[tuple[int, ...], list[int]] = {}

    for window_idx in range(len(token_boxes)):
        for token_idx in range(len(token_boxes[window_idx])):
            bbox_raw = token_boxes[window_idx][token_idx]

            if not _is_valid_bbox(bbox_raw):
                continue

            # Use normalized bbox as key for prediction aggregation
            bbox_key = tuple(bbox_raw)
            prediction = predictions[window_idx][token_idx]

            if bbox_key not in box_prediction_dict:
                box_prediction_dict[bbox_key] = [prediction]
            else:
                box_prediction_dict[bbox_key].append(prediction)

    # Step 3: Resolve predictions using majority voting
    final_predictions: list[int] = []
    box_keys = list(box_prediction_dict.keys())

    for idx, (bbox_key, pred_list) in enumerate(box_prediction_dict.items()):
        # Count occurrences of each prediction
        count_dict: dict[int, int] = {}
        for pred in pred_list:
            count_dict[pred] = count_dict.get(pred, 0) + 1

        max_count = max(count_dict.values())
        max_predictions = [k for k, v in count_dict.items() if v == max_count]

        if len(max_predictions) == 1:
            # Clear winner - no tie
            final_predictions.append(max_predictions[0])
        else:
            # Tie - use resolution strategy
            resolved_pred = _resolve_prediction_tie(
                max_predictions=max_predictions,
                count_dict=count_dict,
                box_prediction_dict=box_prediction_dict,
                box_keys=box_keys,
                current_index=idx,
                id2label=id2label,
            )
            final_predictions.append(resolved_pred)

    # Step 4: Finalize outputs
    # Join subword tokens into complete words
    box_token_dict_joined = {
        bbox: "".join(tokens) for bbox, tokens in box_token_dict.items()
    }

    boxes = list(box_token_dict_joined.keys())
    words = [word.strip() for word in box_token_dict_joined.values()]

    return boxes, final_predictions, words


def repair_bio_labels(labels: list[str]) -> list[str]:
    """Repair BIO label sequences to fix invalid transitions.

    Args:
        labels: List of BIO-tagged labels

    Returns:
        List of repaired BIO labels
    """
    repaired: list[str] = []
    prev_type: str | None = None
    for i, tag in enumerate(labels):
        if tag.startswith("B-"):
            _, curr_type = tag.split("-", 1)
            if prev_type == curr_type:
                # Consecutive B- of same type: convert to I-
                repaired.append(f"I-{curr_type}")
            else:
                repaired.append(tag)
            prev_type = curr_type
        elif tag.startswith("I-"):
            _, curr_type = tag.split("-", 1)
            # If previous type is not the same, or None, treat as B-
            if prev_type != curr_type or prev_type is None:
                repaired.append(f"B-{curr_type}")
            else:
                repaired.append(tag)
            prev_type = curr_type
        else:
            repaired.append(tag)
            prev_type = None
    return repaired

def bio_to_spans(words: list[str], labels: list[str]) -> list[tuple[str, str]]:
    """Convert parallel words and BIO labels into entity spans.

    Args:
        words: List of words
        labels: List of BIO-tagged labels

    Returns:
        List of (entity_type, concatenated_text) tuples
    """
    spans: list[tuple[str, str]] = []
    buff: list[str] = []
    ent_type: str | None = None

    for w, tag in zip(words, labels):
        if tag == "O":
            if buff and ent_type is not None:
                spans.append((ent_type, " ".join(buff)))
                buff, ent_type = [], None
            continue

        if "-" not in tag:  # Handle cases where tag doesn't have BIO prefix
            continue

        prefix, t = tag.split("-", 1)
        if prefix == "B" or (ent_type and t != ent_type):
            if buff and ent_type is not None:
                spans.append((ent_type, " ".join(buff)))
            buff, ent_type = [w], t
        else:  # "I"
            if ent_type is None:
                # Treat as B- (start a new entity)
                buff, ent_type = [w], t
            else:
                buff.append(w)

    if buff and ent_type is not None:
        spans.append((ent_type, " ".join(buff)))
    return spans

def build_page_json(
    words: list[str],
    bboxes: list[BBox],
    labels: list[str]
) -> dict[str, Any]:
    """Build the target JSON structure from BIO-tagged annotations.

    Args:
        words: List of words
        bboxes: List of bounding boxes (currently unused but kept for API compatibility)
        labels: List of BIO-tagged labels

    Returns:
        Dictionary with page_number, deanery, and entries structure

    Expected output format:
    {
      "page_number": "<string | null>",
      "deanery": "<string | null>",
      "entries": [
        {
          "parish": "<string>",
          "dedication": "<string>",
          "building_material": "<string>"
        },
        ...
      ]
    }
    """
    # Sort words in reading order for better parsing
    # if bboxes:
    # words, labels = sort_by_layout(words, bboxes, labels)
    spans = bio_to_spans(words, labels)

    # Initialize result structure
    page_number: str | None = None
    entries: list[dict[str, str | None]] = []
    deanery: str | None = None

    # Running buffer for each parish block
    current: dict[str, str | None] = {
        "deanery": None,
        "parish": None,
        "dedication": None,
        "building_material": None,
    }

    for ent_type, text in spans:
        if ent_type == "page_number":
            page_number = text
        elif ent_type == "parish":
            # Start a new entry - flush previous if it exists
            if current["parish"]:
                entries.append(current)
                current = {
                    "deanery": deanery,
                    "parish": None,
                    "dedication": None,
                    "building_material": None,
                }
            current["parish"] = text
        elif ent_type == "deanery":
            deanery = text
            current["deanery"] = deanery
        elif ent_type == "dedication":
            current["dedication"] = text
        elif ent_type == "building_material":
            current["building_material"] = text

    # Flush last entry if it exists
    if current["parish"]:
        entries.append(current)

    return {
        "page_number": page_number,
        "entries": entries,
    }
