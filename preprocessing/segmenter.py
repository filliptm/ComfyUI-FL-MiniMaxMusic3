import math


def _number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalize_sections(duration, sections):
    duration = max(0.0, float(duration))
    normalized = []
    for item in sections or []:
        if not isinstance(item, dict):
            continue
        start = min(duration, max(0.0, _number(item.get("start_seconds", item.get("start")))))
        end = min(duration, max(start, _number(item.get("end_seconds", item.get("end")), duration)))
        if end - start < 0.1:
            continue
        label = str(item.get("label") or item.get("section") or "section").strip()
        normalized.append({"start": start, "end": end, "label": label})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    result = []
    cursor = 0.0
    for item in normalized:
        start = max(cursor, item["start"])
        if start > cursor + 0.1:
            result.append({"start": cursor, "end": start, "label": "transition"})
        if item["end"] > start + 0.1:
            result.append({"start": start, "end": item["end"], "label": item["label"]})
            cursor = item["end"]
    if cursor < duration - 0.1:
        result.append({"start": cursor, "end": duration, "label": "section"})
    return result


def _split_interval(start, end, label, maximum):
    duration = end - start
    count = max(1, math.ceil(duration / maximum))
    size = duration / count
    return [
        {"start": start + index * size, "end": end if index == count - 1 else start + (index + 1) * size, "labels": [label]}
        for index in range(count)
    ]


def plan_segments(duration, sections, minimum=8.0, target=42.0, maximum=60.0, enabled=True):
    duration = float(duration)
    if duration <= 0:
        return []
    if not enabled or duration <= maximum:
        return [{"index": 0, "start": 0.0, "end": duration, "duration": duration, "labels": ["full track"]}]
    pieces = []
    normalized = normalize_sections(duration, sections)
    if not normalized:
        normalized = [{"start": 0.0, "end": duration, "label": "section"}]
    for item in normalized:
        pieces.extend(_split_interval(item["start"], item["end"], item["label"], maximum))

    segments = []
    current = None
    for piece in pieces:
        if current is None:
            current = dict(piece)
            continue
        combined = piece["end"] - current["start"]
        current_duration = current["end"] - current["start"]
        if combined <= maximum and (current_duration < minimum or combined <= target):
            current["end"] = piece["end"]
            current["labels"].extend(piece["labels"])
        else:
            segments.append(current)
            current = dict(piece)
    if current is not None:
        segments.append(current)
    if len(segments) > 1 and segments[-1]["end"] - segments[-1]["start"] < minimum:
        combined = segments[-1]["end"] - segments[-2]["start"]
        if combined <= maximum:
            tail = segments.pop()
            segments[-1]["end"] = tail["end"]
            segments[-1]["labels"].extend(tail["labels"])
    for index, segment in enumerate(segments):
        segment["index"] = index
        segment["duration"] = segment["end"] - segment["start"]
        segment["labels"] = list(dict.fromkeys(segment["labels"]))
    return segments
