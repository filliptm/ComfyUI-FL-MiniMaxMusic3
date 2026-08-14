import re


SECTION_TAGS = {
    "intro": "Intro",
    "verse": "Verse",
    "pre chorus": "Pre Chorus",
    "pre-chorus": "Pre Chorus",
    "chorus": "Chorus",
    "post chorus": "Post Chorus",
    "post-chorus": "Post Chorus",
    "hook": "Hook",
    "interlude": "Interlude",
    "bridge": "Bridge",
    "break": "Break",
    "breakdown": "Break",
    "build": "Build Up",
    "build up": "Build Up",
    "transition": "Transition",
    "solo": "Solo",
    "outro": "Outro",
    "instrumental": "Inst",
    "inst": "Inst",
}


def _items(value):
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("value") or item.get("name") or item.get("label")
            if item is not None and str(item).strip():
                result.append(str(item).strip())
        return result
    return []


def _sentence(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;")
    return f"{value}." if value else ""


def compile_caption(analysis, segment_caption=None):
    if segment_caption and str(segment_caption).strip():
        return _sentence(segment_caption)
    direct = analysis.get("caption") or analysis.get("description")
    if isinstance(direct, str) and direct.strip():
        return _sentence(direct)
    parts = []
    genres = _items(analysis.get("genres") or analysis.get("genre"))
    moods = _items(analysis.get("moods") or analysis.get("mood"))
    bpm = analysis.get("bpm") or analysis.get("tempo_bpm")
    meter = analysis.get("meter") or analysis.get("time_signature")
    opening = "A " + ", ".join(genres) + " track" if genres else "A music track"
    if bpm:
        opening += f" at approximately {bpm} BPM"
    if meter:
        opening += f" in {meter}"
    if moods:
        opening += " with a " + ", ".join(moods) + " mood"
    parts.append(_sentence(opening))
    instruments = _items(analysis.get("instruments") or analysis.get("instrumentation"))
    if instruments:
        parts.append(_sentence("The arrangement features " + ", ".join(instruments)))
    vocals = analysis.get("vocals")
    if isinstance(vocals, dict) and vocals.get("present") is not False:
        description = ", ".join(_items([vocals.get("language"), vocals.get("character"), vocals.get("delivery")]))
        if description:
            parts.append(_sentence("The vocals use " + description))
    for key in ("harmony", "melody", "production", "arrangement"):
        values = _items(analysis.get(key))
        if values:
            parts.append(_sentence(f"{key.capitalize()}: " + ", ".join(values)))
    return " ".join(part for part in parts if part)


def section_tag(value):
    value = re.sub(r"[^a-z -]", "", str(value or "").lower()).strip()
    if value in SECTION_TAGS:
        return SECTION_TAGS[value]
    for key, tag in SECTION_TAGS.items():
        if key in value:
            return tag
    return "Verse"


def normalize_lyrics(lyrics, analysis):
    if not isinstance(lyrics, dict):
        return {"instrumental": True, "language": "none", "sections": []}
    result = dict(lyrics)
    vocals = analysis.get("vocals") if isinstance(analysis, dict) else None
    language = str(result.get("language") or "").strip().lower()
    if isinstance(vocals, dict) and vocals.get("present") is False:
        result.update({"instrumental": True, "language": "none", "sections": []})
    elif language in {"none", "instrumental", "no vocals"} and not result.get("sections"):
        result["instrumental"] = True
    return result


def _lyric_line(value):
    if isinstance(value, dict):
        value = value.get("text") or value.get("lyrics") or value.get("line") or ""
    return str(value).strip()


def compile_lyrics(lyrics, segment_start=0.0, segment_end=None):
    if not isinstance(lyrics, dict) or lyrics.get("instrumental") is True:
        return "[Inst]"
    blocks = []
    for section in lyrics.get("sections") or []:
        if not isinstance(section, dict):
            continue
        start = float(section.get("start_seconds", section.get("start", 0.0)) or 0.0)
        end_value = section.get("end_seconds", section.get("end"))
        end = float(end_value) if end_value is not None else start
        if segment_end is not None and (end < segment_start or start > segment_end):
            continue
        lines = section.get("lines") or section.get("text") or []
        if isinstance(lines, str):
            lines = lines.splitlines()
        lines = [_lyric_line(line) for line in lines]
        lines = [line for line in lines if line]
        if lines:
            blocks.append("[" + section_tag(section.get("label") or section.get("section")) + "]\n" + "\n".join(lines))
    if blocks:
        return "\n\n".join(blocks)
    text = str(lyrics.get("text") or lyrics.get("transcription") or "").strip()
    return text or "[Inst]"
