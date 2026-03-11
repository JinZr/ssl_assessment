from __future__ import annotations

import re


KNOWN_DIMENSIONS = {
    "breathy_voice_continuous",
    "distorted_vowels",
    "harsh_voice",
    "imprecise_consonants",
    "inappropriate_silences",
    "intelligibility",
    "monoloudness",
    "monopitch",
    "naturalness",
    "pitch_level",
    "reduced_stress",
    "short_rushes_of_speech",
    "variable_rate",
}


CORRECTIONS = {
    "breathy voice (continuous)": "breathy_voice_continuous",
    "distorted vowels": "distorted_vowels",
    "harsh voice": "harsh_voice",
    "imprecise consonants": "imprecise_consonants",
    "inappropriate silences": "inappropriate_silences",
    "intelligbility": "intelligibility",
    "intelligibility": "intelligibility",
    "monoloudness": "monoloudness",
    "monopitch": "monopitch",
    "naturalness": "naturalness",
    "pitch level": "pitch_level",
    "reduced stress": "reduced_stress",
    "short rushes of speech": "short_rushes_of_speech",
    "variable rate": "variable_rate",
}


def normalize_dimension_name(raw_name: str) -> str:
    name = re.sub(r"\s+", " ", (raw_name or "").strip())
    return name


def slugify_dimension(raw_name: str) -> str:
    normalized = normalize_dimension_name(raw_name).lower()
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9() ]+", "", normalized)
    normalized = normalized.replace("(", " ").replace(")", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.replace(" ", "_")


def canonicalize_dimension(raw_name: str) -> tuple[str, bool]:
    normalized = normalize_dimension_name(raw_name)
    lookup_key = normalized.lower()
    if lookup_key in CORRECTIONS:
        canonical = CORRECTIONS[lookup_key]
        return canonical, True
    slug = slugify_dimension(raw_name)
    return slug, slug in KNOWN_DIMENSIONS

