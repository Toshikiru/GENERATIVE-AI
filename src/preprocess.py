"""
Checkpoint 1 - Text Preprocessing

Cleans the raw MIT OpenCourseWare syllabus pages in data/raw/ and writes:
  - data/processed/<name>.txt        cleaned, normalized body text
  - data/processed/tokens/<name>.json word + sentence tokens
  - docs/preprocessing_report.md      before/after examples for the reflection

Raw pages were scraped straight from the live site (see src/fetch_raw_syllabi.py),
so they still contain site chrome: repeated navigation menus, icon-label
artifacts ("notes", "theaters", "assignment"...), a duplicated course-info
sidebar, and a global footer/cookie-modal block. That noise is what gets
stripped here -- the actual syllabus prose (grading policy, schedule,
prerequisites, readings) is what should survive.
"""
import html
import json
import re
from pathlib import Path

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    _NLTK_OK = True
except ImportError:
    _NLTK_OK = False

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"
REPORT_PATH = ROOT / "docs" / "preprocessing_report.md"

# Icon-label / UI artifacts that leak in as standalone lines and carry no
# semantic content (Material Design icon names used by the OCW site).
ICON_NOISE = {
    "notes", "theaters", "assignment", "auto_stories", "grading",
    "co_present", "assignment_turned_in", "menu", "search",
}

FOOTER_NOISE_PATTERNS = [
    r"^Over 2,500 courses & materials$",
    r"^Freely sharing knowledge.*$",
    r"^Learn more$",
    r"^©\s?\d{4}.*Massachusetts Institute of Technology$",
    r"^Accessibility$",
    r"^Creative Commons License$",
    r"^Terms and Conditions$",
    r"^Proud member of:$",
    r"^You are leaving MIT OpenCourseWare$",
    r"^close$",
    r"^Please be advised that external sites.*$",
    r"^including license rights.*$",
    r"^for any content on third party sites.*$",
    r"^of those sites and/or their content\.$",
    r"^Stay Here$",
    r"^Continue$",
    r"^Give Now$", r"^GIVE NOW$", r"^About OCW$", r"^Help\s?&?\s?Faqs$",
    r"^Contact Us$", r"^about ocw$", r"^help\s?&?\s?faqs$", r"^contact us$",
    r"^Download Course$", r"^Browse Course Material$",
    r"^Next$", r"^Previous$", r"^»$", r"^«$",
]
FOOTER_NOISE_RE = re.compile("|".join(FOOTER_NOISE_PATTERNS), re.IGNORECASE)


def split_header(raw_text: str):
    """Separate our own SOURCE/LICENSE metadata lines from the scraped body."""
    lines = raw_text.splitlines()
    meta = {}
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.startswith("SOURCE:"):
            meta["source"] = ln.split("SOURCE:", 1)[1].strip()
        elif ln.startswith("LICENSE:"):
            meta["license"] = ln.split("LICENSE:", 1)[1].strip()
        elif ln.strip() == "" and meta:
            body_start = i + 1
            break
    return meta, "\n".join(lines[body_start:])


def strip_site_chrome(body: str) -> str:
    """
    Every scraped page repeats the word 'Syllabus' three times (top nav item,
    'More Info' submenu item, then the real content heading) and 'Course Info'
    twice (a top sidebar summary, then a duplicate right after the real
    content ends). The real syllabus prose always sits between the LAST
    'Syllabus' line and the LAST 'Course Info' line -- everything before/after
    that window is navigation chrome or the repeated footer.
    """
    lines = [ln.strip() for ln in body.splitlines()]
    syllabus_idx = [i for i, ln in enumerate(lines) if ln == "Syllabus"]
    course_info_idx = [i for i, ln in enumerate(lines) if ln == "Course Info"]
    if not syllabus_idx or not course_info_idx:
        # Fallback: couldn't find the markers, keep everything (better to
        # over-include than silently drop real content).
        content_lines = lines
    else:
        start = syllabus_idx[-1] + 1
        end = course_info_idx[-1]
        content_lines = lines[start:end]
    return "\n".join(content_lines)


def clean_text(body: str) -> str:
    text = html.unescape(body)  # &rsquo; -> ' , &ldquo; -> " , &amp; -> & , etc.
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = []
    for ln in lines:
        if not ln:
            continue
        if ln in ICON_NOISE:
            continue
        if FOOTER_NOISE_RE.match(ln):
            continue
        cleaned.append(ln)
    text = "\n".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def tokenize(text: str):
    if _NLTK_OK:
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+%?", text.lower())
    return sentences, words


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    if _NLTK_OK:
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    report_sections = []
    stats_rows = []

    for raw_path in sorted(RAW_DIR.glob("*.txt")):
        raw_text = raw_path.read_text(encoding="utf-8")
        meta, body = split_header(raw_text)
        stripped = strip_site_chrome(body)
        cleaned = clean_text(stripped)
        sentences, words = tokenize(cleaned)

        out_path = PROCESSED_DIR / raw_path.name
        out_path.write_text(cleaned, encoding="utf-8")

        token_path = TOKENS_DIR / (raw_path.stem + ".json")
        token_path.write_text(
            json.dumps({"sentences": sentences, "words": words}, indent=2),
            encoding="utf-8",
        )

        stats_rows.append(
            f"| {raw_path.name} | {len(raw_text.split())} | {len(cleaned.split())} "
            f"| {len(sentences)} | {len(words)} |"
        )

        if len(report_sections) < 2:  # show full before/after for first 2 docs
            raw_preview = "\n".join(raw_text.splitlines()[:20])
            cleaned_preview = "\n".join(cleaned.splitlines()[:12])
            report_sections.append(
                f"## {raw_path.name}\n\n"
                f"**Before (raw, first 20 lines):**\n```\n{raw_preview}\n```\n\n"
                f"**After (cleaned, first 12 lines):**\n```\n{cleaned_preview}\n```\n"
            )

    report = ["# Checkpoint 1 - Text Preprocessing Report\n",
              "Raw pages were scraped from live MIT OpenCourseWare syllabus pages "
              "(see `src/fetch_raw_syllabi.py`). Cleaning removes site navigation, "
              "duplicated course-info sidebars, icon-label artifacts, and the "
              "global footer/cookie modal, then decodes HTML entities and "
              "normalizes whitespace. What remains is the actual syllabus "
              "prose: prerequisites, grading policy, schedule, and readings.\n",
              "## Before / After Examples\n"]
    report.extend(report_sections)
    report.append("\n## Word / Sentence / Token Counts Per Document\n")
    report.append("| File | Raw words | Cleaned words | Sentences | Word tokens |")
    report.append("|---|---|---|---|---|")
    report.extend(stats_rows)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"Processed {len(stats_rows)} documents.")
    print(f"Cleaned text -> {PROCESSED_DIR}")
    print(f"Tokens       -> {TOKENS_DIR}")
    print(f"Report       -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
