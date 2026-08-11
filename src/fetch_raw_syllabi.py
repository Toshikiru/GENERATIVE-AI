"""
Checkpoint 1 - Data Collection
Downloads real MIT OpenCourseWare syllabus pages (CC-licensed, public) and saves
the extracted page text as-is (including navigation/footer noise) into data/raw/.
This intentionally keeps the text messy -- cleaning happens in the preprocessing
notebook (src/preprocess.py), not here.
"""
import re
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COURSES = [
    ("01_mathematics_for_computer_science",
     "https://ocw.mit.edu/courses/6-1200j-mathematics-for-computer-science-spring-2024/pages/syllabus/"),
    ("02_introduction_to_psychology",
     "https://ocw.mit.edu/courses/9-00sc-introduction-to-psychology-fall-2011/pages/syllabus/syllabus/"),
    ("03_evolutionary_psychology",
     "https://ocw.mit.edu/courses/9-250-evolutionary-psychology-spring-1999/pages/syllabus/"),
    ("04_matrix_methods_data_analysis",
     "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/pages/syllabus/"),
    ("05_machine_learning_6867",
     "https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/pages/syllabus/"),
    ("06_computational_thinking_data_science",
     "https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/pages/syllabus/"),
    ("07_mathematics_of_big_data",
     "https://ocw.mit.edu/courses/res-ll-005-mathematics-of-big-data-and-machine-learning-january-iap-2020/pages/syllabus/"),
    ("08_principles_of_microeconomics",
     "https://ocw.mit.edu/courses/14-01-principles-of-microeconomics-fall-2023/pages/syllabus/"),
    ("09_statistics_for_applications",
     "https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/pages/syllabus"),
    ("10_intro_cs_programming_python",
     "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus"),
    ("11_database_systems",
     "https://ocw.mit.edu/courses/6-830-database-systems-fall-2010/pages/syllabus/"),
]


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"&#\d+;", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def main():
    for slug, url in COURSES:
        print(f"Fetching {slug} ...")
        html = fetch_html(url)
        text = html_to_text(html)
        out_path = RAW_DIR / f"{slug}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: {url}\n")
            f.write("LICENSE: Creative Commons (MIT OpenCourseWare)\n\n")
            f.write(text)
        print(f"  saved {out_path.name} ({len(text.split())} words)")
        time.sleep(1)


if __name__ == "__main__":
    main()
