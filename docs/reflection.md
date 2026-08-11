# Checkpoint 1 Reflection

Our raw data is scraped directly from live MIT OpenCourseWare syllabus pages
rather than clean text dumps, so it came with the kind of mess a real
document ingestion pipeline actually has to deal with.

The biggest issue was site chrome mixed into the content. Every page repeats
its full navigation menu, a course-info sidebar, and a global footer/cookie
modal -- sometimes twice on the same page -- so a naive HTML-to-text
conversion left us with as much boilerplate ("Give Now", "About OCW",
Material Design icon labels like `theaters` and `auto_stories`) as actual
syllabus content. We fixed this by noticing a consistent structural pattern
across every page: the real syllabus prose always sits between the *last*
occurrence of the line "Syllabus" and the *last* occurrence of "Course Info".
That let us extract just the body text programmatically instead of writing
one-off rules per document.

Second, the text had HTML entities (`&rsquo;`, `&ldquo;`, `&amp;`) left over
from the scrape, which we decoded during cleaning so quotes and apostrophes
render correctly instead of as escape codes.

Third, the documents were inconsistent in structure because they span pages
from 1999 to 2024 -- older pages (e.g., Evolutionary Psychology, 1999) are
much shorter and lack a formal grading breakdown, while newer pages have
detailed percentage-based rubrics, calendars, and late-policy sections. One
page (Introduction to Psychology) turned out to be paginated across multiple
sub-URLs, so our single fetch only captured one section plus a stray "Next"
pagination link, which we had to filter out separately. This meant our
cleaning script couldn't assume a fixed template -- we designed it around
structural markers common to the platform rather than exact line positions,
so it generalizes across all 11 documents instead of breaking on the
outliers.
