# Checkpoint 1 - Text Preprocessing Report

Raw pages were scraped from live MIT OpenCourseWare syllabus pages (see `src/fetch_raw_syllabi.py`). Cleaning removes site navigation, duplicated course-info sidebars, icon-label artifacts, and the global footer/cookie modal, then decodes HTML entities and normalizes whitespace. What remains is the actual syllabus prose: prerequisites, grading policy, schedule, and readings.

## Before / After Examples

## 01_mathematics_for_computer_science.txt

**Before (raw, first 20 lines):**
```
SOURCE: https://ocw.mit.edu/courses/6-1200j-mathematics-for-computer-science-spring-2024/pages/syllabus/
LICENSE: Creative Commons (MIT OpenCourseWare)

Syllabus | Mathematics for Computer Science | Electrical Engineering and Computer Science | MIT OpenCourseWare
Browse Course Material
Syllabus
Readings
Lecture Videos
Lecture Notes
Warm Up Problems
Problem Sets
Course Info
Instructors
Prof. Erik Demaine
Dr. Zachary Abel
Dr. Brynmor Chapman
Departments
Electrical Engineering and Computer Science
Mathematics
As Taught In
```

**After (cleaned, first 12 lines):**
```
Course Meeting Times
Lectures: 2 sessions/week; 1.5 hours/session
Recitation: 2 sessions/week; 1 hour/session
Prerequisites
The only prerequisite is
18.01 Single Variable Calculus
. If you have already taken
18.200 Principles of Discrete Applied Mathematics
or
6.1220 Design and Analysis of Algorithms
(formerly 6.046), then you probably should not take 6.1200.
Course Description
```

## 02_introduction_to_psychology.txt

**Before (raw, first 20 lines):**
```
SOURCE: https://ocw.mit.edu/courses/9-00sc-introduction-to-psychology-fall-2011/pages/syllabus/syllabus/
LICENSE: Creative Commons (MIT OpenCourseWare)

Syllabus | Introduction to Psychology | Brain and Cognitive Sciences | MIT OpenCourseWare
Browse Course Material
Syllabus
Meet Professor John Gabrieli
Meet the TAs
Open Textbook
Instructor Insights
Structuring a Broad Survey Course
Crafting Lectures That Inspire and Inform
Bringing Demonstrations into the Classroom
Maintaining Currency in a Rapidly Evolving Field
Teaching Students to Evaluate Research
Introduction
Science & Research
Brain I: Structure and Functions
Brain II: Methods of Research
Discussion: Brain
```

**After (cleaned, first 12 lines):**
```
About this Course
This course is designed to introduce you to the scientific study of human nature. You will learn how psychologists ask questions from several different perspectives: questions about the relation of brain and behavior, about perception, about learning and thinking, about development, about social behavior and personality, and about psychopathology and psychotherapy. You will also learn about the methods psychologists use to find the answers to these questions and become acquainted with many of the important findings and theoretical approaches in the field of psychology. By the time it’s over, we hope that you will have learned to think critically about psychological evidence, and to evaluate its validity and its relevance to important issues in your life.
Meet Prof. John Gabrieli
Prerequisites and Preparation
This introductory college undergraduate course has no specific course prerequisites. It designed to be most useful to people with knowledge of the following subjects at the level typically taught in U.S. high schools:
Mathematics
Experimental data collection and visual representations of data in graphs and tables
Basic probability and statistics: e.g. average, median, distribution, variance
Natural sciences
Biology of the human nervous system
Physics of light and sound
Literacy
```


## Word / Sentence / Token Counts Per Document

| File | Raw words | Cleaned words | Sentences | Word tokens |
|---|---|---|---|---|
| 01_mathematics_for_computer_science.txt | 1131 | 854 | 40 | 1015 |
| 02_introduction_to_psychology.txt | 1354 | 824 | 51 | 943 |
| 03_evolutionary_psychology.txt | 372 | 146 | 6 | 168 |
| 04_matrix_methods_data_analysis.txt | 474 | 205 | 14 | 235 |
| 05_machine_learning_6867.txt | 1270 | 987 | 60 | 1148 |
| 06_computational_thinking_data_science.txt | 1170 | 889 | 46 | 979 |
| 07_mathematics_of_big_data.txt | 657 | 375 | 20 | 446 |
| 08_principles_of_microeconomics.txt | 905 | 642 | 36 | 723 |
| 09_statistics_for_applications.txt | 369 | 144 | 5 | 158 |
| 10_intro_cs_programming_python.txt | 1149 | 818 | 42 | 912 |
| 11_database_systems.txt | 1047 | 657 | 41 | 779 |