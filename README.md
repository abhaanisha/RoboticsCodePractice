# Robotics Practice Code

## 1 Purpose

This repository contains implementations of the standard algorithms of mobile robotics,
written from first principles and in order of increasing difficulty. One algorithm is added
at a time.

Three rules are applied to every entry. The algorithm is written in plain NumPy, so that no
library performs the part of the work that is worth understanding. The algorithm is
separated from the code that draws it, so that the algorithm file can be read on its own
and is usually short. The behaviour is animated, because the difference between a filter
that is working and one that is failing is far easier to recognize in a moving picture than
in a column of numbers.

## 2 Contents

| Topic | Question it answers | State |
|---|---|---|
| [Localization](Localization/) | Where am I | days 1 to 3 of 8 complete |
| Mapping | What does the world look like | not started |
| Simultaneous localization and mapping | Both of the above at once | not started |
| Path planning | How do I get there | not started |
| Motion control | How do I follow the path once it is chosen | not started |

## 3 Organization

Each entry occupies one folder and holds the same four things.

```
Localization/day01_histogram_filter_1d/
    histogram_filter.py     the algorithm, with no drawing code in it
    run_demo.py             the simulated robot and the figures
    README.md               the description, with the animation included
    media/                  the generated animation and figures
```

## 4 Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later is required. NumPy, Matplotlib, and Pillow carry most of the work.
SciPy supplies the statistical distributions from day 2 onward, and filterpy is used only
to check the handwritten filters against an established implementation.
