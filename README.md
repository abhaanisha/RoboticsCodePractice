# Robotics Practice Code

Working through robotics algorithms from scratch, one a day, basic to advanced.

The rule for every entry: **implement the algorithm in plain NumPy with no black-box
library doing the interesting part**, then explain it well enough that someone who has
never seen it can follow, then animate it so the behaviour is obvious at a glance.

## Topics

| Topic | What it answers | Progress |
|---|---|---|
| [**Localization**](Localization/) | *Where am I?* | Day 1 of 8 |
| Mapping | *What does the world look like?* | not started |
| SLAM | *Both at once, from scratch* | not started |
| Path planning | *How do I get there?* | not started |
| Control | *How do I actually follow that path?* | not started |

## Layout

Each day is a self-contained folder:

```
Localization/day01_histogram_filter_1d/
├── histogram_filter.py    # the algorithm, importable, no plotting in it
├── run_demo.py            # simulation + animation
├── README.md              # the explanation, with the GIF embedded
└── media/                 # generated GIF and figures
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. Only NumPy, Matplotlib and Pillow — nothing robotics-specific, on purpose.
