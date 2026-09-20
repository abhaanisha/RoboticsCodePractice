# 1 The Histogram Filter

## 1.1 Introduction

A mobile robot must know where it is before it can decide where to go. In most cases it
cannot measure its position directly. Instead it carries sensors that report indirect and
noisy evidence about the surroundings, and it must combine that evidence with a map in
order to estimate its own position. This is the localization problem.

The histogram filter is the simplest complete solution to this problem. It divides the
world into a finite number of cells and stores one probability for each cell. This
collection of probabilities is called the belief. A belief in which every cell holds the
same value represents a robot that has no information about its position. A belief in
which one cell holds nearly all of the probability represents a robot that is almost
certain. The filter never produces a single answer. It produces a distribution, and the
answer is read off as the largest entry.

The demonstration in this folder places a robot in a corridor of twenty cells. Six of the
cells contain doors and the remainder are blank wall. The robot carries one sensor that
reports whether a door is present at the current cell, and this sensor is correct 92
percent of the time. At each step the robot is commanded to drive one cell to the right,
and the wheels obey the command 90 percent of the time. The robot is given the map of the
corridor but is not told where it started. Figure 1.1 shows the result.

![Histogram filter localizing in a corridor of twenty cells](media/histogram_filter_1d.gif)

**Figure 1.1**
A histogram filter localizing in a corridor of twenty cells. The upper strip is the
corridor, and the orange marker is the true position of the robot, which the robot itself
does not know. The lower bar chart is the belief. The belief begins flat and ends as a
single tall bar over the correct cell.

## 1.2 An everyday version of the problem

A person walking along an unfamiliar corridor in the dark solves the same problem with the
same method. Suppose the floor plan is known but the starting point is not. The only
available observation is whether the hand touches a door or a blank wall.

The first observation is a door. At that moment every blank stretch of the floor plan has
been ruled out, and the set of remaining candidates is the set of doors. The next step
carries the walker forward, and every candidate moves forward with them. The second
observation removes those candidates whose floor plan does not match. After a small number
of steps the sequence of doors and walls that has been passed matches only one place on the
plan, and the position is known.

A robot follows this procedure with one important addition. Its sensor can be wrong and its
wheels can slip, so it never removes a candidate completely. Rather than keeping a list of
possible cells, it keeps a number for every cell and raises or lowers those numbers. A
candidate that disagrees with an observation is made less likely but is never eliminated.
This is what allows the robot to recover when a sensor reading turns out to be false.

## 1.3 The two steps of the filter

Every operation the filter performs is one of two kinds, and the two alternate for as long
as the robot is running. Table 1.1 summarizes them.

**Table 1.1**
The two steps of a Bayes filter

| Step | Input | Effect on the belief | Effect on uncertainty |
|---|---|---|---|
| Sense | a sensor reading | sharpens it | uncertainty falls |
| Move | a motion command | blurs it | uncertainty rises |

This alternation is the central idea of the whole subject. Motion destroys information
because the outcome of a command is never exactly known. Observation creates information
because it favors some cells over others. Localization succeeds when observation creates
information at least as fast as motion destroys it. The Kalman filter and the particle
filter studied later play the same game. Only the storage of the belief changes.

### 1.3.1 The sense step

The sense step applies Bayes rule. Every cell is multiplied by the probability that it
would have produced the reading that was actually obtained, and the results are rescaled so
that they sum to one again.

```
bel(x)  =  p(z | x) . bel(x)  /  p(z)                                          (1.1)
```

In the demonstration the sensor is correct 92 percent of the time. When it reports a door,
every cell that contains a door on the map is multiplied by 0.92 and every cell that does
not is multiplied by 0.08. The division by p(z) is the rescaling, and it is the only part
of the operation that is not obvious. Bayes rule in this setting amounts to multiplying by
how well each cell explains the reading and then normalizing.

The value 0.08 deserves attention. A cell that disagrees with the sensor is reduced but is
never driven to zero, so the correct answer survives a false reading and can recover
afterward. A filter that used zero in place of 0.08 would claim that a disagreeing cell is
impossible. Once a cell reaches zero it can never rise again, because every later operation
multiplies it. A single sensor error would then remove the true position permanently. For
this reason a probability of exactly zero or exactly one is avoided in practice.

### 1.3.2 The move step

The move step applies the theorem of total probability. The belief is shifted by the
commanded distance and then spread over the outcomes the command might actually have
produced.

```
bel(x)  =  sum over x' of  p(x | x', u) . bel(x')                              (1.2)
```

In the demonstration the robot is commanded to drive one cell to the right. It arrives one
cell to the right 90 percent of the time, overshoots by one cell 5 percent of the time and
falls short by one cell 5 percent of the time. The filter therefore takes the whole bar
chart, slides it one cell to the right, and smears a small part of each bar onto the two
neighboring cells. This operation is a convolution. Its visible effect is that tall bars
become shorter and wider, which is correct, because a robot that has moved without looking
knows less than it did before.

## 1.4 A worked example

The arithmetic is short enough to follow once by hand. Consider a world of five cells in
which cells 1 and 2 contain doors.

```
cell    0      1      2      3      4
map    wall   DOOR   DOOR   wall   wall
```

With no information at all, the five cells are equally likely and each holds one fifth.

```
0.200  0.200  0.200  0.200  0.200
```

Suppose the sensor reports a door and is correct 90 percent of the time. Each cell
containing a door is multiplied by 0.9 and each cell without one is multiplied by 0.1.

```
0.020  0.180  0.180  0.020  0.020
```

These five numbers sum to 0.42 rather than to one, so each is divided by 0.42.

```
0.048  0.429  0.429  0.048  0.048
```

One observation has taken the robot from complete ignorance to a strong preference for two
cells. It cannot yet distinguish cell 1 from cell 2, because a single observation does not
contain enough information to do so. Now suppose the robot moves one cell to the right,
arriving correctly 80 percent of the time and missing by one cell in either direction 10
percent of the time.

```
0.048  0.086  0.390  0.390  0.086
```

The pair of peaks has moved one cell to the right and each peak has given up a little of
its height to its neighbors. The following command reproduces both results.

```bash
python -c "
import numpy as np, histogram_filter as hf
w = np.array([0,1,1,0,0], dtype=bool)
b = hf.uniform_belief(5)
print(np.round(hf.sense(b, w, True, 0.9, 0.1), 3))
print(np.round(hf.MotionModel().apply(hf.sense(b, w, True, 0.9, 0.1), 1), 3))
"
```

## 1.5 Implementation

The algorithm is contained in [histogram_filter.py](histogram_filter.py) and consists of
two functions. Equation 1.1 becomes the following.

```python
def sense(belief, world, z, prob_hit, prob_miss):
    likelihood = np.where(world == z, prob_hit, prob_miss)
    posterior  = belief * likelihood
    return posterior / posterior.sum()
```

Equation 1.2 becomes the following.

```python
def predict(belief, u, kernel, offsets):
    prior = np.zeros_like(belief)
    for weight, offset in zip(kernel, offsets):
        prior += weight * np.roll(belief, u + offset)
    return prior
```

Nothing further is required. The function `np.roll` performs the shift, and because the
corridor is treated as cyclic, probability that leaves one end of the array arrives at the
other. The loop over the kernel performs the smearing. The file
[run_demo.py](run_demo.py) contains only the simulated robot and the drawing code, and none
of the algorithm.

## 1.6 Behavior of the demonstration

The run shown in figure 1.1 was chosen deliberately so that two things go wrong during it.
A run in which nothing goes wrong shows that the filter works but does not show why it is
worth using.

At step 3 the sensor fails. The robot is standing at cell 6, which does contain a door, but
the detector reports a blank wall. The filter has no way of knowing this and accepts the
reading, and the probability assigned to the true cell falls to 1 percent. A method that
treated each reading as certain would be lost at this point and would stay lost.

At step 4 the wheels fail. The robot is commanded to drive one cell to the right and does
not move at all. Again the robot has no way of detecting this. The move step had already
spread the belief over the three places the robot might have reached, so the true position
remains covered.

At the same step the sensor works correctly and reports a door. One reliable observation is
enough to undo the earlier error, and the probability assigned to the true cell rises from
1 percent to 52 percent. By step 6 the run of three doors in a row has been recognized and
the probability reaches 79 percent. By step 10 it reaches 98.4 percent and the estimate is
correct.

The value of a Bayesian filter is not that it performs well when every component works. It
is that it degrades gradually when components fail and recovers without intervention
afterward.

![Belief at every step of the run, shown as a heat map](media/belief_evolution.png)

**Figure 1.2**
The same run drawn as a single image. Each row is one step and darker shading indicates
greater belief. The orange line marks the true position of the robot. The wide pale band of
the first four rows narrows onto the true position once the doors are reached.

## 1.7 Running the code

```bash
cd Localization/day01_histogram_filter_1d
python run_demo.py              # prints the trace and writes the figures
python run_demo.py --no-gif     # prints the trace only, and returns immediately
```

The program prints the following.

```
map:  ......DDD..D.DD.....   (D = door)
      01234567890123456789

step  0  true= 3  saw=wall  |------...--.-..-----|  p(truth)=0.069  H=3.99
         move  ->  blur                            H 3.99 -> 4.04  (+0.06)
step  1  true= 4  saw=wall  |------....-.....----|  p(truth)=0.086  H=3.75
         move  ->  blur                            H 3.75 -> 3.80  (+0.05)
step  2  true= 5  saw=wall  |++++++...........-++|  p(truth)=0.104  H=3.51
         move  ->  blur                            H 3.51 -> 3.54  (+0.03)
step  3  true= 6  saw=wall  |++++++............++|  p(truth)=0.010  H=3.28  <- SENSOR GLITCH
         move  ->  blur                            H 3.28 -> 3.31  (+0.03)   <- WHEEL SLIP
step  4  true= 6  saw=DOOR  |------#-.....-.....-|  p(truth)=0.523  H=2.67
         move  ->  blur                            H 2.67 -> 2.84  (+0.17)
step  5  true= 7  saw=DOOR  |......+#+.....-.....|  p(truth)=0.675  H=1.67
         move  ->  blur                            H 1.67 -> 1.88  (+0.20)
step  6  true= 8  saw=DOOR  |.......+#...........|  p(truth)=0.790  H=1.04
         move  ->  blur                            H 1.04 -> 1.32  (+0.27)
step  7  true= 9  saw=wall  |.........#-.........|  p(truth)=0.889  H=0.71
         move  ->  blur                            H 0.71 -> 1.07  (+0.36)
step  8  true=10  saw=wall  |.........-#.........|  p(truth)=0.894  H=0.69
         move  ->  blur                            H 0.69 -> 1.06  (+0.37)
step  9  true=11  saw=DOOR  |...........#........|  p(truth)=0.974  H=0.23
         move  ->  blur                            H 0.23 -> 0.72  (+0.49)
step 10  true=12  saw=wall  |............#.......|  p(truth)=0.984  H=0.16
```

The quantity H is the entropy of the belief measured in bits, and it answers the question
of how lost the robot is using a single number. A uniform belief over twenty cells has an
entropy of log base 2 of 20, which is 4.32 bits. The run ends at 0.16 bits.

This trace is the alternation of table 1.1 expressed in numbers. Every one of the ten move
steps raised the entropy and every one of the eleven sense steps lowered it, without a
single exception. The last few rows are the clearest. Each move is discarding almost half a
bit of information and each observation is recovering more than that.

## 1.8 Exercises

The following changes to `run_demo.py` each demonstrate something the preceding text only
asserts.

1. Set `P_SENSOR_CORRECT = 0.55`. The sensor is now barely better than a coin toss, it
   carries almost no information, and convergence becomes slow and unsteady.

2. Set `P_SENSOR_CORRECT = 1.0`, then alter one reading by hand so that it is wrong. The
   filter assigns the true cell a probability of exactly zero, and it can never recover.
   This is the failure described in section 1.3.1.

3. Set `DOOR_CELLS = [0, 5, 10, 15]`, which places a door every five cells all the way
   around the corridor. Run for thirty steps. The belief settles into four equal peaks of
   19.3 percent each and remains there for as long as the program runs. No amount of
   further travel improves the estimate, because every position has three exact
   look-alikes. This condition is called aliasing, and it is a property of the world rather
   than a defect in the filter. Note that having few landmarks is not by itself enough to
   cause it. The setting `DOOR_CELLS = [6, 12]` provides only two doors, yet the filter
   still reaches 86 percent, because the gaps on either side of those doors have different
   lengths and that asymmetry is itself usable information.

4. Remove the sense step. The belief spreads out and flattens and never recovers. This
   condition is called dead reckoning, and it explains why wheel odometry alone is never
   sufficient.

5. Extend the corridor to two dimensions. The belief now requires N times M numbers rather
   than N. Adding orientation requires N times M times 360. This growth is the subject of
   section 1.9.

## 1.9 Limitations and what follows

The histogram filter is the Bayes filter with no approximation of any kind. It stores a
probability for every possible state, so it can represent a belief of any shape. It can be
flat, or concentrated on one cell, or divided between several widely separated cells that
are equally plausible. For this reason it is the correct starting point and the standard
against which the later methods are compared.

Its weakness is cost. One number for every possible state is reasonable for twenty corridor
cells and impossible for a real robot, whose state is a position and an orientation. Such a
robot would require millions of cells, and every one of them would have to be updated
several times each second.

Each method that follows is a different answer to the question of how the idea can be kept
while the cost is reduced. Table 1.2 states the compromise each one makes.

**Table 1.2**
How each filter stores the belief

| Filter | Representation of the belief | Compromise |
|---|---|---|
| Histogram filter, day 1 | one probability for each cell | exact and unrestricted in shape, but the cost grows beyond use |
| Kalman filter, day 2 | a mean and a variance | very small and very fast, but restricted to a single peak |
| Particle filter, later | several thousand weighted samples | unrestricted in shape again, at a cost the designer chooses |

The restriction in the second row is more serious than it appears. A Kalman filter cannot
represent the statement that the robot is either at the lift or at the fire exit and
nowhere between them. Its belief has one peak, so it would answer that the robot is
somewhere in the middle, which is the one place the robot is known not to be. The four
equal peaks produced by exercise 3 cannot be represented by a Kalman filter at all. On day
2 the belief stops being twenty numbers and becomes two.

## 1.10 Notes on the demonstration

The corridor is cyclic, so that cell 19 is adjacent to cell 0. This is what allows the
motion step to be written with `np.roll` in a single line.

The random seed is set to 3749. It was selected on purpose, because it produces a run
containing both a sensor error and a wheel slip that still converges, and such a run
teaches more than a clean one. Any other seed produces a different and usually less
instructive sequence.

The colors of the figures were checked for readability under color blindness. The worst
pair separates by a color difference of 9.2 under simulated deuteranopia, against a
threshold of 8.
