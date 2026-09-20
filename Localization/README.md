# Localization

## 1 Introduction

Localization is the problem of a robot determining its own position and orientation within
a known environment. It is one of the four basic competences of a mobile robot, alongside
perception, cognition, and motion control, and very little else can be attempted until it
is solved. A robot that does not know where it is cannot follow a path, cannot return to a
charging station, and cannot report where it found something.

The difficulty is that position is almost never measured directly. A satellite navigation
receiver reports a position that wanders by several metres and fails indoors. Wheel
encoders measure how far the wheels turned rather than how far the robot travelled, and the
difference accumulates without limit. A laser scanner reports the shape of the surroundings
rather than a position, and that shape must be compared against a map before it means
anything. Each of these sources is individually inadequate. Localization is the business of
combining them into an estimate that is better than any of them alone.

The methods in this folder are Bayesian filters. A Bayesian filter does not compute a
single position. It maintains a probability distribution over all the positions the robot
might occupy, called the belief, and revises that distribution as new information arrives.
The position reported to the rest of the robot is read off the belief, usually as its most
probable value. The advantage of carrying the whole distribution is that the robot also
knows how much to trust its own estimate, which a single position cannot express.

## 2 The estimation loop

Every method in this folder is the same loop of two steps. Only the way the belief is
stored changes from one method to the next.

```
        +------------------------------------------+
        |                                          |
        v                                          |
   +---------+                   +---------+       |
   | PREDICT |  ---------------> | UPDATE  |  -----+
   |  blur   |                   | sharpen |
   +---------+                   +---------+
  the robot moved              the robot observed
  uncertainty grows            uncertainty shrinks
```

**Figure 1**
The two step loop common to every Bayesian filter. The predict step applies the motion
model and the update step applies the sensor model.

The predict step accounts for what the robot did. The robot is commanded to move, the
command is not carried out exactly, and so the belief is shifted by the intended amount and
then spread out to cover the error. This step always increases uncertainty.

The update step accounts for what the robot observed. Positions that would have produced
the observation are made more probable and positions that would not are made less probable.
This step always reduces uncertainty.

Localization is therefore a contest between the two steps. Motion destroys information at a
rate set by the quality of the wheels, and observation supplies information at a rate set
by the quality of the sensors and the distinctiveness of the surroundings. The estimate
stays usable for as long as the second rate exceeds the first. This single fact explains
most of what happens in the demonstrations that follow, including the failures.

## 3 Contents

| Day | Method | Representation of the belief | State |
|---|---|---|---|
| 1 | [Histogram filter in one dimension](day01_histogram_filter_1d/) | one probability for each cell | complete |
| 2 | [Kalman filter in one dimension](day02_kalman_filter_1d/) | a mean and a variance | complete |
| 3 | Kalman filter, multivariate | a mean vector and a covariance matrix | planned |
| 4 | Extended Kalman filter | the same, linearized with Jacobians | planned |
| 5 | Unscented Kalman filter | the same, propagated through sigma points | planned |
| 6 | Particle filter, or Monte Carlo localization | several thousand weighted samples | planned |
| 7 | Adaptive Monte Carlo localization on a grid map | samples with a laser scan model | planned |
| 8 | Fusion of satellite navigation and inertial measurement | an extended Kalman filter over position, velocity, and sensor bias | planned |

The order is deliberate. Days 1 to 3 are exact and have closed form solutions. Days 4 and 5
are the two standard responses to the fact that real robots do not travel in straight lines
and real sensors do not report position. Days 6 and 7 abandon the assumption of a single
peak and recover the generality of day 1 at a cost the designer selects. Day 8 is the
arrangement that most closely resembles a working vehicle.

## 4 How the belief is represented

The choice of representation is the decision from which everything else follows. Table 1
states the compromise in each case.

**Table 1**
Properties of the three families of filter

| Family | Shapes of belief it can represent | Cost | Can express total uncertainty |
|---|---|---|---|
| Histogram filter | any shape | grows beyond use with added dimensions | yes |
| Kalman filter and its variants | a single peak only | very small and very fast | no |
| Particle filter | any shape | selected by the designer | yes |

The restriction in the second row matters more than its brevity suggests. Consider a robot
in a building that has identified its surroundings as a lift lobby, and suppose the
building contains two identical lift lobbies. The correct belief has two peaks. A histogram
filter or a particle filter can hold that belief and wait for further evidence. A Kalman
filter cannot. Its belief has one peak, so it reports a position midway between the two
lobbies, which is the one place the robot is known not to be.

This limitation is the reason the Kalman filter is not the end of the subject, despite
being faster and smaller than everything else here. It is also the reason day 1 begins with
the histogram filter rather than with the Kalman filter, even though the histogram filter
is never used on a real robot. The histogram filter shows what the correct answer looks
like, and the later methods are best understood as approximations to it.

## 5 Running the code

```bash
pip install -r ../requirements.txt
cd day01_histogram_filter_1d
python run_demo.py
```

Each day is self contained and consists of four parts. There is a file holding the
algorithm alone, with no drawing code in it. There is a file named `run_demo.py` that
simulates a robot and produces the figures. There is a folder of generated figures. There
is a description written to be read without reference to the others.
