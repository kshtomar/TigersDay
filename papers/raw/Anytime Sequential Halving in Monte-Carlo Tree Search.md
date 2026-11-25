## Page 1

Anytime Sequential Halving in
Monte-Carlo Tree Search

arXiv:2411.07171v1 [cs.LG] 11 Nov 2024

Dominic Sagers, Mark H.M. Winands, and Dennis J.N.J. Soemers
Department of Advanced Computing Sciences, Maastricht University,
Paul-Henri Spaaklaan 1, 6229 EN Maastricht, the Netherlands
dominicsagers1@gmail.com,
{m.winands,dennis.soemers}@maastrichtuniversity.nl

Abstract. Monte-Carlo Tree Search (MCTS) typically uses multi-armed
bandit (MAB) strategies designed to minimize cumulative regret, such
as UCB1, as its selection strategy. However, in the root node of the
search tree, it is more sensible to minimize simple regret. Previous work
has proposed using Sequential Halving as selection strategy in the root
node, as, in theory, it performs better with respect to simple regret.
However, Sequential Halving requires a budget of iterations to be predetermined, which is often impractical. This paper proposes an anytime
version of the algorithm, which can be halted at any arbitrary time and
still return a satisfactory result, while being designed such that it approximates the behavior of Sequential Halving. Empirical results in synthetic
MAB problems and ten different board games demonstrate that the algorithm’s performance is competitive with Sequential Halving and UCB1
(and their analogues in MCTS).
Keywords: Sequential Halving · Anytime · Monte-Carlo Tree Search.

1

Introduction

Monte-Carlo Tree Search (MCTS) [13,7] is a search algorithm used for different
sequential decision-making problems. It has been thoroughly studied within the
context of game playing agents, but also seen use in other planning, optimization,
and control problems [4]. The algorithm consists of four strategic steps, each of
which can be implemented using a variety of different strategies [4,19]. Strategies
for the selection step tend to use Multi-Armed Bandit (MAB) algorithms, which
balance exploration (sampling actions that are less explored) with exploitation
(more deeply searching actions that appear more promising).
The most commonly used selection strategy is UCB1 [2], which focuses on
minimizing cumulative regret. Sequential Halving (SH) [12], which focuses on
simple regret, may be argued to be a more suitable choice in MCTS [16,6]. Integrations of SH into MCTS have been described in research on partially observable
games [15], variants of MCTS that take additional guidance from scores learned
through online or offline learning [9], and the state-of-the-art Gumbel AlphaZero
and MuZero [8], which also integrate deep neural networks.

## Page 2

2

D. Sagers et al.

Running through the four strategic steps of MCTS once is referred to as an
iteration, and MCTS typically runs multiple iterations, after which it returns a
final decision (e.g., move to play or action to take). It is common to use either a
time budget, where MCTS keeps running iterations until it runs out of time, or an
iteration budget, where it runs a predetermined number of iterations. SH requires
the number of iterations that can be executed to be known in advance, which
means that MCTS using SH as selection strategy does not have the anytime
property. The algorithm cannot be terminated at any arbitrary point in time
and be expected to have the quality of its final decision smoothly increasing as
processing time increases (barring pathological cases [14]). When dealing with
known games (for which the average number of iterations per unit of time could
be measured) and a fixed per-move time budgets, a reasonable approximation
of a fixed iteration budget may be calculated and used. However, when dealing
with particularly large and varied sets of games [18], automatically generated
games that must be played quickly in the context of an evolutionary search [20],
or agents that automatically manage their time in an intelligent manner [11,3],
the lack of anytime property can be more problematic.
This paper proposes anytime SH : a MAB algorithm with the anytime property, which can be used as selection strategy in (the root node of) MCTS. Its
design was heavily inspired by the original SH, with anytime SH essentially being
the original SH turned inside out. While we leave formal analyses of bounds on
regret for future work, empirical results in synthetic MAB problems as well as a
diverse set of ten board games demonstrate that anytime SH performs competitively with UCB1 (or UCT in games) as well as SH (only used in root node in
games) in practice, whilst—in contrast to SH—retaining the anytime property.

2

Background

The Multi-armed Bandit (MAB) problem assumes an environment in which an
agent repeatedly chooses one from a set of K arms to pull at time steps t =
1, 2, . . . . Each of the arms is associated with a stationary, unknown probability
distribution, and whenever the agent pulls an arm with index it at time t, it
receives a reward sampled from the corresponding distribution with (unknown)
mean µit . The most common measure of performance in MAB problems is the
PT
cumulative regret R(T ) = t=1 (µ∗ − µit ), which accumulates the regret of not
having always picked the optimal arm (with the highest mean µ∗ ) over a sequence
of T time steps. The simple regret r(T ) = µ∗ − µiT +1 , which solely measures
the regret of a single choice made after a period of T exploration steps, is an
alternative measure of performance [1,5,10]. Simple regret is a more appropriate
measure of performance when it is only an ultimate single decision that matters,
with all prior decisions simply serving as a learning or training phase.
UCB1 [2] is a common MAB algorithm, used as the selection strategy in
the canonical UCT [13] variant of MCTS. It is designed primarily to minimize
cumulative regret. At any give time step t, UCB1 selects an arm jt to pull

## Page 3

Anytime Sequential Halving in Monte-Carlo Tree Search

3

Algorithm 1 Sequential Halving
Input total budget T , K arms
1: S1 ← {1, ..., K} , B ← ⌈log2 K⌉
2: for k = 1 . . . B do
3:
Sample each arm i ∈ Sk , nk = |SkT|×B times
4:
Update the average reward for each arm relative to the found rewards
5:
Sk+1 ← the ⌈|Sk | /2⌉ best arms from Sk
6: end for
7: return the sole remaining arm of SB+1

r
using jt = argmax1≤j≤K Xj + C

P
ln( K
k=1 nk )
, where Xj is the average reward
nj

collected from previous pulls of arm j so far, C is a tunable hyperparameter,
which affects the balance between exploration and exploitation, and ni is the
number of times an arm i has been pulled so far.
Sequential Halving (SH) [12] is a different MAB algorithm, designed to optimize for simple regret rather than cumulative regret. It assumes prior knowledge
of the budget of arm pulls (or time steps) T , as well as the fixed number of arms
K. It calculates how many times the number of arms can be halved before narrowing down to a single arm as B = ⌈log2 K⌉, and operates in B separate phases.
The total budget T gets spread out evenly over these B phases. Within every
phase, the algorithm distributes the number of arm pulls uniformly over all remaining arms (in the first phase, this is the full set of all K arms). After every
phase, the worst-performing half of all arms get discarded. Algorithm 1 provides
pseudocode for SH, and Fig. 1 gives a visual depiction of the algorithm.
Outside of affecting rewards that are collected and information that is gathered by an agent, decisions made in earlier time steps have no effect on later
time steps in MAB problems. In contrast, games are examples of sequential
decision-making problems, where actions taken (moves played) in early states
affect which states the agent transitions into. Monte-Carlo Tree Search (MCTS)
[13,7] can handle this sequential nature by gradually building up and traversing through a search tree, with nodes representing different states, connected
by the actions that lead to transitions between the states. When traversing the
tree, MCTS views the problem of selecting a branch from each node it reaches
as a MAB problem. After running many iterations, it will use the information
it has collected from simulations to make a final decision as to which move to
play. Because only this final move selection truly matters—any choices made for
tree traversal during the search itself only result in fictitious rewards—it can be
argued that algorithms that optimize for simple regret (such as SH) are more
suitable than ones that optimize for cumulative regret (such as UCB1) in the
root node [10,16,6]. Outside of the root node, the argument in favor of simple
regret minimization still holds due to the minimax structure of zero-sum games.
However, simultaneously there is an argument to be made in favor of cumulative
regret minimization, as this leads to a more focused best-first search that wastes
less time on seemingly uninteresting parts of the search space—in particular for

## Page 4

4

D. Sagers et al.

Fig. 1. Example of one complete run of Sequential Halving with 15,000 iterations. Only
the root node and its immediate children are depicted. Circles in the same vertical bar
depict the same node, in different rounds of the algorithm. The best half of nodes in
each round are colored. Numbers inside the nodes denote the total number of iterations
allocated to that node up to and including the corresponding round.

small search budgets. Hybrid MCTS (H-MCTS) [16] therefore introduces a parameter that dynamically adjusts from using SH in and close to the root, to using
UCB1 further down the tree, depending on the available budget. As a simplified
version of this, we use H-MCTS with SH only in the root node, and UCB1 in
all other nodes, as a baseline in our experiments.

3

Time-Based and Anytime Sequential Halving

As an initial step towards an anytime variant of SH, we consider Time-based SH.
Standard SH requires a budget of iterations to be known in advance, because it
spreads this number of iterations evenly over all of its phases. If every iteration
takes approximately the same amount of time, a simple but effective algorithm
with a time budget could simply divide the time budget over the phases instead
of the iteration budget. This is a trivial variant of SH, but not one that we have
seen in prior literature. It does not have the anytime property, as it still expects
the time budget to be predetermined.
We propose Anytime SH as an anytime variant of the algorithm, which works
as follows. It starts operating like the standard SH, albeit with an assumed budget of iterations that is only the bare minimum required for a valid, “complete”
run of SH. In the first phase it allocates exactly one iteration to each arm. Then
it halves the arms, allocating two additional iterations to each remaining arm,
and so on, until only a single arm remains. We refer to this as one pass of the

## Page 5

Anytime Sequential Halving in Monte-Carlo Tree Search

5

Fig. 2. Visualization of two passes made by Anytime Sequential Halving. Only the root
node and its immediate children are depicted. Circles in the same vertical bar depict
the same node, in different rounds of the algorithm. The best half of nodes in each
round of each pass are colored. Numbers inside the nodes denote the total number
of iterations allocated to that node up to and including the corresponding pass and
round.

Algorithm 2 Anytime Sequential Halving
Input K arms
1: S ← {1, ..., K}
2: T ← S
3: N ← 1
4: repeat
5:
Every arm in T is sampled N times.
6:
Update the average reward for each arm.
7:
T ← the ⌈|T |/2⌉ best arms from T
8:
N ←N ×2
9:
if |T | = 1 then
10:
T ←S
11:
N ←1
12:
end if
13: until stopping condition
14: return The highest-reward arm from S

algorithm. After the first pass, if there is still time remaining, the algorithm resets back to considering the full set of arms again, and repeats the same process.
If the ordering of arms’ empirically observed mean rewards never changes (for
example, if every arm i deterministically gives the same reward µi on each pull,
with no randomness), this algorithm distributes its pulls over the arms in the
same way as SH would have done in hindsight (if the iteration budget had been
known in advance). If the ordering of arms based on their empirically observed
mean rewards can vary over a run of the algorithm, it is possible to end up with
different allocations of iterations than in the original SH. Algorithm 2 provides
pseudocode for the algorithm, and Fig. 2 visualises two passes of the algorithm.

## Page 6

6

D. Sagers et al.

Table 1. Mappings between time and iteration budgets for 10-armed MAB problems.
Milliseconds 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
Iterations
18500 37000 55500 73000 93000 112500 131000 150500 167500 186500

4

Experiments

This section describes the experiments used to evaluate the novel Anytime SH
algorithm.1 The algorithm is compared to appropriate baselines in synthetic
MAB problems (Subsection 4.1), and in a set of ten different board games (Subsection 4.2). In the board games, the algorithm is not used standalone, but as
selection strategy in the root node for MCTS.
4.1

Synthetic Multi-Armed Bandit Problems

For an evaluation of the raw MAB algorithms, without the added complexity
of sequential decision making and combining with tree search in games, we construct a set of 100 synthetic MAB problems as follows. For every MAB problem,
we sample ten different means µ1 , . . . , µ10 for K = 10 arms from a normal distribution with mean 0 and unit variance. Within each problem, when an arm i
is pulled, a reward is drawn from another normal distribution with mean µi and
unit variance. The following algorithms are compared on this suite of problems:
√ −1
– UCB1: UCB1 [2] using an exploration constant C = 2 , as described in
Section 2.
– Base SH: standard Sequential Halving [12], as described in Section 2.
– Time SH: the trivial Time-based variant of SH, as described in Section 3.
– Anytime SH: the novel algorithm as described in Section 3.
Every algorithm was run on each MAB problem for up to 5000 milliseconds
(or 186,500 iterations in the case of Base SH) using Python 3, on an i7-10700k
Intel CPU. Table 1 lists mappings between time and iteration budgets that were
empirically determined, allowing for a fair comparison between algorithms that
support only certain types of budgets. For each time budget listed in Table 1
(or iteration budget for Base SH), the simple regret incurred by each algorithm
running with that budget is recorded for each of the 100 problems, and averaged over those 100 problems. Fig. 3 depicts 95% confidence intervals for these
averaged simple regret measures.
4.2

Board Games

The second experiment evaluates the performance of Anytime SH, when used as
selection strategy in the root node of MCTS, in a set of ten different board games
(listed in Table 2). Three different agents are considered in this experiment:
1

Source code: https://github.com/dominic-sagers/Anytime-Sequential-Halving

## Page 7

Anytime Sequential Halving in Monte-Carlo Tree Search

7

Fig. 3. Simple regret, averaged over 100 MAB problems, for four MAB algorithms.

√
– UCT: a standard implementation of MCTS, using UCB1 with C = 2 as
selection strategy, adding one new node to the tree per iteration, and running
uniformly random play-outs.
– H-MCTS: a simplified version of Hybrid MCTS [16], using SH as selection
strategy in the root node, but otherwise running as described above for UCT.
– Anytime SH: uses anytime SH as selection strategy for the root node, but
otherwise runs as described above for UCT.
Each of these agents played 150 times (75 as first and 75 as second player) against
each of the other two agents, across ten different board games. For each matchup,
seven different MCTS iteration budgets per move were used, ranging from 1000
to 50,000. We used iteration budgets rather than time budgets, because H-MCTS
can only function with the former, whereas the other two algorithms can handle
either type of budget. Table 2 lists 95% Agresti-Coull confidence intervals for
win percentages, with any draws being counted as half-wins. All listed games
were run using their default settings in the Ludii general game playing system
[17], except for Atari Go and Clobber being played on boards of size 9×9 and
7×7, respectively. In addition to the table with results per game, Fig. 4 depicts
the win percentages when averaging over all ten games.

5

Discussion

The results for the MAB problems in Fig. 3 show that Anytime SH performs on
par with the naive time-based SH and the standard SH in terms of simple regret.
In this setting, adjusting the algorithm to introduce the anytime property does
not harm performance. UCB1 appears to have better performance for some budgets, which is somewhat surprising as this algorithm is not necessarily designed

## Page 8

8

D. Sagers et al.

Table 2. 95% Agresti-Coull confidence intervals for win percentages of row agents
against UCT (top half) or H-MCTS (bottom half) in ten different board games.
MCTS iterations per move:

1000

5000

10,000

20,000

30,000

40,000

50,000

Win percentages against UCT (95% Agresti-Coull confidence intervals)
Amazons
H-MCTS
Anytime SH
Atari Go (9×9)
H-MCTS
Anytime SH
Breakthrough
H-MCTS
Anytime SH
Clobber (7×7)
H-MCTS
Anytime SH
Gomoku
H-MCTS
Anytime SH
Hex
H-MCTS
Anytime SH
Pentalath
H-MCTS
Anytime SH
Reversi
H-MCTS
Anytime SH
Tablut
H-MCTS
Anytime SH
Yavalath
H-MCTS
Anytime SH

46.10 ± 7.88 56.50 ± 7.83 55.85 ± 7.85 51.95 ± 7.90 51.30 ± 7.90 46.75 ± 7.88 55.85 ± 7.85
51.95 ± 7.90 52.60 ± 7.89 50.00 ± 7.90 47.40 ± 7.89 46.10 ± 7.88 48.70 ± 7.90 48.05 ± 7.90
64.30 ± 7.57 56.50 ± 7.83 61.70 ± 7.68 64.30 ± 7.57 65.60 ± 7.51 59.10 ± 7.77 52.60 ± 7.89
44.80 ± 7.86 46.75 ± 7.88 61.05 ± 7.71 53.25 ± 7.88 51.30 ± 7.90 49.35 ± 7.90 45.45 ± 7.87
47.40 ± 7.89 51.30 ± 7.90 51.30 ± 7.90 52.60 ± 7.89 58.45 ± 7.79 48.70 ± 7.90 55.20 ± 7.86
51.95 ± 7.90 53.90 ± 7.88 57.80 ± 7.80 61.05 ± 7.71 55.20 ± 7.86 52.60 ± 7.89 52.60 ± 7.89
54.55 ± 7.87 52.60 ± 7.89 46.10 ± 7.88 56.50 ± 7.83 51.95 ± 7.90 49.35 ± 7.90 48.70 ± 7.90
44.15 ± 7.85 51.95 ± 7.90 49.35 ± 7.90 51.95 ± 7.90 47.40 ± 7.89 51.30 ± 7.90 45.45 ± 7.87
75.35 ± 6.81 78.60 ± 6.48 80.55 ± 6.25 76.65 ± 6.69 79.25 ± 6.41 72.75 ± 7.04 74.05 ± 6.93
55.20 ± 7.86 55.20 ± 7.86 50.00 ± 7.90 45.45 ± 7.87 45.45 ± 7.87 47.40 ± 7.89 59.10 ± 7.77
79.25 ± 6.41 63.00 ± 7.63 57.80 ± 7.80 53.25 ± 7.88 48.70 ± 7.90 59.10 ± 7.77 54.55 ± 7.87
32.45 ± 7.40 40.90 ± 7.77 52.60 ± 7.89 46.10 ± 7.88 53.90 ± 7.88 54.55 ± 7.87 48.05 ± 7.90
62.35 ± 7.66 59.75 ± 7.75 56.50 ± 7.83 50.00 ± 7.90 48.70 ± 7.90 53.90 ± 7.88 52.60 ± 7.89
52.60 ± 7.89 44.80 ± 7.86 51.30 ± 7.90 51.30 ± 7.90 53.25 ± 7.88 57.80 ± 7.80 47.40 ± 7.89
50.98 ± 7.90 51.95 ± 7.90 51.30 ± 7.90 53.25 ± 7.88 46.10 ± 7.88 45.77 ± 7.87 49.35 ± 7.90
45.45 ± 7.87 53.25 ± 7.88 52.28 ± 7.89 55.85 ± 7.85 51.63 ± 7.90 51.95 ± 7.90 49.35 ± 7.90
57.15 ± 7.82 50.65 ± 7.90 50.00 ± 7.90 50.00 ± 7.90 49.35 ± 7.90 50.00 ± 7.90 50.65 ± 7.90
48.05 ± 7.90 50.00 ± 7.90 50.65 ± 7.90 50.00 ± 7.90 51.95 ± 7.90 50.00 ± 7.90 49.35 ± 7.90
57.15 ± 7.82 46.10 ± 7.88 37.00 ± 7.63 40.25 ± 7.75 45.45 ± 7.87 49.67 ± 7.90 48.37 ± 7.90
57.15 ± 7.82 44.15 ± 7.85 43.82 ± 7.84 48.05 ± 7.90 43.17 ± 7.83 53.25 ± 7.88 48.70 ± 7.90
Win percentages against H-MCTS (95% Agresti-Coull confidence intervals)

Amazons
UCT
Anytime SH
Atari Go (9×9)
UCT
Anytime SH
Breakthrough
UCT
Anytime SH
Clobber (7×7)
UCT
Anytime SH
Gomoku
UCT
Anytime SH
Hex
UCT
Anytime SH
Pentalath
UCT
Anytime SH
Reversi
UCT
Anytime SH
Tablut
UCT
Anytime SH
Yavalath
UCT
Anytime SH

53.90 ± 7.88 43.50 ± 7.83 44.15 ± 7.85 48.05 ± 7.90 48.70 ± 7.90 53.25 ± 7.88 44.15 ± 7.85
38.95 ± 7.71 37.00 ± 7.63 41.55 ± 7.79 51.95 ± 7.90 40.90 ± 7.77 44.15 ± 7.85 40.25 ± 7.75
35.70 ± 7.57 43.50 ± 7.83 38.30 ± 7.68 35.70 ± 7.57 34.40 ± 7.51 40.90 ± 7.77 47.40 ± 7.89
33.75 ± 7.47 42.20 ± 7.80 33.75 ± 7.47 45.45 ± 7.87 41.55 ± 7.79 40.25 ± 7.75 50.00 ± 7.90
52.60 ± 7.89 48.70 ± 7.90 48.70 ± 7.90 47.40 ± 7.89 41.55 ± 7.79 51.30 ± 7.90 44.80 ± 7.86
50.65 ± 7.90 52.60 ± 7.89 53.90 ± 7.88 46.75 ± 7.88 50.65 ± 7.90 56.50 ± 7.83 50.65 ± 7.90
45.45 ± 7.87 47.40 ± 7.89 53.90 ± 7.88 43.50 ± 7.83 48.05 ± 7.90 50.65 ± 7.90 51.30 ± 7.90
54.55 ± 7.87 46.75 ± 7.88 47.40 ± 7.89 51.95 ± 7.90 50.65 ± 7.90 53.25 ± 7.88 49.35 ± 7.90
24.65 ± 6.81 21.40 ± 6.48 19.45 ± 6.25 23.35 ± 6.69 20.75 ± 6.41 27.25 ± 7.04 25.95 ± 6.93
32.45 ± 7.40 31.80 ± 7.36 18.15 ± 6.09 10.35 ± 4.81 25.30 ± 6.87 18.15 ± 6.09 26.60 ± 6.98
20.75 ± 6.41 37.00 ± 7.63 42.20 ± 7.80 46.75 ± 7.88 51.30 ± 7.90 40.90 ± 7.77 45.45 ± 7.87
12.30 ± 5.19 22.70 ± 6.62 38.30 ± 7.68 45.45 ± 7.87 51.95 ± 7.90 50.00 ± 7.90 48.70 ± 7.90
37.65 ± 7.66 40.25 ± 7.75 43.50 ± 7.83 50.00 ± 7.90 51.30 ± 7.90 46.10 ± 7.88 47.40 ± 7.89
24.65 ± 6.81 35.70 ± 7.57 44.80 ± 7.86 50.00 ± 7.90 43.50 ± 7.83 54.55 ± 7.87 55.20 ± 7.86
49.02 ± 7.90 48.05 ± 7.90 48.70 ± 7.90 46.75 ± 7.88 53.90 ± 7.88 54.23 ± 7.87 50.65 ± 7.90
50.98 ± 7.90 51.95 ± 7.90 54.23 ± 7.87 48.05 ± 7.90 64.95 ± 7.54 54.23 ± 7.87 59.43 ± 7.76
42.85 ± 7.82 49.35 ± 7.90 50.00 ± 7.90 50.00 ± 7.90 50.65 ± 7.90 50.00 ± 7.90 49.35 ± 7.90
43.50 ± 7.83 47.40 ± 7.89 48.70 ± 7.90 49.35 ± 7.90 48.05 ± 7.90 48.05 ± 7.90 47.40 ± 7.89
42.85 ± 7.82 53.90 ± 7.88 63.00 ± 7.63 59.75 ± 7.75 54.55 ± 7.87 50.33 ± 7.90 51.63 ± 7.90
47.40 ± 7.89 57.15 ± 7.82 58.45 ± 7.79 57.48 ± 7.81 52.60 ± 7.89 51.30 ± 7.90 53.90 ± 7.88

## Page 9

Anytime SH
H-MCTS

Win percentage against UCT

60

55

50

45

40
0

10000

20000

30000

40000

Win percentage against H-MCTS

Anytime Sequential Halving in Monte-Carlo Tree Search

50000

Iteration budget per move

9
Anytime SH
UCT

60

55

50

45

40
0

10000

20000

30000

40000

50000

Iteration budget per move

Fig. 4. Win percentages averaged over all ten games against UCT (left) and H-MCTS
(right). 95% confidence intervals are too small to be visible.

to optimize simple regret. However, for many budgets, UCB1 has substantially
greater variance in its performance level than the three SH-based algorithms do.
When averaging results over all ten board games (Fig. 4), we find that (i)
H-MCTS (which is not an anytime algorithm) tends to perform better than the
other two algorithms, in particular for low iteration budgets, and (ii) anytime
SH appears to be possibly weaker than UCT by a slight margin for low iteration
budgets, but evenly-matched or slightly stronger for medium and large iteration
budgets. At the level of individual games, these trends can differ (see Table 2).
The results in the game of Gomoku stand out in particular, with H-MCTS having
particularly high win percentages of around 75% against UCT for all budgets,
and ranging between approximately 70% and 90% against Anytime SH. While
a sample size of ten games (with 150 plays per game, per pair of agents) is no
less than is customary in the literature, it is still small enough for a single game
with relatively extreme results such as Gomoku to have an outsized effect on a
plot that averages over the games like Fig. 4. Especially in the direct matchup
against H-MCTS (the left subfigure), Anytime SH would be substantially closer
to H-MCTS if the results for Gomoku were excluded (and, plausibly also if the
experiment were extended with more other games, if Gomoku is an outlier).

6

Conclusion and Future Work

This paper proposed Anytime Sequential Halving: an algorithm designed in an
effort to approximate the behavior of the standard Sequential Halving (SH)
algorithm, whilst having the anytime property. Experiments in a set of synthetic
Multi-Armed Bandit problems show that Anytime SH performs on par in terms
of simple regret with the standard SH, as well as a trivial time-based variant
of SH, which can handle time-based budgets (but is not truly anytime). While
the performance in terms of simple regret remains unchanged, Anytime SH does
bring the benefit of having the anytime property. When used as selection strategy
for the root node in Monte-Carlo Tree Search (MCTS) for game playing, Anytime
SH appears to perform slightly below the Hybrid MCTS baseline (which has the
downside of not being an anytime algorithm). Its performance level is competitive

## Page 10

10

D. Sagers et al.

with UCT, appearing to be possibly slightly worse for low search budgets, but
slightly better for higher budgets.
We see potential for ample future work to further improve Anytime SH. The
Anytime SH algorithm as proposed in this paper was kept simple and straightforward. We expect that there is likely room to improve the algorithm, in particular
by more carefully investigating how it should behave in situations where its ordering of arms changes between different passes of the algorithm (a situation
that cannot occur in the standard SH). When Anytime SH “changes its mind”
about ordering of any pair of arms, one of them will have had fewer iterations
so far than it should have had in hindsight (fewer iterations than the standard
SH would have allocated if the current total number of iterations were the full
budget). In our implementation, we ignore this, but it is worth investigating if
correcting for this would be worthwhile. This could potentially make the algorithm more adaptive to the level of complexity of a problem than the standard
SH is, which always halves sets of arms at specific intervals regardless of how
many arms are close or far from each other in terms of expected rewards. Furthermore, it would be interesting to investigate how the algorithm interacts with
different values of hyperparameters, such as the exploration constant still used
for UCB1 in nodes below the root node in MCTS.
Disclosure of Interests. The authors have no competing interests to declare that
are relevant to the content of this article.

References
1. Audibert, J.Y., Bubeck, S.: Best arm identification in multi-armed bandits. In:
Kalai, A.T., Mohri, M. (eds.) COLT 2010 - The 23rd Conf. on Learning Theory.
pp. 41–53 (2010)
2. Auer, P., Cesa-Bianchi, N., Fischer, P.: Finite-time analysis of the multiarmed
bandit problem. Mach. Learn. 47(2–3), 235–256 (2002)
3. Baier, H., Winands, M.H.M.: Time management for Monte Carlo tree search. IEEE
Trans. Comput. Intell. and AI in Games 8(3), 301–314 (2015)
4. Browne, C., Powley, E., Whitehouse, D., Lucas, S., Cowling, P.I., Rohlfshagen, P.,
Tavener, S., Perez, D., Samothrakis, S., Colton, S.: A Survey of Monte Carlo Tree
Search Methods. IEEE Trans. Comput. Intell. AI Games 4(1), 1–49 (2012)
5. Bubeck, S., Munos, R., Stoltz, G.: Pure exploration in finitely-armed and
continuous-armed bandits. Theoretical Comput. Sci. 412(19), 1832–1852 (2011)
6. Cazenave, T.: Sequential halving applied to trees. IEEE Trans. Comput. Intell.
and AI in Games 7(1), 102–105 (2015)
7. Coulom, R.: Efficient selectivity and backup operators in Monte-Carlo tree search.
In: van den Herik, H.J., Ciancarini, P., Donkers, H.H.L.M. (eds.) Computers and
Games. LNCS, vol. 4630, pp. 72–83. Springer Berlin Heidelberg (2007)
8. Danihelka, I., Guez, A., Schrittwieser, J., Silver, D.: Policy improvement by planning with Gumbel. In: Int. Conf. on Learning Representations (ICLR 2022) (2022)
9. Fabiano, N., Cazenave, T.: Sequential halving using scores. In: Browne, C., Kishimoto, A., Schaeffer, J. (eds.) Advances in Computer Games. ACG 2021. Lecture
Notes in Computer Science, vol. 13262, pp. 41–52. Springer, Cham (2022)

## Page 11

Anytime Sequential Halving in Monte-Carlo Tree Search

11

10. Feldman, Z., Domshlak, C.: Simple regret optimization in online planning for
Markov decision processes. Journal of Artif. Intell. Research 51, 165–205 (2014)
11. Huang, S.C., Coulom, R., Lin, S.S.: Time management for Monte-Carlo tree search
applied to the game of Go. In: Proc. 2010 Int. Conf. on Technologies and Applications of Artif. Intell. pp. 462–466 (2010)
12. Karnin, Z., Koren, T., Somekh, O.: Almost optimal exploration in multi-armed
bandits. In: Proc. 30th Int. Conf. Mach. Learn. PMLR, vol. 28, pp. 1238–1246
(2013)
13. Kocsis, L., Szepesvári, C.: Bandit based Monte-Carlo planning. In: Fürnkranz, J.,
Scheffer, T., Spiliopoulou, M. (eds.) Mach. Learn.: ECML 2006, LNCS, vol. 4212,
pp. 282–293. Springer, Berlin, Heidelberg (2006)
14. Nguyen, K.P.N., Ramanujan, R.: Lookahead pathology in Monte-Carlo tree search.
In: Proc. Int. Conf. Autom. Planning and Scheduling. vol. 34, pp. 414–422 (2024)
15. Pepels, T., Cazenave, T., Winands, M.H.M.: Sequential halving for partially
observable games. In: Cazenave, T., Winands, M., Edelkamp, S., Schiffel, S.,
Thielscher, M., Togelius, J. (eds.) Computer Games. CGW GIGA 2015. CCIS,
vol. 614, pp. 16–29. Springer, Cham (2016)
16. Pepels, T., Cazenave, T., Winands, M.H.M., Lanctot, M.: Minimizing simple and
cumulative regret in Monte-Carlo tree search. In: Cazenave, T., Winands, M.H.M.,
Björnsson, Y. (eds.) Computer Games. CGW 2014. CCIS, vol. 504, pp. 1–15.
Springer, Cham (2014)
17. Piette, É., Soemers, D.J.N.J., Stephenson, M., Sironi, C.F., Winands, M.H.M.,
Browne, C.: Ludii – the ludemic general game system. In: Giacomo, G.D., Catala,
A., Dilkina, B., Milano, M., Barro, S., Bugarín, A., Lang, J. (eds.) Proc. 24th Eur.
Conf. Artif. Intell. Frontiers in Artificial Intell. and Appl., vol. 325, pp. 411–418.
IOS Press (2020)
18. Soemers, D.J.N.J., Bams, G., Persoon, M., Rietjens, M., Sladić, D., Stefanov, S.,
Driessens, K., Winands, M.H.M.: Towards a characterisation of Monte-Carlo tree
search performance in different games. In: Proc. IEEE Conf. Games. pp. 1–4 (2024)
19. Świechowski, M., Godlewski, K., Sawicki, B., Mańdziuk, J.: Monte Carlo tree
search: A review of recent modifications and applications. Artificial Intell. Review
56, 2497–2562 (2022)
20. Todd, G., Padula, A., Stephenson, M., Piette, É., Soemers, D.J.N.J., Togelius, J.:
GAVEL: Generating games via evolution and language models. https://arxiv.org/
abs/2407.09388 (2024)
