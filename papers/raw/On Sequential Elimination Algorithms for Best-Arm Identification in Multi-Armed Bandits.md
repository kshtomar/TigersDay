## Page 1

1

On Sequential Elimination Algorithms for Best-Arm
Identification in Multi-Armed Bandits

arXiv:1609.02606v2 [stat.ML] 13 Apr 2017

Shahin Shahrampour, Mohammad Noshad, Vahid Tarokh

Abstract—We consider the best-arm identification problem in
multi-armed bandits, which focuses purely on exploration. A
player is given a fixed budget to explore a finite set of arms,
and the rewards of each arm are drawn independently from
a fixed, unknown distribution. The player aims to identify the
arm with the largest expected reward. We propose a general
framework to unify sequential elimination algorithms, where the
arms are dismissed iteratively until a unique arm is left. Our
analysis reveals a novel performance measure expressed in terms
of the sampling mechanism and number of eliminated arms at
each round. Based on this result, we develop an algorithm that
divides the budget according to a nonlinear function of remaining
arms at each round. We provide theoretical guarantees for the
algorithm, characterizing the suitable nonlinearity for different
problem environments described by the number of competitive
arms. Matching the theoretical results, our experiments show
that the nonlinear algorithm outperforms the state-of-the-art.
We finally study the side-observation model, where pulling an
arm reveals the rewards of its related arms, and we establish
improved theoretical guarantees in the pure-exploration setting.

I. I NTRODUCTION
Multi-Armed Bandits (MAB) is a sequential decisionmaking framework addressing the exploration-exploitation
dilemma [1]–[4]. A player explores a finite set of arms
sequentially, and pulling each of them results in a reward to
the player. The problem has been studied for different rewards
models. In the stochastic MAB, the rewards of each arm are
assumed to be i.i.d. samples of an unknown, fixed distribution.
Then, the player’s goal is to exploit the arm with largest
expected reward as many times as possible to maximize the
gain. In the literature, this objective has been translated to
minimizing the cumulative regret, a comparison measure between the actual performance of the player versus a clairvoyant
knowing the best arm in advance. Early studies on MAB dates
back to a few decades ago, but the problem has surged a lot
of renewed interest due to its modern applications. Though
generally addressing sequential decision making problems,
MAB has been studied in several engineering contexts such as
web search and advertising, wireless cognitive radios, multichannel communication systems, and Big-Data streaming (see
e.g. [5]–[11] and references therein).
Departing from its classical setting (explorationexploitation), many researchers have studied MAB in
the pure-exploration framework. In this case, the player aims
to minimize the simple regret which can be related to finding
S. Shahrampour, M. Noshad, and V. Tarokh are with the
John A. Paulson School of Engineering and Applied Sciences,
Harvard
University,
Cambridge,
MA,
02138
USA.
(e-mail:
{shahin,mnoshad,vahid}@seas.harvard.edu).

the best arm with a high probability [12]. As a result, the
best-arm identification problem has received a considerable
attention in the literature of machine learning [12]–[19]. The
problem has been viewed from two main standpoints: (i) the
fixed-confidence setting, where the objective is to minimize
the number of trials to find the best arm with a certain
confidence, and (ii) the fixed-budget setting, where the player
attempts to maximize the probability of correct identification
given a fixed number of arm pulls. Best-arm identification
has several applications including channel allocation as
originally proposed in [12]. Consider the problem of channel
allocation for mobile phone communication. Before the outset
of communication, a cellphone (player) can explore the set
of channels (arms) to find the best one to operate. Each
channel feedback is noisy, and the number of trials (budget)
is limited. The problem is hence an instance of best-arm
identification, and minimizing the cumulative regret is not the
right approach to the problem.
While both pure-exploration and exploration-exploitation
setups are concerned with finding the best arm, they are quite
different problems in nature. In fact, Bubeck et al. [17] establish that methods designed to minimize the cumulative regret
(exploration-exploitation) can perform poorly for the simpleregret minimization (pure-exploration). More specifically, they
proved that upper bounds on the cumulative regret yield lower
bounds on the simple regret, i.e., the smaller the cumulative
regret, the larger the simple regret. Therefore, one must adopt
different strategies for optimal best-arm recommendation.
A. Our Contribution
In this paper, we address the best-arm identification problem
in the fixed-budget setting. We restrict our attention to a
class of algorithms that work based on sequential elimination.
Recently, it is proved in [20] that some existing strategies
based on the sequential elimination of the arms are optimal.
However, the notion of optimality is defined with respect
to the worst-case allocation of the reward distributions. The
main focus of this paper is not the worst-case scenario. On
the contrary, given certain regimes for rewards, our goal is
to propose an algorithm outperforming the state-of-the-art in
these regimes. We characterize these reward allocations and
prove the superiority of our algorithm both theoretically and
empirically.
Of particular relevance to the current study is the works
of [12], [18], where two iterative algorithms are proposed
for sequential elimination: Successive Rejects (Succ-Rej)
[12] and Sequential Halving (Seq-Halv) [18]. In both algorithms, the player must sample the arms in rounds to discard

## Page 2

2

the arms sequentially, until a single arm is left, one that
is perceived as the best arm. However, we recognize two
distinctions between the algorithms: (i) Succ-Rej eliminates
one arm at each round, until it is left with a single arm,
whereas Seq-Halv discards roughly half of the remaining
arms at each round to identify the best arm. (ii) At each
round, Seq-Halv samples the remaining arms uniformly
(excluding previous rounds), whereas Succ-Rej samples
them uniformly once including the previous rounds.
Inspired by these works, our first contribution is to propose
a general framework to bring sequential elimination algorithms
(including Succ-Rej and Seq-Halv) under the same umbrella. Our analysis reveals a novel performance bound which
relies on the sampling design as well as the number of eliminated arms at each around. Following this general framework,
we extend Succ-Rej to an algorithm that divides the budget
by a nonlinear function of remaining arms at each round,
unlike Succ-Rej that does so in a linear fashion. We prove
theoretically that we can gain advantage from the nonlinearity.
In particular, we consider several well-studied reward regimes
and exhibit the suitable nonlinearity for each environment.
Benefiting from the nonlinearity, our algorithm outperforms
Succ-Rej and Seq-Halv in these regimes. Interestingly,
our numerical experiments support our theoretical results,
while showing that our algorithm is competitive with UCB-E
[12] which requires prior knowledge of a problem-dependent
parameter.
Finally, we consider sequential elimination in the presence
of side observations. In this model, a graph encodes the
connection between the arms, and pulling one arm reveals
the rewards of all neighboring arms [21]. While the impact of
side observations is well-known for exploration-exploitation
[22], [23], we consider the model in the pure-exploration
setting. Given a partition of arms to a few blocks, we propose
an algorithm that eliminates blocks consecutively and selects
the best arm from the final block. Naturally, we provide an
improved theoretical guarantee comparing to the full bandit
setting where there is no side observation.
B. Related Work
Pure-exploration in the PAC-learning setup was studied in
[13], where Successive Elimination for finding an ǫ-optimal
arm with probability 1 − δ (fixed-confidence setting) was
proposed. Seminal works of [14], [15] provide matching lower
bounds for the problem, which present a sufficient number of
arm pulls to reach the confidence 1 − δ. Many algorithms
for pure-exploration are inspired by the classical UCB1 for
exploration-exploitation [2]. For instance, Audibert et al. [12]
propose UCB-E, which modifies UCB1 for pure-exploration.
UCB-E needs prior knowledge of a problem-dependent parameter, so the authors also propose its adaptive counterpart
AUCB-E to address the issue. In addition, Jamieson et al. [24]
propose an optimal algorithm for the fixed confidence setting,
inspired by the law of the iterated logarithm. We refer the
reader to [25] and the references therein for recent advances in
the fixed-confidence setting, while remarking that Gabillon et
al. [19] present a unifying approach for fixed-budget and fixedconfidence settings. As another interesting direction, various

works in the literature introduce information-theoretic measures for best-arm identification. Kaufmann et al. [26] study
the identification of multiple top arms using KL-divergencebased confidence intervals. The authors of [27] investigate
both settings to show that the complexity of the fixed-budget
setting may be smaller than that of the fixed-confidence setting.
Recently, Russo [28] develops three Bayesian algorithms to examine asymptotic complexity measure for the fixed-confidence
setting. There also exists extensive literature on identification
of multiple top arms in MAB (see e.g. [26], [27], [29]–[32]).
Finally, we remark that simple-regret minimization has been
successfully used in the context of Monte-Carlo Tree Search
[33], [34] as well.
C. Organization
The rest of the paper is organized as follows. Section II
is dedicated to nomenclature, problem formulation, and a
summary of our results. In Sections III and IV, we discuss
our main theoretical results and their consequences, while we
extend these results to side-observation model in Section V.
In Section VI, we describe our numerical experiments, and the
concluding remarks are provided in Section VII. We include
the proofs in the Appendix (Section VIII).
II. P RELIMINARIES
Notation: For integer K, we define [K] := {1, . . . , K} to
represent the set of positive integers smaller than or equal
to K. We use |S| to denote the cardinality of the set S.
Throughout, the random variables are denoted in bold letters.
A. Problem Statement
Consider the stochastic Multi-armed Bandit (MAB) problem, where a player explores a finite set of K arms. When
the player samples an arm, only the corresponding payoff or
reward of that arm is observed. The reward sequence for each
arm i ∈ [K] corresponds to i.i.d samples of an unknown
distribution whose expected value is µi . We assume that the
distribution is supported on the unit interval [0, 1], and the
rewards are generated independently across the arms. Without
loss of generality, we further assume that the expected value
of the rewards are ordered as
µ1 > µ2 ≥ · · · ≥ µK ,

(1)

and therefore, arm 1 is the unique best arm. We let ∆i :=
µ1 − µi denote the gap between arm i and arm 1, measuring
the sub-optimality of arm i. We also represent by x̄i,n the
average reward obtained from pulling arm i for n times.
In this paper, we address the best-arm identification setup, a
pure-exploration problem in which the player aims to find the
arm with the largest expected value with a high confidence.
There are two well-known scenarios for which the problem has
been studied: fixed confidence and fixed budget. In the fixedconfidence setting, the objective is to minimize the number of
trials needed to achieve a fixed confidence. However, in this
work, we restrict our attention to the fixed-budget, formally
described as follows:

## Page 3

3

TABLE I
T HE PARAMETERS α AND β FOR THE ALGORITHMS DISCUSSED IN THIS SECTION , WHERE THE MISIDENTIFICATION PROBABILITY FOR EACH OF THEM
DECAYS IN THE FORM OF β exp (−T /α). T HE RELEVANT COMPLEXITY MEASURES USED IN THIS TABLE ARE DEFINED IN (4) AND (5).

α
β

Succ-Rej

Seq-Halv

Our algorithm

H2 logK


0.5K(K − 1) exp K/(H2 logK)

8H2 log2 K

H(p)Cp

3 log2 K

(K − 1) exp (K/H(p)Cp )

Problem 1. Given a total budget of T arm pulls, minimize
the probability of misidentifying the best arm.
It is well-known that classical MAB techniques in the
exploration-exploitation setting, such as UCB1, are not efficient for the identification of the best arm. In fact, Bubeck et
al. have proved in [17] that upper bounds on the cumulative
regret yield lower bounds on the simple regret, i.e., the smaller
the cumulative regret, the larger the simple regret. In particular,
for all Bernoulli distributions on the rewards, a constant L > 0,
and a function f (·), they have proved if an algorithm satisfies
Expected Cumulative Regret ≤ Lf (T ),
after T rounds, we have
Misidentification Probability ≥
Expected Simple Regret ≥ D1 exp(−D2 f (T )),
for two positive constants D1 and D2 . Given that for optimal
algorithms in the exploration-exploitation setting, we have
f (T ) = O(log T ), these algorithms decay polynomially fast
for best-arm identification. However, a carefully designed bestarm identification algorithm achieves an exponentially fast
decay rate (see e.g. [12], [18]).
The underlying intuition is that in the explorationexploitation setting, only playing the best arm matters. For
instance, playing the second best arm for a long time can result
in a dramatically large cumulative regret. Therefore, the player
needs to minimize the exploration time to focus only on the
best arm. On the contrary, in the best-arm identification setting,
player must recommend the best arm at the end of the game.
Hence, exploring the suboptimal arms “strategically” during
the game helps the player to make a better recommendation.
In other words, the performance is measured by the final recommendation regardless of the time spent on the suboptimal
arms. We focus on sequential elimination algorithms for the
best-arm identification in the next section.
B. Previous Performance Guarantees and Our Result
In this work, we examine sequential-elimination type algorithms in the fixed budget setting. We propose a general algorithm that unifies sequential elimination methods, including
celebrated Succ-Rej [12] and Seq-Halv [18]. We then
use a special case of this general framework to develop an
algorithm called Nonlinear Sequential Elimination. We show
that this algorithm is more efficient than Succ-Rej and
Seq-Halv in several problem scenarios.
Any sequential elimination algorithm samples the arms
based on some strategy. It then discards a few arms at each

round and stops when it is only left by one arm. In order
to integrate sequential elimination algorithms proceeding in
R rounds, we use the following key observation: let any such
algorithm play each (remaining) arm for nr times (in total) by
the end of round r ∈ [R]. If the algorithm needs to discard br
arms at round r, it must satisfy the following budget constraint
b1 n1 + b2 n2 + · · · + (bR + 1)nR ≤ T,

(2)

since the br arms eliminated at round r have been played
nr times, and the surviving P
arm has been played nR times.
Alternatively, letting gr := R
i=r bi + 1 denote the number
of remaining arms at the start of round r, one can pose the
budget constraint as
g1 n1 + g2 (n2 − n1 ) + · · · + gR (nR − nR−1 ) ≤ T.

(3)

Our first contribution is to derive a generic performance
measure for such algorithm (Theorem 2), relating the algorithm efficiency to {br }R
r=1 as well as the sampling scheme
determining {nr }R
r=1 . Note that Succ-Rej satisfies br = 1
for R = K − 1 rounds, whereas Seq-Halv is characterized
via gr+1 = ⌈gr /2⌉ with g1 = K for R = ⌈log2 K⌉. While it
is shown in [14] that in many settings for MAB, the quantity
H1 in the following plays a key role,
H1 :=

K
X
1

∆2i
i=2

and

H2 := max
i6=1

i
,
∆2i

(4)

the performance of both Succ-Rej and Seq-Halv is described via the complexity measure H2 , which is equal to H1
up to logarithmic factors in K [12]. In particular, for each
algorithm the bound on the probability of misidentification
can be written in the form of β exp (−T /α), P
where α and β
K
are given in Table I in which log K = 0.5 + i=2 i−1 .
In Succ-Rej, at round r, the K −r+1 remaining arms are
played proportional to the whole budget divided by K − r + 1
which is a linear function of r. Motivated by the fact that this
linear function is not necessarily the best sampling rule, as
our second contribution, we specialize our general framework
to an algorithm which can be cast as a nonlinear extension
of Succ-Rej. This algorithm is called Nonlinear Sequential
Elimination (N-Seq-El), where the term “nonlinear” refers
to the fact that at round r, the algorithm divides the budget by
the nonlinear function (K −r+1)p for a positive real p > 0 (an
input value). We prove (Proposition 3) that the performance
of our algorithm depends on the following quantities
ip
i6=1 ∆2
i

H(p) := max

and

Cp := 2−p +

K
X
r=2

r−p , (5)

## Page 4

4

General Sequential Elimination Algorithm

Nonlinear Sequential Elimination Algorithm

Input: budget T , sequence {zr , br }R
r=1 .

Input: budget T , parameter p > 0.

Initialize: G1 = [K], n0 = 0.

Initialize: G1 = [K], n0 = 0.

Let

Let

R
X
1
br
C=
+
zR r=1 zr


T −K
for r ∈ [R]
nr =
Czr

At round r = 1, . . . , R:
(1) Sample each arm in Gr for nr − nr−1 times.
(2) Let Br be the set of br arms with smallest average
rewards.
(3) Let Gr+1 = Gr \ Br , i.e., discard the set worst br
arms.
Output: GR+1 .

Cp = 2−p +

K
X

r−p

r=2

T −K
nr =
p for r ∈ [K − 1]
Cp (K − r + 1)



At round r = 1, . . . , K − 1:
(1) Sample each arm in Gr for nr − nr−1 times.
(2) Let Br be the set containing the arm with smallest
average reward.
(3) Let Gr+1 = Gr \ Br , i.e., discard the worst single
arm.
Output: GK .

Fig. 1. The algorithm on the left represents a general recipe for sequential elimination, whereas the one on the right is a special case of the left hand side,
which extends Succ-Rej to nonlinear budget allocation.

as described in Table I. We do not use p as a subscript in
the definition of H(p) to avoid confusion over the fact that
H(1) = H2 due to definition (4). Indeed, N-Seq-El with
p = 1 recovers Succ-Rej, but we show that in many regimes
for arm gaps, p 6= 1 provides better theoretical results (Corollary 4). We also illustrate this improvement in the numerical
experiments in Section VI, where we observe that p 6= 1
can outperform Succ-Rej and Seq-Halv in many settings
considered in [12], [18]. Since the value of p is received
as an input, we remark that our algorithm needs tuning to
perform well; however, the tuning is more qualitative rather
than quantitative, i.e., the algorithm maintains a reasonable
performance as long as p is in a certain interval, and therefore,
the value of p needs not be specific. We will discuss this in
the next section in more details.
III. N ONLINEAR S EQUENTIAL E LIMINATION
In this section, we propose our generic sequential elimination method and analyze its performance in the fixed budget
setting. Consider a sequential elimination algorithm given
budget T of arm pulls. The algorithm maintains an active
set initialized by the K arms, and it proceeds for R rounds
to discard the arms sequentially until it is left by a single
arm. Let us use ⌈·⌉ to denote the ceiling function. Then, for a
constant C and a decreasing, positive sequence {zr }R
r=1 , set
nr = ⌈(T − K)/(Czr )⌉ at round r ∈ [R], let the algorithm
sample the remaining arms for nr − nr−1 times, and calculate
the empirical average of rewards for each arm. If the algorithm
dismisses br arms
PRwith lowest average rewards, and we impose
the constraint r=1 br = K − 1 on the sequence {br }R
r=1 , the
algorithm outputs a single arm, the one that it hopes to be the
best arm. This general point of view is summarized in Fig. 1
on the left side.
The choice of C in the algorithm (see Fig. 1) must warrant
that the budget constraint (3) holds. When we substitute nr

into (3), we get

 X

R
T −K
T −K
+
br
nR +
b r nr =
CzR
Czr
r=1
r=1
R
X



R
R
X
X
T −K
T −K
+1+
br +
br
CzR
Czr
r=1
r=1
!
R
X br
T −K
1
=K+
= T,
+
C
zR r=1 zr

≤

PR
where in the last line we used the condition r=1 br = K −1.
The following theorem provides an upper bound on the error
probability of the algorithm.
Theorem 2. Consider the General Sequential Elimination
algorithm outlined in Fig. 1. For any r ∈ [R], let br denote the
number of arms thatPthe algorithm eliminates at round r. Let
R
also gr := |Gr | = i=r bi + 1 be the number of remaining
arms at the start of round r. Given a fixed budget T of arm
R
pulls and the input sequence {zr }R
r=1 , setting C and {nr }r=1
as described in the algorithm, the misidentification probability
satisfies the bound,
P (GR+1 6= {1}) ≤
T −K
R max {br } exp −
min
r∈[R]
C r∈[R]

(

2∆2gr+1 +1
zr

)!

.

It is already quite well-known that the sub-optimality ∆i
of each arm i plays a key role in the identification quality;
however, an important subsequent of Theorem 2 is that the
performance of any sequential elimination algorithm also
relies on the choice of zr which governs the constant C. In
[12], Succ-Rej employs zr = K − r + 1, i.e., at each round
the remaining arms are played equally often in total. This
results in C being of order log K.

## Page 5

5

We now use the abstract form of the generic algorithm
to specialize it to Nonlinear Sequential Elimination delineated in Fig. 1. The algorithm works with br = 1 and
p
zr = (K − r + 1) for a given p > 0, and it is called
“nonlinear” since p is not necessarily equal to one. The choice
of p = 1 reduces the algorithm to Succ-Rej. In Section
IV, we prove that in many regimes for arm gaps, p 6= 1
provides better theoretical results, and we further exhibit the
efficiency in the numerical experiments in Section VI. The
following proposition encapsulates the theoretical guarantee
of the algorithm.
Proposition 3. Let the Nonlinear Sequential Elimination algorithm in Fig. 1 run for a given p > 0, and let Cp and H(p)
be as defined in (5). Then, the misidentification probability
satisfies the bound,


T −K
P (GK 6= {1}) ≤ (K − 1) exp −2
.
Cp H(p)

The performance of the algorithm does depend on the
parameter p, but the choice is more qualitative rather than
quantitative. For instance, if the sub-optimal arms are almost
the same, i.e. ∆i ≈ ∆ for i ∈ [K], noting the definition of
Cp and H(p) in (5), we observe that 0 < p < 1 performs
better than p > 1. In general, larger values for p increase
H(p) and decrease C(p). Therefore, there is a trade-off in
choosing p. We elaborate on this issue in Sections IV and VI,
where we observe that a wide range of values for p can be
used for tuning, and the trade-off can be addressed using either
0 < p < 1 or 1 < p ≤ 2.
We remark that in the proof of Theorem 2, the constant
behind the exponent can be improved if one can avoid the
union bounds. To do so, one needs to assume some structure
in the sequence {gr }R
r=1 . For instance, the authors in [18]
leverage the fact that gr+1 = ⌈gr /2⌉ to avoid union bounds
in Seq-Halv. The idea can be extended to when the ratio of
gr+1 /gr is a constant independent of r.
Finally, while the result of Proposition 3 provides a general
performance bound with respect to reward regimes, in the
next section, we provide two important corollaries of the
proposition to study several regimes for rewards, where we
can simplify the bound and compare our result with other
algorithms.
IV. P ERFORMANCE IN S EVERAL S UB - OPTIMALITY
R EGIMES
Inspired by numerical experiments carried out in the previous works [12], [18], we consider a few instances for suboptimality of arms in this section, and we demonstrate how
N-Seq-El fares in these cases. We would like to distinguish
three general regimes that encompass interesting settings for
sub-optimality and determine the values of p for which we
expect N-Seq-El to achieve faster identification rates:
1 Arithmetic progression of gaps: In this case, ∆i = (i −
1)∆0 for i > 1, where ∆0 is a constant.
2 A large group of competitive, suboptimal arms: We
recognize this case as when there exists a constant1 ε ≥ 0
1 The choice of ε must be constant with respect to i and K.

such that ∆i /∆2 ≤ 1+ε for arms i ∈ S, where |S| grows
linearly as a function of K, and for i ∈
/ S, ∆i /∆2 ≥ i.
3 A small group of competitive, suboptimal arms: This
case occurs when there exists a constant ε ≥ 0 such that
∆i /∆2 ≤ 1 + ε for i ∈ S, where |S| is of constant order
with respect to K, and for i ∈
/ S, ∆i /∆2 ≥ i.
We now state the following corollary of Proposition 3, which
proves to be useful for our numerical evaluations. Note that
the orders are expressed with respect to K.
Corollary 4. Consider the Nonlinear Sequential Elimination
algorithm described in Fig. 1. Let constants p and q be chosen
such that 1 < p ≤ 2 and 0 < q < 1. Then, for the
three settings given above, the bound on the misidentification
probability presented in Proposition 3 satisfies
Regime 1
H(p)Cp = O(1)

Regime 2
H(q)Cq = O(K)

Regime 3
H(p)Cp = O(1)

We can now compare our algorithm with Succ-Rej and
Seq-Halv using the result of Corollary 4. Returning to Table
I and calculating H2 for Regimes 1 to 3, we can observe the
following table, which shows that with a right choice of p for
TABLE II
T HE MISIDENTIFICATION PROBABILITY FOR ALL THREE ALGORITHMS
DECAYS IN THE FORM OF β exp (−T /α). T HE TABLE REPRESENTS THE
PARAMETER α FOR EACH ALGORITHM IN R EGIMES 1 TO 3. F OR R EGIME
2, WE SET 0 < p < 1, AND FOR R EGIMES 1 AND 3, WE USE 1 < p ≤ 2.

Regime 1
Regime 2
Regime 3

Succ-Rej
O(log K)
O(K log K)
O(log K)

Seq-Halv
O(log K)
O(K log K)
O(log K)

Our algorithm
O(1)
O(K)
O(1)

N-Seq-El, we can save a O(log K) factor in the exponential
rate comparing to other methods. Though we do not have a
prior information on gaps to categorize them into one of the
Regimes 1 to 3 (and then choose p), the choice of p is more
qualitative rather than quantitative. Roughly speaking: if the
sub-optimal arms are almost the same 0 < p < 1 performs
better than p > 1, and if there are a few real competitive
arms, p > 1 outperforms 0 < p < 1. Therefore, the result
of Corollary 4 is of practical interest, and we will show using
numerical experiments (Section VI) that a wide range of values
for p can potentially result in efficient algorithms.
One should observe that the number of competitive arms is
a key factor in tuning p. We now provide another corollary of
Proposition 3, which presents the suitable range of parameter
p given the growth rate of competitive arms as follows.
Corollary 5. Let the number of competitive arms be an
arbitrary function fK of total arms, i.e., there exists a constant
ε ≥ 0 such that ∆i /∆2 ≤ 1 + ε for i ∈ S, where
|S| = fK , and for i ∈
/ S, ∆i /∆2 ≥ i. Then, there exists
a suitable choice of p for which N-Seq-El outperforms
Succ-Rej and Seq-Halv in the sense that the misidentification probability decays with a faster rate, i.e., we have

## Page 6

6

O(H(p)Cp ) ≤ O(H2 log K). For different conditions on the
growth of fK , the suitable choice of parameter p is presented
in Table III.
TABLE III
G IVEN THAT THE MISIDENTIFICATION PROBABILITY DECAYS IN THE
FORM OF β exp (−T /α), THE TABLE REPRESENTS THE CHOICE OF p FOR
WHICH THE PARAMETER α IN N-Seq-El IS SMALLER THAN THOSE OF
Succ-Rej AND Seq-Halv.

Condition on fK

Suitable Range of Parameter p

1 ≤ fK ≤ log K

1<p≤2

K
log K < fK < log
K

log K
K
< p < 1 + log
1 − loglog
K
log f

K
≤ fK ≤ K − 1
log K

0<p<1

K

log f
K

The corollary above indicates that a side information in the
form of the number of competitive arms (in the order) can
help us tune the input parameter p. The corollary exhibits a
smooth interpolation between when the competitive arms are
small versus when they are large. Perhaps, the most interesting
regime is the middle row, where the choice of p is given as
a function of the number of arms K. Consider the following
example where we calculate the choice of p for fK = K γ ,
where γ ∈ {0.3, 0.5, 0.7}.
TABLE IV
T HE TABLE SHOWS THE SUITABLE INTERVAL OF p FOR
K ∈ {40, 120, 5 ∗ 102 , 5 ∗ 104 , 5 ∗ 106 , 5 ∗ 108 } GIVEN THE GROWTH
RATE OF COMPETITIVE ARMS AS DERIVED IN C OROLLARY 5.

PP
PPfK
K
P
P

K 0.3

K 0.5

K 0.7

40

(0.5, 2)

(0.3, 1.7)

(0, 1.5)

120

(0.53, 2)

(0.35, 1.65)

(0, 1.47)

5 ∗ 102

(0.58, 1.97)

(0.42, 1.58)

(0.03, 1.42)

5 ∗ 104

(0.69, 1.73)

(0.56, 1.44)

(0.27, 1.31)

5 ∗ 106

(0.75, 1.59)

(0.65, 1.35)

(0.41, 1.25)

5 ∗ 108

(0.79, 1.5)

(0.7, 1.3)

(0.5, 1.21)

As we can see in the table above, since the condition on
p depends in logarithmic orders on K and fK , we have
flexibility to tune p even for very large K. However, we only
use K ∈ {40, 120} for the numerical experiments in Section
VI, since the time-complexity of Monte Carlo simulations is
prohibitive on large number of arms.
V. S IDE O BSERVATIONS
In the previous sections, we considered a scenario in which
pulling an arm yields only the reward of the chosen arm.
However, there exist applications where pulling an arm can
additionally result in some side observations. For a motivative
example, consider the problem of web advertising, where an

ad placer offers an ad to a user and receives a reward only if
the user clicks on the ad. In this example, if the user clicks
on a vacation ad, the ad placer receives the side information
that the user could have also clicked on ads for rental cars.
The value of side observations in the stochastic MAB was
studied in [22] for exploration-exploitation setting. The sideobservation model is described via an undirected graph that
captures the relationship between the arms. Once an arm is
pulled, the player observes the reward of the arm as well as
its neighboring arms. In exploration-exploitation settings, the
analysis of MAB with side observations relies on the cliques of
the graph [21], [22]. In this section, we would like to consider
the impact in the pure-exploration setting.
In the pure-exploration, we minimize the simple regret
rather than the cumulative regret, and therefore, the player’s
best bets are the most connected arms resulting in more
observations. Now consider a partition of the set [K] into
M blocks {Vi }M
i=1 such that each Vi contains a star graph,
i.e., there exists an arm in Vi connected to all other arms.
Given such partition, the player can follow a simple rule to
leverage side observations and eliminate blocks one by one.
The idea is to sample the central arm in each block, which
reveals the rewards for all the arms in that block. At round r,
sample the remaining blocks (the central arms of each block)
for nr − nr−1 times, and find the arm with largest average
reward in each block. With the best arms of all blocks at hand,
remove the block whose best arm is the worst comparing to
other blocks. Continue the sampling until only one block is
left, and output the arm with largest average reward in the final
block as the best arm. The algorithm, called Sequential Block
Elimination, is summarized in Fig. 2, and its performance is
characterized in the following theorem.
Theorem 6. Let the Sequential Block Elimination algorithm
in Fig. 2 run given the input sequence {zr }M
r=1 and blocks
{Vi }M
.
Define
V
:=
max
{|V
|}
to
be
the maximum
i
i∈[M]
i=1
cardinality among the blocks. Then, the misidentification probability satisfies the bound
 2


∆M+1−r
T −M
.
2 min
P ({J} 6= {1}) ≤ V M exp −
C
zr
r∈[M]
In particular, the choice of zr = (M + 1 − r)p for a given
p > 0, yields


T −M
P ({J} 6= {1}) ≤ V M exp −2
,
CHM,p
n po
where H(M, p) := maxi∈[M]\{1} ∆i 2 .
i

The theorem indicates that once we can partition the arms
into M blocks, the complexity measure H(p) need not be
maximized over K arms. Instead, it is maximized over the
top M arms. The improvement must be more visible in the
settings that arms are roughly as competitive as each other,
since we can slow down the linear growth of K by M .
As a final note, M = K recovers the fully bandit (no side
observation) setting with V = 1, and we observe that in such
case, we recover the result of Proposition 3 from Theorem 6 in
the rate of exponential decay. However, the constant behind the

## Page 7

7

Sequential Block Elimination Algorithm
M
Input: budget T , sequence {zr }M
r=1 , blocks {Vi }i=1 .

Initialize: G1 = {V1 , . . . , VM }, n0 = 0.
m
l
P
T −M
−1
for r ∈ [M ]
Let C = M
z
and
n
=
r
r=1 r
Czr
At round r = 1, . . . , M − 1:
(1) Sample each block in Gr for nr − nr−1 times.
(2) For any block Vm ∈ Gr , let ym ← maxi∈Vm {x̄i,nr }.
(3) Calculate j ← argminm {ym } to locate the block with smallest maximum average.
(4) Let Br = {Vj }, and eliminate the block Vj , i.e. Gr+1 = Gr \ Br .
If r = M : Stop at (2), and let J be the arm index with the largest empirical average in the only
remaining block.
Output: {J}
Fig. 2. Sequential Block Elimination discards the blocks one by one and selects the best arm in the last block.

exponential would change from K −1 in Proposition 3 to K in
Theorem 6. This is the artifact of an extra term contributing to
the upper bound, which can be removed for the case M = K.
VI. N UMERICAL E XPERIMENTS
In this section, we empirically evaluate our algorithm on the
settings perviously studied in [12], [18]. In these experiments,
we compare N-Seq-El with Succ-Rej, Seq-Halv, and
UCB-E. Though we include UCB-E proposed in [12] as a
benchmark, we remark that the algorithm requires a prior
knowledge of a parameter that depends on H1 defined in
(4). The adaptive version of UCB-E was also developed in
[12]. The algorithm (called AUCB-E) does not need prior
knowledge of H1 , and it calculates the parameter online. The
experiments in [18] suggest that for T ≈ H1 , Succ-Rej,
Seq-Halv, and UCB-E outperform AUCB-E, and it is not
surprising that the prior knowledge of H1 must give UCB-E
an advantage over AUCB-E.
We consider Bernoulli distribution on the rewards, assuming
that the expected value of Bernoulli distribution for the best
arm is µ1 = 0.7. In what follows, we use the notation x :y to
denote integers in [x, y]. We examine the following setups for
two values of arm numbers K ∈ {40, 120}:
1 One group of suboptimal arms: µ2 :K = 0.6.
2
2 Two groups of suboptimal
 arms:
µ2:m = 0.7 − K ,
K
µm+1:K = 0.4, and m = log 2 + 1 .
2
,
3 Three groups of suboptimal arms: µ2:m = 0.7 − K
4
µ
=
0.7
−
,
µ
=
0.4,
and
m
=
m+1:2m
2m+1:K
K


log K
2 +1 .
for i = 2 :K.
4 Arithmetic progression: ∆i = 0.6(i−1)
K−1
4 i−2
5 Geometric progression: ∆i = 0.01(1 + K
)
for i =
2 :K.
1
and µ3:K =
6 One real competitive arm: µ2 = 0.7 − 2K
0.2.
We run 4000 experiments for each setup with specific K,
and the misidentification probability is averaged out over the
experiment runs. The budget T considered in each setup is
equal to ⌈H1 ⌉ in the corresponding setup following [12], [18].

Fig. 3 illustrates the performance of each algorithm in different
setups. The height of each bar depicts the misidentification
probability, and the index guideline is as follows: (i) indices
1-4: N-Seq-El with parameter p = 0.75, 1.35, 1.7, 2. (ii)
index 5: Succ-Rej. (iii) index 6: Seq-Halv. (iv) indices
7-9: UCB-E with parameter a = cT /H1 , for c = 1, 2, 4. The
legends are the same for all the plots, and we remove some
of them to avoid clutter.
The results are consistent with Corollary 4, and the following comments are in order:
• Setup 1 perfectly relates to Regime 2 in Corollary 4
(ε = 0). While any choice of 0 < p < 1 gives an
O(K) rate, the rate deteriorates to O(K p ) by choosing
1 < p ≤ 2. Therefore, only the the choice of p = 0.75
is appropriate for N-Seq-El. This choice joined with
Succ-Rej (which amounts to our algorithm with p = 1)
outperform others even UCB-E.
• Setup 4 defines Regime 1 in Corollary 4. Accordingly,
choice of 1 < p ≤ 2 outperforms Succ-Rej and
Seq-Halv, and p = 1.7, 2 prove to be competitive to
UCB-E.
• In Setups 2-3, the number of competitive arms grows
with O(log K), but they can be considered close to
Regime 3 in Corollary 4 as the growth is sub-linear. Also,
Setup 6 is the ideal match for Regime 3 in Corollary 4.
Therefore, any choice of 1 < p ≤ 2 should be suitable
in these cases. We observe that for p = 1.35, 1.7, 2,
N-Seq-El outperforms Succ-Rej and Seq-Halv,
while being quite competitive to UCB-E in Setups 2-3
for K = 40.
• In Setup 5, we were expecting to observe good performance for 1 < p ≤ 2. While we do see that
for K = 40, N-Seq-El outperforms Succ-Rej and
Seq-Halv with p = 1.35, 1.7, the performance is not
quite satisfactory for K = 120 (only p = 1.35 performs
well). A potential reason is that we want to keep the
rewards bounded in [0, 1], thereby choosing the geometric
rate for ∆i so slow that it is dominated by ip rate in (5).
To support our argument, we present a complementary

## Page 8

8

Error Probability

1-4: N-Seq-El
5: Succ-Rej
6: Seq-Halv
7-9: UCB-E

0.5
0.4
0.3
0.2
0.1
0
2

3

5

6

7

8

0.3
0.2
0.1
2

3

0.4
0.3
0.2
0.1

4

5

6

7

8

9

7

8

9

7

8

9

Setup 2 (K=120)

0.6

0.5

0.5
0.4
0.3
0.2
0.1
0

1

2

3

4

5

6

7

8

9

1

Setup 3 (K=40)

2

3

0.5
0.4
0.3
0.2
0.1
0

4

5

6

Setup 3 (K=120)

0.6

Error Probability

0.6

Error Probability

0.4

1

0

0.5
0.4
0.3
0.2
0.1
0

2

3

4

5

6

7

8

Setup 4 (K=40)

0.6

9
1-4: N-Seq-El
5: Succ-Rej
6: Seq-Halv
7-9: UCB-E

0.5
0.4
0.3
0.2
0.1

1

2

3

0

4

5

6

Setup 4 (K=120)

0.6

Error Probability

1

Error Probability

0.5

9

Error Probability

Error Probability

4

Setup 2 (K=40)

0.6

1-4: N-Seq-El
5: Succ-Rej
6: Seq-Halv
7-9: UCB-E

0.5
0.4
0.3
0.2
0.1
0

1

2

3

4

5

6

7

8

9

1

Setup 5 (K=40)

2

3

0.5
0.4
0.3
0.2
0.1
0

4

5

6

7

8

9

7

8

9

7

8

9

Setup 5 (K=120)

0.6

Error Probability

0.6

Error Probability

1-4: N-Seq-El
5: Succ-Rej
6: Seq-Halv
7-9: UCB-E

0
1

0.5
0.4
0.3
0.2
0.1
0

1

2

3

4

5

6

7

8

9

1

Setup 6 (K=40)

2

3

0.5
0.4
0.3
0.2
0.1
0

4

5

6

Setup 6 (K=120)

0.6

Error Probability

0.6

Error Probability

Setup 1 (K=120)

0.6

Error Probability

Setup 1 (K=40)

0.6

0.5
0.4
0.3
0.2
0.1
0

1

2

3

4

5

6

7

8

9

1

2

3

4

5

6

Fig. 3. The figure shows the misidentification probability for N-Seq-El, Succ-Rej, Seq-Halv, and UCB-E algorithms in six different setups. The six
plots on the left hand side relate to the case K = 40, and the six plots on the right hand side are associated with K = 120. The height of each bar depicts the
misidentification probability, and each index (or color) represents one algorithm tuned with a specific parameter in case the algorithm is parameter dependent.

## Page 9

9

Geometric Setup for K=7

0.4

1-4: N-Seq-El
5: Succ-Rej
6: Seq-Halv
7-9: UCB-E

Error Probability

0.3

0.2

(iii) and establishing a theoretical result on the value of side
observations in the pure-exploration setting.
Having established that we gain advantage from nonlinear
budget allocation, an important future direction is to propose
a method that starts with a specific nonlinear rate and finetunes the rate according to the problem environment. The
main challenge is that the quantity C should perhaps be timevarying, and its value needs to be cautiously controlled by
the algorithm, so that the algorithm does not overspend the
budget.

0.1

ACKNOWLEDGMENTS
0
1

2

3

4

5

6

7

8

9

Fig. 4. The figure shows the misidentification probability for N-Seq-El,
Succ-Rej, Seq-Halv, and UCB-E algorithms for an experiment with K =
7. The height of each bar depicts the misidentification probability, and each
index (or color) represents one algorithm tuned with a specific parameter in
case the algorithm is parameter dependent.

We gratefully acknowledge the support of DARPA under
grant numbers N66001-15-C-4028 and W911NF-14-1-0508.
We thank Prof. Masahiro Yukawa (Keio University) for his
helpful comments. We also thank Dr. Ahmad Beirami and Dr.
Hamed Farhadi for many helpful discussions.

VIII. A PPENDIX
evaluation of this case for a small K, where we can use
faster geometric growth.
• As a final remark, note that in consistent with Table
II, for relevant cases we observe an improvement of
performance when K increases. For instance, in Setup 1
with p = 0.75, the ratio of misidentification probability of
N-Seq-El to Seq-Halv increases from 1.84 (K = 40)
to 2.06 (K = 120), or in Setup 4 with p = 2, the
ratio of misidentification probability of N-Seq-El to
Succ-Rej increases from 1.43 (K = 40) to 1.63 (K =
120).
A. Complementary Numerical Experiment for Geometric Suboptimality Setup
As we remarked in the very last comment of numerical
experiments, choosing small number of arms, we can space
the expected value of arm rewards such that in (5) the suboptimality term ∆i dominates ip for 1 < p ≤ 2. We propose
a simple experiment for K = 7 arms, where µ1 = 0.7 and
∆i = (0.6)8−i in Fig. 4. As expected, N-Seq-El tuned with
p > 1 is competitive, and it achieves its best performance with
p = 1.7, winning the competition against others, while almost
equalizing with UCB-E for c = 2.
VII. C ONCLUSION
We considered best-arm identification in the stochastic
multi-armed bandits, where a player is given a certain budget
to explore a finite number of arms. The player’s objective is
to detect the arm with largest expected reward. We contribute
to the literature of best-arm identification by (i) unifying
sequential elimination algorithms under a general framework,
which introduces a novel performance metric for this class of
algorithms, (ii) developing a nonlinear sequential elimination
algorithm with provable theoretical and practical guarantees,

Fact 1. (Hoeffding’s inequality) Let W1 , . . . , Wn be independent random variables with
P support on the unit interval with
probability one. If Sn = ni=1 Wi , then for all a > 0, it holds
that


−2a2
.
P (Sn − E[Sn ] ≥ a) ≤ exp
n
P ROOF OF T HEOREM 2
Recall that Gr denotes the set of arms not eliminated by
the start of round r with gr = |Gr | being its cardinality. Also,
Br represents the set of arms that we decide to discard after
playing round r with br = |Br | denoting its cardinality. It
evidently holds that Gr+1 = Gr \ Br for r ∈ [R]. Therefore,
R
 X
=
P (1 ∈ Br ) ,
P (GR+1 6= {1}) = P 1 ∈ ∪R
B
r
r=1
r=1

(6)

since the sets of removed arms at each round are disjoint, i.e.
Bi ∩ Bj = ∅ for i 6= j. We can then write
P (1 ∈ Br ) =

X

P (1 ∈ Br | Gr = Gr ) P (Gr = Gr ) .

Gr

(7)

Now for any particular Gr , consider the worst br arms,
i.e., the bottom br arms when arms are ordered in terms
of (true) expected value. If the best arm (arm 1) is set to
be eliminated at the end of round r, its empirical average
must be less than at least one of these br arms. In the case
that Gr = {1, 2, . . . , gr }, the bottom br arms would be
{gr − br + 1, . . . , gr }. Therefore, recalling that x̄i,n denotes

## Page 10

10

the average reward of pulling arm i for n times, and using
Hoeffding’s inequality (Fact 1), we get
P (1 ∈ Br | Gr = {1, 2, . . . , gr })
gr
X
P (x̄1,nr ≤ x̄i,nr )
≤
i=gr −br +1

−2nr ∆2gr −br +1

≤ br exp





= br exp −2nr ∆2gr+1 +1 ,

(8)

where the last step is due to the fact that gr+1 = gr − br . In
any other case for Gr the best of the worst br arms cannot be
better than arm gr − br + 1. As a result, combining (7) and
(8), we obtain


X
P (1 ∈ Br ) ≤
br exp −2nr ∆2gr+1 +1 P (Gr = Gr )
Gr



= br exp −2nr ∆2gr+1 +1 .

Then, in view of (6) we derive
P (GR+1 6= {1}) ≤

R
X



n

≤ R max {br } max exp −2nr ∆2gr+1 +1
r∈[R]

Noting the fact that nr =
to conclude that

r∈[R]

l

T −K
Czr

and the maximum order can be achieved as the number of arms
close to the second best arm grows linearly in K. Combining
with Cq = O(K 1−q ), the product Cq H(q) = O(K).
For Regime 3, if i ∈ S, we have
ip
max 2 = O(1),
i∈S ∆i
since the cardinality of S is of constant order with respect to
K. On the other hand, since 1 < p ≤ 2, we have
ip
ip−2
ip
≤
max
=
max
= O(1).
i/
∈S i2 ∆2
i/
∈S ∆2
i/
∈S ∆2
2
2
i

m

o

.

−K
≥ TCz
, we can use above
r

P (GR+1 6= {1}) ≤


T −K
2
2∆gr+1 +1
,
R max {br } max exp −
Czr
r∈[R]
r∈[R]


which completes the proof.
P ROOF OF P ROPOSITION 3
We point out that the algorithm is a special case of General
Sequential Elimination where R = K − 1, br = 1, gr =
K − r + 1, and zr = (K − r + 1)p . The proof then follows
immediately from the result of Theorem 2.
P ROOF OF C OROLLARY 4
The proof follows by substituting each case in (5). We need
to understand the order of Cp and H(p) for different regimes
of p. Let us start by
Cp = 2−p +

which is of constant order with respect to K. Therefore, the
product Cp H(p) = O(1).
For Regime 2, we have
iq
Kq
iq
H(q) = max 2 ≤ max 2 ≤ 2 ,
i6=1 ∆2
i6=1 ∆i
∆2

max



br exp −2nr ∆2gr+1 +1

r=1

For Regime 1, since ∆i = (i − 1)∆0 , for p ∈ (1, 2), we have
p

(i − 1)p−2
1
H(p) = max 1 +
i6=1
i−1
∆20

p
1
1
1.5p
≤ max 1 +
= 2,
2
i6=1
i−1
∆0
∆0

K
X

r−p ,

r=2

and noting that for any p > 1, Cp is a convergent sum when
K → ∞. Therefore, for the regime p > 1, the sum is a
constant, i.e., Cp = O(1). On the other hand, consider q ∈
(0, 1), and note that the sum is divergent, and for large K we
have Cq = O(K 1−q ). Now, let us analyze
ip
.
i6=1 ∆2
i

H(p) = max

Therefore, H(p) is of constant order, and combining with
Cp = O(1), the product Cp H(p) = O(1).
P ROOF OF C OROLLARY 5
According to calculations in the previous proof, we have
that Cp = O(K 1−p ) for 0 < p < 1, and Cp = O(1) for
1 < p ≤ 2. In order to find out the order of H(p) and H(2),
we should note that




ip
ip
p
,
O(H(p)) = O max 2 = O max 2 = fK
i∈S ∆2
i∈S ∆i

and we simply have O(H2 ) = O(H(1)) = fK . Now returning
p
O(Cp ) and
to Table I, we must compare O(H(p)Cp ) = fK
O(H2 log K) = fK log K for all three the rows given in Table
III.
Row 1: In the case that 1 ≤ fK ≤ log K, for any choice
of 1 < p ≤ 2, since Cp = O(1), we always have
p
p−1
fK
≤ fK log K ⇔ fK
≤ log K.

Row 3: In the case that logKK ≤ fK ≤ K − 1, for any
choice of 0 < p < 1, since Cp = O(K 1−p ), we have
1−p

K
p
p
1−p
≤ fK log K.
fK O(Cp ) = fK K
= fK
fK

Row 2: In this case, we have that log K < fK < logKK . To
prove the claim in the Table III, we have to break down the
analysis into two cases:
Row 2 – Case 1: First, consider the case 1 − loglogK K <
log

p < 1. Since Cp = O(K

1−p

), we have that
1−p

K
p
p
1−p
fK O(Cp ) = fK K
= fK
fK
  loglog K
K log fK
K
= fK log K.
≤ fK
fK

fK

## Page 11

11

Row 2 – Case 2: Second, consider the case 1 < p <
log K
1 + log
log fK . Since Cp = O(1), we have that
log K
1+ log
log fK
p
p
fK
O(Cp ) = fK
≤ fK

log log K
log fK

= fK fK

= fK log K.

Therefore, for all of the conditions on fK , we showed proper
p
choice of p which guarantees fK
O(Cp ) ≤ fK log K.
P ROOF OF T HEOREM 6
Recall that the elements of Gr are the set of M + 1 − r
blocks not eliminated by the start of round r ∈ [M ], and Br
only contains the block we decide to discard after playing
round r. Also, recall that arm 1 is the best arm, located in the
block V1 , and V := maxi∈[M] {|Vi |} denotes the maximum
cardinality of the blocks.
If the algorithm does not output the best arm, the arm is
either eliminated with block V1 in one the rounds in [M − 1],
or not selected at the final round M . Therefore,

P (J 6= 1) ≤ P argmaxi∈V1 {x̄i,nM } 6= 1 | GM = {V1 }
+

M−1
X

P (Br = {V1 }) .

(9)

r=1

The first term can be bounded simply by Hoeffding’s inequality

P argmaxi∈V1 {x̄i,nM } 6= 1 | GM = {V1 } ≤


|V1 | exp −2nM ∆22 = |V1 | exp −2nM ∆21 ,

(10)

using the convention ∆1 := ∆2 . On the other hand, for any
r ∈ [M − 1]
P (Br = {V1 }) =
X
P (Br = {V1 } | Gr = Gr ) P (Gr = Gr ) . (11)
Gr

Note that, without loss of generality, we sort the blocks based
on their best arm as
µ1 = max µi > max µi ≥ · · · ≥ max µi ,
i∈V1

i∈V2

i∈VM

so the best possible arm in Vm cannot be better than arm
m in view of above and (1). We remove block V1 after
execution of round r only if it is the worst among all other
candidates. Therefore, consider the particular case that Gr =
{V1 , V2 , . . . , VM+1−r } contains the best possible M + 1 − r
blocks that one can keep until the start of round r. In such
case,
P (Br = {V1 } | Gr = {V1 , V2 , . . . , VM+1−r })

≤ V exp −2nr ∆2M+1−r .

In any other case for Gr the best possible arm in the worst
block cannot be better than arm M + 1 − r. Therefore,
combining above with (11), we obtain
X

exp −2nr ∆2M+1−r P (Gr = Gr )
P (Br = {V1 }) ≤ V
G

r

= V exp −2nr ∆2M+1−r .

Incorporating above and (10) into (9), we derive
P (J 6= 1) ≤ |V1 | exp −2nM ∆21
+

M−1
X



V exp −2nr ∆2M+1−r

r=1
M
X

≤V

exp −2nr ∆2M+1−r

r=1





 2


∆M+1−r
T −M
2 min
,
≤ V M exp −
C
zr
r∈[M]

noticing the choice of nr in the algorithm (Fig. 2). Having
finished the first part of the proof, when we set zr =
p
(M + 1 − r) for r ∈ [M − 1] and zM = 2p , we have


T −M
,
P (J 6= 1) ≤ V M exp −2
CHM,p
n po
where H(M, p) := maxi∈[M]\{1} ∆i 2 .
i

R EFERENCES

[1] T. L. Lai and H. Robbins, “Asymptotically efficient adaptive allocation
rules,” Advances in applied mathematics, vol. 6, no. 1, pp. 4–22, 1985.
[2] P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time analysis of the
multiarmed bandit problem,” Machine learning, vol. 47, no. 2-3, pp.
235–256, 2002.
[3] P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire, “The nonstochastic multiarmed bandit problem,” SIAM Journal on Computing,
vol. 32, no. 1, pp. 48–77, 2002.
[4] S. Bubeck and N. Cesa-Bianchi, “Regret analysis of stochastic and
nonstochastic multi-armed bandit problems,” Machine Learning, vol. 5,
no. 1, pp. 1–122, 2012.
[5] A. Mahajan and D. Teneketzis, “Multi-armed bandit problems,” in
Foundations and Applications of Sensor Management. Springer, 2008,
pp. 121–151.
[6] K. Liu and Q. Zhao, “Distributed learning in multi-armed bandit with
multiple players,” IEEE Transactions on Signal Processing, vol. 58,
no. 11, pp. 5667–5681, 2010.
[7] K. Wang and L. Chen, “On optimality of myopic policy for restless
multi-armed bandit problem: An axiomatic approach,” IEEE Transactions on Signal Processing, vol. 60, no. 1, pp. 300–309, 2012.
[8] S. Vakili, K. Liu, and Q. Zhao, “Deterministic sequencing of exploration
and exploitation for multi-armed bandit problems,” IEEE Journal of
Selected Topics in Signal Processing, vol. 7, no. 5, pp. 759–767, 2013.
[9] D. Kalathil, N. Nayyar, and R. Jain, “Decentralized learning for multiplayer multiarmed bandits,” IEEE Transactions on Information Theory,
vol. 60, no. 4, pp. 2331–2345, 2014.
[10] S. Bagheri and A. Scaglione, “The restless multi-armed bandit formulation of the cognitive compressive sensing problem,” IEEE Transactions
on Signal Processing, vol. 63, no. 5, pp. 1183–1198, 2015.
[11] K. Kanoun, C. Tekin, D. Atienza, and M. Van Der Schaar, “Big-data
streaming applications scheduling based on staged multi-armed bandits,”
IEEE Transactions on Computers, 2016.
[12] J.-Y. Audibert and S. Bubeck, “Best arm identification in multi-armed
bandits,” in COLT-23th Conference on Learning Theory-2010, 2010, pp.
13–p.
[13] E. Even-Dar, S. Mannor, and Y. Mansour, “PAC bounds for multi-armed
bandit and markov decision processes,” in Computational Learning
Theory. Springer, 2002, pp. 255–270.
[14] S. Mannor and J. N. Tsitsiklis, “The sample complexity of exploration
in the multi-armed bandit problem,” The Journal of Machine Learning
Research, vol. 5, pp. 623–648, 2004.
[15] E. Even-Dar, S. Mannor, and Y. Mansour, “Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning
problems,” The Journal of Machine Learning Research, vol. 7, pp. 1079–
1105, 2006.
[16] S. Bubeck, R. Munos, and G. Stoltz, “Pure exploration in multi-armed
bandits problems,” in Algorithmic Learning Theory. Springer, 2009,
pp. 23–37.

## Page 12

12

[17] ——, “Pure exploration in finitely-armed and continuous-armed bandits,” Theoretical Computer Science, vol. 412, no. 19, pp. 1832–1852,
2011.
[18] Z. Karnin, T. Koren, and O. Somekh, “Almost optimal exploration in
multi-armed bandits,” in Proceedings of the 30th International Conference on Machine Learning (ICML-13), 2013, pp. 1238–1246.
[19] V. Gabillon, M. Ghavamzadeh, and A. Lazaric, “Best arm identification:
A unified approach to fixed budget and fixed confidence,” in Advances
in Neural Information Processing Systems, 2012, pp. 3212–3220.
[20] A. Carpentier and A. Locatelli, “Tight (lower) bounds for the fixed budget best arm identification bandit problem,” in 29th Annual Conference
on Learning Theory, 2016, pp. 590–604.
[21] S. Mannor and O. Shamir, “From bandits to experts: On the value
of side-observations,” in Advances in Neural Information Processing
Systems, 2011, pp. 684–692.
[22] S. Caron, B. Kveton, M. Lelarge, and S. Bhagat, “Leveraging side
observations in stochastic bandits,” Uncertainty in Artificial Intelligence
(UAI), 2012.
[23] S. Buccapatnam, A. Eryilmaz, and N. B. Shroff, “Stochastic bandits
with side observations on networks,” in The 2014 ACM international
conference on Measurement and modeling of computer systems, 2014,
pp. 289–300.
[24] K. Jamieson, M. Malloy, R. Nowak, and S. Bubeck, “lil’ucb: An optimal
exploration algorithm for multi-armed bandits,” in Proceedings of The
27th Conference on Learning Theory, 2014, pp. 423–439.
[25] K. Jamieson and R. Nowak, “Best-arm identification algorithms for
multi-armed bandits in the fixed confidence setting,” in Information
Sciences and Systems (CISS), 2014 48th Annual Conference on. IEEE,

2014, pp. 1–6.
[26] E. Kaufmann and S. Kalyanakrishnan, “Information complexity in bandit
subset selection,” in Conference on Learning Theory, 2013, pp. 228–251.
[27] E. Kaufmann, O. Cappé, and A. Garivier, “On the complexity of bestarm identification in multi-armed bandit models,” Journal of Machine
Learning Research, vol. 17, no. 1, pp. 1–42, 2016.
[28] D. Russo, “Simple bayesian algorithms for best arm identification,” in
29th Annual Conference on Learning Theory, 2016.
[29] S. Kalyanakrishnan and P. Stone, “Efficient selection of multiple bandit
arms: Theory and practice,” in Proceedings of the 27th International
Conference on Machine Learning (ICML-10), 2010, pp. 511–518.
[30] S. Kalyanakrishnan, A. Tewari, P. Auer, and P. Stone, “PAC subset
selection in stochastic multi-armed bandits,” in Proceedings of the 29th
International Conference on Machine Learning (ICML-12), 2012, pp.
655–662.
[31] S. Bubeck, T. Wang, and N. Viswanathan, “Multiple identifications
in multi-armed bandits,” in Proceedings of The 30th International
Conference on Machine Learning, 2013, pp. 258–265.
[32] Y. Zhou, X. Chen, and J. Li, “Optimal PAC multiple arm identification
with applications to crowdsourcing,” in Proceedings of the 31st International Conference on Machine Learning (ICML-14), 2014, pp. 217–225.
[33] T. Pepels, T. Cazenave, M. H. Winands, and M. Lanctot, “Minimizing
simple and cumulative regret in monte-carlo tree search,” in Computer
Games. Springer, 2014, pp. 1–15.
[34] Y.-C. Liu and Y. Tsuruoka, “Regulation of exploration for simple
regret minimization in monte-carlo tree search,” in IEEE Conference
on Computational Intelligence and Games (CIG), 2015, pp. 35–42.
