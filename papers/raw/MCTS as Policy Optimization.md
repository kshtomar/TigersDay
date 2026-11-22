## Page 1

Monte-Carlo tree search as regularized policy optimization

Jean-Bastien Grill * 1 Florent Altché * 1 Yunhao Tang * 1 2 Thomas Hubert 3 Michal Valko 1
Ioannis Antonoglou 3 Rémi Munos 1

Abstract
The combination of Monte-Carlo tree search
(MCTS) with deep reinforcement learning has
led to significant advances in artificial intelligence. However, AlphaZero, the current stateof-the-art MCTS algorithm, still relies on handcrafted heuristics that are only partially understood. In this paper, we show that AlphaZero’s
search heuristics, along with other common ones
such as UCT, are an approximation to the solution of a specific regularized policy optimization
problem. With this insight, we propose a variant
of AlphaZero which uses the exact solution to
this policy optimization problem, and show experimentally that it reliably outperforms the original
algorithm in multiple domains.

1. Introduction
Policy gradient is at the core of many state-of-the-art deep
reinforcement learning (RL) algorithms. Among many successive improvements to the original algorithm (Sutton et al.,
2000), regularized policy optimization encompasses a large
family of such techniques. Among them trust region policy
optimization is a prominent example (Schulman et al., 2015;
2017; Abdolmaleki et al., 2018; Song et al., 2019). These algorithmic enhancements have led to significant performance
gains in various benchmark domains (Song et al., 2019).
As another successful RL framework, the AlphaZero family
of algorithms (Silver et al., 2016; 2017b;a; Schrittwieser
et al., 2019) have obtained groundbreaking results on challenging domains by combining classical deep learning (He
et al., 2016) and RL (Williams, 1992) techniques with
Monte-Carlo tree search (Kocsis and Szepesvári, 2006).
To search efficiently, the MCTS action selection criteria
takes inspiration from bandits (Auer, 2002). Interestingly,
*

Equal contribution 1 DeepMind, Paris, FR 2 Columbia University, New York, USA 3 DeepMind, London, UK. Correspondence
to: Jean-Bastien Grill <jbgrill@google.com>.
Proceedings of the 37 th International Conference on Machine
Learning, Vienna, Austria, PMLR 119, 2020. Copyright 2020 by
the author(s).

AlphaZero employs an alternative handcrafted heuristic to
achieve super-human performance on board games (Silver
et al., 2016). Recent MCTS-based MuZero (Schrittwieser
et al., 2019) has also led to state-of-the-art results in the
Atari benchmarks (Bellemare et al., 2013).
Our main contribution is connecting MCTS algorithms,
in particular the highly-successful AlphaZero, with MPO,
a state-of-the-art model-free policy-optimization algorithm (Abdolmaleki et al., 2018). Specifically, we show that
the empirical visit distribution of actions in AlphaZero’s
search procedure approximates the solution of a regularized
policy-optimization objective. With this insight, our second
contribution a modified version of AlphaZero that comes
significant performance gains over the original algorithm,
especially in cases where AlphaZero has been observed to
fail, e.g., when per-search simulation budgets are low (Hamrick et al., 2020).
In Section 2, we briefly present MCTS with a focus on AlphaZero and provide a short summary of the model-free
policy-optimization. In Section 3, we show that AlphaZero
(and many other MCTS algorithms) computes approximate
solutions to a family of regularized policy optimization
problems. With this insight, Section 4 introduces a modified
version of AlphaZero which leverages the benefits of the
policy optimization formalism to improve upon the original algorithm. Finally, Section 5 shows that this modified
algorithm outperforms AlphaZero on Atari games and continuous control tasks.

2. Background
Consider a standard RL setting tied to a Markov decision
process (MDP) with state space X and action space A. At a
discrete round t ≥ 0, the agent in state xt ∈ X takes action
at ∈ A given a policy at ∼ π(·|st ), receives reward rt ,
and transitions to a next state xt+1 ∼ p(·|xt , at ). The RL
problem consists in finding a policy
P which maximizes the
discounted cumulative return Eπ [ t≥0 γ t rt ] for a discount
factor γ ∈ (0, 1). To scale the method to large environments,
we assume that the policy πθ (a|x) is parameterized by a
neural network θ.

## Page 2

MCTS as regularized policy optimization

2.1. AlphaZero
We focus on the AlphaZero family, comprised of AlphaGo (Silver et al., 2016), AlphaGo Zero (Silver
et al., 2017b), AlphaZero (Silver et al., 2017a), and
MuZero (Schrittwieser et al., 2019), which are among the
most successful algorithms in combining model-free and
model-based RL. Although they make different assumptions, all of these methods share the same underlying search
algorithm, which we refer to as AlphaZero for simplicity.
From a state x, AlphaZero uses MCTS (Browne et al., 2012)
to compute an improved policy π̂(·|x) at the root of the
search tree from the prior distribution predicted by a policy
network πθ (·|x)1 ; see Eq. 3 for the definition. This improved policy is then distilled back into πθ by updating θ
as θ ← θ − η∇θ Ex [D(π̂(·|x), πθ (·|x))] for a certain divergence D. In turn, the distilled parameterized policy πθ
informs the next local search by predicting priors, further
improving the local policy over successive iterations. Therefore, such an algorithmic procedure is a special case of
generalized policy improvement (Sutton and Barto, 1998).
One of the main differences between AlphaZero and previous MCTS algorithms such as UCT (Kocsis and Szepesvári,
2006) is the introduction of a learned prior πθ and value
function vθ . Additionally, AlphaZero’s search procedure
applies the following action selection heuristic,
"

pP

arg max Q(x, a) + c · πθ (a|x) ·
a

b n(x, b)

1 + n(x, a)

πθ0 , arg max QTπθ y − R(y, πθ ),

(2)

y∈S

where Qπθ is an estimate of the Q-function, S is the |A|dimensional simplex and R : S 2 → R a convex regularization term (Neu et al., 2017; Grill et al., 2019; Geist et al.,
2019). Intuitively, Eq. 2 updates πθ to maximize the value
QTπθ y while constraining the update with a regularization
term R(y, πθ ).
Without regularizations, i.e., R = 0, Eq. 2 reduces to policy
iteration (Sutton and Barto, 1998). When πθ is updated
using a single gradient ascent step towards the solution
of Eq. 2, instead of using the solution directly, the above
formulation reduces to (regularized) policy gradient (Sutton
et al., 2000; Levine, 2018).
Interestingly, the regularization term has been found to stabilize, and possibly to speed up the convergence of πθ .
For instance, trust region policy search algorithms (TRPO,
Schulman et al., 2015; MPO Abdolmaleki et al., 2018; VMPO, Song et al., 2019), set R to be the KL-divergence
between consecutive policies KL[y, πθ ]; maximum entropy
RL (Ziebart, 2010; Fox et al., 2015; O’Donoghue et al.,
2016; Haarnoja et al., 2017) sets R to be the negative entropy of y to avoid collapsing to a deterministic policy.

#
,

(1)

2

where c is a numerical constant, n(x, a) is the number of
times that action a has been selected from state x during
search, and Q(x, a) is an estimate of the Q-function for
state-action pair (x, a) computed from search statistics and
using vθ for bootstrapping.
Intuitively, this selection criteria balances exploration and
exploitation, by selecting the most promising actions (high
Q-value Q(x, a) and prior policy πθ (a|x)) or actions that
have rarely been explored (small visit count n(x, a)). We
denote by Nsim the simulation budget, i.e., the search is
run with Nsim simulations. A more detailed presentation of
AlphaZero is in Appendix A; for a full description of the
algorithm, refer to Silver et al. (2017a).
2.2. Policy optimization
Policy optimization aims at finding a globally optimal policy πθ , generally using iterative updates. Each iteration
1

updates the current policy πθ by solving a local maximization problem of the form

We note here that terminologies such as prior follow Silver
et al. (2017a) and do not relate to concepts in Bayesian statistics.
2
Schrittwieser
P et al. (2019) uses a c that has a slow-varying
dependency on b n(x, b), which we omit here for simplicity, as
it was the case of Silver et al. (2017a).

3. MCTS as regularized policy optimization
In Section 2, we presented AlphaZero that relies on modelbased planning. We also presented policy optimization, a
framework that has achieved good performance in modelfree RL. In this section, we establish our main claim
namely that AlphaZero’s action selection criteria can be
interpreted as approximating the solution to a regularized
policy-optimization objective.
3.1. Notation
First, let us define the empirical visit distribution π̂ as
π̂(a|x) ,

1 + n(x, a)
P
·
|A| + b n(x, b)

(3)

Note that in Eq. 3, we consider an extra visit per action
compared to the acting policy and distillation target in the
original definition (Silver et al., 2016). This extra visit is
introduced for convenience in the upcoming analysis (to
avoid divisions by zero) and does not change the generality
of our results.
We also define the multiplier λN as
pP
b nb ,
P
λN (x) , c ·
|A| + b nb

(4)

## Page 3

MCTS as regularized policy optimization

where theP
shorthand notation na is used for n(x, a), and
N (x) , b nb denotes the number of visits to x during
search. With this notation, the action selection formula of
Eq. 1 can be written as selecting the action a? such that


πθ (a|x)
a? (x) , arg max Q(x, a) + λN ·
· (5)
π̂(a|x)
a
Note that in Eq. 5 and in the rest of the paper (unless otherwise specified), we use Q to denote the search Q-values,
i.e., those estimated by the search algorithm as presented
in Section 2.1. For more compact notation, we use bold
fonts to denote vector quantities, with the convention that
u[a]
u
v [a] = v[a] for two vectors u and v with the same dimension. Additionally, we omit the dependency of quantities
on state x when the context is clear. In particular, we use
q ∈ R|A| to denote the vector of search Q-function Q(x, a)
such that qa = Q(x, a). With this notation, we can rewrite
the action selection formula of Eq. 5 simply as3
h
πθ i
a? , arg max q + λN
·
(6)
π̂
3.2. A related regularized policy optimization problem
We now define π̄ as the solution to a regularized policy
optimization problem; we will see in the next subsection
that the visit distribution π̂ is a good approximation of π̄.
Definition 1 (π̄). Let π̄ be the solution to the following
objective
π̄ , arg max[qT y − λN KL[πθ , y]],

(7)

y∈S

where S is the |A|−dimensional simplex and KL is the
KL-divergence.4
We can see from Eq. 2 and Definition 1 that π̄ is the solution
to a policy optimization problem where Q is set to the search
Q-values, and the regularization term R is a reversed KLdivergence weighted by factor λN .
In addition, note that π̄ is as a smooth version of the arg max
associated to the search Q-values q. In fact, π̄ can be computed as (Appendix B.3 gives a detailed derivation of π̄)
πθ ,
π̄ = λN
α−q

Remark The factor λN is√a decreasing function of N .
Asymptotically, λN = Õ(1/ N ). Therefore, the influence
of the regularization term decreases as the number of simulation increases, which makes π̄ rely increasingly more on
search Q-values q and less on the policy prior πθ . As we
explain next, λN follows the design choice of AlphaZero,
and may be justified by a similar choice done in bandits
(Bubeck et al., 2012).
3.3. AlphaZero as policy optimization
We now analyze the action selection formula of AlphaZero
(Eq. 1). Interestingly, we show that this formula, which
was handcrafted5 independently of the policy optimization
research, turns out to result in a distribution π̂ that closely
relates to the policy optimization solution π̄.
The main formal claim of this section that AlphaZero’s
search policy π̂ tracks the exact solution π̄ of the regularized
policy optimization problem of Definition 1. We show that
Proposition 1 and Proposition 2 support this claim from two
complementary perspectives.
First, with Proposition 1, we show that π̂ approximately
follows the gradient of the concave objective for which π̄ is
the optimum.
Proposition 1. For any action a ∈ A, visit count n ∈ RA ,
policy prior πθ > 0 and Q-values q,


∂
?
T
a = arg max
(q π̂ − λN KL[πθ , π̂]) , (9)
∂na
a
with a? being the action realizing
Eq. 1 as defined in Eq. 5
P
and π̂ = (1 + n)/(|A| + b nb ) as defined in Eq. 3, is a
function of the count vector extended to real values.
The only thing that the search algorithm eventually influences through the tree search is the visit count distribution.
If we could do an infinitesimally small update, then the
greedy update maximizing Eq. 8 would be in the direction
of the partial derivative of Eq. 9. However, as we are restricted by a discrete update, then increasing the visit count
as in Proposition 1 makes π̂ track π̄. Below, we further
characterize the selected action a? and assume πθ > 0.
Proposition 2. The action a? realizing Eq. 1 is such that

(8)

where α ∈ R is such that π̄ is a proper probability vector. This is slightly different from the softmax distribution
obtained with KL[y, πθ ], which is written as


q
T
arg max[q y − λN KL[y, πθ ]] ∝ πθ exp
·
λN
y∈S
3
When the context is clear, we simplify for any x ∈ R|A| , that
arg max [x] , arg maxa {x[a], a ∈ A}.
P
x[a]
4
We apply the definition KL[x, y] , a x[a] log y[a]
·

π̂(a? |x) ≤ π̄(a? |x).

(10)

To acquire intuition from Proposition 2, note that once a?
is selected, its count na? increases and so does the total
count N . As a result, π̂(a? ) increases (in the order of
O(1/N )) and further approximates π̄(a? ). As such, Proposition 2 shows that the action selection formula encourages
5

Nonetheless, this heuristic could be interpreted as loosely
inspired by bandits (Rosin, 2011), but was adapted to accommodate
a prior term πθ .

## Page 4

MCTS as regularized policy optimization

the shape of π̂ to be close to that of π̄, until in the limit the
two distributions coincide.
Note that Proposition 1 and Proposition 2 are a special
case of a more general result that we formally prove in
Appendix D.1. In this particular case, the proof relies on
noticing that
"
#
pP
b nb
arg max qa + c · πθ (a) ·
(1)
1 + na
a



1
1
= arg max πθ (a) ·
−
·
(11)
π̂(a) π̄(a)
a
P
P
Then, since a π̂(a) = a π̄(a) and π̂ > 0 and π̄ > 0,
there exists at least one action for which 0 < π̂(a) ≤ π̄(a),
i.e., 1/π̂(a) − 1/π̄(a) ≥ 0.
To state a formal statement on π̂ approximating π̄, in Appendix D.3 we expand the conclusion under the assumption
that π̄ is a constant. In this case we can derive a bound
for the convergence rate of these two distributions as N
increases over the search,
|A| − 1 ,
||π̄ − π̂||∞ ≤
|A| + N

xi · y i is an f -divergence6

i

s
λUCT
N (x) , c ·

and

P
log b nb
P
·
|A| + b nb

 √ 
Similar to AlphaZero, λUCT
behaves7 as Õ 1/ N and
N
therefore the regularization gets weaker as N increases.
We can also derive tracking properties between π̄UCT and
the UCT empirical visit distribution π̂UCT as we did for
AlphaZero in the previous section, with Proposition 3; as
in the previous section, this is a special case of the general
result with any f -divergence in Appendix D.1.
Proposition 3. We have that

s
arg maxqa + c ·
a


P
log( b nb ) 
πθ (a) ·
1 + na

"
p
= arg max
πθ (a) ·

(12)

1
p

π̂(a)

1

!#

−p
π̄UCT (a)

and

3.4. Generalization to common MCTS algorithms
Besides AlphaZero, UCT (Kocsis and Szepesvári, 2006) is
another heuristic with a selection criteria inspired by UCB,
defined as


s
P
log( b nb ) 
·
(13)
arg maxqa + c ·
1 + na
a
Contrary to AlphaZero, the standard UCT formula does
not involve a prior policy. In this section, we consider a
slightly modified version of UCT with a (learned) prior πθ ,
as defined in Eq. 14. By setting the prior πθ to the uniform
distribution, we recover the original UCT formula,


s
P
log( b nb ) 
arg maxqa + c · πθ (a) ·
·
(14)
1 + na
a
Using the same reasoning as in Section 3.3, we now show
that this modified UCT formula also tracks the solution to a
regularized policy optimization problem, thus generalizing
our result to commonly used MCTS algorithms.
First, we introduce π̄UCT , which is tracked by the UCT visit
distribution, as:
y∈S

X√

a

with O(1/N ) matching the lowest possible approximation
error (see Appendix D.3) among discrete distributions of
the form (ki /N )i for ki ∈ N.

π̄UCT , arg max qT y − λUCT
N D(y, πθ ),

where D(x, y) , 2 − 2

(15)

a?UCT = arg max
a





∂
qT π̂ − λUCT
D[π
,
π̂]
.
θ
N
∂na

(16)

To sum up, similar to the previous section, we show that
UCT’s search policy π̂UCT tracks the exact solution π̄UCT of
the regularized policy optimization problem of Eq. 15.

4. Algorithmic benefits
In Section 3, we introduced a distribution π̄ as the solution to a regularized policy optimization problem. We then
showed that AlphaZero, along with general MCTS algorithms, select actions such that the empirical visit distribution π̂ actively approximates π̄. Building on this insight,
below we argue that π̄ is preferable to π̂, and we propose
three complementary algorithmic changes to AlphaZero.
4.1. Advantages of using π̄ over π̂
MCTS algorithms produce Q-values as a by-product of the
search procedure. However, MCTS does not directly use
search Q-values to compute the policy, but instead uses the
visit distribution π̂ (search Q-values implicitly influence π̂
by guiding the search). We postulate that this degrades the
performance especially at low simulation budgets Nsim for
several reasons:
6
In particular D(x, y) ≥ 0, D(x, y) = 0 =⇒ x = y and
D(x, y) is jointly convex in x and y (Csiszár, 1964; Liese and
Vajda, 2006).
7
We ignore logarithmic terms.

## Page 5

MCTS as regularized policy optimization

1. When a promising new (high-value) leaf is discovered,
many additional simulations might be needed before
this information is reflected in π̂; since π̄ is directly
computed from Q-values, this information is updated
instantly.
2. By definition (Eq. 3), π̂ is the ratio of two integers and
has limited expressiveness when Nsim is low, which
might lead to a sub-optimal policy; π̄ does not have
this constraint.
3. The prior πθ is trained against the target π̂, but the latter
is only improved for actions that have been sampled
at least once during search. Due to the deterministic
action selection (Eq. 1), this may be problematic for
certain actions that would require a large simulation
budget to be sampled even once.
The above downsides cause MCTS to be highly sensitive
to simulation budgets Nsim . When Nsim is high relative
to the branching factor of the tree search, i.e., number of
actions, MCTS algorithms such as AlphaZero perform well.
However, this performance drops significantly when Nsim
is low as showed by Hamrick et al. (2020); see also e.g.,
Figure 3.D. by Schrittwieser et al. (2019).
We illustrate the effect of simulation budgets in Figure 1,
where x-axis shows the budgets Nsim and y-axis shows the
episodic performance of algorithms applying π̂ vs. π̄; see
the details of these algorithms in the following sections. We
see that π̂ is highly sensitive to simulation budgets while π̄
performs consistently well across all budget values.
4.2. Proposed improvements to AlphaZero
We have pointed out potential issues due to π̂. We now
detail how to use π̄ as a replacement to resolve such issues.8
Appendix B.3 shows how to compute π̄ in practice.
ACT: acting with π̄ AlphaZero acts in the real environment by sampling actions according to a ∼ π̂(·|xroot ). Instead, we propose to to sample actions sampling according
to a ∼ π̄(·|xroot ). We label this variant as ACT.
S EARCH: searching with π̄ During search, we propose
to stochastically sample actions according to π̄ instead of
the deterministic action selection rule of Eq. 1. At each
node x in the tree, π̄(·) is computed with Q-values and total
visit counts at the node based on Definition 1. We label this
variant as S EARCH.
L EARN: learning with π̄ AlphaZero computes locally
improved policy with tree search and distills such improved
8

Recall that we have identified three issues. Each algorithmic
variant below helps in addressing issue 1 and 2. Furthermore, the
L EARN variant helps address issue 3.

Figure 1. Comparison of the score (median score over 3 seeds) of
MuZero (red: using π̂) and A LL (blue: using π̄) after 100k learner
steps as a function of Nsim on Cheetah Run of the Control Suite.

policy into πθ . We propose to use π̄ as the target policy
in place of π̂ to train our prior policy. As a result, the
parameters are updated as
h
i
θ ← θ − η∇θ Exroot KL[π̄(·|xroot ), πθ (·|xroot )] , (17)
where xroot is sampled from a prioritized replay buffer as in
AlphaZero. We label this variant as L EARN.
A LL: combining them all We refer to the combination
of these three independent variants as A LL. Appendix B
provides additional implementation details.
Remark Note that AlphaZero entangles search and learning, which is not desirable. For example, when the action selection formula changes, this impacts not only intermediate search results but also the root visit distribution
π̂(·|xroot ), which is also the learning target for πθ . However,
the L EARN variant partially disentangles these components.
Indeed, the new learning target is π̄(·|xroot ) which is computed from search Q-values, rendering it less sensitive to
e.g., the action selection formula.
4.3. Connections between AlphaZero and model-free
policy optimization.
Next, we make the explicit link between proposed algorithmic variants and existing policy optimization algorithms.
First, we provide two complementary interpretations.
L EARN as policy optimization For this interpretation,
we treat S EARCH as a blackbox, i.e., a subroutine that takes
a root node x and returns statistics such as search Q-values.
Recall that policy optimization (Eq. 2) maximizes the objective ≈ QTπθ y with the local policy y. There are many modelfree methods for the estimation of Qπθ , ranging from
P MonteCarlo estimates of cumulative returns Qπθ ≈ t≥0 γ t rt

## Page 6

MCTS as regularized policy optimization

(Schulman et al., 2015; 2017) to using predictions from a
Q-value critic Qπθ ≈ qθ trained with off-policy samples
(Abdolmaleki et al., 2018; Song et al., 2019). When solving π̄ for the update (Eq. 17), we can interpret L EARN as a
policy optimization algorithm using tree search to estimate
Qπθ . Indeed, L EARN could be interpreted as building a
Q-function9 critic qθ with a tree-structured inductive bias.
However, this inductive bias is not built-in a network architecture (Silver et al., 2017c; Farquhar et al., 2017; Oh
et al., 2017; Guez et al., 2018), but constructed online by an
algorithm, i.e., MCTS. Next, L EARN computes the locally
optimal policy π̄ to the regularized policy optimization objective and distills π̄ into πθ . This is exactly the approach
taken by MPO (Abdolmaleki et al., 2018).

S EARCH as policy optimization We now unpack the algorithmic procedure of the tree search, and show that it can
also be interpreted as policy optimization.
During the forward simulation phase of S EARCH, the action
at each node x is selected by sampling a ∼ π̄(·|x). As a
result, the full imaginary trajectory is generated consistently
according to policy π̄. During backward updates, each encountered node x receives a backup value from its child
node, which is an exact estimate of Qπ̄ (x, a). Finally, the
local policy π̄(·|x) is updated by solving the constrained optimization problem of Definition 1, leading to an improved
policy over previous π̄(·|x). Overall, with Nsim simulated
trajectories, S EARCH optimizes the root policy π̄(·|xroot )
and root search Q-values, by carrying out Nsim sequences of
MPO-style updates across the entire tree.10 A highly related
approach is to update local policies via policy gradients
(Anthony et al., 2019).
By combining the above two interpretations, we see that
the A LL variant is very similar to a full policy optimization
algorithm. Specifically, on a high level, A LL carries out
MPO updates with search Q-values. These search Q-values
are also themselves obtained via MPO-style updates within
the tree search. This paves the way to our major revelation
stated next.
Observation 1. A LL can be interpreted as regularized policy optimization. Further, since π̂ approximates π̄, AlphaZero and other MCTS algorithms can be interpreted as
approximate regularized policy optimization.
9

During search, because child nodes have fewer simulations
than the root, the Q-function estimate at the root slightly underestimates the acting policy Q-function.
10
Note that there are several differences from typical model-free
implementations of policy optimization: most notably, unlike a
fully-parameterized policy, the tree search policy is tabular at each
node. This also entails that the MPO-style distillation is exact.

Figure 2. Comparison of median scores of MuZero (red) and A LL
(blue) at Nsim = 5 (dotted line) and Nsim = 50 (solid line) simulations per step on Ms Pacman (Atari). Averaged across 8 seeds.

5. Experiments
In this section, we aim to address several questions: (1)
How sensitive are state-of-the-art hybrid algorithms such
as AlphaZero to low simulation budgets and can the A LL
variant provide a more robust alternative? (2) What changes
among ACT, S EARCH, and L EARN are most critical in this
variant performance? (3) How does the performance of the
A LL variant compare with AlphaZero in environments with
large branching factors?

Baseline algorithm Throughout the experiments, we take
MuZero (Schrittwieser et al., 2019) as the baseline algorithm. As a variant of AlphaZero, MuZero applies tree
search in learned models instead of real environments,
which makes it applicable to a wider range of problems.
Since MuZero shares the same search procedure as AlphaGo, AlphaGo Zero, and AlphaZero, we expect the performance gains to be transferable to these algorithms. Note
that the results below were obtained with a scaled-down
version of MuZero, which is described in Appendix B.1.

Hyper-parameters The hyper-parameters of the algorithms are tuned to achieve the maximum possible performance for baseline MuZero on the Ms Pacman level of the
Atari suite (Bellemare et al., 2013), and are identical in all
experiments with the exception of the number of simulations
per step Nsim .11 In particular, no further tuning was required
for the L EARN, S EARCH, ACT, and A LL variants, as was
expected from the theoretical considerations of Section 3.
11
The number of actors is scaled linearly with Nsim to maintain
the same total number of generated frames per second.

## Page 7

MCTS as regularized policy optimization

(a) Alien

(b) Asteroids

(a) 5 simulations
(c) Gravitar

(d) Seaquest

Figure 3. Comparison of median score (solid lines) over 6 seeds
of MuZero and A LL on four Atari games with Nsim = 50. The
shaded areas correspond to the range of the best and worst seed.
A LL (blue) performs consistently better than MuZero (red).

5.1. Search with low simulation budgets
Since AlphaZero solely relies on the π̂ for training targets, it
may misbehave when simulation budgets Nsim are low. On
the other hand, our new algorithmic variants might perform
better in this regime. To confirm these hypotheses, we
compare the performance of MuZero and the A LL variant
on the Ms Pacman level of the Atari suite at different levels
of simulation budgets.
Result In Figure 2, we compare the episodic return of A LL
vs. MuZero averaged over 8 seeds, with a simulation budget
Nsim = 5 and Nsim = 50 for an action set of size |A| ≤ 18;
thus, we consider that Nsim = 5 and Nsim = 50 respectively
correspond to a low and high simulation budgets relative
to the number of actions. We make several observations:
(1) At a relatively high simulation budget, Nsim = 50, same
as Schrittwieser et al. (2019), both MuZero and A LL exhibit reasonably close levels of performance; though A LL
obtains marginally better performance than MuZero; (2) At
low simulation budget, Nsim = 5, though both algorithms
suffer in performance relative to high budgets, A LL significantly outperforms MuZero both in terms of learning speed
and asymptotic performance; (3) Figure 6 in Appendix C.1
shows that this behavior is consistently observed at intermediate simulation budgets, with the two algorithms starting
to reach comparable levels of performance when Nsim ≥ 24
simulations. These observations confirm the intuitions from
Section 3. (4) We provide results on a subset of Atari games
in Figure 3, which show that the performance gains due to π̄
over π̂ are also observed in other levels than Ms Pacman; see
Appendix C.2 for results on additional levels. This subset

(b) 50 simulations
Figure 4. Ablation study at 5 and 50 simulations per step on Ms
Pacman (Atari); average across 8 seeds.

of levels are selected based on the experiment setup in Figure S1 of Schrittwieser et al. (2019). Importantly, note that
the performance gains of A LL are consistently significant
across selected levels, even at a higher simulation budget of
Nsim = 50.
5.2. Ablation study
To better understand which component of the A LL contributes the most to the performance gains, Figure 4 presents
the results of an ablation study where we compare individual
component L EARN, S EARCH, or ACT.
Result The comparison is shown in Figure 4, we make
several observations: (1) At Nsim = 5 (Figure 4a), the main
improvement comes from using the policy optimization solution π̄ as the learning target (L EARN variant); using π̄ during
search or acting leads to an additional marginal improvement; (2) Interestingly, we observe a different behavior at
Nsim = 50 (Figure 4b). In this case, using π̄ for learning or
acting does not lead to a noticeable improvement. However,

## Page 8

MCTS as regularized policy optimization

the superior performance of A LL is mostly due to sampling
according to π̄ during search (S EARCH).
The improved performance when using π̄ as the learning
target (L EARN) illustrates the theoretical considerations of
Section 3: at low simulation budgets, the discretization
noise in π̂ makes it a worse training target than π̄, but this
advantage vanishes when the number of simulations per
step increases. As predicted by the theoretical results of Section 3, learning and acting using π̄ and π̂ becomes equivalent
when the simulation budget increases.
On the other hand, we see a slight but significant improvement when sampling the next node according to π̄ during
search (S EARCH) regardless of the simulation budget. This
could be explained by the fact that even at high simulations
budget, the S EARCH modification also affect deeper node
that have less simulations.

Figure 5. Comparison of the median score over 3 seeds of MuZero
(red) and A LL (blue) at 4 (dotted) and 50 (solid line) simulations
per step on Cheetah Run (Control Suite).

5.3. Search with large action space – continuous control
The previous results confirm the intuitions presented in
Sections 3 and 4; namely, the A LL variation greatly improves performance at low simulation budgets, and obtain
marginally higher performance at high simulation budgets.
Since simulation budgets are relative to the number of action,
these improvements are critical in tasks with a high number
of actions, where MuZero might require a prohibitively high
simulation budgets; prior work (Dulac-Arnold et al., 2015;
Metz et al., 2017; Van de Wiele et al., 2020) has already
identified continuous control tasks as an interesting testbed.
Benchmarks We select high-dimensional environments
from DeepMind Control Suite (Tassa et al., 2018). The
observations are images and action space A = [−1, 1]m
with m dimensions. We apply an action discretization
method similar to that of Tang and Agrawal (2019). In
short, for a continuous action space m dimensions, each dimension is discretized into K evenly spaced atomic actions.
With proper parameterization of the policy network (see,
e.g., Appendix B.2), we can reduce the effective branching
factor to Km  K m , though this still results in a much
larger action space than Atari benchmarks. In Appendix C.2,
we provide additional descriptions of the tasks.
Result In Figure 5, we compare MuZero with the A LL
variant on the CheetahRun environment of the DeepMind
Control Suite (Tassa et al., 2018). We evaluate the performance at low (Nsim = 4), medium (Nsim = 12) and “high”
(Nsim = 50) simulation budgets, for an effective action
space of size 30 (m = 6, K = 5). The horizontal line
corresponds to the performance of model-free D4PG also
trained on pixel observations (Barth-Maron et al., 2018),
as reported in (Tassa et al., 2018). Appendix C.2 provides
experimental results on additional tasks. We again observe

that A LL outperforms the original MuZero at low simulation
budgets and still achieves faster convergence to the same
asymptotic performance with more simulations. Figure 1
compares the asymptotic performance of MuZero and A LL
as a function of the simulation budget at 100k learner steps.

6. Conclusion
In this paper, we showed that the action selection formula
used in MCTS algorithms, most notably AlphaZero, approximates the solution to a regularized policy optimization problem formulated with search Q-values. From this theoretical
insight, we proposed variations of the original AlphaZero
algorithm by explicitly using the exact policy optimization
solution instead of the approximation. We show experimentally that these variants achieve much higher performance
at low simulation budget, while also providing statistically
significant improvements when this budget increases.
Our analysis on the behavior of model-based algorithms
(i.e., MCTS) has made explicit connections to model-free
algorithms. We hope that this sheds light on new ways of
combining both paradigms and opens doors to future ideas
and improvements.
Acknowledgements The authors would like to thank Alaa
Saade, Bernardo Avila Pires, Bilal Piot, Corentin Tallec, Daniel Guo, David Silver, Eugene Tarassov, Florian
Strub, Jessica Hamrick, Julian Schrittwieser, Katrina McKinney, Mohammad Gheshlaghi Azar, Nathalie Beauguerlange, Pierre Ménard, Shantanu Thakoor, Théophane Weber,
Thomas Mesnard, Toby Pohlen and the rest of the DeepMind team.

## Page 9

MCTS as regularized policy optimization

References
Abdolmaleki, A., Springenberg, J. T., Tassa, Y., Munos,
R., Heess, N., and Riedmiller, M. (2018). Maximum a posteriori policy optimisation. arXiv preprint
arXiv:1806.06920.
Andrychowicz, O. M., Baker, B., Chociej, M., Jozefowicz,
R., McGrew, B., Pachocki, J., Petron, A., Plappert, M.,
Powell, G., Ray, A., et al. (2020). Learning dexterous inhand manipulation. The International Journal of Robotics
Research, 39(1):3–20.
Anthony, T., Nishihara, R., Moritz, P., Salimans, T., and
Schulman, J. (2019). Policy gradient search: Online
planning and expert iteration without search trees. arXiv
preprint arXiv:1904.03646.
Auer, P. (2002). Using confidence bounds for exploitationexploration trade-offs. Journal of Machine Learning
Research, 3(Nov):397–422.
Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney, W.,
Horgan, D., TB, D., Muldal, A., Heess, N., and Lillicrap,
T. (2018). Distributional policy gradients. In International Conference on Learning Representations.
Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M.
(2013). The arcade learning environment: An evaluation platform for general agents. Journal of Artificial
Intelligence Research, 47:253–279.
Boyd, S. and Vandenberghe, L. (2004). Convex optimization.
Cambridge university press.
Browne, C. B., Powley, E., Whitehouse, D., Lucas, S. M.,
Cowling, P. I., Rohlfshagen, P., Tavener, S., Perez, D.,
Samothrakis, S., and Colton, S. (2012). A survey of
monte carlo tree search methods. IEEE Transactions on
Computational Intelligence and AI in games, 4(1):1–43.
Bubeck, S., Cesa-Bianchi, N., et al. (2012). Regret analysis
of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends R in Machine Learning,
5(1):1–122.
Csiszár, I. (1964). Eine informationstheoretische ungleichung und ihre anwendung auf beweis der ergodizitaet
von markoffschen ketten. Magyer Tud. Akad. Mat. Kutato
Int. Koezl., 8:85–108.
Dhariwal, P., Hesse, C., Klimov, O., Nichol, A., Plappert,
M., Radford, A., Schulman, J., Sidor, S., Wu, Y., and
Zhokhov, P. (2017). Openai baselines.
Dulac-Arnold, G., Evans, R., van Hasselt, H., Sunehag, P.,
Lillicrap, T., Hunt, J., Mann, T., Weber, T., Degris, T., and
Coppin, B. (2015). Deep reinforcement learning in large
discrete action spaces. arXiv preprint arXiv:1512.07679.

Farquhar, G., Rocktäschel, T., Igl, M., and Whiteson,
S. (2017). TreeQN and ATreeC: Differentiable treestructured models for deep reinforcement learning. arXiv
preprint arXiv:1710.11417.
Fox, R., Pakman, A., and Tishby, N. (2015). Taming the
noise in reinforcement learning via soft updates. arXiv
preprint arXiv:1512.08562.
Geist, M., Scherrer, B., and Pietquin, O. (2019). A theory
of regularized markov decision processes. arXiv preprint
arXiv:1901.11275.
Google (2020).
Cloud TPU — Google Cloud.
https://cloud.google.com/tpu/.
Grill, J.-B., Domingues, O. D., Ménard, P., Munos, R.,
and Valko, M. (2019). Planning in entropy-regularized
Markov decision processes and games. In Neural Information Processing Systems.
Guez, A., Weber, T., Antonoglou, I., Simonyan, K.,
Vinyals, O., Wierstra, D., Munos, R., and Silver, D.
(2018). Learning to search with mctsnets. arXiv preprint
arXiv:1802.04697.
Haarnoja, T., Tang, H., Abbeel, P., and Levine, S. (2017).
Reinforcement learning with deep energy-based policies.
In Proceedings of the 34th International Conference on
Machine Learning-Volume 70, pages 1352–1361. JMLR.
org.
Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Pfaff,
T., Weber, T., Buesing, L., and Battaglia, P. W. (2020).
Combining Q-learning and search with amortized value
estimates. In International Conference on Learning Representations.
He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep
residual learning for image recognition. In Proceedings
of the IEEE conference on computer vision and pattern
recognition, pages 770–778.
Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel,
M., van Hasselt, H., and Silver, D. (2018). Distributed
prioritized experience replay. In International Conference
on Learning Representations.
Kocsis, L. and Szepesvári, C. (2006). Bandit based MonteCarlo planning. In European conference on machine
learning, pages 282–293. Springer.
Levine, S. (2018). Reinforcement learning and control
as probabilistic inference: Tutorial and review. arXiv
preprint arXiv:1805.00909.
Liese, F. and Vajda, I. (2006). On divergences and informations in statistics and information theory. IEEE Transactions on Information Theory, 52(10):4394–4412.

## Page 10

MCTS as regularized policy optimization

Metz, L., Ibarz, J., Jaitly, N., and Davidson, J. (2017). Discrete sequential prediction of continuous actions for deep
rl. arXiv preprint arXiv:1705.05035.
Neu, G., Jonsson, A., and Gómez, V. (2017). A unified view
of entropy-regularized Markov decision processes. arXiv
preprint arXiv:1705.07798.
O’Donoghue, B., Munos, R., Kavukcuoglu, K., and Mnih,
V. (2016). Combining policy gradient and Q-learning.
arXiv preprint arXiv:1611.01626.
Oh, J., Singh, S., and Lee, H. (2017). Value prediction
network. In Advances in Neural Information Processing
Systems, pages 6118–6128.
Rosin, C. D. (2011). Multi-armed bandits with episode context. Annals of Mathematics and Artificial Intelligence,
61(3):203–230.
Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan,
K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., et al. (2019). Mastering Atari, go,
chess and shogi by planning with a learned model. arXiv
preprint arXiv:1911.08265.
Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz,
P. (2015). Trust region policy optimization. In International conference on machine learning, pages 1889–1897.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L.,
Van Den Driessche, G., Schrittwieser, J., Antonoglou, I.,
Panneershelvam, V., Lanctot, M., et al. (2016). Mastering
the game of go with deep neural networks and tree search.
nature, 529(7587):484.
Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai,
M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., et al. (2017a). Mastering chess and shogi by
self-play with a general reinforcement learning algorithm.
arXiv preprint arXiv:1712.01815.
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou,
I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M.,
Bolton, A., et al. (2017b). Mastering the game of go
without human knowledge. Nature, 550(7676):354–359.
Silver, D., van Hasselt, H., Hessel, M., Schaul, T., Guez, A.,
Harley, T., Dulac-Arnold, G., Reichert, D., Rabinowitz,
N., Barreto, A., et al. (2017c). The predictron: End-toend learning and planning. In Proceedings of the 34th
International Conference on Machine Learning-Volume
70, pages 3191–3199. JMLR. org.

Song, H. F., Abdolmaleki, A., Springenberg, J. T., Clark, A.,
Soyer, H., Rae, J. W., Noury, S., Ahuja, A., Liu, S., Tirumala, D., et al. (2019). V-MPO: On-policy maximum a
posteriori policy optimization for discrete and continuous
control. arXiv preprint arXiv:1909.12238.
Sutton, R. S. and Barto, A. G. (1998). Reinforcement Learning: An Introduction. MIT Press.
Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour,
Y. (2000). Policy gradient methods for reinforcement
learning with function approximation. In Advances in
neural information processing systems, pages 1057–1063.
Tang, Y. and Agrawal, S. (2019). Discretizing continuous
action space for on-policy optimization. arXiv preprint
arXiv:1901.10500.
Tassa, Y., Doron, Y., Muldal, A., Erez, T., Li, Y., Casas, D.
d. L., Budden, D., Abdolmaleki, A., Merel, J., Lefrancq,
A., et al. (2018). DeepMind control suite. arXiv preprint
arXiv:1801.00690.
Van de Wiele, T., Warde-Farley, D., Mnih, A., and Mnih,
V. (2020). Q-learning in enormous action spaces via
amortized approximate maximization. arXiv preprint
arXiv:2001.08116.
Williams, R. J. (1992). Simple statistical gradient-following
algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229–256.
Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy.
PhD thesis, Carnegie Mellon University, USA.
