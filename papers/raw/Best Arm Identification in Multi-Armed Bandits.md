## Page 1

Best Arm Identification in Multi-Armed Bandits

Jean-Yves Audibert
Imagine, Université Paris Est
&
Willow, CNRS/ENS/INRIA, Paris, France
audibert@imagine.enpc.fr

Sébastien Bubeck, Rémi Munos
SequeL Project, INRIA Lille
40 avenue Halley,
59650 Villeneuve d’Ascq, France
{sebastien.bubeck, remi.munos}@inria.fr

Abstract
We consider the problem of finding the best arm in a stochastic multi-armed bandit game. The
regret of a forecaster is here defined by the gap between the mean reward of the optimal arm
and the mean reward of the ultimately chosen arm. We propose a highly exploring UCB policy
and a new algorithm based on successive rejects. We show that these algorithms are essentially
optimal since their regret decreases exponentially at a rate which is, up to a logarithmic factor, the
best possible. However, while the UCB policy needs the tuning of a parameter depending on the
unobservable hardness of the task, the successive rejects policy benefits from being parameter-free,
and also independent of the scaling of the rewards. As a by-product of our analysis, we show
that identifyingP
the best arm (when it is unique) requires a number of samples of order (up to a
log(K) factor) i 1/∆2i , where the sum is on the suboptimal arms and ∆i represents the difference
between the mean reward of the best arm and the one of arm i. This generalizes the well-known
fact that one needs of order of 1/∆2 samples to differentiate the means of two distributions with
gap ∆.

1

Introduction

In the multi-armed bandit problem, at each stage, an agent (or forecaster) chooses one action (or arm), and
receives a reward from it. In its stochastic version, the reward is drawn from a fixed probability distribution
given the arm. The usual goal is to maximize the cumulative sum of rewards, see Robbins (1952); Auer et al.
(2002) among many others. Since the forecaster does not know the distributions, he needs to explore (try)
the different actions and yet, exploit (concentrate its draws on) the seemingly most rewarding arms. In this
paper, we adopt a different viewpoint. We assume that after a given number of pulls, the forecaster is asked to
output a recommended arm. He is then only evaluated by the average payoff of his recommended arm. This
is the so-called pure exploration problem, Bubeck et al. (2009).
The distinguishing feature from the classical multi-armed bandit problem described above is that the
exploration phase and the evaluation phase are separated. Thus, there is no explicit trade-off between the
exploration and the exploitation while pulling the arms. The target of Hoeffding and Bernstein races, see
Maron and Moore (1993); Mnih et al. (2008) among others, is more similar to ours. However, instead of
trying to extract from a fixed number of rounds the best action, racing algorithms try to identify the best
action at a given confidence level while consuming the minimal number of pulls. They optimize the budget
for a given confidence level, instead of optimizing the quality of the recommendation for a given budget
size. Another variant of the best arm identification task is the problem of minimal sampling times required to
identify an ε-optimal arm, see in particular Domingo et al. (2002) and Even-Dar et al. (2006).
We now illustrate why this is a natural framework for numerous applications. Historically, the first occurrence of multi-armed bandit problems was given by medical trials, see Robbins (1952). In the case of
a severe disease, ill patients only are included in the trial and the cost of picking the wrong treatment is
high. It is important to minimize the cumulative regret, since the test and cure phases coincide. However,
for cosmetic products, there exists a test phase separated from the commercialization phase, and one aims
at minimizing the regret of the commercialized product rather than the cumulative regret in the test phase,
which is irrelevant.
Another motivating example concerns channel allocation for mobile phone communications. During a
very short time before the communication starts, a cellphone can explore the set of channels to find the
best one to operate. Each evaluation of a channel is noisy and there is a limited number of evaluations

## Page 2

Parameters available to the forecaster: the number of rounds n and the number of arms K.
Parameters unknown to the forecaster: the reward distributions ν1 , . . . , νK of the arms.
For each round t = 1, 2, . . . , n;
(1) the forecaster chooses It ∈ {1, . . . , K},
(2) the environment draws the reward XIt ,TIt (t) from νIt and independently of the past
given It .
At the end of the n rounds, the forecaster outputs a recommendation Jn ∈ {1, . . . , K}.

Figure 1: The pure exploration problem for multi-armed bandits.
before the communication starts. The connection is then launched on the channel which is believed to be the
best. Opportunistic communication systems rely on the same idea. Again the cumulative regret during the
exploration phase is irrelevant since the user is only interested in the quality of its communication starting
after the exploration phase.
More generally, the pure exploration problem addresses the design of strategies making the best possible
use of available resources in order to optimize the performance of some decision-making task. That is, it
occurs in situations with a preliminary exploration phase in which costs are not measured in terms of rewards
but rather in terms of resources that come in limited budget (the number of patients in the test phase in the
clinical trial setting and the time to connect in the communication example).

2

Problem setup

A stochastic multi-armed bandit game is parameterized by the number of arms K, the number of rounds (or
budget) n, and K probability distributions ν1 , . . . , νK associated respectively with arm 1, . . . , arm K. These
distributions are unknown to the forecaster. For t = 1, . . . , n, at round t, the forecaster chooses an arm It
in the set of arms {1, . . . , K}, and observes a reward drawn from νIt independently from the past (actions
and observations). At the end of the n rounds, the forecaster selects an arm, denoted Jn , and is evaluated in
terms of the difference between the mean reward of the optimal arm and the mean reward of Jn . Precisely, let
µ1 , . . . , µK be the respective means of ν1 , . . . , νK . Let µ∗ = maxk∈{1,...,K} µk . The regret of the forecaster
is
rn = µ∗ − µJn .
For sake of simplicity, we will assume that the rewards are in [0, 1] and that there is a unique optimal arm.
Let i∗ denote this arm (so, µi∗ = µ∗ ). For i 6= i∗ , we introduce the following suboptimality measure of arm i:
∆i = µ∗ − µi .
For reasons that will be obvious later, we also define ∆i∗ as the minimal gap
∆i∗ = min∗ ∆i .
i6=i

We introduce the notation (i) ∈ {1, . . . , K} to denote the i–th best arm (with ties break arbitrarily), hence
∆i∗ = ∆(1) = ∆(2) ≤ ∆(3) ≤ . . . ≤ ∆(K) .
Let en denote the probability of error, that is the probability that the recommendation is not the optimal one:
en = P(Jn 6= i∗ ).
We have Ern =

P

i6=i∗ P(Jn = i)∆i , and consequently

∆i∗ en ≤ Ern ≤ en .
As a consequence of this equation, up to a second order term, en and Ern behave similarly, and it does not
harm to focus on the probability en .
For each arm i and all time rounds t ≥ 1, we denote by Ti (t) the number of times arm i was pulled
bi,s =
from rounds 1 to t, and by Xi,1 , Xi,2 , . . . , Xi,Ti (t) the sequence of associated rewards. Introduce X

## Page 3

1
s

Ps

t=1 Xi,t the empirical mean of arm i after s pulls. In the following, the symbol c will denote a positive
numerical constant which may differ from line to line.
The goal of this work is to propose allocation strategies with small regret, and possibly as small as the best
allocation strategy which would know beforehand the distributions ν1 , . . . , νK up to a permutation. Before
going further, note that the goal is unachievable for all distributions ν1 , . . . , νK : a policy cannot perform
as well as the “oracle” allocation strategy in every particular cases. For instance, when the supports of
ν1 , . . . , νK are disjoint, the oracle forecaster almost surely identifies an arm by a single draw of it. As a
consequence, it has almost surely zero regret for any n ≥ K. The generic policy which does not have any
knowledge on the K distributions cannot reproduce this performance for any K-tuple of disjointly supported
distributions. In this work, the above goal of deciding as well as an oracle will be reached for the set of
Bernoulli distributions with parameters in (0, 1), but the algorithms are defined for any distributions supported
in [0, 1].
We would like to mention that the case K = 2 is unique and simple since, as we will indirectly see,
it is optimally solved by the uniform allocation strategy consisting in drawing each arm n/2 times (up to
rounding problem), and at the end recommending the arm with the highest empirical mean. Therefore, our
main contributions concern more the problem of the budget allocation when K ≥ 3. The hardness of the task
will be characterized by the following quantities

H1 =

K
X
1

∆2i
i=1

and

H2 =

max
i∈{1,...,K}

i∆−2
(i) .

These quantities are equivalent up to a logarithmic factor since we have (see Section 6.1)
H2 ≤ H1 ≤ log(2K)H2 .
(1)
Intuitively, we will show that these quantities are indeed characteristic of the hardness of the problem, in the
sense that they give the order of magnitude of the number of samples required to find the best arm with a
reasonable probability. This statement will be made precise in the rest of the paper, in particular through
Theorem 2 and Theorem 4.
Outline. In Section 3, we propose a highly exploring policy based on upper confidence bounds, called
UCB-E (Upper Confidence Bound Exploration), in the spirit of UCB1 Auer et al. (2002). We prove that
this algorithm, provided
that it is appropriately tuned, has an upper bound on the probability of error en of

order exp − c Hn1 . The core problem of this policy is the tuning of the parameter. The optimal value of
the parameter depends on H1 , which has no reason to be known beforehand by the forecaster, and which, to
our knowledge, cannot be estimated from past observations with sufficiently high confidence in order that the
resulting algorithm still satisfies a similar bound on en .
To get round this limitation, in Section 4, we propose a simple new policy called SR (Successive Rejects)
that progressively rejects the arms which seem to be suboptimal.
This algorithm is parameter-free and its

n
probability of error en is at most of order exp − log(2K)H
. Since H2 ≤ H1 ≤ log(2K)H2 , up to at most
2
a logarithmic term in K, the algorithm performs as well as UCB-E while not requiring the knowledge of H1 .
In Section 5, we prove that H1 and H2 truly represent the hardness of the problem (up to a logarithmic
factor). Precisely, we consider a forecaster which knows the reward distributions of the arms up to a permutation. When these distributions are of Bernoulli type with parameter in [p, 1 − p] for some p > 0, there
exists a permutation of the 
distributions for which the probability of error of the (oracle) forecaster is lower
cn
bounded by exp − p(1−p)H
.
2
Section 6 gathers some of the proofs. Section 7 provides some experiments testing the efficiency of the
proposed policies and enlightening our theoretical results. We also discuss a modification of UCB-E where
we perform a non-trivial online estimation of H1 . We conclude in Section 8.
Example. To put in perspective the results
 we just mentioned, letus consider a specific example with
Bernoulli distributions. Let ν1 = Ber 21 , and νi = Ber 12 − K1i for i ∈ {2, . . . , K}. Here, one can
easily check that H2 = 2K 2K
 . Thus, in this case, the probability of missing the best arm of SR is at most of
n
order exp − 2 log(2K)K
2K . Moreover, in Section 5, we prove that there does not exist any forecaster (even
with the knowledge
 of the distributions up to a permutation) with a probability of missing the best arm smaller
than exp − K11n
for infinitely many n. Thus, our analysis finds that, for this particular reward distributions,
2K
the number of samples required to find the best arm is at least (of order of) K 2K , and SR actually finds it
with (of order of) log(K)K 2K samples.

3

Highly exploring policy based on upper confidence bounds

In this section, we propose and study the algorithm UCB-E described in Figure 2. When a is taken of
order log n, the algorithm essentially corresponds to the UCB1 policy introduced in Auer et al. (2002), and

## Page 4

Parameter: exploration parameter a > 0.
p
bi,s + a for s ≥ 1 and Bi,0 = +∞.
For i ∈ {1, . . . , K}, let Bi,s = X
s
For each round t = 1, 2, . . . , n:
Draw It ∈ argmaxi∈{1,...,K} Bi,Ti (t−1) .
bi,T (n) .
Let Jn ∈ argmaxi∈{1,...,K} X
i

Figure 2: UCB-E (Upper Confidence Bound Exploration) algorithm.
its cumulative regret is of order log n. Bubeck et al. (2009) have shown that algorithms having at most
logarithmic cumulative regret, have at least a (non-cumulative) regret of order n−γ for some γ > 0. So taking
a of order log n is inappropriate to reach exponentially small probability of error. For our regret notion, one
has to explore much more and typically use a parameter which is essentially linear in n. Precisely, we have
the following result, the proof of which can be found in Section 6.2.
n−K
Theorem 1 If UCB-E is run with parameter 0 < a ≤ 25
36 H1 , then it satisfies


2a
en ≤ 2nK exp −
.
25

n−K
n−K
In particular for a = 25
36 H1 , we have en ≤ 2nK exp − 18H1 .

The theorem shows that the probability of error of UCB-E is at most of order exp(−ca) for a ≥ log n.
n−K
In fact, one can easily show a corresponding lower bound. In view of this, as long as a ≤ 25
36 H1 , we can
essentially say: the more we explore (i.e., the larger a is), the smaller the regret is. Besides, the smallest upper
bound on the probability of error is obtained for a of order n/H1 , and is therefore exponentially decreasing
with n. The constant H1 depends not only on how close the mean rewards of the two best arms are, but
also on the number of arms and how close their mean reward is to the optimal mean reward. This constant
should be seen as the order of the minimal number n for which the recommended arm is the optimal one with
high probability. In Section 5, we will show that H1 is
 indeed a good measure of the hardness of the task by
cn
showing that no forecaster satisfies en ≤ exp − H
for any distributions ν1 , . . . , νK , where we recall that
2
H2 satisfies H2 ≤ H1 ≤ log(2K)H2 .
One interesting
probability at least 1 −
 message to take from the proof of Theorem 1 is that, with −2
2nK exp − 2a
25 , the number of draws of any suboptimal arm i is of order a∆i . This means that the
optimal arm will be played at least n − caH1 , showing that for too small a, UCB-E “exploits” too much
in view of our regret target. Theorem 1 does not specify how the algorithm performs when a is larger than
25 n−K
36 H1 . Nevertheless, similar arguments than the ones in the proof show that for large a, with high probability, only low rewarding arms are played of order a∆−2
times, whereas the best ones are all drawn the same
i
number of times up to a constant factor. The number of these similarly drawn arms grows with a. In the limit,
when a goes to infinity, UCB-E is exactly the uniform allocation strategy studied in Bubeck et al. (2009).
In general1 , the uniform allocation has a probability of error which can be lower and upper bounded by a
n∆2 
n−K
quantity of the form exp − c Ki∗ . It consequently performs much worse than UCB-E for a = 25
36 H1 ,
−2
since H1 ≤ K∆−2
i∗ , and potentially H1  K∆i∗ for very large number of arms with heterogeneous mean
rewards.
One straightforward idea to cope with the absence of an oracle telling us the value of H1 would be to
estimate online the parameter H1 and use this estimation in the algorithm. Unfortunately, we were not able
to prove, and do not believe that, this modified algorithm generally attains the expected rate of convergence.
Indeed, overestimating H1 leads to low exploring, and in the event when the optimal arm has given abnormally low rewards, the arm stops being drawn by the policy, its estimated mean reward is thus not corrected,
and the arm is finally not recommended by the policy. On the contrary, underestimating H1 leads to draw
too much the suboptimal arms, precluding a sufficiently accurate estimation of the mean rewards of the best
arms. For this last case, things are in fact much more subtle than what can be retranscribed in these few lines,
1

We say “in general” to rule out some trivial cases (like when the reward distributions are all Dirac distributions) in
which the probability of error en would be much smaller.

## Page 5

P
1
Let A1 = {1, . . . , K}, log(K) = 12 + K
i=2 i , n0 = 0 and for k ∈ {1, . . . , K − 1},


1
n−K
nk =
.
log(K) K + 1 − k
For each phase k = 1, 2, . . . , K − 1:
(1) For each i ∈ Ak , select arm i for nk − nk−1 rounds.
bi,n (we only remove one element from Ak , if there
(2) Let Ak+1 = Ak \ arg mini∈Ak X
k
is a tie, select randomly the arm to dismiss among the worst arms).
Let Jn be the unique element of AK .

Figure 3: SR (Successive Rejects) algorithm.
and we notice that keeping track of a lower bound on H1 would lead to the correct rate only under appropriate assumptions on the decrease of the sequence ∆(k) , k ∈ {1, . . . , K}. In Section 7 we push this idea and
propose a way to estimate online H1 , however we solely justify the corresponding algorithm by experiments.
In the next section we propose an algorithm which does not suffer from these limitations.

4

Successive Rejects algorithm

In this section, we describe and analyze a new algorithm, SR (Successive Rejects), see Figure 3 for its precise
description. Informally it proceeds as follows. First the algorithm divides the time (i.e., the n rounds) in
K − 1 phases. At the end of each phase, the algorithm dismisses the arm with the lowest empirical mean.
During the next phase, it pulls equally often each arm which has not been dismissed yet. The recommended
arm Jn is the last surviving arm. The length of the phases are carefully chosen toobtain an optimal
(up to

1
n−K
a logarithmic factor) convergence rate. More precisely, one arm is pulled n1 = log(K)
times,
one
K
 1 n−K 
 1 n−K 
n2 = log(K) K−1 times, ..., and two arms are pulled nK−1 = log(K) 2
times. SR does not exceed
PK 1
1
the budget of n pulls, since, from the definition log(K) = 2 + i=2 i , we have
!
K−1
n−K 1 X
1
n1 + . . . + nK−1 + nK−1 ≤ K +
+
= n.
K +1−k
log(K) 2
k=1

For K = 2, up to rounding effects, SR is just the uniform allocation strategy.
Theorem 2 The probability of error of SR satisfies


n−K
K(K − 1)
exp −
.
en ≤
2
log(K)H2
Proof: We can assume that the sequence of rewards for each arm is drawn before the beginning of the game.
Thus the empirical reward for arm i after s pulls is well defined even if arm i has not been actually pulled s
times. During phase k, at least one of the k worst arms is surviving. So, if the optimal arm i∗ is dismissed
bi∗ ,n ≤ maxi∈{(K),(K−1),...,(K+1−k)} X
bi,n . By a union bound and
at the end of phase k, it means that X
k
k
Hoeffding’s inequality, the probability of error en = P(AK 6= {i∗ }) thus satisfies
en ≤

K−1
X

K
X

bi∗ ,n ≤ X
b(i),n )
P(X
k
k

k=1 i=K+1−k

≤

K−1
X

K
X

b(i),n − µ(i) + µ∗ − X
bi∗ ,n ≥ ∆(i) )
P(X
k
k

k=1 i=K+1−k

≤

K−1
X

K
X

k=1 i=K+1−k


 K−1


X
exp −nk ∆2(i) ≤
k exp −nk ∆2(K+1−k) .
k=1

## Page 6

We conclude the proof by noting that by definition of nk and H2 , we have
nk ∆2(K+1−k) ≥

n−K
1
n−K
≥
.
log(K) (K + 1 − k)∆−2
log(K)H
2
(K+1−k)

(2)

The following theorem provides a deeper understanding of how SR works. It lower bounds the sampling
times of the arms and shows that at the end of phase k, we have a high-confidence estimation of ∆(K+1−k)
up to numerical constant factor. This intuition will prove to be useful in Section 7, see in particular Figure 4.

3
n−K
Theorem 3 With probability at least 1 − K2 exp − 4log(K)H
, for any arm j, we have
2

n−K
.
(3)
4log(K)H2 ∆2j

n−K
With probability at least 1 − K 3 exp − 32log(K)H
, for any k ∈ {1, . . . , K − 1}, the dismissed arm
2
`k = Ak+1 \ Ak at the end of phase k satisfies
Tj (n) ≥

1
1
bm,n − X
b` ,n ≤ 3 ∆` ≤ 3∆(K+1−k) .
∆(K+1−k) ≤ ∆`k ≤ max X
k
k
k
m∈Ak
4
2
2 k

(4)

Proof: We consider the event E on which for any k ∈ {1, . . . , K − 1}, for any arm ` in the worst k arms, and
any arm j such that 2∆j ≤ ∆` , we have
bj,n − X
b`,n > 0.
X
k
k

3
n−K
, since, from Hoeffding’s inequality, a
This event holds with probability at least 1 − K2 exp − 4log(K)H
2
union bound and (2), we have


K−1
X
X
bj,n − X
b`,n ≤ 0
P X
k

k

k=1 `∈{(K),(K−1),...,(K+1−k)}
j:2∆j ≤∆`

≤

K−1
X

X

exp − nk (∆` − ∆j )2



k=1 `∈{(K),(K−1),...,(K+1−k)}
j:2∆j ≤∆`

≤

K−1
X
k=1

kK exp − nk

∆2(K+1−k) 
4

K3
≤
exp
2




n−K
−
.
4log(K)H2

During phase k, at least one of the k worst arms is surviving. On the event E, this surviving arm has an empirical mean at the end of the phase which is smaller than the one of any arm j satisfying 2∆j ≤ ∆(K+1−k) .
So, at the end of phase k, any arm j satisfying 2∆j ≤ ∆(K+1−k) cannot be dismissed. Now, for a given arm
j, we consider two cases depending whether there exists m ∈ {1, . . . , K} such that ∆(m−1) ≤ 2∆j ≤ ∆(m) .
n−K
First case. If no such m exists, then we have ∆2j Tj (n) ≥ 14 ∆2(K) n1 ≥ 4log(K)H
, so that (3) holds.
2
Second case. If such m exists, then, from the above argument, the arm j cannot be dismissed before the end
of the phase K + 2 − m (since there exists K + 1 − m arms ` such that ∆` ≥ 2∆j ). From (2), we get
∆2j Tj (n) ≥ ∆2j nK+2−m ≥

∆2j
n−K
n−K
≥
,
2
∆(m−1) log(K)H2
4log(K)H2

which ends the proof of (3). We have seen that at the end of phase k, any arm j satisfying 2∆j ≤ ∆(K+1−k)
cannot be dismissed. Consequently, at the end of phase k, the dismissed arm `k = Ak+1 \ Ak satisfies the
left inequality of
1
∆(K+1−k) ≤ ∆`k ≤ 2∆(K+1−k) .
(5)
2
Let us now prove the right inequality by contradiction. Consider k such that 2∆(K+1−k) < ∆`k . Arm `k
thus belongs to the k − 1 worst arms. Hence, in the first k − 1 rejects, say at the end of phase k 0 , an arm
j with ∆j ≤ ∆(K+1−k) is dismissed. From the left inequality of (5), we get ∆(K+1−k0 ) ≤ 2∆j < ∆`k .

## Page 7

bj,n 0 − X
b` ,n 0 > 0 (since `k belongs to the k 0 worst arms by the previous
On the event E, we thus have X
k
k
k
inequality). This contradicts the fact that j is rejected at phase k 0 . So (5) holds.
bj,n − µj ≤ ∆(K+1−k) .
Now let E 0 be the event on which for any arm j, and any k ∈ {1, . . . , K − 1} X
k
8
Using again Hoeffding’s inequality,
a union bound and (2), this event holds with probability at least 1 −

n−K
2K(K − 1) exp − 32log(K)H
. We now work on the event E ∩ E 0 , which holds with probability at least
2

n−K
. From (5), the dismissed arm `k at the end of phase k satisfies
1 − K 3 exp − 32log(K)H
2

b` ,n − µ` ≤ ∆(K+1−k) ≤ ∆`k .
X
k
k
k
8
4
Besides, we also have
bm,n − µ(1) ≤
max X
k

m∈Ak

∆(K+1−k)
∆`k
≤
.
8
4

Consequently, at the end of phase k, we have
1
1
bm,n − X
b` ,n ≤ 3 ∆` ≤ 3∆(K+1−k) .
∆(K+1−k) ≤ ∆`k ≤ max X
k
k
k
m∈Ak
4
2
2 k

5

Lower bound

In this section, we provide a general and somewhat surprising lower bound. We prove that, when the reward
distributions are Bernoulli distributions with variances bounded away from 0, then for any forecaster, one can
permute the distributions on the arms
 (before the game starts) so that the probability of missing the best arm
cn
will be at least of order exp − H
. Note that, in this formulation, we allow the forecaster to know the reward
2
distributions up to a permutation of the indices! However, as the lower bound expresses it, whatever Bernoulli
distributions with variances bounded away from 0 are considered, the quantity H2 is a good measure of the
hardness of finding the best arm.
Theorem 4 (Lower Bound) Let ν1 , . . . , νK be Bernoulli distributions with parameters in [p, 1 − p], p ∈
(0, 1/2). For any forecaster, there exists a permutation σ : {1, . . . , K} → {1, . . . , K} such that the probability error of the forecaster on the bandit problem defined by ν̃1 = νσ(1) , . . . , ν̃K = νσ(K) satisfies


(5 + o(1))n
en ≥ exp −
,
p(1 − p)H2
where the o(1) term depends only on K, p and n and goes to 0 when n goes to infinity (see the end of the
proof).
The proof of this result is quite technical. However, it is simple to explain why we can expect such a
bound to hold. Assume (without loss of generality) that the arms are ordered, i.e., µ1 > µ2 ≥ . . . ≥ µK , and
that all rewards Xi,t , t ∈ {1, . . . , n}, i ∈ {1, . . . , K}, are drawn before the game starts. Let i ∈ {2, . . . , K}.
b1,n/i < X
bi,n/i ≤ X
bj,n/i for all j ∈ {2, . . . , i − 1}, then it seems reasonable that a good forecaster
If X
should not pull arm 1 more than n/i times, and furthermore not select it as its recommendation. One can see
that, the probability of the event we just described is of order of exp(−c(n/i)∆2i ). Thus, with probability at
least exp(−cn/ max2≤i≤K i∆−2
i ), the forecaster makes an error, which is exactly the lower bound we propose. However, note that this argument does not yield a reasonable proof strategy, in particular we assumed a
“good” forecaster with a “reasonable” behavior. For instance, it is obvious that the proof has to permute the
arms, since a forecaster could, despite all, choose arm 1 as its recommendation, which imply a probability
error of 0 as soon as the best arm is in position 1.
The main idea of our proposed proof goes as follows. A bandit problem is defined by a product distribution ν = ν1 ⊗ · · · ⊗ νK . One can consider that at the beginning of the game, n K-tuples of rewards are
sampled from this product distribution. This defines a table of nK rewards. A forecaster will explore a subpart of this table. We want to find a permutation σ of {1, . . . , K} so that the indices of the best arm for ν and
ν̃ = νσ(1) ⊗ · · · ⊗ νσ(K) are different and such that the likelihood ratio of the explored part of the table of nK
rewards under ν and ν̃ is at least of order exp(−cn/H2 ) with probability with respect to ν ⊗n lower bounded
by a positive numerical constant. This would imply the claimed bound. Remark that, the “likelihood cost” of
moving distribution νi to arm j depends on both the (Kullback-Leibler) distance between the distributions νi
and νj , and the number of times arm j is pulled. Thus, we have to find the right trade-off between moving

## Page 8

a distribution to a “close” distribution, and the fact that the target arm should not be pulled too much. To do
this, we “slice” the set of indices in a non-trivial (and non-intuitive) way. This “slicing” depends only on the
reward distributions, and not on the considered forecaster. Then, to put it simply, we move the less drawn arm
from one slice to the less drawn arm in the next slice. Note that the preceding sentence is not well defined,
since by doing this we would get a random permutation (which of course does not make sense to derive a
lower bound). However, at the cost of some technical difficulties, it is possible to circumvent this issue.
To achieve the program outlined above, as already hinted, we use the Kullback-Leibler divergence, which
is defined for two probability distributions ρ, ρ0 on [0, 1] with ρ absolutely continuous with respect to ρ0 as:




Z 1
dρ
dρ
KL(ρ, ρ0 ) =
log
(x)
dρ(x)
=
E
log
(X)
.
X∼ρ
dρ0
dρ0
0


c i,t (ρ, ρ0 ) = Pt log dρ0 (Xi,s ) . In particular,
Another quantity of particular interest for our analysis is KL
s=1
dρ
note that, if arm i has distribution ρ, then this quantity represents the (non re-normalized) empirical estimation
of KL(ρ, ρ0 ) after t pulls of arm i. Let Pν and Eν the probability and expectation signs when we integrate
with respect to the distribution ν ⊗n . Another important property is that for any two product distributions
ν, ν 0 , which differ only on index i, and for any event A, one has:


c i,T (n) (νi0 , νi ) ,
Pν (A) = Eν 0 1A exp −KL
(6)
i

QTi,n dνi
0
c
since we have s=1
dνi0 (Xi,s ) = exp − KLi,Ti (n) (νi , νi ) .
Proof: First step: Notations. Without loss of generality we can assume that ν is ordered in the sense that
µ1 > µ2 ≥ . . . ≥ µK . Moreover let L ∈ {2, . . . , K} such that H2 = L/∆2L , that is for all i ∈ {1, . . . , K},
i/∆2i ≤ L/∆2L .
We define now recursively the following sets. Let k1 = 1,

Σ1 = i : µL ≤ µi ≤ µL +

(7)
∆L
k
L1/2 1


,

and for j > 1,


∆L

kj−1 < µi ≤ µL +

∆L



,
k
L1/2 j
L1/2
where kj is the smallest integer (if it exists, otherwise set kj = +∞) such that |Σj | > 2|Σj−1 |. Let
` = max{j : kj < +∞}. We define now the random variables Z1 , . . . , Z` corresponding to the indices of
the less sampled arms of the respective slices Σ1 , . . . , Σ` : for j ∈ {1, . . . , `},
Σj =

i : µL +

Zj ∈ argmin Ti (n).
i∈Σj

Finally let Z`+1 ∈ argmini∈{1,...,L}\{Jn } Ti (n).
Second step: Controlling TZj (n), j ∈ {1, . . . , ` + 1}. We first prove that for any j ∈ {1, . . . , `},
1
−1
2 j+1

1− k

3|Σj | ≥ L

.

(8)

To do so let us note that, by definition of kj+1 , we have
n
o
kj+1 −1
kj
2|Σj | ≥
i : µL + ∆L /L1/2 < µi ≤ µL + ∆L /L1/2
n
o
kj+1 −1
≥
i : µi ≤ µL + ∆L /L1/2
− (|Σ1 | + . . . + |Σj−1 |).
Now remark that, by definition
again, we have |Σ1 | + . . . o
+ |Σj−1 | ≤ (2−(j−1) + . . . + 2−1 )|Σj | ≤ |Σj |.
n
kj+1 −1

Thus we obtain 3|Σj | ≥ i : µi ≤ µL + ∆L /L1/2
calculation, which makes use of (7). For any v ≥ 1,

. We finish the proof of (8) with the following

|{i : µi ≤ µL + ∆L /v}| = |{i : ∆i ≥ ∆L (1 − 1/v)}|
( r
)
i
∆L ≥ ∆L (1 − 1/v)
≥ i:
L



= i : i ≥ L(1 − 1/v)2 ≥ L 1 − (1 − 1/v)2 ≥ L/v.

## Page 9

Now (8) directly entails (since a minimum is smaller than an average), for j ∈ {1, . . . , `},
1
−1 X
TZj (n) ≤ 3L 2kj+1 −1
Ti (n).

(9)

i∈Σj

Besides, since Z`+1 is the less drawn arm among L − 1 arms, we trivially have
n
TZ`+1 (n) ≤
.
(10)
L−1
Third step: A change of measure. Let ν 0 = νL ⊗ ν2 ⊗ · · · ⊗ νK be a modified product distribution where
we replaced the best distribution by νL . Now let us consider the event
n
Cn = ∀t ∈{1, . . . , n}, i ∈ {2, . . . , L}, j ∈ {1, . . . , L},
o
c i,t (νi , νj ) ≤ t KL(νi , νj ) + on
c 1,t (νL , νj ) ≤ t KL(νL , νj ) + on ,
KL
and
KL
p
where on = 2 log(p−1 ) n log(2L). From Hoeffding’s maximal inequality, one can prove
that we have

P
Pν 0 (Cn ) ≥ 1/2. We thus have 1≤z1 ,...,z`+1 ≤L Pν 0 Cn ∩ {Z1 = z1 , . . . , Z`+1 = z`+1 } ≥ 1/2. Moreover
note that Z1 , . . . , Z` are all distinct. Thus there exist ` + 1 constants z1 , . . . , z`+1 such that, for An =
Cn ∩ {Z1 = z1 , . . . , Z`+1 = z`+1 }, we have
1
.
(11)
Pν 0 (An ) ≥
2L × L!
Since, by definition Z`+1 6= Jn , we have
An ⊂ {Jn 6= z`+1 }.

(12)

In the following we treat differently the cases z`+1 = 1 and z`+1 6= 1. First, let us assume that z`+1 = 1.
Then, an application of (6) and (12) directly gives, by definition of An ,


c
en (ν) = Pν (Jn 6= 1) = Eν 0 1Jn 6=1 exp − KL1,T1 (n) (νL , ν1 )


c
≥ Eν 0 1An exp − KL1,T1 (n) (νL , ν1 )


≥ Eν 0 1An exp − on − TZ`+1 (n)KL(νL , ν1 )


n
1
exp − on −
KL(νL , ν1 ) ,
≥
2L × L!
L−1
where we used (10) and (11) for the last equation. Now, for any p, q ∈ [0, 1], the KL divergence between
Bernoulli distributions of parameters p and q satisfies
KL(Ber(p), Ber(q)) ≤

(p − q)2
.
q(1 − q)

(13)

This can be seen by using log u ≤ u − 1 on the two logarithmic terms in KL(Ber(p), Ber(q)). In particular,
∆2L
it implies KL(νL , ν1 ) ≤ p(1−p)
, which concludes the proof in the case z`+1 = 1.
Assume now that z`+1 6= 1. In this case we prove that the lower bound holds for a well defined permuted
product distribution ν̃ of ν. We define it as follows. Let m be the smallest j ∈ {1, . . . , ` + 1} such that
zm = z`+1 . Now we set ν̃ as follows: ν̃zm = ν1 , ν̃zm−1 = νzm , . . ., ν̃z1 = νz2 , ν̃1 = νz1 , and ν̃j = νj for
other values of j in {1, . . . , K}. Remark that ν̃ is indeed the result of a permutation of the distributions of ν.
Again, an application of (6) and (12) gives, by definition of An ,
en (ν̃) = Pν̃ (Jn 6= zm )

= Eν 0 1Jn 6=zm exp

c 1,T (n) (νL , νz ) −
− KL
1
1

m−1
X


c
c
KLzj ,Tzj (n) (νzj , νzj+1 ) − KLzm ,Tzm (n) (νzm , νz1 )

j=1


≥ Eν 0 1An exp

− (m + 1)on − T1 (n)KL(νL , νZ1 ) −

m−1
X

TZj (n)KL(νZj , νZj+1 )

j=1


− TZm (n)KL(νZm , νZ1 ) .

(14)

## Page 10

From (13), the definition of Σj , and since the parameters of the Bernoulli distributions are in [p, 1 − p],
∆2

∆2

1
L
L
we have KL(νL , νZ1 ) ≤ p(1−p)
L , KL(νZm , νZ1 ) ≤ p(1−p) , and for any j ∈ {1, . . . , m − 1},

1
KL(νZj , νZj+1 ) ≤
p(1 − p)



∆L
k

L1/2 j+1

2
.

Reporting these inequalities, as well as (9), (10) and (11) in (14), we obtain:



m−1
X X
nL
∆2L
T1 (n) +
Ti (n) +
en (ν̃) ≥ Eν 0 1An exp − (m + 1)on − 3
p(1 − p)L
3(L − 1)
j=1 i∈Σj



∆2L
L
1
exp − L on − 3n
1+
≥
2L × L!
p(1 − p)L
3(L − 1)

Since L ≤ K and 2K × K! ≤ exp 2K log(K) and from the definitions of on and L, we obtain


p
n
en (ν̃) ≥ exp −2K log(K) − 2K log(p−1 ) n log(2K) − 5
,
p(1 − p)H2
which concludes the proof.

6

Proofs

6.1

Proof of Inequalities (1)
PK
Let log(K) = 12 + i=2 1i . Remark that log(K +1)−1/2 ≤ log(K) ≤ log(K)+1/2 ≤ log(2K). Precisely,
we will prove
H2 ≤ H1 ≤ log(K) H2 ,
√
which is√tight to the extent that the right inequality is an equality when for some 0 < c ≤ 1/ K, we have
∆(i) = ic for any i 6= i∗ , and the left inequality is an equality if all ∆i ’s are equal.
PK
Pi
−2
−2
Proof: The left inequality follows from: for any i ∈ {1, . . . , K}, H1 = k=1 ∆−2
k=1 ∆(i) ≥ i∆(i) .
(k) ≥
PK
P
K 1
−2
−2
−2
The right inequality directly comes from i=1 ∆−2
i=2 i i∆(i) ≤ log(K) maxi∈{1,...,K} i∆(i) .
(i) = ∆(2) +
6.2

Proof of Theorem 1

First step. Let us consider the event
r 

1 a
b
ξ = ∀i ∈ {1, . . . , K}, s ∈ {1, . . . , n}, |Xi,s − µi | <
.
5 s

From Hoeffding’s inequality and a union bound, we have P(ξ) ≥ 1 − 2nK exp − 2a
25 . In the following, we
prove that on the event ξ we have Jn = i∗ , which concludes the proof. Since Jn is the empirical best arm,
and given that we are on ξ, it is enough to prove that
r
1
a
∆i
≤
, ∀i ∈ {1, . . . , K},
5 Ti (n)
2
or equivalently:
Ti (n) ≥

4 a
, ∀i ∈ {1, . . . , K}.
25 ∆2i

(15)

Second step. Firstly we prove by induction that
Ti (t) ≤

36 a
+ 1, ∀i 6= i∗ .
25 ∆2i

(16)

It is obviously true at time t = 1. Now assume that the formula is true at time t − 1. If It 6= i then
Ti (t) = Ti (t − 1) and the formula still holds. On the other hand, if It = i, then in particular it means
that Bi,Ti (t−1) ≥ Bi∗ ,Ti∗ (t−1) . Moreover, since we are on ξ, we have Bi∗ ,Ti∗ (t−1) ≥ µ∗ and Bi,Ti (t−1) ≤
q
q
a
a
µi + 56 Ti (t−1)
. Thus, we have 56 Ti (t−1)
≥ ∆i . By using Ti (t) = Ti (t − 1) + 1, we obtain (16).

## Page 11

Parameter: exploration rate c > 0.
 1

n−K
Definitions: For k ∈ {1, . . . , K − 1}, let nk = log(K)
, t0 = 0, t1 = Kn1 , and
K+1−k
for k > 1, tk = n1 + . . . nk−1 + (K − k + 1)nk .
p
bi,s + a for s ≥ 1 and Bi,0 = +∞.
For i ∈ {1, . . . , K} and a > 0, let Bi,s (a) = X
s
Algorithm: For each phase k = 0, 1, . . . , K − 1:
b 1,k = K if k = 0, and otherwise
Let H
b 1,k =
H

max

K−k+1≤i≤K

b −2
i∆
<i> ,


b i = max1≤j≤K X
bj,T (t ) − X
bi,T (t ) and < i > is an ordering such that
where ∆
j k
i k
b
b
∆<1> ≤ . . . ≤ ∆<K> .
For t = tk + 1, . . . , tk+1 :
b 1,k ).
Draw It ∈ argmaxi∈{1,...,K} Bi,Ti (t−1) (cn/H
bi,T (n) .
Recommendation: Let Jn ∈ argmaxi∈{1,...,K} X
i

Figure 4: Adaptive UCB-E algorithm. Its intuitive justification goes as follows: The time points tk correspond to the moments where the Successive Rejects algorithm would dismiss an arm. Intuitively, in light
of Theorem 3, one can say that at time tk a good algorithm should have reasonable approximation of the
gaps between the best arm and the k worst arms, that is the quantities ∆(K−k+1) , . . . , ∆(K) . Now with these
quantities, one can build a lower estimate of H2 and thus also of H1 . We use this estimate between the time
points tk and tk+1 to tune the parameter a of UCB-E.
Now we prove an other useful formula:
Ti (t) ≥

4
min
25




a 25
∗ (t) − 1)
,
(T
, ∀i 6= i∗ .
i
∆2i 36

(17)

With the same inductive argument as the one to get equation (16), we only need to prove that this formula
holds when It = i∗ . By definition of the algorithm, and since we are on ξ, when It = i∗ we have for all i:
r
r
a
a
4
6
∗
≥ µi +
,
µ +
5 Ti∗ (t − 1)
5 Ti (t − 1)
which implies
Ti (t − 1) ≥

16

25

∆i + 65

a
q

a
Ti∗ (t−1)

2 .

We then obtain (17) by using u + v ≤ 2 max(u, v), Ti (t) = Ti (t − 1) and Ti∗ (t − 1) = Ti∗ (t) − 1.
Third step. Recall that we want to prove equation (15). From (17), we only have to show that
a
25
(Ti∗ (n) − 1) ≥ 2 ,
36
∆i∗
where we recall that ∆i∗ is the minimal gap ∆i∗ = mini6=i∗ ∆i . Using equation (16) we obtain:
X
36 X −2
36 −2
Ti∗ (n) − 1 = n − 1 −
Ti (n) ≥ n − K − a
∆i ≥
a∆ ∗ ,
25
25 i
∗
∗
i6=i

i6=i

where the last inequality uses 36
25 H1 a ≤ n − K. This concludes the proof.

7

Experiments

We propose a few simple experiments to illustrate our theoretical analysis. As a baseline comparison we
use the Hoeffding Race algorithm, see Maron and Moore (1993), and the uniform strategy, which pulls
equally often each arm and recommend the arm with the highest empirical mean, see Bubeck et al. (2009) for
its theoretical analysis. We consider only Bernoulli distributions, and the optimal arm always has parameter
1/2. Each experiment corresponds to a different situation for the gaps, they are either clustered in few groups,

## Page 12

or distributed according to an arithmetic or geometric progression. In each experiment we choose the number
of samples (almost) equal to H1 (except for the last experiment where we run it twice, the second time with
2H1 samples). If our understanding of the meaning of H1 is sound, in each experiment the strategies SR and
UCB-E should be able to find the best arm with a reasonable probability (which should be roughly of the
same order in each experiment). We report our results in Figure 5. The parameters for the experiments are as
follows:
– Experiment 1: One group of bad arms, K = 20, µ2:20 = 0.4 (meaning for any j ∈ {2, . . . , 20}, µj = 0.4)
– Experiment 2: Two groups of bad arms, K = 20, µ2:6 = 0.42, µ7:20 = 0.38.
– Experiment 3: Geometric progression, K = 4, µi = 0.5 − (0.37)i , i ∈ {2, 3, 4}.
– Experiment 4: 6 arms divided in three groups, K = 6, µ2 = 0.42, µ3:4 = 0.4, µ5:6 = 0.35.
– Experiment 5: Arithmetic progression, K = 15, µi = 0.5 − 0.025i, i ∈ {2, . . . , 15}.
– Experiment 6: Two good arms and a large group of bad arms, K = 20, µ2 = 0.48, µ3:20 = 0.37.
– Experiment 7: Three groups of bad arms, K = 30, µ2:6 = 0.45, µ7:20 = 0.43, µ21:30 = 0.38.
The different graphics should be read as follows: Each bar represents a different algorithm and the bar’s
height represents the probability of error of this algorithm. The correspondence between algorithms and bars
is the following:
– Bar 1: Uniform sampling strategy.
– Bar 2-4: Hoeffding Race algorithm with parameters δ = 0.01, 0.1, 0.3.
– Bar 5: Successive Rejects strategy.
– Bar 6-9: UCB-E with parameter a = cn/H1 where respectively c = 1, 2, 4, 8.
– Bar 10-14: Adaptive UCB-E (see Figure 4) with parameters c = 1/4, 1/2, 1, 2, 4.

8

Conclusion

This work has investigated strategies for finding the best arm in a multi-armed bandit problem. It has proposed
a simple parameter-free algorithm, SR, that attains optimal guarantees up to a logarithmic term (Theorem 2
and Theorem 4). A precise understanding of both SR (Theorem 3) and a UCB policy (Theorem 1) lead
us to define a new algorithm, Adaptive UCB-E. It comes without guarantee of optimal rates (see end of
Section 3), but performs better than SR in practice (for c = 1, Adaptive UCB-E outperformed SR on all the
experiments we did, even those done to make it fail). One possible explanation is that SR is too static: it does
not implement more data driven arguments such as: in a phase, a surviving arm performing much worse than
the other ones is still drawn until the end of the phase even if it is clear that it is the next dismissed arm.
Extensions of this work may concentrate on the following problems. (i) What is a good measure of
hardness when one takes into account the (empirical) variances? Do we have a good scaling with respect
to the variance with the current algorithms or do we need to modify them ? (ii) Is it possible to derive a
natural anytime version of Successive Rejects (without using a doubling trick)? (iii) Is it possible to close
the logarithmic gap between the lower and upper bounds? (iv) How should we modify the algorithm and the
analysis if one is interested in recommending the top m actions instead of a single one?

References
P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine
Learning Journal, 47(2-3):235–256, 2002.
S. Bubeck, R. Munos, and G. Stoltz. Pure exploration in multi-armed bandits problems. In Proc. of the 20th
International Conference on Algorithmic Learning Theory, 2009.
C. Domingo, R. Gavaldà, and O. Watanabe. Adaptive sampling methods for scaling up knowledge discovery
algorithms. Data Mining and Knowledge Discovery, 6(2):131–152, 2002.
E. Even-Dar, S. Mannor, and Y. Mansour. Action elimination and stopping conditions for the multi-armed
bandit and reinforcement learning problems. The Journal of Machine Learning Research, 7:1079–1105,
2006.
O. Maron and A. W. Moore. Hoeffding races: Accelerating model selection search for classification and
function approximation. In NIPS, pages 59–66, 1993.
V. Mnih, Cs. Szepesvári, and J.-Y. Audibert. Empirical bernstein stopping. In ICML, volume 307, pages
672–679, 2008.
H. Robbins. Some aspects of the sequential design of experiments. Bulletin of the American Mathematics
Society, 58:527–535, 1952.

## Page 13

Experiment 1, n=2000

Experiment 4, n=600

0.5

0.25

0.45

Probability of error

0.4
0.35
0.3
0.25
0.2
0.15

1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.2

Probability of error

1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.1

0.15

0.1

0.05

0.05
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

10

11

12

13

0

14

1

2

3

Experiment 2, n=2000

7

8

9

10

11

12

13

14

0.35
0.3
0.25
0.2
0.15

1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.35

Probability of error

Probability of error

6

0.4
1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.4

0.1

0.3
0.25
0.2
0.15
0.1
0.05

0.05

1

2

3

4

5

6

7

8

9

10

11

12

13

0

14

1

2

3

Experiment 3, n=2000

4

5

6

7

8

9

10

11

12

13

14

Experiment 6, n=6000

0.25

0.35

0.15

0.1

1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.3

Probability of error

1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.2

Probability of error

5

Experiment 5, n=4000

0.45

0

4

0.25

0.2

0.15

0.1

0.05
0.05

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

10

11

12

13

0

14

1

2

3

Experiment 7, n=6000

6

7

8

9

10

11

12

13

14

0.35

0.5

0.4

0.3

0.2

0.1

1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.3

Probability of error

1 : Unif
2−4 : HR
5 : SR
6−9 : UCB−E
10−14 : Ad UCB−E

0.6

Probability of error

5

Experiment 7, n=12000

0.7

0

4

0.25

0.2

0.15

0.1

0.05

1

2

3

4

5

6

7

8

9

10

11

12

13

14

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

10

11

12

13

14

Figure 5: These results support our theoretical findings in the following sense: Despite the fact that the
experiments are very different, one can see that since we use a number of samples (almost) equal to the
hardness H1 , in all of them we get a probability of error of the same order, and moreover this probability is
small enough to say that we identified the best arm. Note that the Successive Rejects algorithm represents in
all cases a substantial improvement over both the naive uniform strategy and Hoeffding Race. These results
also justify experimentally the algorithm Adaptive UCB-E. Indeed one can see that with the constant c = 1,
we obtain better results than SR in all experiments, even in experiment 6 which was designed to be a difficult
instance of Adaptive UCB-E.
