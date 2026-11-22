## Page 1

Policy-Value Alignment and Robustness in Search-based Multi-Agent Learning

arXiv:2301.11857v2 [cs.AI] 6 Feb 2023

Niko A. Grupen1 , Michael Hanlon1 , Alexis Hao1 , Daniel D. Lee1 , Bart Selman1
1
Cornell University

Abstract
Large-scale AI systems that combine search and
learning have reached super-human levels of performance in game-playing, but have also been
shown to fail in surprising ways. The brittleness
of such models limits their efficacy and trustworthiness in real-world deployments. In this work,
we systematically study one such algorithm, AlphaZero, and identify two phenomena related to
the nature of exploration. First, we find evidence
of policy-value misalignment—for many states,
AlphaZero’s policy and value predictions contradict each other, revealing a tension between accurate move-selection and value estimation in AlphaZero’s objective. Further, we find inconsistency within AlphaZero’s value function, which
causes it to generalize poorly, despite its policy
playing an optimal strategy. From these insights
we derive VISA-VIS: a novel method that improves
policy-value alignment and value robustness in AlphaZero. Experimentally, we show that our method
reduces policy-value misalignment by up to 76%,
reduces value generalization error by up to 50%,
and reduces average value error by up to 55%.

1

Introduction

Learning algorithms that combine self-play reinforcement
learning and search have grown in popularity in recent years
due to their ability to reach superhuman levels of performance in combinatorially complex environments [Silver et
al., 2017b]. More recently, an examination of AlphaZero
[Silver et al., 2017a] revealed that it obeys neural scaling laws
with respect to play strength and network capacity [Neumann
and Gros, 2022]—further suggesting that hybrid algorithms,
as in other domains (e.g. large language models [Kaplan et
al., 2020]), may have significant room to scale further. As
these algorithms continue to scale towards real-world complexity, it calls into question whether state-of-the-art hybrid
systems are ready for real-world deployment.
To date, researchers and practitioners have a very limited understanding of why the neural networks underpinning these hybrid algorithms make the decisions that they
do—consider the infamous “move 37” in AlphaGo vs. Lee

Sedol [Metz, 2016]—and what they might learn about the
environment and other agents within it. Though significant
progress has been made in the area of concept-based interpretability, where post-hoc analysis has shown that many
human-understandable concepts can be accurately regressed
from AlphaZero’s representations [McGrath et al., 2022],
opaque artifacts of self-play learning still remain. For example, studies leveraging adversarial game-play have shown that
AlphaZero is surprisingly susceptible to minor perturbations
in opponent behavior [Lan et al., 2022] and can be exploited
to lose games that could be easily won by amateur human
players [Wang et al., 2022].
In this work, we discover two phenomena that are related
to the nature of AlphaZero’s exploration and may underlie its
documented failure modes. First, we identify a novel emergent phenomenon, policy-value misalignment, in which AlphaZero’s policy and value predictions contradict each other.
Policy-value misalignment is a direct consequence of AlphaZero’s loss function, which includes separate terms for policy and value errors individually, but no constraint on their
consistency. We introduce an information-theoretic definition
of policy-value misalignment and show how it can be measured for any AlphaZero network. Next, we find evidence of
inconsistency within AlphaZero’s value function with respect to symmetries in the environment in which it is trained.
As a result, AlphaZero’s value predictions generalize surprisingly poorly to infrequently-visited and unseen states.
Crucially, we show these phenomena are masked by
search, as it is possible for both to be present even when AlphaZero plays optimally within its search process. This work
therefore demonstrates an interesting tension: search allows
AlphaZero to achieve unprecedented scale and performance,
but may also hide deficiencies in its learned components (i.e.
its policy and value network).
To address these issues, we propose two modifications
to AlphaZero aimed directly at policy-value alignment and
value function consistency. First, we introduce ValueInformed Selection (VIS): a modified action-selection rule
in which action probabilities are computed using a one-step
lookahead with AlphaZero’s value network. VIS then alternates stochastically between this value action selection
rule and AlphaZero’s standard policy (which is derived from
search probabilities). During training, this forces alignment
between AlphaZero’s policy and value predictions. Next, we

## Page 2

Figure 1: An overview of our proposed method. VISA-VIS combines two improvements to the AlphaZero algorithm: (i) Value-Informed
Selection (VIS), where action selection alternates stochastically between sampling actions from MCTS search probabilities (at ∼ πp )
and sampling from a set of action probabilities derived from AlphaZero’s value function (at ∼ πv ); and (ii) Value-Informed Symmetric
Augmentation (VISA), a form of data augmention that targets uncertainty in AlphaZero’s value function. Specifically, before adding a state
st to the replay buffer D, VISA generates a set of symmetric state transformations sσ , evaluates each transformed state sσi with its value
function, and returns the state s∗σ that differs most from AlphaZero’s value estimate for the original state st . Both st and s∗σ are added to D.

introduce Value-Informed Symmetric Augmentation (VISA):
a data augmentation technique that takes into account uncertainty in AlphaZero’s value network. Unlike data augmentation techniques that naively add states to a network’s replay
buffer, VISA considers symmetric transformations of a given
state and stores only the transformed state whose value differs most from the network’s value estimate for the original,
un-transformed state.
VISA and VIS can improve both policy-value alignment
and value robustness in AlphaZero simultaneously. We refer
to this joint method as VISA-VIS. In experiments across a variety of symmetric, two-player zero-sum games, we show that
VISA-VIS reduces policy-value misalignment by up to 76%,
reduces value generalization error by up to 50%, and reduces
average value prediction error by up to 55%; as compared to
AlphaZero.
Contributions: In sum, we offer five primary contributions:
• We systematically study AlphaZero and identify two
phenomena–policy-value misalignment and value function
inconsistency–that are masked by AlphaZero’s search.
• We define an information-theoretic measure of policy-value
misalignment that can be used to evaluate any trained AlphaZero network.
• We propose Value-Informed Selection (VIS): A modified
action-selection rule for AlphaZero that forces alignment
between policy and value predictions.
• We propose Value-Informed Symmetric Augmentation
(VISA): A data augmentation technique that leverages

symmetry and value function uncertainty to learn a more
consistent value network.
• We introduce a new method, VISA-VIS, that combines VIS
and VISA into a single algorithm that extends AlphaZero.
Experimentally, we show that VISA-VIS reduces policyvalue misalignment by up to 76%, reduces value generalization error by up to 50%, and reduces average value predictions error by up to 55% compared to vanilla AlphaZero.

2

Related Work

The continued success of hybrid algorithms that combine
search and self-play RL has drawn research interest into their
robustness and interpretability.
Adversarial Examples In supervised learning settings, it
is well-documented that both synthetic [Goodfellow et al.,
2014] and naturally-occuring [Hendrycks et al., 2021] adversarial examples can induce sub-optimal behavior from neural network-based classifiers. Recently, similar tactics have
proven effective in identifying the weaknesses of RL agents.
For example, it has been shown that small perturbations of
an agent’s observations can cause large decreases in reward
[Huang et al., 2017; Kos and Song, 2017] and that performance degradation can be exacerbated by leveraging knowledge of an agent’s behavior [Lin et al., 2017].
In multi-agent settings, RL agents are particularly vulnerable to adversarial behavior from other agents in the environment [Gleave et al., 2019]. Of particular relevance is the work
of Wang et al. [2022], which showed that a professional-level

## Page 3

AlphaZero Go agent can be tricked into losing games by simple adversarial moves that are easily countered by amateur
human players. Further studies have shown that AlphaZero is
susceptible to other state- and action-space perturbation [Lan
et al., 2022]. We posit that these surprising failure modes
are a symptom of the phenomena we describe in this work—
policy-value misalignment and value function inconsistency.
Interpretability RL interpretability methods are divided
into those that improve the transparency of agent decisionmaking by either (i) introducing intrinsically-interpretable
policy and value network architectures; or (ii) generating
post-hoc explanations for previously-trained networks. Beyond more traditional approaches that are derived directly
from supervised learning counterparts–such as saliency maps
[Greydanus et al., 2018; Selvaraju et al., 2017]—advances
in reward-agnostic behavioral analysis [Omidshafiei et al.,
2022] and intrinsically-interpretable policy learning schemes
[Grupen et al., 2022] have greatly improved the interpretability of an RL agent’s decision-making (and are applicable to
multi-agent settings).
For large-scale AI systems such as AlphaZero, seminal
work on post-hoc, concept-based interpretability has shown
that many human-understandable concepts are encoded in AlphaZero’s policy network during training [McGrath et al.,
2022]. Additional studies have employed model probing and
behavioral analysis to understand the concepts represented by
AlphaZero for the game of Hex [Forde et al., 2022]. These
studies highlight the search-learning conundrum proposed in
our work—it is possible for AlphaZero to learn complex playing strategies and sophisticated human-understandable concepts while simultaneously demonstrating inconsistencies in
its policy and value predictions.

3

Background

3.1

AlphaZero

AlphaZero is a hybrid algorithm that combines self-play
RL and Monte-Carlo Tree Search (MCTS). We review AlphaZero’s search and learning processes below, as both are
crucial to understanding our method.

encourages the exploration of lesser-visited states (within the
search tree); and c is a coefficient determining the level of exploration. When a leaf node sL is encountered, it is expanded,
evaluated by AlphaZero’s network to produce (pL , vL ) ∼
fθ (sL ), and initialized as follows:
N (s, a) = 0, W (s, a) = 0, Q(s, a) = 0, P (s, a) = pL,a
(3)
Then, in a backward pass up the search tree, the statistics
of each traversed edge are updated as follows: N (s, a) =
N (s, a) + 1, W (s, a) = W (s, a) + vL , Q(s, a) =
W (s, a)/N (s, a).
Action Selection After MCTS, AlphaZero constructs a policy π, where the probability π(a|s) of selecting action a from
state s is computed directly from visitation counts:
N (s, a)1/τ
π(a|s) = P
1/τ
b N (s, b)

(4)

and τ is a temperature parameter that controls exploration.
During both training and test-time rollouts, AlphaZero samples actions a ∼ π according to this policy.
Learning Starting from an initial state s0 (e.g. an empty
board), AlphaZero proceeds with self-play, using the aforementioned search process to select actions until a terminal
state sT is reached. A score z is produced for sT according
to the rules of the game and tuples (st , at , z) are stored in a
replay buffer D for each time-step t ∈ [0, T ] in the trajectory.
AlphaZero’s neural network is updated to minimize the loss
function:
l = (z − v)2 − π T log p + λ||θ||2

(5)

where (p, v) ∼ fθ (s) as in Equation (1), π follows
from Equation (4), and λ is a coefficient weighting L2regularization. Intuitively, Equation (5) encourages the network to (i) minimize the error between predicted values v
and game outcomes z; and (ii) maximize similarity between
predicted policy probabilities p and search probabilities π.

4

Method

Setup AlphaZero maintains a search tree in which each
node is a state s and each edge (s, a) represents taking an
action a from s. Each edge is described by a tuple of statistics {N (s, a), W (s, a), Q(s, a), P (s, a)}, holding the edge’s
visit count, total action value, mean action value, and selection probability, respectively. AlphaZero uses a two-headed
neural network fθ with parameters θ to compute policy probabilities p and a value estimate v for a state s as follows:

In this section, we introduce the proposed VISA-VIS method
that combines two modifications to the AlphaZero algorithm:
Value-Informed Selection (VIS) and Value-Informed Symmetric Augmentation (VISA). We describe each in detail in
the subsections below, and an overview of our proposed approach is presented in Figure 1.

(p, v) = fθ (s)

Recall AlphaZero’s loss function from Equation (5). Though
this objective incentivizes accurate policy and value predictions individually, it imposes no constraint on their consistency. It is therefore possible for AlphaZero’s network to output policy probabilities p that contradict its value prediction
v—e.g. during inference, p places large probability mass on a
winning action while v predicts that the current state is a losing one. We find that this occurs frequently in practice. To enforce policy-value consistency, we introduce Value-Informed
Selection (VIS). VIS is a novel action selection rule in which

(1)

MCTS uses fθ to evaluate states during the search process.
Search In each state s, AlphaZero executes n simulations
of MCTS tree search. In each simulation, MCTS traverses
the game tree by selecting edges that maximizes the upper
confidence bound Q(s, a) + U (s, a), where:
pP
b N (s, b)
U (s, a) = cP (s, a)
(2)
1 + N (s, a)

4.1

Value-Informed Selection

## Page 4

AlphaZero alternates between selecting actions with its policy (as informed by the search probabilities in Equation (4))
and selecting actions with its value function.
More formally, let s0 be the set of successor states from a
given state s, where each successor s0a ∈ s0 is the result of
taking a legal action a from s. We then define πv to be the
action probabilities obtained by computing a one-step value
lookahead from s:
0

ev(sa )
πv (a|s) = P v(s0 )
e b

(6)

4.3

Measuring Policy-Value Misalignment

We also introduce an information-theoretic measure of
policy-value misalignment. Given AlphaZero’s standard policy πp and the value-informed policy πv from Section 4.1, it
is possible to compute policy-value misalignment as the KLdivergence:
DKL (πp ||πv )
(8)
Intuitively, a model with strong policy-value alignment will
yield a lower divergence term, whereas policy and value networks that are inconsistent will result in higher divergence.

s0b ∈S 0

where v(s) is the value of state s, as predicted by AlphaZero’s
value network. Renaming the search policy defined in Equation (4) to be πp , VIS then defines the action selection rule:

πp (a|s) if η < 
(7)
πVIS (a|s) =
πv (a|s) otherwise.
where η∼U (0, 1) is drawn uniformly and  is a threshold determining the likelihood of policy vs. value action selection.
When used in conjunction with Equation (5), VIS forces AlphaZero’s policy and value predictions to be aligned by swapping πp and πv stochastically. Intuitively, VIS pushes AlphaZero to align its policy and value function because it never
knows which it will have to use for any given action selection.

4.2

Value-Informed Symmetric Augmentation

Exhaustively searching a combinatorial state-space is impractical, so during training, AlphaZero relies upon MCTS to reduce the set of states it encounters. However, this also places
the onus on AlphaZero’s network to generalize to states that
are skipped by MCTS during training. We find that AlphaZero’s value predictions generalize poorly to such states.
To improve the generalization capabilities of AlphaZero’s
value network, we propose Value-Informed Symmetric Augmentation (VISA): a form of data augmention that explicitly
targets uncertainty in AlphaZero’s value function. For each
state s encountered during training, VISA first generates a
set sσ = {sσ1 , ..., sσN } of N additional states, where each
sσi ∈ sσ is the result of applying a symmetric transformation σi (s) to s. VISA then finds the transformed state s∗σ with
maximum value uncertainty as:
s∗σ = max[(v(s) − v(sσi ))2 ]
i

and adds both s and s∗σ to AlphaZero’s replay buffer. Note
that, by adding only s∗σ , VISA is an improvement over both
random data augmentation—it targets the augmentation for
which the value network is least certain—and exhaustive data
augmentation—it is more memory efficient than storing all of
sσ in the replay buffer.
In this work, we consider three principal symmetric transformations: rotation and reflection from the dihedral group,
and inversion. Intuitively, in the board game setting (as in
Figure 1), these transformations correspond to rotating the
game board, mirroring the game board (vertically or horizontally), or flipping each player’s pieces. Importantly, we note
that board inversion also requires inverting the ground-truth
game-tree value z when computing Equation (5).

5

Experiments

In this section, we first demonstrate the presence of policyvalue misalignment and value function inconsistency in AlphaZero. Then, we evaluate VISA-VIS and examine the extent
to which it alleviates these phenomena.
Environments: We consider three solved, symmetric twoplayer zero-sum games of increasing complexity: Tic-TacToe, 4x4 Tic-Tac-Toe, and Connect Four. We emphasize the
importance of studying solved games to our analysis. In order
to properly measure the scope of AlphaZero’s value generalization error, we need a suitable ground truth with which to
compare AlphaZero’s predictions. Solved games fulfill that
need. In each game, the outcome can take on a value of
z = 1, 0, and −1 for wins, losses, and draws, respectively.
Training Details: AlphaZero is trained for a fixed number
of time-steps following the algorithm in Section 3.1. AlphaZero’s network consists of a ResNet [He et al., 2016] backbone with separate heads for policy and value prediction. All
details are the same for VISA-VIS training runs, but following the modifications outlined in Section 4. Additional details, including hyperparameters, are outline in Appendix A.

5.1

AlphaZero in an Enumerable Game Tree

We first present a systematic study of AlphaZero’s performance in an enumerable game-tree: Tic-Tac-Toe. Due to the
small state-space of Tic-Tac-Toe, it is possible to evaluate AlphaZero’s predictions against every legal state, providing us
with a holistic view of AlphaZero’s behavior.
After training, the performance of each AlphaZero model
is verified in two ways: First, we play each model against
an oracle opponent for 1, 000 matches across five random
seeds and score each match according to the rules of the
game (z ∈ {1, 0, −1}). For completeness, we include both
games in which AlphaZero moves first, and those in which
AlphaZero moves second. Next, we verify the accuracy of
AlphaZero’s value predictions by iterating through the legal
state-space and computing the mean-squared error between
AlphaZero’s value predictions and the true game-tree value.
The results of this analysis are presented in Figure 2.
From the training curves in Figure 2a, it is clear that AlphaZero converges to a strong game-playing strategy very
quickly—the expected outcome is a tie under optimal play—
which is expected. The result of the value prediction analysis,
on the other hand, is quite surprising. In Figure 2b, we find
that, despite AlphaZero’s ability to play optimally, its value
predictions are not always accurate. In fact, when the error

## Page 5

Figure 2: Evaluation of AlphaZero in an enumerable game tree. a) AlphaZero performance during training. For a simple game, AlphaZero
quickly converges to an optimal strategy. b) Despite playing optimally, an exhaustive analysis of AlphaZero’s value function performance
shows that it makes many incorrect and high-error predictions (relative the true game-tree value for each state). c) Computing value error as
a function of state visitation reveals that AlphaZero’s value function is failing to generalize to infrequently visited and unseen states. d) A
qualitative example where AlphaZero’s policy and value predictions are both correct. e) A qualitative example demonstrating policy-value
misalignment—AlphaZero’s policy prediction picks a winning move, but its value prediction is incorrect (predicts a loss).

of AlphaZero’s value predictions is quantized, there is a noticeable error bump when e > 1.0—indicating AlphaZero is
predicting a loss when the true game-tree value is a win (or
vice versa) and another error bump when e > 3.5—indicating
that AlphaZero is predicting a loss very confidently when the
true game-tree value is a win (or vice versa).
When plotted as a function of the number of visits made
by AlphaZero to each state during training (see Figure 2c),
it becomes clear that the error of AlphaZero’s value predictions is highest for infrequently-visited or unseen states, indicating that AlphaZero has failed to generalize. Moreover,
as shown in the qualitative examples from Figure 2e, which
includes both AlphaZero’s policy predictions and its corresponding value estimates, we find that, in many instances,
AlphaZero’s policy predictions are correct (they choose the
winning move), despite value predictions that are wildly inaccurate (they predict a loss, despite the winning opportunity).
Altogether, this analysis provides evidence that (i) AlphaZero’s value predictions are inconsistent and generalize
poorly; and (ii) AlphaZero’s value predictions are not aligned
with its policy predictions (i.e. AlphaZero suffers from
policy-value misalignment).

5.2

Ablation: VISA, VIS, and VISA-VIS

Next, we investigate how well our proposed method, VISAVIS, corrects AlphaZero’s susceptibility to generalization error and policy-value misalignment. We perform the same
test-time analysis over trained VISA-VIS models and, to

gauge the contributions of each component of our method,
perform an ablation over both VISA and VIS separately.
Aside from AlphaZero and VISA-VIS, we report results
for an AlphaZero model that is trained from random initial
states (each training rollout begins from a random legal, nonterminal state). Though training with random initial states
does not scale to more complex games, it provides a lowerbound for value error in this setting (i.e. how well can AlphaZero do when it is allowed to visit all states many times).
Policy-Value Misalignment First, we compute the average
policy-value misalignment exhibited by each method across
all environmental states, as defined by Equation (8). As
shown in Figure 3, VIS, and therefore VISA-VIS, significantly improve the consistency of the model’s policy and
value predictions, reducing policy-value misalignment by
50% compared to AlphaZero and by over 60% as compared
to AlphaZero Random. These results also provide a great
example of how value prediction accuracy and policy-value
misalignment differ in nature. Consider AlphaZero Random,
for example. Despite observing all game states directly, AlphaZero Random exhibits a high policy-value misalignment.
This is because value estimation alone does not tell the whole
story—it is still possible for AlphaZero’s value function to be
inconsistent with its policy.
Value Error Value error results for this ablation are reported in Figure 4. There we see that VISA, VIS, and VISAVIS each significantly improve the quality of AlphaZero’s
value predictions. In particular, VISA-VIS reduces the per-

## Page 6

Figure 3: Policy-value misalignment of each method, as given by
the information-theoretic measure introduced in Equation 8 (lower
is better). VIS reduces policy-value misalignment by over 50%
over AlphaZero (AZ) and maintains alignment when combined with
VISA (VISA-VIS).

centage of states with a medium value error (e > 1.0) by
almost 10% as compared to AlphaZero and produces a high
value error prediction (e > 3.0) for only 0.6% of states (6.9×
fewer than AlphaZero), which is extremely close to the perfect score obtained by the version of AlphaZero that observes
all states (AZ Random). In terms of generalization, we find
that VISA-VIS reduces average value generalization error by
over 50% (from an average of 1.44 to 0.64). Notably, for
both high-error predictions and generalization error, we see
that the performance improvements of VISA and VIS are
complementary—VISA and VIS both decrease the number
high-error states, but value error is decreased the most when
they are combined (and likewise for generalization error).

5.3

Scaling Up

Here we demonstrate the scalability of our method through
evaluation in significantly more complex games. The results
of this analysis are presented in Figure 5.
Adversarial State Detection In more complex environments, it is not possible to enumerate the state-space and evaluate each state individually. We therefore make two modifications to our evaluation strategy to support more complex games. First, we focus primarily on endgame states.
Endgame states allow us to target AlphaZero’s most important decisions—a single move may determine the game’s
outcome—while still allowing us to extract a ground truth
outcome from the game-tree, which we can use to verify AlphaZero’s predictions.
Second, inspired by prior work on adversarial examples for
AlphaZero [Wang et al., 2022], we define an adversarial strategy that aids in uncovering states for which AlphaZero makes
inconsistent predictions. Specifically, at test-time, after computing search-informed action probabilities with Equation
(4), we force AlphaZero to take the action at = mina [π(a|s)]
with minimum assigned probability. Because π encodes the

Figure 4: Value error for each method (lower is better). VISA-VIS
significantly reduces the percentage of medium (e > 1.0) and high
error (e > 3.0) value predictions and reduces the average generalization error of AlphaZero’s value network by over 50%.
Table 1: Adversarially-detected States.

Game
4x4 Tic-Tac-Toe
Connect Four

Adversarial States

Avg. Value Error

10,241
52,768

2.63
2.36

expected value of taking action a from state s, we are effectively forcing AlphaZero to play the worst move available. During each rollout, this adversarial strategy nudges AlphaZero into subsets of the state-space that are most likely to
have been overlooked by search during training. This simple
adversarial strategy proves effective at generating states that
are poorly evaluated by AlphaZero. In each of our environments, we played a trained AlphaZero model against itself for
100, 000 matches using the aforementioned adversarial detection strategy and collected all unique endgame states for
which AlphaZero made a high-error (e > 1.0) value prediction. In 4x4 Tic-Tac-Toe, the adversarial detector uncovered
over 10, 000 states that received high-error value estimates in
each of three separately trained AlphaZero models. The
detector similarly uncovered over 50, 000 states for Connect
Four. A summary of these statistics is available in Table 1.
In Figure 5a, which shows the results for our adversarial analysis in 4x4 Tic-Tac-Toe, we see that regular AlphaZero displays both poor value function consistency—
with an average value error of 2.63 for these states—and
significant policy-value misalignment—with average KLdivergence DKL (πp ||πv ) = 1.41. Our method VISA-VIS
corrects both of these issues, decreasing average value error

## Page 7

Figure 5: AlphaZero vs. VISA-VIS in more complex environments. a) In 4x4 Tic-Tac-Toe, An adversarial state analysis reveals over 10,000
states in which AlphaZero performs poorly (high value error, high policy-value misalignment). VISA-VIS largely corrects AlphaZero’s errors
in these states, reducing average value error by 55% and policy-value misalignment by 76%. Moreover, During training, AlphaZero and
VISA-VIS converge at a similar rate, indicating that VISA-VIS attains with no corresponding increase in sample complexity. b) In Connect
Four, the adversarial analysis yields qualitatively similar results. AlphaZero again demonstrates value function inconsistency—average value
error of 2.36 across adversarial states—and policy-value misalignment—average KL-divergence of 0.975. These errors are again reduced by
VISA-VIS, which averages a value error of 1.29 and a policy-value misalignment score of 0.425 over the same adversarially-generated states
(representing a 45% and 56.5% reduction, respectively). Moreover, in terms of sample complexity, VISA-VIS converges more quickly than
AlphaZero on average (in the Connect Four game). This reveals an additional benefit of VISA-VIS—symmetric augmentation reduces the
time it takes to learn an optimal policy.

55% to 1.20 and reducing policy-value misalignment by 76%
down to DKL (πp ||πv ) = 0.327.
Figure 5b presents our adversarial analysis in Connect
Four. Here we find a qualitatively similar result to the 4x4
Tic-Tac-Toe case. Our adversarial detector is able to identify over 50,000 states that yield high-error predictions from
AlphaZero. As before, VISA-VIS corrects the high-error predictions. More specifically, the adversarial states result in an
average value error of 2.36 and an average policy-value misalignment score of 0.975 from AlphaZero. VISA-VIS compares favorably with an average error of 1.29 and an average policy-value misalignment score of 0.425—representing
a 45% and 56.5% reduction over AlphaZero, respectively.
Altogether, these results not only show that policy-value
misalignment and value function inconsistency are general
phenomena that span multiple environments, but they also
provide evidence that our method, VISA-VIS, can reliably
combat both phenomena simultaneously.
Sample Complexity We also plot the game-playing performance of AlphaZero and VISA-VIS throughout training. As
in the 3x3 case, the expected outcome of 4x4 Tic-Tac-Toe is
a tie under optimal play (z = 1). As shown in Figure 5a,
we find evidence that the time-to-convergence of AlphaZero
and VISA-VIS are roughly the same. Overall, VISA-VIS
is able to attain the aforementioned improvements in policyvalue alignment and value functon consistency without a loss
in sample complexity.

In the case of Connect Four, we see that VISA-VISA has
a more significant positive impact on sample complexity. In
Figure 5b), it is clear that VISA-VIS converges to an optimal strategy more quickly than AlphaZero on average (the
expected outcome for the first player is a win in Connect
Four). This result suggests that, in this more complex game,
the symmetric data augmentation of VISA-VIS reduces the
sample complexity needed to learn an optimal policy. This
highlights yet another benefit of VISA-VISA.

6

Conclusion

In this work, we presented a systematic study of AlphaZero, which uncovered two phenomena—policy-value
misalignment and value function inconsistency—that hurt AlphaZero’s robustness. Crucially, both of these phenomena are
masked by search, indicating that it is still possible for AlphaZero to play optimally despite their existence and may
contribute to AlphaZero’s susceptibility to adversarial examples. We introduced VISA-VIS, which combines ValueInformed Selection (VIS)—a novel action selection criteria
for AlphaZero—and Value-Informed Symmetric Augmentation (VISA)—a novel data augmentation that targets value
function uncertainty with respect to symmetry. Together,
VISA-VIS leads to significant improvements for both policyvalue alignment and value function robustness in AlphaZero,
as demonstrated in our experiments.

## Page 8

Ethical Statement
This work examines the robustness and reliability of AlphaZero. As large-scale AI systems like AI proliferate in realworld use cases, it is important to understand their limitations
and potential failure modes. Algorithms that leverage selfplay reinforcement learning are particularly susceptible to potential harms (e.g. safety, reliability, interpretability), because
they do not observe any human-provided data during training.
We posit that, in order to ensure that such systems are generally helpful, and to mitigate unforeseen or unintended harms
during deployment, we must ensure their robustness to potential attacks (e.g. adversarial inputs) and overall reliability.
The goal of this work was to take a small step in that direction. This work does not make use of sensitive data or include
as part of its results a sensitive task or study.

References
[Forde et al., 2022] Jessica Zosa Forde, Charles Lovering,
George Konidaris, Ellie Pavlick, and Michael L Littman.
Where, when & which concepts does alphazero learn?
lessons from the game of hex. In AAAI Workshop on Reinforcement Learning in Games, 2022.
[Gleave et al., 2019] Adam Gleave, Michael Dennis, Cody
Wild, Neel Kant, Sergey Levine, and Stuart Russell. Adversarial policies: Attacking deep reinforcement learning.
arXiv preprint arXiv:1905.10615, 2019.
[Goodfellow et al., 2014] Ian Goodfellow, Jonathon Shlens,
and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.
[Greydanus et al., 2018] Samuel Greydanus, Anurag Koul,
Jonathan Dodge, and Alan Fern. Visualizing and understanding atari agents. In International conference on machine learning. PMLR, 2018.
[Grupen et al., 2022] Niko Grupen, Natasha Jaques, Been
Kim, and Shayegan Omidshafiei. Concept-based understanding of emergent multi-agent behavior. In Deep Reinforcement Learning Workshop NeurIPS 2022, 2022.
[He et al., 2016] Kaiming He, Xiangyu Zhang, Shaoqing
Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer
vision and pattern recognition, 2016.
[Hendrycks et al., 2021] Dan Hendrycks, Kevin Zhao,
Steven Basart, Jacob Steinhardt, and Dawn Song. Natural
adversarial examples. In Proceedings of the Conference
on Computer Vision and Pattern Recognition, 2021.
[Huang et al., 2017] Sandy Huang, Nicolas Papernot, Ian
Goodfellow, Yan Duan, and Pieter Abbeel. Adversarial attacks on neural network policies. arXiv preprint
arXiv:1702.02284, 2017.
[Kaplan et al., 2020] Jared Kaplan, Sam McCandlish, Tom
Henighan, Tom B Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
Scaling laws for neural language models. arXiv preprint
arXiv:2001.08361, 2020.

[Kos and Song, 2017] Jernej Kos and Dawn Song. Delving
into adversarial attacks on deep policies. arXiv preprint
arXiv:1705.06452, 2017.
[Lan et al., 2022] Li-Cheng Lan, Huan Zhang, Ti-Rong Wu,
Meng-Yu Tsai, I Wu, Cho-Jui Hsieh, et al. Are alphazerolike agents robust to adversarial perturbations? arXiv
preprint arXiv:2211.03769, 2022.
[Lanctot et al., 2019] Marc Lanctot, Edward Lockhart, JeanBaptiste Lespiau, Vinicius Zambaldi, Satyaki Upadhyay,
Julien Pérolat, Sriram Srinivasan, Finbarr Timbers, Karl
Tuyls, Shayegan Omidshafiei, Daniel Hennes, Dustin
Morrill, Paul Muller, Timo Ewalds, Ryan Faulkner, János
Kramár, Bart De Vylder, Brennan Saeta, James Bradbury,
and David Dingand others. OpenSpiel: A framework for
reinforcement learning in games. 2019.
[Lin et al., 2017] Yen-Chen Lin, Zhang-Wei Hong, YuanHong Liao, Meng-Li Shih, Ming-Yu Liu, and Min Sun.
Tactics of adversarial attack on deep reinforcement learning agents. arXiv preprint arXiv:1703.06748, 2017.
[McGrath et al., 2022] Thomas McGrath, Andrei Kapishnikov, Nenad Tomašev, Adam Pearce, Martin Wattenberg,
Demis Hassabis, Been Kim, Ulrich Paquet, and Vladimir
Kramnik. Acquisition of chess knowledge in alphazero.
Proceedings of the National Academy of Sciences, 2022.
[Metz, 2016] Cade Metz. In two moves, alphago and lee
sedol redefined the future, 2016.
[Neumann and Gros, 2022] Oren Neumann and Claudius
Gros. Scaling laws for a multi-agent reinforcement learning model. arXiv preprint arXiv:2210.00849, 2022.
[Omidshafiei et al., 2022] Shayegan Omidshafiei, Andrei
Kapishnikov, Yannick Assogba, Lucas Dixon, and Been
Kim. Beyond rewards: a hierarchical perspective on
offline multiagent behavioral analysis. arXiv preprint
arXiv:2206.09046, 2022.
[Selvaraju et al., 2017] Ramprasaath R Selvaraju, Michael
Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi
Parikh, and Dhruv Batra. Grad-cam: Visual explanations
from deep networks via gradient-based localization. In
Proceedings of the international conference on computer
vision, 2017.
[Silver et al., 2017a] David Silver, Thomas Hubert, Julian
Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur
Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran,
Thore Graepel, et al. Mastering chess and shogi by
self-play with a general reinforcement learning algorithm.
arXiv preprint arXiv:1712.01815, 2017.
[Silver et al., 2017b] David Silver, Julian Schrittwieser,
Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur
Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian
Bolton, et al. Mastering the game of go without human
knowledge. nature, 2017.
[Wang et al., 2022] Tony Tong Wang, Adam Gleave, Nora
Belrose, Tom Tseng, Joseph Miller, Michael D Dennis,
Yawen Duan, Viktor Pogrebniak, Sergey Levine, and Stuart Russell. Adversarial policies beat professional-level go
ais. arXiv preprint arXiv:2211.00241, 2022.

## Page 9

Table 2: Hyperparameters for AlphaZero and VISA-VIS.

Table 3: Search parameters for AlphaZero and VISA-VIS.

Training Parameters and Hyperparameters

Search Parameters

Parameter
Name

Value
(TTT)

Value
(4x4 TTT)

Value
(Connect Four)

Parameter
Name

Value
(TTT)

Value
(4x4 TTT)

Value
(Connect Four)

Num. Games
Batch Size
Learning Rate
Network Depth
Network Width
λ

5e5
64
1e−3
2
128
1e−4

1.75e6
128
1e−4
4
128
1e−4

7.5e6
256
1e−4
4
128
1e−4

MCTS Simulations
c
Initial τ
τ drop (steps)

25
2.0
1.0
5

25
2.0
1.0
9

50
2.0
1.0
21

A

Training Details (cont’d)

Here we provide additional training details, including the
search parameters, training hyper parameters, and other
learning details that are used when training AlphaZero and
VISA-VIS.
Training parameters Both algorithm’s were trained for
500, 000 games, 1.75M games, 1.75M games, and 7.5M
games for Tic-Tac-Toe, 4x4 Tic-Tac-Toe, and Connect Four,
respectively. The joint policy-value network was represented
by a ResNet backbone with separate heads outputting policy and value predictions, as in Equation (1). Each fullyconnected layer was initialized with a width of 128 neurons
and the depth of the network backbone was scaled to account
for game complexity—-d = 2 ResNet modules for Tic-TacToe, d = 4 for 4x4 Tic-Tac-Toe and Connect Four. Batch size
and learning rate also varied for each game: (i) Tic-Tac-Toe
used a learning rate of 1e−3 and batch size of 64; (ii) 4x4 TicTac-Toe used a learning rate of 1e−4 and batch size of 128;
and (iii) Connect Four used a learning rate of 1e−4 and batch
size of 256. A coefficient of λ = 1e−4 was used to weight
L2-regularization in all games, as described in Equation (5).
Search parameters Recall from Section 3.1 that there are
a number of parameters that influence the search process of
AlphaZero (and therefore VISA-VIS). During training, AlphaZero used 25 simulations of MCTS search per time-step
for Tic-Tac-Toe and 4x4 Tic-Tac-Toe, and 50 simulations per
time-step for Connect Four. For each simulation, the withintree exploration parameter c = 2.0 was used to guide exploration in the upper confidence bound specified by Equation
(2) in all games. The temperature parameter τ , which guides
exploration in the action selection step specified by Equation
(4), was set to an initial value of τ = 1.0 for all games. During each training rollout, τ was dropped after a fixed number
of steps—5, 9, and 21 for Tic-Tac-Toe, 4x4 Tic-Tac-Toe, and
Connect Four, respectively—as in Silver et al. [2017b].
Implementation Our implementation is based on the
open-source AlphaZero provided by DeepMind’s OpenSpiel
project [Lanctot et al., 2019]. VISA-VIS extends the implementation available in OpenSpiel. The most significant implementation detail is the choice of each game’s board representation. For each game, the board state was represented as
a stack of 3 HxW frames—where H and W are the height
and width of the game board, respectively—representing the

pieces of each player (noughts and crosses for both Tic-TacToe variants, red and yellow pieces for Connect Four) and
an extra plane to encode turns (i.e. which player’s turn it is
to move). Each game also encoded the game rules into the
action space, which are used to specify legal moves at each
turn—illegal moves are masked out and assigned a probability of zero by the value network.
Resources All experiments were run on a cluster with 2x18
core Intel Xeon Skylake 6154 CPUs an Nvidia V100 GPU
with 16GB of memory.
