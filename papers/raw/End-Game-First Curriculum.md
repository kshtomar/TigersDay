## Page 1

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

1

Improved Reinforcement Learning with Curriculum
Joseph West1 , Frederic Maire1 , Member, IEEE, Cameron Browne2 and Simon Denman1 , Member, IEEE
1 School of Electrical Engineering and Computer Science, Queensland University of Technology, Brisbane, QLD, 4500, Australia

arXiv:1903.12328v2 [cs.LG] 10 Jun 2019

2 Department of Data Science and Knowledge Engineering, Maastricht University, Maastricht, 6211, Netherlands

Humans tend to learn complex abstract concepts faster if examples are presented in a structured manner. For instance, when
learning how to play a board game, usually one of the first concepts learned is how the game ends, i.e. the actions that lead to
a terminal state (win, lose or draw). The advantage of learning end-games first is that once the actions which lead to a terminal
state are understood, it becomes possible to incrementally learn the consequences of actions that are further away from a terminal
state - we call this an end-game-first curriculum. Currently the state-of-the-art machine learning player for general board games,
AlphaZero by Google DeepMind, does not employ a structured training curriculum; instead learning from the entire game at all
times. By employing an end-game-first training curriculum to train an AlphaZero inspired player, we empirically show that the rate
of learning of an artificial player can be improved when compared to a player not using a training curriculum.
Index Terms—Curriculum Learning, Reinforcement Learning, Monte Carlo Tree Search, General Game Playing.

I. I NTRODUCTION
N artificial game playing agent can utilise either a
knowledge-based method or a brute-force method to
determine which move to make when playing a game [1].
Traditionally, brute-force methods like uninformed tree-search
are effective for games with low state-space complexity whilst
pure knowledge-based methods like direct-coding or neuralnetworks are generally best for games with a low decision
complexity [1]. The current state-of-the-art game playing
agent, Google DeepMind’s AlphaZero [2] uses a combination of brute-force and knowledge-based methods by using
a neural-network to guide a tree-search to decide its moves
when playing a game. Whilst AlphaZero has demonstrated
superhuman performance on a number of different games, we
address a weakness in its method and propose an approach
which results in an improvement in the time the agent takes
to learn.
AlphaZero generates its own training examples as part of
its learning loop through self-play and the generated examples
continually improve as the network learns via reinforcement
learning [3]–[5]. Neural-networks learn via reinforcement
learning by minimising the difference between the network’s
prediction for a state at one time and a future prediction for
the same state at some later time, called temporal difference
(TD) learning [4]. The future prediction can be approximated
by conducting a tree-search to look-ahead, this variation
of TD learning is called TD-Leaf [6] and underpins AlphaZero’s combined tree-search/neural-network design, that is;
AlphaZero uses the difference between the neural-network’s
prediction and the outcome of a tree-search to train its network.
AlphaZero does not use its neural-network to directly make
move decisions, instead it is used to identify the most promising actions for the search to explore and to also estimate
the values of states which are non-terminal. The benefit of

A

Manuscript
received
jd.west@hdr.qut.edu.au

xxxx,

2019;

Corresponding

Author:

this design is that as the neural-network improves the treesearch also improves, however, a characteristic of reinforcement learning is that the network’s predictions are initially
inaccurate. AlphaZero’s method is effective, however many
of the early training examples are of inherently poor quality
as they are generated using an inadequately trained1 neuralnetwork.
The neural-network used in AlphaZero has the net effect of
identifying an initial set of most promising paths for the treesearch to explore. As the search is conducted, if the initial most
promising paths are poorly selected by the neural-network they
can be overridden by a sufficiently deep search, resulting in
a good move decision despite the poor initial most promising
set. A weakness with this approach arises when:
the network is inadequately trained, typically during the
early stages of training; and
• the tree search is less likely to discover terminal states,
typically in the early stages of a game.

•

If the network is inadequately trained, then the set of moves
that the network selects, which are intended to be promising
moves, will likely be random or at best poor. If the treesearch does not find sufficient terminal states then the expansion of the tree is primarily determined by the neuralnetwork, meaning that there may not be enough actual gameenvironment rewards for the tree-search to correct any poor
network predictions. If both of these situations occur together
then we say that the resulting decision is uninformed, which
results in an uninformed training example that has little or
no information about the game that we are trying to learn.
In this paper we demonstrate how employing an end-gamefirst training curriculum to exclude expectedly uninformed
training experiences results in an improved rate of learning for
a combined tree-search/neural-network game playing agent.
1 A network can be inadequately trained either due to poor training or
insufficient training.


## Page 2

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

A. Outline of Experiment
The effect of an end-game-first training curriculum can be
achieved by discarding a fraction of the early-game experiences in each self-play game, with the fraction dependent on
the number of training epochs which have occurred. By selectively discarding the early-game experiences, we hypothesise
that the result is a net improvement in the quality of experiences used for training which leads to the observed improvement in the rate of learning of the player. We demonstrate the
effectiveness of the end-game-first training curriculum on two
games: modified Racing-Kings and Reversi. During training,
we compare the win-ratio’s of two AlphaZero inspired game
playing agents: a baseline player without any modification,
and a player using an end-game-first curriculum (curriculum
player); against a fixed reference opponent.
We find that by using only the late-game data during the
early training epochs, and then expanding to include earlier
game experiences as the training progresses, the curriculum
player learns faster compared to the baseline player. Whilst we
empirically demonstrate that this method improves the player’s
win-ratio over the early stages of training, the curricula used
in this paper were chosen semi-arbitrarily, and as such we do
not claim that the implemented curricula are optimal. We do
however show that an end-game-first curriculum learning approach improves the training of a combined tree-search/neuralnetwork game-playing reinforcement learning agent like AlphaZero.
The structure of the remainder of this paper is as follows.
We review the related work in Section I-B; then outline the
design of our system in Section II; present our evaluation in
Section III; and finally we conclude the paper in Section IV.
For ease of understanding we explain in this paper how our
proposed approach enhances AlphaZero’s method and for the
sake of conciseness only revisit key concepts from [2] where
they are needed to the explain our contribution.
B. Related Work
1) AlphaZero’s Evolution
Google Deepmind’s AlphaGo surprised many researchers
by defeating a professional human player at the game of
Go in 2016 [7], however it was highly customised to the
game of Go. In 2017 AlphaGo Zero was released, superseding
AlphaGo with a generic algorithm with no game-specific
customisation or hand-crafted features, although only tested
on Go [8]. The AlphaGo Zero method was quickly validated
as being generic with the release of AlphaZero, an adaption
of AlphaGo Zero for the games of Chess and Shogi [2].
Both AlphaGo Zero and AlphaGo had a three stage training
pipeline; self-play, optimization and evaluation as shown in
Figure 3 and explained in Section II-B3. AlphaZero differs
primarily from AlphaGo Zero in that no evaluation step is
conducted as explained in Section II-B1b. While AlphaGo
utilises a neural-network to bias a monte-carlo-tree-search
(MCTS) expansion [9], [10], AlphaGo Zero and AlphaZero
completely exclude conducting monte-carlo rollouts during the
tree-search, instead obtaining value estimates from the neuralnetwork. This MCTS process is explained in Section II-B1c.

2

2) Using a Curriculum for Machine Learning
Two predominant approaches are outlined in the literature
for curriculum learning in relation to machine learning: reward
shaping and incrementally increasing problem complexity [11].
Reward shaping is where the reward is adjusted to encourage behaviour in the direction of the goal-state, effectively
providing sub-goals which lead to the final goal. For example,
an agent with an objective of moving to a particular location
may be rewarded for the simpler task of progressing to a
point in the direction of the final goal, but closer to the
point of origin. When the sub-goal can be achieved, the
reward is adjusted to encourage progression closer to the target
position [12]. Reward shaping has been used successfully to
train an agent to control the flight of an autonomous helicopter
in conducting highly non-linear manoeuvres [13]–[15].
Another approach to curriculum learning for a neuralnetwork is to incrementally increase the problem complexity
as the neural-network learns [16]. This method has been
used to train a network to classify geometric shapes by
using a two step training process. In this approach, a simpler
training set was used to initially train the network, before
further training the network with the complete dataset which
contained additional classes. This approach resulted in an
improved rate of learning [17] for a simple neural-network
classifier. Both reward shaping and incrementally increasing
problem complexity rely on prior knowledge of the problem
and some level of human customisation [18], [19].
Florensa et al. [20] demonstrated how an agent can learn
by reversing the problem-space, by positioning an agent at the
goal-state and then exploring progressively further away from
the goal. Their method, reverse curriculum, sets the agent’s
starting position to the goal-state and noisy actions are taken
to obtain other valid states in close proximity to the goal. As
the agent learns, the actions result in the agent moving further
and further from the goal. Their method was demonstrated
to improve the time taken to train the agent in a number of
single-goal reinforcement learning problems including a robot
navigating a maze to a particular location and a robot arm
conducting a single task such as placing a ring on a peg.
The set of problems which the reverse curriculum is suited
to is constrained due to the requirement of a known goal
state, and that the goal states themselves be non-terminal;
i.e. on reaching the goal-state legal actions still exist which
move the agent away from the goal. Having a known goal
state permits the reverse curriculum to be particularly useful
if the problem’s goal state is unlikely to be discovered through
random actions. The weakness with the reverse curriculum is
that if the problem has a number of distinct goal-states, then
focusing on training near a single known goal would result in
the agent being over-fitted to the selected known goal at the
exclusion of all others. The reverse curriculum is not suitable
for games, as games have multiple terminal states which are
not known in advance, and once a terminal state is reached
there are no further legal actions that can be taken.
We define an end-game-first curriculum as one where the
initial focus is to learn the consequences of actions near a
terminal state/goal-state and then progressively learn from
experiences that are further and further from the terminal state.


## Page 3

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

We consider the reverse curriculum to be a special case of the
end-game-first curriculum due to the additional requirements
outlined above. It is not a requirement of the end-game-first
curriculum that the agent commence near a terminal/goal state,
or that one is even known; but in the course of exploration
when a terminal/goal state is discovered the distance to any
visited non-goal states can be calculated and a decision can
be made as to whether those states will be used for training
the agent depending on their distance.
The end-game-first curriculum differs from incrementally
increasing the problem complexity in that the consequences
of actions leading to a terminal state may in fact be more
complex to learn than the earlier transitions. The end-gamefirst curriculum does however temporarily reduce the size
of the problem space by initially training the network on a
smaller subset of the overall problem space. The advantage
of an end-game-first curriculum is that it doesn’t rely on any
prior knowledge of the problem - including requiring a known
terminal state. By first focusing near a terminal state, the agent
is trained to recognise the features and actions which give rise
to environmental rewards (terminal states) then progressively
learns how to behave further and further from these states.
We demonstrate in this paper that using an end-game-first
curriculum for training a combined tree-search/neural-network
game playing agent can improve the rate at which the agent
learns.
II. S YSTEM D ESIGN
The system consists of two core modules: the game environment and the players. Both the environment and the players
utilise a standard framework, regardless of the game or the
type of player. The system is designed in such a way as to
reduce the variability between compared players. The agent’s
performance during both training and gameplay is traded off
in favour of reducing the variability of the experiments. For
example, the training pipeline used for the neural-network is
conducted sequentially, however better performance could be
obtained by conducting the training in parallel. Likewise, when
two players are compared they are both trained on the same
system simultaneously to further reduce the impact of any
variance in processor load. The result of this design decision
is that the complexity of the agent, the difficulty of the games
and the quality of the opponents are constrained to permit
the simultaneous training of two AI agent’s, using sequential
processes, within a period of time which is reasonable yet
is still sufficiently complex to demonstrate the effectiveness
of the presented method. Our focus is not on the absolute
performance of the agent with respect to any particular game,
but instead the comparative improvement of using an endgame-first training curriculum.
A. Game Environment
The game environment encodes the rules and definitions
of the game as well as maintains the progress of the game,
permitting players to make moves and updating the game
state accordingly. The players query the game environment
to inform their decisions. A game has a set of m possible

3

actions, A = {a1 , .., am }. For any game state, the environment
provides:
• a tensor of sufficient size to represent the current state s;
• a bit array b(s) of length m with bits representing the
legal actions set to 1;
• a vector representing the set of legal actions L(s) ⊂ A
in a format accepted by the environment’s move function
- internally mapped to b(s); NB that A is used to denote
the set of all actions while a is used to denote a single
action; and
• a scalar w indicating if the game: is ongoing, a draw,
player 1 wins, or player 2 wins.
1) Games
We evaluate the effectiveness of using an end-game-first
curriculum on two games; a modified version of Racing Kings
[21] and Reversi [22]. The games selected have fundamental
differences in how a player makes their move. Reversi is a
game which the board starts nearly empty and as the game
progresses, tiles are placed on the board filling the board up
until the game ends; tiles are not relocated once they are
placed. Games like Tic-Tac-Toe, Connect Four, Hex and Go
have the same movement mechanics as Reversi and with the
exception of the occasional piece removal in Go, these games
also fill the board as the game progresses. Racing Kings was
chosen as the representative of the class of games that have
piece mobility, i.e. where a piece is moved by picking it up
from one cell and placing it at another. Like Racing Kings,
often games with piece mobility also have the characteristic of
piece removal through capture. Games with similar movement
mechanics as Racing Kings includes the many Chess variants,
Shogi and Nim.
a) Racing Kings: Racing Kings is a game played on
a Chess board with Chess pieces. Pieces are placed on a
single row at one end of the board with the aim being to
be the first player to move their King to the other end of the
board. Checkmate is not permitted, and neither is castling,
however pieces can be captured and removed from the board.
We modify the full Racing Kings game by using less pieces
and adjusting the starting position of the pieces by placing
them in the middle of the board, as shown in Figure 1, instead
of on the first rank. As the Racing Kings game library is part
of a suite of Chess variants [23] we maintain the environment
as it would be for Chess. A state s is represented as a tensor of
size 8 × 8 × 12; the width and height dimensions representing
cells on the board and the 12 planes representing each of
the player’s pieces; King, Queen, Rook, Bishop, Knight and
Pawn. As the pieces are moved from one cell to another the
total number of actions includes all possible pick and place
options, 64 × 64 = 4096, excluding any additional actions
such as promotion of pawns. There are 44 possible from/to
pawn promotion movements in Chess but this also needs to
be multiplied by the number of pieces which a pawn can
be promoted to. We only consider promotion to a Knight
or a Queen giving 88 possible promotion actions. The total
number of actions for the Racing Kings game environment is
m = 4184.
b) Reversi: Reversi “is a strategic boardgame which
involves play by two parties on an eight-by-eight square grid


## Page 4

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

(a) Racing Kings

(b) Reversi

Fig. 1: Starting board positions for Modified Racing Kings
and Reversi. The aim of Racing Kings is to move your King
to the final rank (top row), while the aim of Reversi is to have
more pieces on the board when neither player can no longer
move.
with pieces that have two distinct sides ... a light and a dark
face. There are 64 identical pieces called ‘disks’. The basic
rule of Reversi, if there are players discs between opponents
discs, then the discs that belong to the player become the
opponents discs” [24]. The winner is the player with the most
tiles when both players have no further moves. For the Reversi
game environment a state s is represented as a tensor of size
8 × 8 × 2; the width and height dimensions representing cells
on the board and each plane representing the location of each
players’ pieces, with 64 possible different actions m = 64.
Our Reversi environment is based on [25].
B. Player Architecture
Two Artificial Intelligence (AI) players: a baseline player
without any modifications, and a curriculum player using a
specified end-game-first training curriculum, are tested in a
contest against a game specific reference opponent.
1) Artificial Intelligence Players
The architecture of both players is a combined neuralnetwork/MCTS reinforcement learning system which chooses
an action a from the legal moves L(s) for any given state
s. The neural-network with parameters θ is trained by selfplay reinforcement learning. As the players are inspired by
AlphaZero [2], only the components relevant to this paper will
be covered in this section.
a) Neural-Network: A deep residual convolutional
neural-network is used for both players with parameters as
shown in Tables I and II. Deep residual convolutional neuralnetworks have been found to be stable and less prone to
overfitting than traditional convolutional neural-networks [26]
for large networks. The architecture is shown in Figure 2.
The input to the neural-network is a state tensor s from
the game environment. The neural-network has two outputs,
a policy vector pθ (s), and a scalar value estimate vθ (s) for
the given s. pθ (s) is an m sized vector, indexed by actions a,
representing the probability distribution of the best actions to
take from s.

4

b) Training Pipeline: The training pipeline consists of
two independent processes; self-play and optimisation. The
initial neural-network weights θi=0 are randomised and after
each training iteration, i, are updated yielding θi+1 . The
self-play process plays games against itself using the latest
weights to generate training examples, filling an experience
buffer. The optimisation process trains the network using
these experiences via batched gradient descent. The process
is shown in Figure 3(a).
Unlike AlphaZero where the processes are executed in
parallel across multiple systems for maximum performance,
we seek to reduce the variability of training by conducting
the training process sequentially on a single system. When the
experience buffer contains the experiences from a set number
of games, self-play stops and optimisation begins. After optimisation has finished all experiences from a percentage of the
earliest games are removed from the experience buffer, and
self-play recommences.
Training experiences are generated during self-play using
the latest network weights. Each training experience X is
comprised of a set X := {s, π(s), z(s)} where s is the tensor
representation of the state, π(s) is a probability distribution
of most likely actions (policy) indexed by a obtained from
the MCTS, and z(s) is the scalar reward from the perspective
of the player for the game’s terminal state. During self-play,
an experience is saved for every ply2 in the game, creating
an experience buffer full of experiences from a number of
different self-play games. z(s) = rT (s) where rT (s) is the
reward for the terminal state of the game and is −1 for a loss,
+1 for a win, and −0.5 for a draw. We use a slightly negative
reward for a draw instead of 0 to discourage the search from
settling on a draw, instead preferring exploration of other
nodes which are predicted as having a slightly negative value.
During training, the experiences are randomised, and parameters θ are updated to minimise the difference between vθ (s)
and z(s), and maximise similarities between pθ (s) and π(s)
using the loss function shown in Equation 1. One training
step is the presentation of a single batch of experiences for
training, while an epoch t is completed when all experiences
in the buffer have been utilised. 20 epochs are conducted for
each training iteration i. The experiences are stored in the
experience buffer in the order in which they were created. At
the conclusion of each iteration the buffer is partially emptied
by removing all experiences from a portion of the oldest
games.
l = (z(s) − vθ (s))2 − π(s) · log(pθ (s)) + c · ||θ||2

(1)

where:
z(s)
= The reward from the game’s terminal state (return).
c · ||θ||2 = L2 weight regularisation.
vθ (s) = Neural-network value inference.
π(s)
= Policy from the Monte-Carlo Tree Search (MCTS).
pθ (s) = Neural-network policy inference.
2 The term ‘ply’ is used to disambiguate one player’s turn which has
different meanings in different games. One ply is a player’s single action.


## Page 5

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

5

Fig. 2: Architecture for the AI players with input s from the environment and outputs pθ (s) and vθ (s). Network parameters
are shown in Tables I and II. Each residual block has 3x3 convolution has 512 filters when playing Reversi and 256 filters
when playing Racing Kings.

The value of the node V (s).
The average action-value for a particular action from a
node Q(s, a).
MCTS is conducted as follows:
• Selection. The tree is traversed from the root node by
calculating the upper confidence bound U (s, a) using
Equation 2 and selecting actions ai = argmax(U (s, a))
•

•

a

Fig. 3: Two training loops used to demonstrate effectiveness
of curriculum learning. a. AlphaZero inspired and b. AlphaGo
Zero inspired.

until a leaf node is found [2]. Note the use of pθ (s) from
the neural-network in Equation 2.
• Expansion. When a leaf node is found vθ (s) and pθ (s)
are obtained from the neural-network and a node is added
to the tree.
• Evaluation. If the node is terminal the reward r is
obtained from the environment for the current player and
V (s) := r otherwise V (s) := vθ (s). Note that there is
no Monte Carlo roll-out.
• Backpropogation. Q(s, a) is updated by the weighted
average of the old value and the new value using Equation
4.
s
m
P
N (s, aj )

For completeness we also separately conduct an additional
experiment using AlphaGo Zero’s method of adding a third
process to evaluate the best weights θ? [7]. The primary
difference for this experiment is that self-play is conducted
with the best weights θ? instead of the current weights, as
shown in Figure 3(b).
c) Monte-Carlo Tree Search (MCTS): The MCTS builds
an asymmetric tree with the states s as nodes, and actions a
as edges. At the conclusion of the search the policy π(s) is
generated by calculating the proportion of the number of visits
to each edge.
During the tree-search the following variables are stored:
The number of times an action was taken N (s, a).
• The neural-network’s estimation of the optimum policy
pθ (s) and the network’s estimate of the node’s value
vθ (s).

•

U (s, a) = Q(s, a) + 3 · p0a (s)

j=1

1 + N (s, a)

(2)

p0 (s) = pθ (s) + δ

(3)

N (s, a) · Q(s, a) + V (s)
Q(s, a) ←
N (s, a) + 1

(4)

where
Q(s, a)
= Average action value.
pθ (s)
= Policy for s from neural-network.
δ
= The Dirichlet noise function.
p0 (s)
= pθ (s) with added Dirichlet noise.
m
P
N (s, aj ) = Total visits to parent node.
j=1

N (s, a)
V (s)

= Number of visits to edge (s, a).
= The estimated value of the node.

At the conclusion of the search a probability distribution
π(s) is calculated from the proportion of visits to each edge


## Page 6

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

N (s,a )

j
πaj (s) = P N (s,A)
, j = 1, .., m. The tree is retained for reuse
in the player’s next turn after it is trimmed to the relevant
portion based on the opponent’s action.

d) Move Selection: To ensure move diversity during each
self-play game, for a given number of ply as detailed in Tables
I and II, actions are chosen probabilisticly based on π(s) after
the MCTS is conducted. After a given number of ply, actions
are then chosen in a greedy manner by selecting the action with
the largest probability from π(s). In a competition against a
reference opponent, moves are always selected greedily.
2) Curriculum Player
The difference between the baseline player and the curriculum player is that some experiences are excluded from
the experience buffer during training of the curriculum player.
A curriculum function ζ(t) is introduced which indicates the
percentage of experiences to be retained for a given game,
depending on the number of epochs t which have occurred.
When storing a game’s experiences the first (1 − ζ(t))% of
experiences are excluded from a game. This can be achieved
by either trimming the game’s experiences or not generating
the experience in the first place. The baseline player has an
equivalent function ζ(t) = 1 for all t; that is retaining 100% of
a game’s experiences. The curriculum used for Racing Kings
is shown in Equation 5, while the curriculum used for Reversi
is shown in Equation 6.
We exploit the fact that the curriculum player excludes some
early-game experiences during training by not conducting a
tree-search for actions which will result in an experience
that is going to be excluded. Instead of naively conducting
a full search and discarding the experiences, we randomly
choose the actions for the estimated number that would have
been discarded. We manage the number of random actions
by maintaining the average ply, av, in a game, and playing
(1 − ζ(t))% · av of the actions randomly. After these random
actions, we then use the full MCTS to choose the remaining
actions to play the game. If a terminal state is found during
random play then the game is rolled back ζ(t)% ply and
MCTS is used for the remaining actions.

0.1 t = 0





0.33 0 < t < 100



0.5
100 ≤ t < 500
ζracingkings (t) =
(5)

0.66 500 ≤ t < 800





0.8
800 ≤ t < 1000



1.0
1000 ≤ t

0.25 t = 0



0.5
0 < t < 100
ζreversi (t) =
(6)

0.75 100 ≤ t < 500



1.0
500 ≤ t
3) AlphaGo Zero Inspired Player
The AlphaGo Zero inspired player is very similar to the
player explained above (see Section II-B2), with the exception
of an evaluation step in the training loop. The evaluation step
plays a two player competition between the current best player

6

with weights θ? and a challenger using the latest weights
θi . The competition is stopped when the lower bound of
the 95% percentile Wilson confidence score with continuity
correction [27] is above 0.5, or the upper bound is below 0.5
allowing a competition winner to be declared. The competition
is also stopped when the difference between the upper and
lower confidence interval is less than 0.1, in which case no
replacement is conducted. If the challenger is declared the
winner of the competition, then it’s weights become the best
weights and are used for subsequent self-play until they are
replaced after a future evaluation competition. Although this
method for stopping does not provide exactly 95% confidence
[28], it provides sufficient precision for determining which
weights to use to create self-play training examples.
4) Reference Opponent
The reference opponent provides a fixed performance opponent for testing the quality of the AI players. Stockfishmultivariant [29] is used as the reference opponent for Racing
Kings; whilst an MCTS player using 200 simulations [10] is
used for Reversi.
III. E XPERIMENTS
Two AI players are trained: one using the proposed curriculum learning approach (curriculum player), the other without
(baseline player). A competition is periodically conducted during training between the AI players and a reference opponent,
and the players’ respective win ratios yθi are recorded. A
competition consists of a minimum of 30 games for each
randomly selected θi to obtain the win ratio. The experiment is
conducted in full three times and the results are combined. The
moving average of the win ratio yθi from all three experiments
is plotted against time, steps and epochs. A training step is
completed after the presentation of one batch of experiences
and a training epoch is completed after all experiences have
been presented once. As training begins after a set number
of games, the number of steps for each epoch varies from
experiment to experiment, likewise the time taken to play a
game is also completely unique making the time, epochs and
steps independent from experiment to experiment. As such,
when combining data from multiple experiments the three
plots may have different appearances.
Whilst yθi vs time is the measure which we are primarily
interested in, measuring against steps and epochs are also
informative. We mitigate the potential differences which might
arise from differing system loads by training both AI players
simultaneously on one dual-GPU system.
A. Results
Figure 4 shows the win ratio of the AlphaZero inspired
players in a Racing Kings competition against the Stockfish
opponent; Figure 5 shows the win ratio of the AlphaZero
inspired players playing Reversi against the MCTS opponent;
and Figure 6 shows the win ratio whilst playing Racing Kings
against the Stockfish opponent but using an AlphaGoZero
inspired player with the added evaluation step.
The win ratio of the player using the end-game-first training
curriculum exceeds the baseline player during the early stages


## Page 7

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

7

(a)

(a)

(b)

(b)

(c)
Fig. 4: Win ratio for AlphaZero inspired player vs Stockfish
level 2 playing Racing Kings with and without training curriculum from Equation 5. Note that improvement is seen in
both the time and the steps figures, indicating that the performance improvement is more than the time saved by conducting
random moves during self-play. Note the unlearning which
occurs around 1000 epochs for the player with a curriculum.
This plot is the 20 point moving average of 3 independent
training runs.

(c)
Fig. 5: Win ratio for AlphaZero inspired player vs MCTS with
200 simulations playing Reversi with and without the training
curriculum from Equation 6. Note that the epoch win ratio
of both players is similar despite the player with curriculum
learning having less training examples per epoch due to the
dropped experiences. This plot is the 20 point moving average
of 3 independent training runs.


## Page 8

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

8

of training in all cases when measured against time. This
was also observed across multiple training runs with differing
network parameters and differing curricula.
B. Discussion

(a)

(b)

(c)
Fig. 6: Win ratio for AlphaGoZero inspired player vs
Stockfish level 2 playing Racing Kings with and without
training curriculum from Equation 5. This plot is the 20 point
moving average of 3 independent training runs.

Our results indicate that a player trained using an end-gamefirst curriculum learns faster than a player with no curriculum
during the early training periods. The improvement over time
can, in part, be attributed to the increased speed of selfplay moves being chosen randomly by the curriculum player
instead of conducting a full search; however the win ratio
improvement is also observed when compared over training
steps indicating that a more subtle benefit is obtained. The
win ratio of the two players when compared against training
epoch shows similar performance for the two players.
In this section we compare and contrast the performance
of the curriculum player and the baseline player against the
reference opponents and separately discuss the results with
respect to time (Subsection III-B1), training steps (Subsection
III-B2) and epochs (Subsection III-B3).
1) Win ratio vs Time comparison
Subfigure (a) of Figures 4, 5 and 6 shows the win ratio
of the AI players vs Time. The curriculum player does not
retain experiences generated early in a game during the early
training periods. This is achieved by selecting these moves
randomly instead of using a naive approach of conducting
a full search, and dropping the early-game experience as
explained in Section II-B2. The time saved as a result of
randomly selecting moves instead of conducting a tree-search
can be significant, however by using random move selection
no experience is added to the experience buffer meaning that a
randomly selected move contributes in no way to the training
of the player. The curriculum needs to balance the time saved
by playing random moves with the reduction in generating
training experiences.
Whilst the curriculum player leads in win ratio compared
to the baseline player during the early time periods, the win
ratios of both players converge in all experiments. For a
given neural network configuration there is some expected
maximum performance threshold against a fixed opponent;
in the ideal case this would be a 100% win ratio but if the
network is inadequate it may be less. Although we expect that
the win ratios of the two players would converge eventually,
it appears that convergence occurs prior to the maximum
performance threshold as shown in Figure 4(a) near 4000
minutes and Figure 6(a) near 5000 minutes; while Figure 5(a)
shows convergence near 6000 minutes at what appears to be
the networks maximum performance threshold. For an optimal
curriculum, the convergence of the two players would be
expected to occur only at the maximum performance threshold.
2) Win ratio vs Steps comparison
Subfigure (b) in Figures 4, 5 and 6 shows the win ratio of
the AI players vs the number of training steps. Although the
win ratio improvement over time for the curriculum player can
be attributed in part to the use of random move selection, a
win ratio improvement is also observed when measured against
training steps. A training step is when one batch of experiences


## Page 9

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

is presented to the optimise module, making a training step
time independent; i.e. how long it takes to create an experience
is not a factor for training steps.
Given that the win ratio for the curriculum player outperforms the baseline player when measured over training steps,
we argue that there is a gain which relates directly to the
net quality of the training experiences. Consider, for example,
the baseline player’s very first move decision in the very first
self-play game for a newly initialised network. With a small
number of MCTS simulations relative to the branching factor
of the game, the first AI player’s decision will not build a
tree of sufficient depth to reach a terminal node, meaning
that the decision will be derived solely from the untrained
network weights - recall that the neural-network is used for
non-terminal value estimates. The resulting policy π which is
stored in the experience buffer has no relevance to the game
environment as it is solely reflective of the untrained network.
We posit that excluding these uninformed experiences results
in a net improvement in the quality of examples in the training
buffer. Later in that first game, terminal states will eventually
be explored, over-ruling the inaccurate network estimates, and
the resulting policy will become reflective of the actual game
environment - these are experiences which should be retained.
As the training progresses the network is able to make more
accurate predictions further and further from a terminal state
creating a visibility horizon which becomes larger after each
epoch. The optimum curriculum would match the change of
the visibility horizon.
3) Win ratio vs Epoch comparison
Subfigure (c) in Figures 4, 5 and 6 shows the win ratio
of the players vs the number of training epochs. Recall that
an epoch is when all experiences in the experience buffer
have been presented to the optimise module. Since training
is only conducted when the experience buffer has sufficient
games, an epoch is directly proportional to the number of
self-play games played, but is independent of the number of
experiences in the buffer and the time it takes to play a game.
When applying the curriculum, fewer experiences per game
are stored during the early training periods compared to the
baseline player, meaning that the curriculum player is trained
with fewer training experiences during early epochs.
The plots of the win ratio vs epoch shows the curriculum
player outperforming the baseline player for the early epochs
in Figure 4(c) with the two win ratios converging rapidly;
while Figures and 5(c) and 6(c) show the two players performing similarly. The similarity of the results when measured
against epochs shows that despite the curriculum player excluding experiences, no useful information is lost in doing so.
In recognising that useful information is not excluded during
these early epochs it supports our view that the net quality of
the data in the experience buffer is improved by applying the
specified end-game-first curriculum.
It is expected that due to a combination of the game
mechanics and the order in which a network learns there may
be some learning resistance3 which could result in plateaus
3 To our knowledge, the term learning resistance is not defined in relation
to machine learning. We define it to mean a short term resistance to network
improvement.

9

in the win ratio plot, allowing the trailing player to catch up
temporarily. We expect a sub-optimal curriculum to result in
additional learning resistance or in the extreme case learning
loss which would predominantly be observed immediately
following a change in the curriculum value. For Reversi
the final curriculum increment from Equation 6 occurs after
500 Epochs and Figure 5(c) shows a training plateau shortly
after this change, albeit at the network’s maximum learning
limit. Likewise in Figure 4(c) a learning loss is observed at
an average of 900 epochs shortly after the curriculum has
changed to 80% as shown in Equation 5, although of the three
experiments that comprise the data for this plot two of them
have a learning loss around 800 epochs and the other at 1000
epochs - the final step in the curriculum. Figure 4(c) appears
to indicate that on average the learning loss is caused by the
final steps of the curriculum, however the learning loss was
not observed in all training runs. The presence of this loss
in the average of training runs, but the absence from some
individual training runs highlights the importance of the order
in which learning occurs and its impact on the effectiveness
of the curriculum.
4) Curriculum Considerations
Although it is expected that the baseline player’s performance would converge with the curriculum player’s performance, ideally this would occur near the maximum win ratio
or at some training plateau. The fact that the player’s win
ratios converge before a clear training plateau has been reached
indicates that the curriculum is sub-optimal, and given that
the curriculum implemented is semi-arbitrary this is expected.
While curriculum learning is shown to be beneficial during the
early stages of training, the gain can be lost if the curriculum
changes too slowly or too abruptly.
When designing a curriculum, consideration needs to be
given to the speed at which the curriculum changes. At
one extreme is the current practice where the full game is
attempted to be learnt at all times, i.e. the curriculum is too
fast by immediately attempting to learn from 100% of a game
at epoch 0. At the other extreme is where the curriculum
is too slow, which can result in the network overfitting to a
small portion of the game space or discarding examples which
contain useful information. We argue that each training run for
each game could have its own optimum curriculum profile, due
to the different training examples which are generated.
IV. C ONCLUSION
The rate at which an AI player learns when using a
combined neural network-MCTS architecture can be improved
by using an end-game-first training curriculum. Although the
hand-crafted curricula used in this study are not optimal, a
fixed curriculum is not likely to be optimal at all times as the
order and the composition of the experiences are themselves
a factor. The following considerations are required for an
optimal curriculum;
• Balancing the time saved by random moves with the loss
of training experiences.
• Minimising training plateaus that are not related to game
complexity.


## Page 10

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

Parameter
Resnet Blocks
Batch Size
Value function size
CNN Filters
Simulations per action
Drop draw chance

Value
10
350
256
512
200
0.5

Min Buffer Size

350; 500; 750;
1000;
1500;
2000
0.1
30

Experience replacement
Number probabilistic turns

10

Comment
Number of Residual Network Blocks
Training Batch size
Neurons before vθ(s)
How many filters are used
How many nodes are added to the tree during the MCTS tree search
If a game is a draw then it has this chance of being dropped. Used to mitigate if
a game has a lot of draws from having too many experiences which are draws
Minimum number of games in the training buffer to stop self-play and
commence training. For Chess the number of required games was incremented
to allow training to commence earlier
Proportion of experiences removed from training buffer after self-play
Number of players actions which are selected probabilistically from the policy
before switching to selecting moves greedily

TABLE I: Parameters for Racing Kings player
Parameter
Resnet Blocks
Batch Size
Value function size
CNN Filters
Simulations per action
Drop draw chance

Value
10
350
256
512
200
0.1

Min Buffer Size

2000

Experience replacement
Number probabilistic turns

0.1
16

Comment
Number of Residual Network Blocks
Training Batch size
Neurons before vθ(s)
How many filters are used
How many nodes are added to the tree during the MCTS tree search
If a game is a draw then it has this chance of being dropped. Used to mitigate if
a game has a lot of draws from having too many experiences which are draws
Minimum number of games in the training buffer to stop self-play and
commence training
Proportion of experiences removed from training buffer after self-play
Number of players actions which are selected probabilistically from the policy
before switching to selecting moves greedily

TABLE II: Parameters for Reversi player

The curriculum profile changes are not too fast as to
include uninformed examples.
• The curriculum profile changes are not too slow as to
cause the network to overfit to a smaller portion of the
environment space.
• The curriculum profile changes are not too slow as to result in discarding examples that are sufficiently informed.
To address these requirements, the curriculum profile should
relate to the visibility horizon of the tree search, not just the
number of training iterations. Our future research will explore
how a curriculum can be automated based on the visibility
horizon of the player’s search.
•

ACKNOWLEDGMENT
Computational resources and services used in this work
were provided by the HPC and Research Support Group,
Queensland University of Technology, Brisbane, Australia.
R EFERENCES
[1] H. van den Herik, J. W. Uiterwijk, and J. van Rijswijck, “Games solved:
Now and in the future,” Artificial Intelligence, vol. 134, no. 1-2, pp.
277–311, Jan. 2002.
[2] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez,
M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. P. Lillicrap, K. Simonyan, and D. Hassabis, “Mastering chess and shogi by self-play with
a general reinforcement learning algorithm,” CoRR, vol. abs/1712.01815,
2017.
[3] E. J. Sondik, “The optimal control of partially observable markov processes over the infinite horizon: Discounted costs,” Operations Research,
vol. 26, no. 2, pp. 282–304, 1978.
[4] R. S. Sutton, “Learning to predict by the methods of temporal differences,” in Machine Learning. Kluwer Academic Publishers, 1988, pp.
9–44.

[5] R. S. Sutton and B. Tanner, “Temporal-difference networks,”
arXiv:1504.05539 [cs], 2015.
[6] J. Baxter, A. Tridgell, and L. Weaver, “TDLeaf(lambda): Combining
temporal difference learning with game-tree search,” arXiv:cs/9901001,
1999.
[7] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den
Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever,
T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis,
“Mastering the game of go with deep neural networks and tree search,”
Nature, vol. 529, no. 7587, pp. 484–489, 2016.
[8] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang,
A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap,
F. Hui, L. Sifre, G. van den Driessche, T. Graepel, and D. Hassabis,
“Mastering the game of go without human knowledge,” Nature, vol.
550, no. 7676, pp. 354–359, 2018.
[9] L. Kocsis and C. Szepesvári, “Bandit based monte-carlo planning,” in
Proceedings of the 17th European Conference on Machine Learning, ser.
ECML’06. Berlin, Heidelberg: Springer-Verlag, 2006, pp. 282–293.
[10] C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling,
P. Rohlfshagen, S. Tavener, D. Perez, S. Samothrakis, and S. Colton, “A
Survey of Monte Carlo Tree Search Methods,” IEEE Transactions on
Computational Intelligence and AI in Games, vol. 4, no. 1, pp. 1–43,
Mar. 2012.
[11] J. L. Elman, “Learning and development in neural networks: the importance of starting small,” Cognition, vol. 48, no. 1, pp. 71–99, 1993.
[12] H. Asoh, S. Hayamizu, I. Hara, Y. Motomura, and S. Akaho, “Socially
embedded learning of the office-conversant mobile robot jijo-2,” in
Proceedings of the Fifteenth International Joint Conference on Artificial
Intelligence, vol. 2, 1997, pp. 880–885.
[13] A. Y. Ng, D. Harada, and S. J. Russell, “Policy invariance under
reward transformations: Theory and application to reward shaping,”
in Proceedings of the Sixteenth International Conference on Machine
Learning, ser. ICML ’99. Morgan Kaufmann Publishers Inc., 1999,
pp. 278–287.
[14] A. Y. Ng, H. J. Kim, M. I. Jordan, and S. Sastry, “Autonomous helicopter ight via reinforcement learning,” Neural Information Processing
Systems, vol. 16, p. 8, 2004.
[15] A. Y. Ng, A. Coates, M. Diel, V. Ganapathi, J. Schulte, B. Tse, E. Berger,


## Page 11

IEEE TRANSACTIONS ON GAMES, VOL. XX, NO. X, MARCH 2019

and E. Liang, “Autonomous inverted helicopter flight via reinforcement
learning,” in Experimental Robotics IX, ser. Springer Tracts in Advanced
Robotics, M. H. Ang and O. Khatib, Eds. Springer Berlin Heidelberg,
2006, pp. 363–372.
[16] Y. J. Lee and K. Grauman, “Learning the easy things first: Self-paced
visual category discovery,” in CVPR 2011. IEEE, 2011, pp. 1721–1728.
[17] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum
learning,” in Proceedings of the 26th Annual International Conference
on Machine Learning - ICML ’09. ACM Press, 2009, pp. 1–8.
[18] L. Jiang, D. Meng, Q. Zhao, S. Shan, and A. G. Hauptmann, “Selfpaced curriculum learning,” in AAAI Publications, Twenty-Ninth AAAI
Conference on Artificial Intelligence, 2015, p. 7.
[19] M. P. Kumar, B. Packer, and D. Koller, “Self-paced learning for latent
variable models,” Neural Information Processing Systems, p. 9, 2010.
[20] C. Florensa, D. Held, and M. Wulfmeier, “Reverse curriculum generation
for reinforcement learning,” p. 14, 2017.
[21] Racing kings. [Online]. Available: https://lichess.org/variant/racingKings
[22] T. Landau. Othello: Brief & basic. [Online]. Available: http:
//www.tlandau.com/files/Othello-B&B.pdf
[23] N. Fiekas. A pure python chess library with move generation and
validation. [Online]. Available: https://github.com/niklasf/python-chess
[24] Gunawan, H. Armanto, J. Santoso, D. Giovanni, F. Kurniawan, R. Yudianto, and Steven, “Evolutionary neural network for othello game,”
Procedia - Social and Behavioral Sciences, vol. 57, pp. 419–425, 2012.
[25] K. Morishita. Reversi reinforcement learning by AlphaGo zero
methods.: mokemokechicken/reversi-alpha-zero. [Online]. Available:
https://github.com/mokemokechicken/reversi-alpha-zero
[26] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” CoRR, vol. abs/1512.03385, 2015.
[27] J. F. Reed, “Better Binomial Confidence Intervals,” Journal of Modern
Applied Statistical Methods, vol. 6, no. 1, pp. 153–161, May 2007.
[28] J. Frey, “Fixed-width sequential confidence intervals for a proportion,”
The American Statistician, vol. 64, no. 3, pp. 242–249, 2010.
[29] D. Dugovic. Multi-variant fork of popular UCI chess engine. [Online].
Available: https://github.com/ddugovic/Stockfish

Joseph West completed Bachelor of Engineering
and Bachelor of Computer Science degrees from the
University of New South Wales - Australian Defence
Force Academy, Canberra, ACT, Australia in 2005
and a Masters in Engineering Science in 2009. He
is currently researching General Game Playing using
Reinforcement Learning at Queensland University of
Technology for his Ph.D.

Frederic Maire received the Ph.D. degree in discrete mathematics from the Universite Pierre et
Marie Curie, Paris 6, France, in 1993, after completing M.Sc. degrees in pure mathematics and computer
science engineering in 1989. Currently, he is a senior
lecturer at the School of Electrical Engineering and
Computer Science, Queensland University of Technology, Brisbane, Australia. His research interests
include computer vision and robotics. Dr. Maire is
a member of the IEEE.

11

Cameron Browne is an Associate Professor at
Maastricht University’s Department of Data Science and Knowledge Engineering (DKE), where he
is running the e2m ERC-funded Digital Ludeme
Project over the next five years. He received the
Ph.D. degree from QUT in 2009, producing the
world’s first published computer-generated games.
Dr Browne is the author of the books Hex Strategy,
Connection Games and Evolutionary Game Design,
which won the 2012 GECCO Humies award for
human-competitive results in evolutionary computation. He is a Section Editor of the IEEE Transactions on Games and the
International Computer Games Association (ICGA) journal, and is the founder
and Editor-in-Chief of the Game & Puzzle Design journal.

Simon Denman received a BEng (Electrical), BIT,
and PhD in the area of object tracking from the
Queensland University of Technology (QUT) in
Brisbane, Australia. He is currently a Senior Lecturer with the School of Electrical Engineering and
Computer Science at the Queensland University of
Technology. His active areas of research include
computer vision and spatio-temporal machine learning. Dr. Denman is a member of the IEEE.


