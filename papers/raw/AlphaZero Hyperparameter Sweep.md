## Page 1

Hyper-Parameter Sweep on AlphaZero General

arXiv:1903.08129v1 [cs.LG] 19 Mar 2019

Hui Wang, Michael Emmerich, Mike Preuss, Aske Plaat
Leiden Institute of Advanced Computer Science, Leiden University,
Leiden, the Netherlands
h.wang.13@liacs.leidenuniv.nl
http://www.cs.leiden.edu

Abstract. Since AlphaGo and AlphaGo Zero have achieved breakground
successes in the game of Go, the programs have been generalized to solve
other tasks. Subsequently, AlphaZero was developed to play Go, Chess
and Shogi. In the literature, the algorithms are explained well. However, AlphaZero contains many parameters, and for neither AlphaGo,
AlphaGo Zero nor AlphaZero, there is sufficient discussion about how
to set parameter values in these algorithms. Therefore, in this paper, we
choose 12 parameters in AlphaZero and evaluate how these parameters
contribute to training. We focus on three objectives (training loss, time
cost and playing strength). For each parameter, we train 3 models using
3 different values (minimum value, default value, maximum value). We
use the game of play 6×6 Othello, on the AlphaZeroGeneral open source
re-implementation of AlphaZero. Overall, experimental results show that
different values can lead to different training results, proving the importance of such a parameter sweep. We categorize these 12 parameters
into time-sensitive parameters and time-friendly parameters. Moreover,
through multi-objective analysis, this paper provides an insightful basis
for further hyper-parameter optimization.
Keywords: AlphaZero, parameter sweep, parameter evaluation, multiobjective analysis

1

Introduction

In recent years, many researchers are interested in deep reinforcement learning.
The AlphaGo series algorithms [1,2,3] are representative achievements of deep
reinforcement learning, which have been greatly promoting the development and
application of artificial intelligence technologies. AlphaGo [1] achieves the superhuman ability of playing game of Go. AlphaGo applies tree search to evaluate
the positions and selects moves from the trained neural networks. However, this
neural network is trained from human experts’ game playing data. AlphaGo
Zero [2] is the update version of AlphaGo and masters game of Go without human knowledge. AlphaGo Zero collects game playing data purely from self-play
for training neural network and also evaluates the positions and selects moves
from the trained neural network. AlphaZero [3] is a generalized version of AlphaGo Zero and claims a general framework of playing different games (such as


## Page 2

2

Hui Wang et al.

Go, Chess and Shogi) without human knowledge. AlphaZero framework presents
the strong adaptability of such a kind of deep reinforcement learning algorithm
which combines self-play, neural network and tree search to solve game playing
problems. Therefore, inspired by AlphaGo series algorithms, many analysis reviews, applications and optimization methods [4,5,6] have been proposed and
made the deep reinforcement learning be a more and more hot and practical
research field.
However, although a lot of research work have been doing based on AlphaGo
series approaches and showing the practicalness and adaptiveness in many different application backgrounds [7,8]. There is insufficient discussion of how to set
the parameters in AlphaGo series algorithms. As for an algorithm, apparently,
different parameters setting might lead to different results. A proper parameter
setting should be found to guarantee the expected capability of the algorithm. In
Deep-Mind’s work, we can only find some simple and straightforward sentences
to give the values of a part of important parameters. Also, few works from others indicate the parameter setting for these algorithms. Therefore, the parameter optimization seems to be necessary. In our work, we choose the most general
framework algorithm in aforementioned AlphaGo series algorithms—AlphaZero,
to study.
We use a lightweight re-implementation of AlphaZero: AlphaZeroGeneral,
https://github.com/suragnair/alpha-zero-general.
In order to optimize parameters, it is very important to understand the
roles (functions and contributions) of each parameter appeared in the algorithm.
Through analyzing the AlphaZeroGeneral framework, in a single iterative loop,
it can be divided into three stages: self-play, training neural network and arena
comparison. Except for the parameters of designing the structure of neural networks, there are 12 parameters (see section 3.2) in AlphaZeroGeneral.
Therefore, in this paper, we intend to sweep 12 parameters by configuring
3 different values (minimum value, default value and maximum value) to find
most promising parameters1. In each single run of experiment of training AlphaZeroGeneral to play 6×6 Othello [9], we change the value of one parameter
and keep the values of the rest parameters as default values(see Table 1). We
observe 3 objectives (see section 3.3): training loss, elo rating and time cost in
each single run. Based on these different objective observations, this paper gives
an intuitive view of evaluating contributions of these parameters and provides
an insightful basis for further hyper parameter optimization.
Our contributions can be summarized as follows:
1. We give time cost function of the AlphaZeroGeneral algorithm, see Equation
2.
2. We sweep 12 parameters in AlphaZeroGeneral and provide detailed training
results and analysis about loss, elo rating and time cost, respectively for
every parameters, see Fig. 1–13 and Table. 2.
1

each experiment only observes one parameter, so in a specific experiment, the values
of other parameters are set as their own default values.


## Page 3

Hyper-Parameter Sweep on AlphaZero General

3

3. We summarize most promising parameters based on different objective for
further optimization, see Table. 3.

2

Related work

Parameter tuning by optimization is very important for many practical algorithms. In reinforcement learning, for instance, the ǫ-greedy strategy of classical
Q-learning is used to balance the exploration and exploitation. Different ǫ values lead to different learning performance [10]. Another well known example of
parameter tuning in reinforcement learning is the parameter Cp in UCT (Upper
Confidence Bound Apply to Tree) formula [11]. There are a lot of work to tune
Cp for different kinds of tasks and provide a lot of insight of setting its value
for Monte Carlo Tree Search (MCTS) to balance the exploration and exploitation [12]. In deep reinforcement learning, neural network shows black-box but
strong capability [13] of training, such as training convolutional neural network
to play Go [14]. Since Mnih et al. reported their work on human-level control
through deep reinforcement learning [15] in 2015, the performance of deep Qnetwork (DQN) shows us an amazing impression on playing Atari 2600 Games.
Thereafter, the applications based on DQN have shown surprising ability of
learning to solve game playing problems. For example, Silver et al. [1] applied
tree search and neural network to train and enhance training model to play game
of Go based on human experts’ playing data. Then, based on the research of selfplay in reinforcement learning [16,17,18], Silver et al. continued to use self-play
to generate training data instead of training from human data, which saves a
lot of work of collecting and labeling data from human experts [2]. Soon after,
Silver et al. [3] generalized their approach as a framework for dealing with more
general tasks (such as Go, Shogi and Chess in their work). Although there are so
many impressive achievements of AlphaGo series algorithms, few works discuss
on parameter tuning of these algorithms. Therefore, it is necessary to optimize
parameters for these algorithms. Instead of optimizing directly, since there are
too many parameters in these algorithms, and we do not know if there are any
correlations among these parameters, in our paper, as the first step, filtering for
the most promising parameters in order to further optimize these is pursued as
an alternative.

3

AlphaZero

3.1

AlphaZero Algorithm

According to [3], the structure of AlphaZero algorithm is an iteratively loop as a
whole, which can be divided into three stages within the loop. During a pure iteration, the first stage is self-play. The player plays several games against to itself
to generate games data for further training. In each step of a game (episode), the
→
−
player runs Monte Carlo Tree Search (MCTS) to obtain the playing policy ( pi).
In MCTS, Cpuct is used to balance the exploration and exploitation of game


## Page 4

4

Hui Wang et al.

tree search. mctssimulation is the number of searching times from the root node
to build the game tree, where the current best neural network model (nnm)
→
−
provides the value of the states for MCTS. Once the player get the pi based on
→
−
MCTS, in the game, the player always chooses the best move according to pi after tempThreshold steps, before tempThreshold steps, the player always chooses
→
−
a random move based on the probability distribution from pi. These games
→
examples (normalized as a form of (st , −
π t , zt )) are appended to the trainingExamplesList. If the iteration count surpasses the retrainlength, the first iterations
examples in the queue will be popped out. The second stage is training neural
network using games data from self-play. While training, it is not enough to
pull through a training examples set into neural network only once, and it is not
possible to pull through the whole training examples set at once time. Therefore,
several epochs are needed. In each epoch, training examples are divided into several small batches according to the specific batchsize. Neural network is trained
to optimize (minimize) [19] the value of loss function. The last stage is comparing the newly trained neural network model with the previous neural network
model (arena comparison), the player will adopt the better model for the next
iteration. In order to achieve this, newly trained neural network model (nnnw )
and previous best neural network model (pnnw ) are compared in an arena. In
the arena, players should pit against to the opponent for arenacompare games.
If the nnnw wins more than a proportion of updateThreshold, the nnnw will be
accepted to replace the pnnw. Otherwise, the nnnw should be rejected and the
pnnw will still be used as current best neural network model. In order to present
this process intuitively, we introduce the pseudo code as Algorithm 1, where all
12 parameters are presented in their corresponding positions.
3.2

Parameters in AlphaZero Algorithm

In order to make it clear, in this paper, parameters are presented as the order
of their first showing up order in the Algorithm 1.
iteration is used to set the number of iteration for the whole training process.
episode is used to set the number of episode for self-play. One episode means a
whole circle from game start to the end.
tempThreshold is used to judge that the player should take the best action or
not. In each game (episode), if the game step is smaller than tempThreshold, the
player will choose an action randomly according to the probability distribution
of all legal actions, otherwise, the player will always choose the best action.
mctssimu is used to set the number of MCTS Simulation. This number controls
the building of search tree according to the neural network.
Cpuct is used in UCT formula, which is very useful to balance the exploration
and exploitation in the MCTS.
retrainlength is used to set the number of last iterations. The neural network
will also be trained using training examples from these iterations again and
again. Except from the training examples of the current iteration, other training
examples are trained one or more times, so this is called retrain. Retrain is used
to avoid over fitting while training the neural network.


## Page 5

Hyper-Parameter Sweep on AlphaZero General

5

Algorithm 1 AlphaZero Algorithm
1: function AlphaZero(initial neural network model: nnm, parameter setting: ps)
2:
while current iteration<ps.iteration do
⊲ stage 1
3:
while current episode<ps.episode do
4:
while !game terminates do
→
−
5:
π ← MCTS(nnm, ps.Cpuct, ps.mctssimulation, s);
6:
if current game step<ps.tempThreshold then
→
7:
action∼ −
π;
8:
else
→
9:
action← arg maxa −
π;
10:
trainingExamplesList← append examples;
11:
if iteration of trainingExamplesList>ps.retrainlength then
12:
trainingExamplesList.pop(0);
13:
while current epoch<ps.epoch do
⊲ stage 2
14:
batch← trainingExamples.size/ps.batchsize;
15:
while current batch<batch do
16:
loss pi, loss v, nnnw← trainNNT( ps.learningrate, ps.dropout, training examples of current batch) by optimizing the loss function;
17:
while current arenacompare<ps.arenacompare do
⊲ stage 3
18:
while !game terminates do
19:
if player==player1 then
−
→ ← MCTS(nnnm, ps.Cpuct, ps.mctssimulation, s);
20:
π
1
→;
21:
action← arg maxa −
π
1
22:
else
−
→ ← MCTS(pnnm, ps.Cpuct, ps.mctssimulation, s);
23:
π
2
→;
24:
action← arg maxa −
π
2
25:
if player1.win/ps.arenacompare≥ps.updateThreshold then
26:
nnm←nnnm;
27:
return nnm;

epoch is used to set the number of epoch. It is called one epoch while all training
examples pass through the neural network. Usually, one epoch is not enough this
is because too few epoch may lead under fitting. However, too many epochs, in
the contrast, will lead over fitting.
batchsize is used to set the number of size of the training example batch. It
is always impossible to make all training examples pass through the Neural
Network at one time, so dividing the training examples into smaller batch is an
applicable way [20].
learningrate is used to control the learning rate of the neural network model.
dropout is used to set the probability. The neural network will ignore some
nodes in the hidden layer randomly based on this probability. Such mechanism
is also used to reduce over fitting [21].
arenacompare is used to set the number of games in arena to compare the new
neural network model with the old one.
updateThreshold is used to determine that the training process should accept
the new neural network model or not. In the arena, if the win rate of the new


## Page 6

6

Hui Wang et al.

neural network is higher than updateThreshold, the training process will accept
the new neural network model as the temporary best model.
3.3

Objectives

In our design, we observe training loss, elo rating and time cost for the learning
player based on Algorithm 1.
training loss function consists of policy loss (loss pi) and value loss (loss v ).
The neural network fθ is parameterized by θ. fθ takes the game board state s as
input, and the value vθ ∈ [−1, 1] of s and a policy probability distribution vector
→
−
→
−
pi over all legal actions as outputs. pi θ is the policy provided by fθ to guide
MCTS for playing games. After performing MCTS, there is an improvement
→
→
estimate −
π , one of training targets is to make −
π be more and more similar
→
−
as pi. This can be achieved by minimizing the cross entropy of two distribu→
−
→
tions. Therefore, the loss pi can be defined as −−
π log( pi θ (st )). The other target
of training neural network is to minimizing the difference between the output
value (vθ (st )) of the s according to fθ and the real outcome (zt ∈ {−1, 1}) of the
game. This can be achieved by minimizing the variances of two output values.
Therefore, loss v can be defined as (vθ (st ) − zt )2 . Overall, the total loss function
can be defined as equation 1
→
−
→
π log( pi θ (st )) + (vθ (st ) − zt )2
loss pi + loss v = −−

(1)

elo rating function is developed as a method for calculating the relative skill
levels of players in games [22]. Following [3], in our design, we also adopt bayesian
elo system [23] to compute the elo increasing curve of the learning player during
the whole training iterations. In zero-sum games, there are two players, player
A and B. If player A has an elo rating of RA and B has an elo rating of RB ,
then the expectation of that player A wins the next game can be calculated by
EA = 1+10(RB1−RA )/400 . If the real outcome of the next game is SA , then the
updated elo rating of player A can be calculated by RA = RA + K(SA − EA ),
where K is the factor of the maximum possible adjustment per game. In practice,
K should be set as a bigger value for weaker players but a smaller value for
stronger players.
time cost function is a prediction function we summarized from Algorithm 1.
According to Algorithm 1, the whole training process consists of several iterations. In every single iteration, there are all three steps as we introduced in
section 3.1. For self-play, there are three loops. The outer one is for playing
episodes, the middle one is the game steps, the inner one is for MCTS simulations in each step. For neural network training, there are also two loops.
The outer one is for epochs, the inner one is for batches. For arena comparison, it is similar as self-play, but outer loop invariance is called arenacompare. Overall, in ith iteration (1 ≤ i ≤ iteration), if we assume that in jth
episode (1 ≤ j ≤ episode), for kth game step (the size of k mainly depends on
the game complexity), the time cost of lth MCTS (1 ≤ l ≤ mctssimulation)


## Page 7

Hyper-Parameter Sweep on AlphaZero General

7

(i)

simulation is tjkl , and assume that for uth epoch (1 ≤ u ≤ epoch), the time
cost of pulling vth batch (1 ≤ v ≤ trainingExampleList.size/batchsize2)
(i)
through the neural network is tuv , and assume that in wth arena comparison (1 ≤ w ≤ arenacompare), for xth game step, the time cost of yth MCTS
(i)
simulation (1 ≤ y ≤ mctssimulation) is txyw , the time cost of whole training
process can be summarized equation 2
self −play

z X}|X {
XX
(i)
tjkl +
(
t=
i

j

k

l

arena comparison

train neural network

z X
}|
X
u

v

{

t(i)
uv

+

zX X}|
X
x

y

{
t(i)
xyw )

(2)

w

From Algorithm 1 and equation 2, we can easily know that the parameters,
such as iteration, episode, mctssimulation, epoch, batchsize, retrainlength, arena(i)
compare etc., will obviously influence the time cost of training. Besides, tjkl and
(i)

(i)

txyw are both the simulation time cost, they rely on the hardware capacity. tuv
also relies on the structure of neural network. In our experiments, all neural network models share the same structure, which consists of 4 convolutional neural
network and 2 fully connected layers.

4

Set Up

Our experiments are run on one of our college server, which has 128G RAM,
3TB local storage, 20 Intel Xeon E5-2650v3 CPUs (2.30GHz, 40 threads), 2
NVIDIA Titanium GPUs (each with 12GB memory) and 6 NVIDIA GTX 980
Ti GPUs (each with 6GB memory). In order to keep using the same GPUs, we
deploy each run of experiments on the NVIDIA GTX 980 Ti GPU.
Othello (also called Reversi) is a classic board game and usually played on
the 8×8 board. In our approach, first, in order to save time, we sweep 12 parameters introduced in section 3.2 by training to play 6×6 Othello. In each run of
our experiments, we only observe one of these parameters. Since each run of experiments is time-consuming, we only set three different values (called minimum
value, default value and maximum value, respectively) for each parameter. As a
baseline, we set all parameters as their own default values to run experiments.
For example, if we observe Cpuct, for instance, we set Cpuct as minimum value
for the first run (then maximum value for the second run) and keep all other
parameters values as their default value to run experiments. All necessary data
generated during experiments, such as loss value, arenacompare results and time
cost, should be recorded properly.
In order to train a player to play Othello based on Algorithm 1, we set values
for each parameters in Table 1.
2

the size of trainingExampleList is also relative to the game complexity


## Page 8

8

Hui Wang et al.
Table 1. Default Parameter Setting
Parameter
Minimum Value Default Value Maximum Value
iteration
50
100
150
episode
10
50
100
tempThreshold 10
15
20
mctssimu
25
100
200
Cpuct
0.5
1.0
2.0
retrainlength
1
20
40
epoch
5
10
15
batchsize
32
64
96
learningrate
0.001
0.005
0.01
dropout
0.2
0.3
0.4
arenacompare
20
40
100
updateThreshold 0.5
0.6
0.7

5

Experiment Results

In order to show the training process clearly, first, we present the results of
default setting as Fig. 1.

3.5
3.0
2.5
2.0
1.5

Loss_pi + loss_v

4.0

1.0

80
0

1

60

2

3

4 5
Epoch

r
ite
ing
n
i
Tra

40
6

7

20
8

9

0

on
ati

Fig. 1. Single Run 3D Training Loss

In Fig. 1, we plot the training loss of each epoch in every iteration. The
results show that, with reasonable fluctuation, (1) in each iteration, the training


## Page 9

Hyper-Parameter Sweep on AlphaZero General

9

loss decreases along with the increasing of epoch, and that (2) the training loss
also decreases with the increasing of iteration until to a relatively stable level.
iteration: In order to find an optimal value for iteration, we train 3 different models to play 6×6 Othello by setting iteration as minimum, default and
maximum value respectively and keeping rest parameters as their default values.
Overall, either from Fig. 2(a) or Fig. 2(b), the training loss decreases and elo rating increases both to the relatively stable level. However, after 110th iteration,
the training loss unexpectedly increases then deceases again to the same level as
only training 100 iteration. This unexpected result could be caused by too big
learning rate, improper update threshold, uncertainty from random and so on.
Theoretically, we always belief more iteration must lead to better performance,
this unexpected result proves the importance of setting proper parameter values.

6000

1.4

Iteration=50
Iteration=100(default)

Iteration=50
Iteration=100(default)
Iteration=150

2000
1.0

Elo rating

Loss_pi + loss_v

1.2

4000

Iteration=150

0

0.8

−2000
0.6

0.4
0

−4000

20

40

60

80

100

Training iteration

(a) Training Loss

120

140

160

−6000
0

20

40

60
80
100
Training iteration

120

140

160

(b) Elo Rating

Fig. 2. Training Loss and Elo Rating with Different Iteration

episode: Since more episode means more training examples. It is believed
that more training examples lead to more accurate results. However, collecting
more training examples needs more resources. This paradox also suggests that
parameter optimization is necessary to find a reasonable value of episode. In
Fig. 3(a), for episode=10, the training loss curve is higher than other 2 curves.
However, it is difficult to judge which one is better for other 2 curves, which
shows that to much training data does not get significant training results.


## Page 10

10

2.0
1.8

Hui Wang et al.

4000

Episode=10
Episode=50(default)
Episode=100

3000
2000
1000

1.4
Elo rating

Loss_pi + loss_v

1.6

1.2
1.0

0
−1000
−2000

0.8

−3000

0.6
0.4
0

Episode=10
Episode=50(default)
Episode=100

−4000
20

40
60
Training iteration

80

−5000
0

100

20

(a) Training Loss

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 3. Training Loss and Elo Rating with Different Episode

tempThreshold: Diversity training examples provide more features, but
using too deterministic policy to generate training examples will lead to too many
similar (even the same) training examples in self-play. An option of solving this
problem is using tempThreshold to control it. However, too small tempThreshold
makes policy more deterministic, too big tempThreshold makes policy more
different from the model. From Fig. 4(a) and Fig. 4(b), we see tempThreshold=10
is the best value.

1.6
1.4

4000

TempThreshold=10
TempThreshold=15(default)
TempThreshold=20

3000
2000
1000

1.0

Elo rating

Loss_pi + loss_v

1.2

TempThreshold=10
TempThreshold=15(default)
TempThreshold=20

0.8

0
−1000
−2000

0.6

−3000
0.4
0.2
0

−4000
20

40
60
Training iteration

(a) Training Loss

80

100

−5000
0

20

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 4. Training Loss and Elo Rating with Different TempThreshold

mctssimu: In theory, more mctssimu can provide better policy for performing the best move. However, more mctssimu means more time to get such good
policy. Besides, In Algorithm 1, tree search is also used to generate an estimation
→
improvement −
π to calculate the cross entropy for loss function. In Fig. 5(a) and


## Page 11

Hyper-Parameter Sweep on AlphaZero General

11

Fig. 5(b), 200 times simulation achieves best performance in 60th iteration then
unexpectedly performs worse and reaches the same level as 100 simulation times
in 100th iteration.

1.4

4000
3000

Mctssimu=25
Mctssimu=100(default)
Mctssimu=200

2000

1.0

Elo rating

Loss_pi + loss_v

1.2

5000

Mctssimu=25
Mctssimu=100(default)
Mctssimu=200

0.8

1000
0
−1000

0.6

−2000
0.4

−3000
0.2
0

20

40
60
Training iteration

80

−4000
0

100

20

(a) Training Loss

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 5. Training Loss and Elo Rating with Different Mctssimu

Cpuct: This parameter is used to balance the exploration and exploitation
during tree search. Practically, it is suggested to set as 1.0. However, in Fig. 6(a),
our experimental results show that more exploitation can get smaller training
loss.

1.4

4000

Cpuct=0.5
Cpuct=1.0(default)
Cpuct=2.0

2000
1.0

Elo rating

Loss_pi + loss_v

1.2

6000

Cpuct=0.5
Cpuct=1.0(default)
Cpuct=2.0

0.8

0
−2000

0.6

0.4
0

−4000

20

40
60
Training iteration

(a) Training Loss

80

100

−6000
0

20

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 6. Training Loss and Elo Rating with Different Cpuct

retrainlength: In order to reduce overfitting, it is important to retrain model
using previous training examples. Finding an optimal retrain length of historical


## Page 12

12

Hui Wang et al.

training examples is necessary to reduce waste of time. In Fig. 7(a), we see
that, only using training examples from most recent 1 iteration achieves smallest
training loss, which seems unexpected. However, in Fig. 7(b), we find that using
training examples from previous 40 iterations gets a stable elo rating quickly (in
60th iteration). In fact, less retrain examples leads to overfitting so that an
unexpected result shows up.

1.4

4000
3000

Retrainlength=1
Retrainlength=20(default)
Retrainlength=40

2000

1.0

Elo rating

Loss_pi + loss_v

1.2

5000

Retrainlength=1
Retrainlength=20(default)
Retrainlength=40

0.8

1000
0
−1000

0.6

−2000
0.4

−3000
0.2
0

20

40
60
Training iteration

80

−4000
0

100

20

(a) Training Loss

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 7. Training Loss and Elo Rating with Different Retrainlength

epoch: The training loss of different epoch is presented as Fig. 8(a). As
we can see, while epoch=15, the training loss curve is the lowest. The smallest
training loss in the 90th iteration, which is about 0.2. The experimental results
prove that along with the increasing of epoch, the training loss decreases, which
is in expected.

2.5

4000

Epochs=5
Epochs=10(default)
Epochs=15

2000
1.5

Elo rating

Loss_pi + loss_v

2.0

6000

Epoch=5
Epoch=10(default)
Epoch=15

1.0

0
−2000

0.5

0.0
0

−4000

20

40
60
Training iteration

(a) Training Loss

80

100

−6000
0

20

40
60
Training iteration

(b) Elo Rating

Fig. 8. Training Loss and Elo Rating with Different Epoch

80

100


## Page 13

Hyper-Parameter Sweep on AlphaZero General

13

batchsize: On the one hand, smaller batchsize makes the number of batches
become larger, which will lead to more time cost. However, on the other hand,
smaller batchsize means less training examples in each batch, which could cause
more fluctuation (larger variance) of training loss. From Fig. 9(a), batchsize=32
achieves the smallest training loss in 94th iteration. And in Fig. 9(b), batchsize=32 reaches the highest stable elo rating more quick than others.

1.6
1.4

6000

Batchsize=32
Batchsize=64(default)
Batchsize=96

4000
2000

1.0

Elo rating

Loss_pi + loss_v

1.2

0.8

0
−2000

0.6

−4000

0.4
0.2
0

Batchsize=32
Batchsize=64(default)
Batchsize=96

20

40
60
Training iteration

(a) Training Loss

80

100

−6000
0

20

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 9. Training Loss and Elo Rating with Different Batchsize

learningrate: In order to avoid falling into a local optimal, normally, a
smaller learning rate is suggested. However, a smaller learning rate learns (accepts) new knowledge slowly. In Fig. 10(a), learningrate=0.001 achieves the
lowest training loss which is in 93th iteration. From Fig. 10(b), while learningrate=0.01, the elo rating is in expected to get higher quickly but falls down
at last unluckily.


## Page 14

14

1.6

Learningrate=0.001
Learningrate=0.005(default)
Learningrate=0.01

4000

Learningrate=0.001
Learningrate=0.005(default)
Learningrate=0.01

2000

1.2

Elo rating

Loss_pi + loss_v

1.4

Hui Wang et al.

1.0

0.8

0

−2000

0.6

−4000
0.4
0

20

40
60
Training iteration

80

0

100

20

(a) Training Loss

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 10. Training Loss and Elo Rating with Different Learningrate

dropout: Dropout prevents overfitting and provides a way of approximately
combining exponentially many different neural network architectures efficiently.
Srivastava et al. claims that dropping out 20% of the input units and 50% of the
hidden units is often found to be optimal [21]. In Fig. 10(a), we find dropout=0.3
achieves the lowest training loss in 91th iteration, which is about 0.26.

1.4
1.3

4000

Dropout=0.2
Dropout=0.3(default)
Dropout=0.4

2000

1.1
1.0

Elo rating

Loss_pi + loss_v

1.2

Dropout=0.2
Dropout=0.3(default)
Dropout=0.4

0.9
0.8

0
−2000

0.7
0.6
0.50

−4000
20

40
60
Training iteration

(a) Training Loss

80

100

0

20

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 11. Training Loss and Elo Rating with Different Dropout

arenacompare: Obviously, this parameter is a key factor of time cost in
arena comparison. Too small value can not avoid coincidence and too large value
is time-sensitive and not necessary at all. Our experimental results in Fig. 12(a)
show that there is no significant difference. A combination with updateThreshold
can be used to determine the acceptance or rejection of newly learnt model. In


## Page 15

Hyper-Parameter Sweep on AlphaZero General

15

order to reduce time cost, a little small arenacompare combines with a litter big
updateThreshold might be the proper choice.

1.4

4000

Arenacompare=20
Arenacompare=40(default)
Arenacompare=100

2000
1.0

Elo rating

Loss_pi + loss_v

1.2

6000

Arenacompare=20
Arenacompare=40(default)
Arenacompare=100

0.8

0
−2000

0.6

0.4
0

−4000

20

40
60
Training iteration

80

−6000
0

100

20

(a) Training Loss

40
60
Training iteration

80

100

(b) Elo Rating

Fig. 12. Training Loss and Elo Rating with Different Arenacompare

updateThreshold: Normally, in two-player games, the condition of judging
A player is better than B player is that A must win more than 50% games.
In order to avoid coincidence, a higher win rate condition is helpful. However,
if we set a too high win rate, then it is more difficult to iteratively update to
better models. Fig. 13(a) shows that updateThreshold=0.7 is too high so that
the model gets better too slowly than others, which is evidenced by the highest
training loss curve. From Fig. 13(b), we find that updateThreshold=0.6 is the
best value to achieve the highest elo rating.

1.4

3000
2000

UpdateThreshold=0.5
UpdateThreshold=0.6(default)
UpdateThreshold=0.7

1000
1.0

Elo rating

Loss_pi + loss_v

1.2

4000

UpdateThreshold=0.5
UpdateThreshold=0.6(default)
UpdateThreshold=0.7

0.8

0
1000
2000
3000

0.6

4000
0.4
0

20

40
60
Training iteration

(a) Training Loss

80

100

5000
0

20

40
60
Training iteration

80

(b) Elo Rating

Fig. 13. Training Loss and Elo Rating with Different UpdateThreshold

100


## Page 16

16

Hui Wang et al.

We present the final time cost of each experiment using different parameter
values in Table 2. From the table, for parameter iteration, episode, mctssimu,
retrainlength, arenacompare, smaller values lead to less time cost, which is in
expectation. For batchsize, bigger value results to less time cost. The rest parameters are time-friendly, changing their values will not lead to significant different
time cost. Therefore, tuning these time-friendly parameters could be effective.

Table 2. Time Cost (hr) of Different Parameter Setting
Parameter
Minimum Value Default Value Maximum Value Type
iteration
23.8
44.0
60.3
time-sensitive
episode
17.4
44.0
87.7
time-sensitive
tempThreshold 41.6
44.0
40.4
time-friendly
mctssimu
26.0
44.0
64.8
time-sensitive
Cpuct
50.7
44.0
49.1
time-friendly
retrainlength
26.5
44.0
50.7
time-sensitive
epoch
43.4
44.0
55.7
time-sensitive
batchsize
47.7
44.0
37.7
time-sensitive
learningrate
47.8
44.0
40.3
time-friendly
dropout
51.9
44.0
51.4
time-friendly
arenacompare
33.5
44.0
57.4
time-sensitive
updateThreshold 39.7
44.0
40.4
time-friendly

Based on the aforementioned experimental results and corresponding analysis, we summarize the importance by evaluating contributions of each parameter
to training loss, elo rating and time cost, respectively, in Table 3. For training
loss, different value of arenacompare and updateThreshold can not make significant difference. For elo rating, different value of arenacompare can not make
significant difference. For time cost, different value of time-friendly parameters
contribute almost the same to the whole training, while usually, bigger value of
time-sensitive parameters requires more time to the whole training.


## Page 17

Hyper-Parameter Sweep on AlphaZero General

17

Table 3. A Summary of Importance in Different Dimensions
Parameter
Default Value Loss
Elo Rating Time Cost
iteration
100
100
100
50
episode
50
10
10
10
tempThreshold 15
10
10
similar
mctssimu
100
200
200
25
Cpuct
1.0
0.5
0.5(2.0) similar
retrainlength
20
1
40
1
epoch
10
15
15
5
batchsize
64
96
32
96
learningrate
0.005
0.001
0.001
similar
dropout
0.3
0.3
0.2
similar
arenacompare
40
insignificant 20
updateThreshold 0.6
insignificant 0.6
similar

6

Conclusion

In this paper, since there is few work discussed how to set the parameters values
in AlphaGo series algorithms. Based on AlphaZero, we analyze 12 parameters
in this general framework. First, we introduce the roles and functions of every
parameters by showing these parameters’ particular places appeared within the
algorithm framework. Second, we introduce three objectives (i.e. training loss
function, time cost function and elo rating function) to optimize. Based on an
implementation of AlphaZero from github [25], we configure the algorithm with
our default parameter setting to do comparison experiments based on 6×6 Othello. 1) We classified time-friendly parameters and time-sensitive parameters, see
Table 2. 2) In addition, we find that although some time-sensitive parameters,
such as mctssimulation, iteration and episode etc, can make significant difference
with different values. Interestingly, not always bigger values result to positive significant improvement, which proves that there exists the optimal value for a given
time cost for these parameters. 3). For time-friendly parameters, different values
will not make significant difference of time cost, so tuning these parameters can
be regarded as a Pareto Improvement process. Our these findings support the
applications of AlphaZero and provide us a basis to optimize parameter setting
values and transfer the optimal parameter setting values to different games and
tasks within AlphaZero framework.
In the future, we can apply automatically optimization framework [26,27]
techniques to replace manually optimizing parameter in this paper. In addition,
relationship among parameters may be much more complicate. Therefore, it is
worthy to digging out the co-efficiency. What’s more, transfer learning in AlphaZero still has much work to do for us. For instance, we can also transfer models
to different tasks to get preliminary and abstract knowledge within AlphaZero
framework.


## Page 18

18

Hui Wang et al.

Acknowledgments. Hui Wang acknowledges financial support from the China
Scholarship Council (CSC), CSC No.201706990015.

References
1. Silver D, Huang A, Maddison C J, et al: Mastering the game of Go with deep
neural networks and tree search. Nature 529(7587), 484–489 (2016)
2. Silver D, Schrittwieser J, Simonyan K, et al: Mastering the game of go without
human knowledge. Nature 550(7676), 354–359 (2017)
3. Silver D, Hubert T, Schrittwieser J, et al: Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning Algorithm. arXiv preprint
arXiv:1712.01815, (2017).
4. Granter S R, Beck A H, Papke Jr D J: AlphaGo, deep learning, and the future
of the human microscopist. Archives of pathology & laboratory medicine 141(5),
619–621 (2017)
5. Wang F Y, Zhang J J, Zheng X, et al: Where does AlphaGo go: From churchturing thesis to AlphaGo thesis and beyon. IEEE/CAA Journal of Automatica
Sinica 3(2), 113–120 (2016)
6. Fu M C: AlphaGo and Monte Carlo tree search: the simulation optimization perspective. Proceedings of the 2016 Winter Simulation Conference. IEEE Press pp.
659–670 (2016)
7. Tao J, Wu L, Hu X: Principle Analysis on AlphaGo and Perspective in Military
Application of Artificial Intelligence. Journal of Command and Control 2(2), 114–
120 (2016)
8. Zhang Z: When doctors meet with AlphaGo: potential application of machine
learning to clinical medicine. Annals of translational medicine 4(6), (2016)
9. Iwata S, Kasai T. The Othello game on an n×n board is PSPACE-complete. Theoretical Computer Science. 123(2), 329–340 (1994)
10. Wang H, Emmerich M, Plaat A. Monte Carlo Q-learning for General Game Playing.
arXiv preprint arXiv:1802.05944 (2018)
11. Browne C B, Powley E, Whitehouse D, et al: A survey of monte carlo tree search
methods. IEEE Transactions on Computational Intelligence and AI in games 4(1),
1–43 (2012)
12. B Ruijl, J Vermaseren, A Plaat, J Herik: Combining Simulated Annealing and
Monte Carlo Tree Search for Expression Simplification. In: Béatrice Duval, H.
Jaap van den Herik, Stéphane Loiseau, Joaquim Filipe. Proceedings of the 6th
International Conference on Agents and Artificial Intelligence 2014, vol. 1, pp.
724–731. SciTePress, Setúbal, Portugal (2014)
13. Schmidhuber J: Deep learning in neural networks: An overview. Neural networks
61 85–117 (2015)
14. Clark C, Storkey A. Training deep convolutional neural networks to play go. International Conference on Machine Learning. pp. 1766–1774 (2015)
15. Mnih V, Kavukcuoglu K, Silver D, et al: Human-level control through deep reinforcement learning. Nature 518(7540), 529–533 (2015)
16. Heinz E A: New self-play results in computer chess. International Conference on
Computers and Games. Springer, Berlin, Heidelberg. pp. 262–276 (2000)
17. Wiering M A: Self-Play and Using an Expert to Learn to Play Backgammon with
Temporal Difference Learning. Journal of Intelligent Learning Systems and Applications 2(2), 57–68 (2010)


## Page 19

Hyper-Parameter Sweep on AlphaZero General

19

18. Van Der Ree M, Wiering M: Reinforcement learning in the game of Othello: Learning against a fixed opponent and learning from self-play. In Adaptive Dynamic
Programming And Reinforcement Learning. pp. 108–115 (2013)
19. Kingma D P, Ba J: Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980 (2014)
20. Ioffe S, Szegedy C: Batch normalization: accelerating deep network training by
reducing internal covariate shift. Proceedings of the 32nd International Conference
on International Conference on Machine Learning-Volume 37. pp. 448–456 (2015)
21. Srivastava N, Hinton G, Krizhevsky A, et al: Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research. 15(1),
1929–1958 (2014)
22. Albers P C H, Vries H. Elo-rating as a tool in the sequential estimation of dominance strengths. Animal Behaviour 489–495 (2001)
23. Coulom R. Whole-history rating: A Bayesian rating system for players of timevarying strength. International Conference on Computers and Games. Springer,
Berlin, Heidelberg, 113–124, 2008
24. Schneider M O, Rosa J L G: Neural connect 4-A connectionist approach to the
game. Neural Networks SBRN 2002. Proceedings. VII Brazilian Symposium on.
IEEE pp. 236–241 (2002)
25. Surag Nair, https://github.com/suragnair/alpha-zero-general
26. Birattari M, Stützle T, Paquete L, et al. A racing algorithm for configuring metaheuristics. Proceedings of the 4th Annual Conference on Genetic and Evolutionary
Computation. Morgan Kaufmann Publishers Inc. 11-18 (2002)
27. Hutter F, Hoos H H, Leyton-Brown K: Sequential model-based optimization for
general algorithm configuration. International Conference on Learning and Intelligent Optimization. Springer, Berlin, Heidelberg, pp. 507–523 (2011)


