## Page 1

On the SDEs and Scaling Rules
for Adaptive Gradient Algorithms

arXiv:2205.10287v3 [cs.LG] 1 Nov 2024

Sadhika Malladi∗

Kaifeng Lyu∗
Abhishek Panigrahi
Sanjeev Arora
Department of Computer Science
Princeton University
{smalladi,klyu,ap34,arora}@cs.princeton.edu

Abstract
Approximating Stochastic Gradient Descent (SGD) as a Stochastic Differential
Equation (SDE) has allowed researchers to enjoy the benefits of studying a continuous optimization trajectory while carefully preserving the stochasticity of SGD.
Analogous study of adaptive gradient methods, such as RMSprop and Adam, has
been challenging because there were no rigorously proven SDE approximations
for these methods. This paper derives the SDE approximations for RMSprop and
Adam, giving theoretical guarantees of their correctness as well as experimental
validation of their applicability to common large-scaling vision and language settings. A key practical result is the derivation of a square root scaling rule to adjust
the optimization hyperparameters of RMSprop and Adam when changing batch
size, and its empirical validation in deep learning settings.

1

Introduction

Distributed synchronous optimization environments have enabled very rapid training of models
(in terms of wall-clock time) by allowing a large batch size. Understanding large-batch stochastic
optimization is crucial to enjoying the speed-up of this setting. In this context, Krizhevsky (2014);
Goyal et al. (2017) empirically discovered the linear scaling rule (LSR) for Stochastic Gradient
Descent (SGD). It calls for scaling learning rate proportionately to the batch size while fixing the
number of epochs. It was recognized that the validity of this scaling rule stems from the benefits
to generalization due to noise from mini-batch gradient estimation. But naive analysis, as done
in Hoffer et al. (2017), suggested that the scaling rule for SGD ought to be square root instead of
linear. Subsequently, Jastrz˛ebski et al. (2017) pointed out that since the phenomenon involves varying
the LR even down to zero, the correct analysis should invoke a continuous view, namely a stochastic
differential equation (SDE). The SDE view helps identify the correct scaling of the noise and leads to
a derivation of the linear scaling rule (see Section 2.2).
However, extending the SDE approach—i.e., continuous-time approximations—to popular adaptive
optimization algorithms, like RMSprop (Tieleman and Hinton, 2012) and Adam (Kingma and Ba,
2015), has been challenging due to their use of coordinate-wise normalization. By ignoring gradient
noise, Ma et al. (2022) derived intuitive continuous approximations for full-batch RMSprop and
Adam; however, this deterministic view precludes a scaling rule.
Recent papers have suggested a square root scaling rule for adaptive algorithms: set the learning
rate proportionately to the square root of the batch size while fixing the number of epochs. Based
on perturbation theory, Granziol et al. (2022) proposed the square root scaling rule for RMSprop
and Adam but could only reason about optimization behavior near convergence, not along the entire
trajectory. A square root scaling rule was also empirically discovered for another adaptive gradient
∗

Equal Contribution.

36th Conference on Neural Information Processing Systems (NeurIPS 2022).


## Page 2

method called LAMB (You et al., 2020), which is an optimization method with layerwise adaptive
learning rates, designed for better optimization and generalization in large-batch training. Instead
of tuning learning rates while increasing batch size, LAMB used the square root scaling rule to
automatically adjust the learning rate and achieve strong performance on vision and language tasks.
In this paper, we make the following contributions.
1. We propose new SDE approximations for two popular adaptive optimization algorithms, RMSprop and Adam (Definitions 4.1 and 4.4) in Theorems 4.2 and 4.5. We prove that these
approximations are 1st-order weak approximations (Definition 2.4), providing a calculus-based
guarantee of the approximation strength between the SDEs and their corresponding discrete
processes as was done for SGD and its SDE in Li et al. (2019).
2. Our SDE approximations immediately yield square-root scaling rules (Definitions 5.1 and 5.2)
for adjusting the optimization hyperparameters of Adam and RMSprop when changing batch
size to ensure that the resulting discrete trajectories are all 1st-order weak approximations of the
same SDE (Theorem 5.3). Experiments (Figures 1 and 2 and Appendix I) validate the scaling
rules in the vision and language modeling domains.
3. We provide efficient experimental verification of the validity of the SDE approximation for the
adaptive algorithms in realistic deep nets (Definitions 5.1 and 5.2). Direct simulation of the
SDE, e.g., Euler-Maruyama, is prohibitively expensive because it requires computing the full
gradient and noise covariance at fine-grained intervals. Hence we adapt (Definition 6.2) the
new and efficient SVAG simulation for SGD (Li et al., 2021) for use with our proposed SDEs
and rigorously prove its correctness (Theorem 6.3). Using SVAG, we provide evidence that
the proposed SDE approximations track the analogous discrete trajectories in many common
large-scale vision and language settings (Figure 3 and Appendix H).

2

Preliminaries

√
We use v ⊙ u, v 2 , v to denote coordinate-wise operators for multiplication, taking squares, taking
square roots. For ease of exposition we modify RMSprop and Adam to use vk in the update for θk
instead of using vk+1 .2 We also assume that v0 is nonzero if ϵ is 0 to avoid division by zero.
Definition 2.1. RMSprop (Tieleman and Hinton, 2012) is an algorithm that updates θk as follows,
√
θk+1 = θk − ηgk ⊙ ( vk + ϵ)−1 , vk+1 = βvk + (1 − β)gk2 ,
where θk is the parameter, gk is the stochastic gradient at step k, and vk is an estimate for the second
moment of gk .
Definition 2.2. Adam (Kingma and Ba, 2015) is an algorithm that updates θk as follows,
mk+1 = β1 mk + (1 − β1 )gk ,

vk+1 = β2 vk + (1 − β2 )gk2 ,

c
mk+1 = mk+1 ⊙ (1 − β1k+1 )−1 ,
p
bk + ϵ)−1 ,
θk+1 = θk − ηc
mk+1 ⊙ ( v

bk+1 = vk+1 ⊙ (1 − β2k+1 )−1 ,
v

where θk is the parameter, gk is the stochastic gradient at step k, mk is the momentum, and vk is an
estimate for the second moment of gk .
2.1

Noisy Gradient Oracle with Scale Parameter

We abstract the stochastic gradient as being provided by a noisy oracle for the full gradient. We
formulate the oracle to highlight the phenomenon of interest: changing the batch size affects the scale
of the noise.
Definition 2.3. A Noisy Gradient Oracle with Scale Parameter (NGOS) is characterized by a
tuple Gσ = (f, Σ, Zσ ). Given a (noise) scale parameter σ > 0, Gσ takes an input θ and returns
g = ∇f (θ) + σz, where ∇f (θ) is the gradient of f at θ, z is the gradient noise drawn from the
probability distribution Zσ (θ) with mean zero and covariance matrix Σ(θ). We use Gσ (θ) to denote
the distribution of g given σ and θ. The probability distribution Zσ (θ) can change with the scale σ,
but the covariance matrix Σ(θ) is fixed across different noise scales.
2

Experiments in Appendix G.2 show that this change does not significantly impact performance.

2


## Page 3

For all gk in Definitions 2.1 and 2.2, we assume that gk is drawn from a noisy gradient oracle Gσ .
In our setting, as is common when batches are
√ sampled with replacement, σ is primarily controlled
through the batch size; in particular, σ ∼ 1/ B (see Appendix F.1 for a derivation). For sampling
with replacement on a finite dataset of size n, where f1 (θ), . . . , fn (θ) are the loss functions for
the n data points (and the average of these n functions is f (θ)), this covariance matrix for a given
parameter θ can be explicitly written as
n

Σ(θ) =

2.2

1X
(∇fi (θ) − ∇f (θ))(∇fi (θ) − ∇f (θ))⊤ .
n i=1

(1)

SDE Approximation and Scaling Rules

Under appropriate conditions it becomes possible to approximate SGD via an Itô SDE, which uses
Brownian motion to model the noise and has the following general form, where Wt is a Wiener
process: dXt = b(Xt )dt + σ(Xt )dWt . The SGD update rule for a loss f is xk+1 = xk − ηgk ,
where η is the learning rate and gk is given by the NGOS on input xk . The following is the canonical
interpretation of SGD as an SDE:
√
dXt = −∇f (Xt )dt + ηΣ1/2 (Xt )dWt .
(2)
Equation (2) hints at a relationship between learning rate and gradient noise—specifically, the linear
scaling rule—since scaling batch size by factor κ scales the noise covariance by 1/κ, which can
be canceled by scaling η by κ as well (Jastrz˛ebski et al., 2017). Therefore, the linear scaling rule
ensures the SDE approximation does not change when using a different batch size. With the same
methodology, the current paper studies the SDE approximations for adaptive gradient algorithms to
derive the square root scaling rule for them.
2.3

Quality of SDE Approximation and Theoretical Assumptions

The quality of the SDE approximation can be measured empirically (Section 6) and bounded theoretically using a calculus-based guarantee, which was initiated in the context of deep learning in Li et al.
(2019). In particular, the theoretical guarantee uses the following notion of approximation between
discrete and continuous stochastic processes by comparing iteration k in the discrete process with
continuous time kηe , where ηe > 0 is the (effective) step size of the discrete process.
Definition 2.4 (Order-1 Weak Approximation, Li et al. (2019)). Let {Xtηe : t ∈ [0, T ]} and
⌊T /η ⌋
{xηke }k=0 e be families of continuous and discrete stochastic processes parametrized by ηe . We say
ηe
{Xt } and {xηke } are order-1 weak approximations of each other if for every test function g with at
most polynomial growth (Definition B.1), there exists a constant C > 0 independent of ηe such that
max

k=0,...,⌊T /ηe ⌋

ηe
|Eg(xηke ) − Eg(Xkη
)| ≤ Cηe
e

A function g : Rd → R is said to have polynomial growth if there exist positive integers κ1 , κ2 > 0
d
2
such that |g(x)| ≤ κ1 (1 + ∥x∥2κ
2 ) for all x ∈ R (Definition B.1). The above definition measures
the strength of the approximation by the closeness of a test function g computed on the iterates of
both trajectories. The approximation becomes stronger in this sense as ηe becomes smaller. In the
SDE approximation of SGD, ηe = η and k steps correspond to continuous time kη. A key difference
between SGD and adaptive algorithms is that ηe = η 2 for both RMSprop and Adam, which means k
steps correspond to continuous time kη 2 . We validate this time-scaling theoretically in Section 4.
Now we formalize the assumptions needed in the theory. Since our analysis framework is based upon
calculus, it becomes necessary to assume differentiability conditions on the NGOS (Definition 2.5).
Similar differentiability conditions also appear in prior SDE works (Li et al., 2019, 2021), and we
note that lately it has become clear that restricting to differentiable losses (via differentiable node
activations such as Swish (Ramachandran et al., 2017)) does not hinder good performance.
Definition 2.5 (Well-behaved NGOS). The loss function f and covariance matrix function Σ in
a NGOS Gσ are well-behaved if they satisfy3 : (1) ∇f (θ) is Lipschitz and C ∞ -smooth; (2) The
square root of covariance matrix Σ1/2 (θ) is bounded, Lipschitz, and C ∞ -smooth; and (3) All partial
3

Note: C ∞ -smoothness can be relaxed using the mollification technique from Li et al. (2019).

3


## Page 4

derivatives of ∇f (θ) and Σ1/2 (θ) up to and including the 4-th order have polynomial growth. We
also say that the NGOS Gσ is well-behaved if f and Σ are well-behaved.
Deriving an SDE approximation also requires conditions on the noise distribution in the NGOS.
It is allowed to be non-Gaussian, but not heavy-tailed. We require an upper bound on the third
moment of the noise so that the distribution is not very skewed. For other higher order moments, we
require Ez∼Zσ [∥z∥p2 ]1/p , namely the Lp -norm of random variable ∥z∥2 , to grow at most linearly as
a function of θ (which is implied by ensuring an upper bound on all even order moments). We note
that the following conditions are standard in prior work using Itô SDEs to study SGD.
Definition 2.6 (Low Skewness Condition). The NGOS Gσ is said to satisfy the low skewness condition
if there is a function K3 (θ) of polynomial growth (independent of σ) such that |Ez∼Zσ (θ) [z ⊗3 ]| ≤
K3 (θ)/σ for all θ ∈ Rd and all noise scale parameters σ.
Definition 2.7 (Bounded Moments Condition). The NGOS Gσ is said to satisfy the bounded moments
condition if for all integers m ≥ 1 and all noise scale parameters σ, there exists a constant C2m
1
d
2m ≤ C
(independent of σ) such that Ez∼Zσ (θ) [∥z∥2m
2m (1 + ∥θ∥2 ) for all θ ∈ R .
2 ]
For well-behaved loss f (θ) and covariance Σ(θ), the above two conditions are satisfied when Zσ is
the Gaussian distribution with mean zero and covariance Σ(θ). That is, the Gaussian NGOS g ∼
N (∇f (θ), σ 2 Σ(θ)) satisfies the low skewness and bounded moments conditions. The low skewness
condition holds because the odd moments of a Gaussian are all zeros, and the bounded moments
1
2m 1
1/2
2m ≤ E
(θ)∥2
condition can be verified since Ez∼N (0,Σ(θ)) [∥z∥2m
w∼N (0,I) [∥w∥2 ] 2m · ∥Σ
2 ]
1/2
and Σ (θ) is Lipschitz.
The Gaussian NGOS with σ = √1B can be seen as an approximation of the gradient noise in a
mini-batch training with batch size B, if Σ(θ) is set to match with (1). But this does not directly
imply that the gradient noise in mini-batch training satisfies the low skewness and bounded moments
conditions, as the noise is not exactly Gaussian. Assuming that the gradient of the loss function fi (θ)
at every data point is Lipschitz, these two conditions can indeed be verified for all batch sizes B ≥ 1.
2.4

Discussion on Heavy-Tailed Gradient Noise

We note that Definitions 2.6 and 2.7 allow some non-Gaussianity in the noise, but K3 (θ) and C2m
could be large in practice. In this case, higher order moments of the gradient noise have non-negligible
effects on training that the Itô SDE cannot capture. Zhang et al. (2020) and Simsekli et al. (2019)
presented experimental evidence that the noise is heavy-tailed. This motivated Zhou et al. (2020) to
consider a Lévy SDE approximation (instead of Itô SDE) to study the (worse) generalization behavior
of Adam. However, the quality of the Lévy SDE approximation was not formally guaranteed (e.g.,
in the sense of Definition 2.4), so finding a guaranteed approximation for adaptive optimization
algorithms remains an open problem. Moreover, Li et al. (2021); Xie et al. (2021) highlighted issues
with the evidence, noting that the measurements used in Simsekli et al. (2019) are intended only for
scalar values. When applied to vector valued distributions the measurement can (incorrectly) identify
a multidimensional Gaussian distribution as heavy-tailed too (Li et al., 2021). It is in general difficult
to estimate the moments of the noise distribution from samples, so the heavy-tailedness of real-world
gradient noise remains an open question.
Our empirical results suggest that our assumptions in Definitions 2.6 and 2.7 are not too strong. In
Section 6, we efficiently simulate the Itô SDE using an algorithm analogous to SVAG (Li et al., 2021).
The simulation closely approximates the test accuracy achieved by the discrete trajectory, suggesting
that even if heavy-tailed noise is present during training, it is not crucial for good generalization
(Appendix H). We remain interested in exploring the heavy-tailed analogs of our Itô SDEs. However,
efficient simulation of such SDEs remains intractable and formal approximation guarantees are
difficult to prove, so we are limited in assessing the utility of such approximations. We leave it for
future work to investigate if and how heavy-tailed noise plays a role in adaptive optimization.

3

Related Work

We defer a full discussion of empirical and theoretical works on adaptive gradient methods to
Appendix A.1 and only discuss works relevant to SDEs and scaling rules here.
4


## Page 5

Applications of SDE Approximations. There are applications of the SDE approximation beyond
the derivation of a scaling rule. Li et al. (2020) and Kunin et al. (2021) assumed that the loss has
some symmetry (i.e., scale invariance) and studied the resulting dynamics. Furthermore, Li et al.
(2020) used this property to explain the phenomenon of sudden rising error after LR decay in training.
Xie et al. (2021) analyzed why SGD favors flat minima using an SDE-motivated diffusion model.
Past Work on Square Root Scaling Rule. As mentioned before, square root scaling was incorrectly
believed for a few years to be theoretically justified for SGD. Granziol et al. (2022) decomposed
the stochastic Hessian during batch training into a deterministic Hessian and stochastic sampling
perturbation and assumes one of the components to be low rank to propose a square root scaling rule.
(You et al., 2020) empirically discovered a square root scaling rule for language models trained by
LAMB, a layer-wise variant of Adam. Xie et al. (2022) heuristically derived, but did not show an
approximation bound for, a second-order SDE for approximating Adam, and they applied the SDE to
study the time needed for Adam to escape sharp minima. Xie et al. (2022) did not discuss a scaling
rule, though their proposed SDE may admit one. Similarly, Zhou et al. (2020) derived a Lévy SDE
for Adam, but no approximation bounds are given in the paper.

4

SDEs for Adaptive Algorithms

An SDE approximation operates in continuous time and thus implicitly considers the limit η → 0.
In adaptive algorithms, the moment averaging parameters β, β1 , β2 and η must be taken to limits
such that the adaptivity and stochasticity can still be studied. For example, if β, β1 , β2 remain fixed
when η → 0, then the algorithm computes the moving averages in a very small neighborhood,
which averages out the effects of gradient noise and gradient history, causing the flow to turn into
deterministic SignGD (Ma et al., 2022). We will need to assume β, β1 , β2 → 1 as η → 0, which
implies the impact of the history grows as the learning rate decreases, and thus the adaptive features
of these algorithms can still be studied in the continuous approximation (Ma et al., 2022). To keep the
stochastic nature of the flow, we require the noise from mini-batching dominate the gradient updates.
4.1

Warm-Up: Linear loss

To build intuition for the SDE and the scaling rule, we first study a simplified setting. In particular,
consider a linear loss f (θ) = ⟨θ, ḡ⟩ and isotropic noise in the NGOS, namely gk ∼ N (ḡ, σ 2 I). The
Pk−1
RMSprop update vk+1 = βvk + (1 − β)gk2 can be expanded as vk = β k v0 + (1 − β) j=0 β j gj2 .
By linearity of expectation,
E[vk ] = β k v0 + (1 − β)

k−1
X

β j (ḡ 2 + σ 2 1) = β k v0 + (1 − β k )(ḡ 2 + σ 2 1).

j=0

This suggests that E[vk ] is approximately ḡ 2 + σ 2 1 after a sufficient number of steps. Setting
v0 = ḡ 2 + σ 2 1, we see that the approximation E[vk ] = ḡ 2 + σ 2 1 becomes exact for all k ≥ 0.
Using the linearity of variance (for independent variables), the standard deviation of each coordinate
of vk is of scale O((1 − β)σ 2 ). Moreover, the expectation E[vk ] is of scale O(σ 2 ), so we know that
vk is nearly deterministic and concentrates around its expectation ḡ 2 + σ 2 1 when β is close to 1.
Therefore, we take the approximation vk ≈ ḡ 2 + σ 2 1 for all k ≥ 0. Ignoring ϵ, the RMSprop update
rule becomes:
θk+1 ≈ θk − ηgk ⊙ (ḡ 2 + σ 2 1)−1/2 .

(3)

These dynamics depend on the relative magnitudes of ḡ and σ. We show that when σ ≪ ∥ḡ∥,
no SDE approximation can exist in Appendix F.2. Here, we explore the case where σ ≫ ∥ḡ∥
which implies θk+1 ≈ θk − ση gk . Noting that gk ∼ N (ḡ, σ 2 I) gives θk+1 − θk ∼ N ( ση ḡ, η 2 I)
approximately. Since Gaussian variables are additive, we can take a telescoping sum to obtain the
marginal distribution of θk : θk ∼ N (kη/σ)ḡ, kη 2 I approximately.
If an SDE approximation of RMSprop exists, then θk can be closely approximated by a fixed random
variable from the corresponding stochastic process at a fixed (continuous) time t. Thus, to keep the
SDE unchanged upon changing the noise scale σ, the hyperparameters must be adjusted to keep
kη
1
1
2
σ and kη unchanged, which implies η ∼ σ and k ∼ η 2 . These observations intuitively yield
5


## Page 6

√
the square
root scaling rule: noting that σ changes with mini-batch size B as σ ∼ 1/ B suggests
√
η ∼ B, and k ∼ 1/B.
4.2

SDE Approximations for Adaptive Algorithms

Having established intuition in a highly simplified setting for adaptive algorithms, we now return
to a more general and realistic setting. We derive the SDEs that are order-1 approximations of
the discrete RMSprop and Adam algorithms under Definition 2.5, where the SDE states consist of
both the parameters θ and moment estimates. From the example of Section 4.1, we see that SDE
approximations may exist if σ ∼ 1/η 2 and σ ≫ ∥ḡ∥ (see Appendix G.1 for empirical validation of
this assumption). In this case, we can prove that k ∼ η 2 is true not only for the simple setting above
but also in general. This is a key difference to the SDE for SGD — each step in RMSprop or Adam
corresponds to a time interval of η 2 in SDEs, but each SGD step corresponds to a time interval of η.
In Section 4.1, v ∼ σ 2 ∼ 1/η 2 grows to infinity as η → 0. This also happens in the general setting,
so we track uk ≜ vk /σ 2 (instead of vk directly) in the SDEs.
Definition 4.1 (SDE for RMSprop). Let σ0 ≜ ση, ϵ0 ≜ ϵη, and c2 ≜ (1 − β)/η 2 . Define the state
of the SDE as Xt = (θt , ut ) and the dynamics as


dθt = −Pt−1 ∇f (θt )dt + σ0 Σ1/2 (θt )dWt ,
dut = c2 (diag(Σ(θt )) − ut )dt
where Pt := σ0 diag(ut )1/2 + ϵ0 I is a preconditioner matrix, and Wt is the Wiener process.
Theorem 4.2 (Informal version of Theorem C.2). Let uk ≜ vk /σ 2 and define the state of the
discrete RMSprop trajectory with hyperparameters η, β, ϵ (Definition 2.1) as xk = (θk , uk ). Then,
for a well-behaved NGOS (Definition 2.3) satisfying the skewness and bounded moments conditions
(Definitions 2.6 and 2.7), the SDE in Definition 4.1 satisfies
max 2 |Eg(xk ) − Eg(Xkη2 )| ≤ Cη 2
k=0,...,⌊T /η ⌋

where g and T are defined as in Definition 2.4 and the initial condition of the SDE is X0 = x0 .
Remark 4.3. Section 4.1 suggested that the SDE approximation can only exist when σ ≫ ∥ḡ∥. This
condition is reflected by keeping σ0 = ση a constant and C depends on σ0 . When η → 0, σ scales as
1/η, so σ ≫ ∥g∥2 .
We need to find continuous approximations of the normalization constants in Adam (Definition 2.2).
As in the RMSprop case, we take (1 − β2 )/η 2 = c2 . Then, we can estimate the normalization
2
constant 1 − β2k in continuous time t = kη 2 as 1 − β2k = 1 − (1 − c2 η 2 )t/η ≈ 1 − exp(−c2 t). We
can do the analogous approximation for the other normalization constant 1 − β1k in Adam. Taking
(1 − β1 )/η 2 = c1 , we can approximate it as 1 − β1k ≈ 1 − exp(−c1 t). This is a heuristic approach
to deal with the normalization constants, but we can indeed justify it in theory.
Definition 4.4 (Adam SDE). Let c1 ≜ (1 − β1 )/η 2 , c2 ≜ (1 − β2 )/η 2 and define σ0 , ϵ0 as in
Definition 4.1. Let γ1 (t) ≜ 1 − exp(−c1 t) and γ2 (t) ≜ 1 − exp(−c2 t). Define the state of the SDE
as Xt = (θt , mt , ut ) and the dynamics as
√
γ2 (t) −1
dθt = − γ1 (t)
Pt mt dt,
dmt = c1 (∇f (θt ) − mt )dt + σ0 c1 Σ1/2 (θt )dWt ,
where Pt := σ0 diag(ut )

1/2

dut = c2 (diag(Σ(θt )) − ut )dt,
p
+ ϵ0 γ2 (t)I is a preconditioner matrix, Wt is the Wiener process.

Our main theorem for Adam is given below. The initial steps of the discrete Adam trajectory can
be discontinuous and noisy because of the normalization constants changing drastically. Hence, we
introduce a constant t0 and show that for any t0 , we can construct an SDE to be a weak approximation
for Adam starting from the ⌈t0 /η 2 ⌉-th step of Adam.
Theorem 4.5 (Informal version of Theorem D.2). Define uk = vk /σ 2 and let xk = (θk , mk , uk ) ∈
R3d be the state of the discrete Adam trajectory with hyperparameters η, β1 , β2 , ϵ. Then, for a
well-behaved NGOS (Definition 2.3) satisfying the skewness and bounded moments conditions
(Definitions 2.6 and 2.7) and any t0 > 0, the SDE in Definition 4.4 satisfies
max
|Eg(xk ) − Eg(Xkη2 )| ≤ Cη 2
2
2
k=⌈t0 /η ⌉,...,⌊T /η ⌋

where g and T are defined as in Definition 2.4 and the initial condition of the SDE is Xt0 = x⌈t0 /η2 ⌉ .
6


## Page 7

Proof Sketch. We provide a proof sketch for our SDE approximations here and defer the technical
details to Theorems C.2 and D.2. The proof follows the same steps as Li et al. (2019): first, we
compute the approximation error of the continuous trajectory after one step in discrete time. Then, we
use the single step error to bound the error over a finite interval of time. The proof extends standard
SDE techniques in several ways. The given SDEs do not satisfy the Lipschitzness and smoothness
conditions because the denominator can be unbounded. We thus construct an auxiliary SDE with
an equivalent distribution to the desired SDE (Theorem C.5) but with better smoothness conditions,
and we prove this SDE to be an order-1 weak approximation of the discrete trajectory. Moreover,
the SDE coefficients are time-dependent for Adam, unlike the ones for SGD, so we need to extend
existing results to cover this case (see Appendix B.1).

5

Square Root Scaling Rule

The SDE approximations in Definitions 4.1 and 4.4 motivate scaling rules for how to adjust the
optimization hyperparameters when changing the batch size. In order for σ0 , c1 , c2 , and ϵ0 to remain
constant, as required by the SDEs, one needs to change η, β, β1 , β2 , and ϵ accordingly.
Definition 5.1 (RMSprop Scaling Rule). When
running RMSprop (Definition 2.1) with batch size
√
B ′ = κB, use the hyperparameters η ′ = η κ, β ′ = 1 − κ(1 − β), and ϵ′ = √ϵκ .
Definition 5.2 (Adam Scaling Rule).
running Adam (Definition 2.2) with batch size B ′ = κB,
√ When
′
′
use the hyperparameters η = η κ, β1 = 1 − κ(1 − β1 ), β2′ = 1 − κ(1 − β2 ), and ϵ′ = √ϵκ .
Theorem 5.3 (Validity of the Scaling Rules). Suppose we have a well-behaved NGOS satisfying the
low skewness and bounded moments conditions.
(B)

1. Let xk be the discrete RMSprop (Definition 2.1) trajectory with batch size B and hyperparame(κB)
ters η, β, and ϵ. Let xk
be the trajectory with batch size κB and hyperparameters adjusted
(B)
(κB)
according to Definition 5.1. If xk and xk
start from the same initial point, then with g and T
defined as in Definition 2.4,
max

k=0,...,⌊T /η 2 ⌋

(B)

(κB)

Eg(xk ) − Eg(x⌊k/κ⌋ ) ≤ C(1 + κ)η 2 .

(B)

2. Fix t0 > 0. Let xk be the discrete Adam (Definition 2.2) trajectory with batch size B and hyper(κB)
parameters η, β1 , β2 , and ϵ. Let xk
be the trajectory with batch size κB and hyperparameters
(B)
(κB)
adjusted according to Definition 5.2. If x⌈t0 /η2 ⌉ and x⌈κt0 /η2 ⌉ are equal, then with g and T
defined as in Definition 2.4,
max

k=⌈t0 /η 2 ⌉,...,⌊T /η 2 ⌋

(B)

(κB)

|Eg(xk ) − Eg(x⌊k/κ⌋ )| ≤ C(1 + κ)η 2 .

Proof. By the linearity
of covariance, scaling the batch size by κ only modifies the NGOS by
√
scaling σ by 1/ κ. Hence, both scaling rules ensure that σ0 , c1 , c2 , and ϵ0 (and thus, the SDEs) are
unchanged when the batch size changes. √
The approximation bounds in Theorems 4.2 and 4.5 are in
terms of η 2 , and since η is scaled here√by κ, the same method gives an upper bound Cκη 2 . Adding
the approximation bounds for η and κη together gives C(1 + κ)η 2 .
Remark 5.4. The t0 condition on the Adam rule, a holdover from the condition in Theorem 4.5,
implies that our theory only directly applies when there is a warm-start phase of ⌈t0 /η 2 ⌉, where the
marginal distribution of the trainable parameters at the end of the phase is the same across different
learning rates η. Regardless, the scaling rules are shown to work in practice even without this phase.
The scaling rules depend on maintaining the same SDE approximation, so the bounded moments
and low skewness conditions are sufficient (but not necessary) for this scaling rule to work. Li et al.
(2021) provided an analogous discussion for SGD, and they show the scaling rule can hold even if
there is heavy-tailed noise. We leave a study of heavy-tailed gradient noise in adaptive algorithms as
future work.
7


## Page 8

Test Accuracy

100
80

B=256
B=512
B=1024
B=2048
B=4096
B=8192

60
40
20 0

200

400

Continuous time = k * 2

(a) VGG-16 with Adam

100

Test Accuracy

Test Accuracy

100

80

80

80

60

60

60

40

40

40

20 0

200

400

Continuous time = k * 2

(b) ResNet-50 with
Adam

20 0

200

Test Accuracy

100

400

Continuous time = k * 2

(c) VGG-16 with
RMSprop

20 0

200

400

Continuous time = k * 2

(d) ResNet-50 with
RMSprop

Figure 1: Square root scaling rule experiments on CIFAR-10 with VGG-16 and ResNet-50 (details in
Appendix J). We plot the mean and variance of 3 random seeds. Same color legend has been used
across all the plots. The performance gap between B = 256 and B = 8192 is at most 3% in all cases.
80 Top-1 Validation Accuracy
70
60
B=256
50
B=512
B=1024
40
B=2048
30
B=4096
B=8192
20
B=16384
10
B=32768
0 0.00 0.01 0.02 0.03 0.04

Continuous time = k * 2

(a) ResNet-50 on ImageNet

Validation Log Perplexity
B=256
B=1024
B=2048
B=4096
B=8192

10
8
6
4

10

Validation Log Perplexity
B=512
B=1024
B=2048
B=4096
B=8192

9
8
7
6

2
0.00

0.02

0.04

Continuous time = k * 2

(b) RoBERTa on Wikipedia +
Bookcorpus

5
0.000

0.005

0.010

Continuous time = k * 2

(c) GPT on WikiText-103

Figure 2: Large scale square root scaling rule experiments (details in Appendix J). Small and large
batch models differ by at most 1.5% test accuracy in vision and 0.5 perplexity in language.
Experiments. Figures 1 and 2 show the square root scaling rule applied to ResNet-50 (He et al.,
2016) and VGG-16 (Simonyan and Zisserman, 2014) trained on CIFAR-10 (Krizhevsky et al.),
RoBERTa-large (Liu et al., 2019) trained on the Wiki+Books corpus (Zhu et al., 2015), 12-layer
GPT (Brown et al., 2020) on WikiText-103 (Merity et al., 2017) and ResNet-50 trained on ImageNet (Deng et al., 2009). We use the efficient language model pre-training recipe outlined in Izsak
et al. (2021). Appendix I has many additional settings, including ablations against other scaling rules
(Appendix I.1).

6

SVAG for Adaptive Algorithms

Validating the approximation strength captured in Definition 2.4 involves comparing the discrete
algorithms and their SDEs on a set of test functions. However, obtaining the SDE solution through
traditional simulations, e.g., Euler-Maruyama, is computationally intractable.4
Li et al. (2021) proposed an efficient simulation, SVAG, of the SDE for SGD in the finite LR regime:
scale the constant LR by 1/ℓ and take the limit ℓ → ∞. In practice the simulation converges for
a small value of ℓ. We adapt SVAG technique to simulate our proposed SDEs, which requires
additionally adjusting the moment averaging hyperparameters (i.e., β, β1 , β2 ) and ϵ.
Definition 6.1 (SVAG Operator). Given an NGOS Gσ = (f, Σ, Zσ ) with scale σ (Definition 2.3)
and hyperparameter ℓ ≥ 1, the SVAG operator transforms Gσ into an NGOS Gbℓσ = (f, Σ, Zbℓσ ) with
scale ℓσ. The NGOS Gbℓσ takes an input θ and returns ĝ = r1 (ℓ)g1 + r2 (ℓ)g2 , where g1 , g2 are two
4

One can also simulate the SDE by constructing 1st-order weak approximations while taking η → 0 along
the scaling rules, but the batch size cannot be smaller than 1 and hence η cannot go arbitrarily close to the limit.

8


## Page 9

Train Accuracy
100
95
90
85
80
75
70
65
60 0.00

0.02

0.04

Test Accuracy
95
90
85
80
75
70
65
0.06 600.00

212

||zk(1)||2 Mean (Log Scale)

26

||gk(1)|| Mean (Log Scale)

211
210

=1
=2
=4
=8
0.02

0.04

25

29
28

0.06

24
0.00

Continuous Time = k * 2/ 2

0.02

0.04

0.00

0.02

Figure 3: SVAG on the Adam trajectory when training ResNet-50 on CIFAR-10 matches the
discrete trajectory (ℓ = 1) on various test functions (see Appendix J for details). The closeness
of the trajectories with respect to various test functions for different values of ℓ implies the SDE
approximation (Definition 4.4) is a 1st-order weak approximation of Adam (Theorem 4.5).
√
stochastic gradients from Gσ (θ) and ri (ℓ) = 12 (1 + (−1)i 2ℓ2 − 1) for i ∈ {1, 2}. The probability
distribution Zbℓσ is defined such that ĝ has the same distribution as ∇f (θ) + ℓσz when z ∼ Zbℓσ (θ).
b ℓσ does indeed compute stochastic gradients for f with covariance Σ.
Lemma E.1 verifies that G
Applying the SVAG operator to mini-batch training amplifies the noise scale by ℓ. We then apply the
square root scaling rule to adjust the learning rates and other hyperparameters accordingly, which
yields the SVAG algorithm.
Definition 6.2 (SVAG Algorithm). For a loss f , SVAG operator hyperparameter ℓ > 0, and optimization hyperparameters η, β, β1 , β2 , and ϵ, compute the stochastic gradient as ĝ = r1 (ℓ)gγ1 + r2 (ℓ)gγ2 ,
where r1 and r2 are defined as in Definition 6.1, and scale the optimization hyperparameters:
1. For RMSprop, set η ← η/ℓ, β ← 1−(1−β)/ℓ2 , and ϵ ← ϵℓ and apply updates as in Definition 2.1.
2. For Adam, set η ← η/ℓ, β1 ← 1 − (1 − β1 )/ℓ2 , β2 ← 1 − (1 − β2 )/ℓ2 and ϵ ← ϵℓ and apply
updates as in Definition 2.2.
The SVAG algorithm describes a discrete trajectory that is a 1st-order approximation of the corresponding SDE (Definitions 4.1 and 4.4), thereby providing an efficient simulation of the SDEs.
Theorem 6.3 (SVAG algorithm approximates SDE). Assume the NGOS is well-behaved and satisfies
the bounded moments condition (Definitions 2.5 and 2.7).
1. Let Xt be the state of the RMSprop SDE (Definition 4.1) with hyperparameters η, β, and ϵ. Let
xk be the state of the analogous discrete SVAG algorithm with hyperparameter ℓ.
2. Let Xt be the state of the Adam SDE (Definition 4.4) with hyperparameters η, β1 , β2 , and ϵ. Let
xk be the state of the analogous discrete SVAG algorithm with hyperparameter ℓ.
In both 1 and 2, following holds for g and T as in Definition 2.4.
max

k=0,...,⌊ℓ2 T /η 2 ⌋

|Eg(xk ) − Eg(Xkη2 /ℓ2 )| ≤ Cη 2 /ℓ2

Proof. The main idea of the proof is to show that the SVAG operator transforms the noise distribution
of a well-behaved NGOS satisfying the bounded moments condition into one that is well-behaved
and satisfies the bounded moments and the low skewness conditions (Lemma E.2). With these three
conditions satisfied, we can directly apply Theorems 4.2 and 4.5 to complete the proof.
Because the SDE scales time as k = t/η 2 , we must run SVAG for ℓ2 steps to match a single step
of the discrete trajectories. Nevertheless, we note that in our setting and in Li et al. (2021), the
approximation guarantee is strong enough for small ℓ, so this simulation is still more efficient than
Euler-Maruyama.
9

0.04


## Page 10

Experiments. Figure 3 compares the Adam SVAG trajectories (Definition 6.2) up to ℓ = 8 to
the discrete one (ℓ = 1) on CIFAR-10 (Krizhevsky et al.) with ResNet-50 (He et al., 2015b). We
use Tr(Σ(θk )) and ∥gk ∥ as mathematically well-behaved test functions to test the approximation
strength (see Definition 2.4). We also measure the train and test accuracies, even though they are not
differentiable (and hence, not covered by the theory). The converged SVAG trajectories are close to
the discrete ones under these test functions, suggesting the SDE approximations are applicable to
realistic deep learning settings. Additional details and settings, including large language models, are
in Appendix H.

7

Conclusion

We derive SDEs that are provable 1st-order approximations of the RMSprop and Adam trajectories,
immediately
yielding formal derivations of square root scaling rules: increase the learning rate by
√
κ and adjust the adaptive hyperparameters when increasing batch size by κ. Experiments in the
vision and language domains verify that applying these rules ensures that the values of several test
functions, including test accuracy, are preserved. We furthermore design an efficient simulation for
the SDEs, allowing us to directly validate the applicability of these SDEs to common vision and
language settings. These SDEs can lead to a deeper understanding of how adaptivity and stochasticity
impact optimization and generalization, and we hope to extend our results to formal identification of
necessary and sufficient conditions for the approximation and its consequences to hold.

Acknowledgements
We thank Zhiyuan Li and Chao Ma for helpful discussion. We also thank Tianyu Gao and Alexander
Wettig for helping us run language modeling experiments. This work is funded by NSF, ONR, Simons
Foundation, DARPA and SRC.

References
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
Jinghui Chen, Dongruo Zhou, Yiqi Tang, Ziyan Yang, Yuan Cao, and Quanquan Gu. Closing the
generalization gap of adaptive gradient methods in training deep neural networks. In Proceedings
of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI’20, 2021.
ISBN 9780999241165.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition,
pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.
Timothy Dozat. Incorporating Nesterov momentum into Adam. In Workshop, 2016.
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and
stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121–2159, 2011.
Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola,
Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet
in 1 hour. arXiv preprint arXiv:1706.02677, 2017.
Diego Granziol, Stefan Zohren, and Stephen Roberts. Learning rates as a function of batch size: A
random matrix theory approach to neural network training. Journal of Machine Learning Research,
23(173):1–65, 2022.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing
human-level performance on ImageNet classification. In The IEEE International Conference on
Computer Vision (ICCV), December 2015a.
10


## Page 11

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing
human-level performance on ImageNet classification. In The IEEE International Conference on
Computer Vision (ICCV), December 2015b.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition,
pages 770–778, 2016.
Elad Hoffer, Itay Hubara, and Daniel Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. In I. Guyon, U. V. Luxburg, S. Bengio,
H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information
Processing Systems 30, pages 1731–1741. Curran Associates, Inc., 2017.
Peter Izsak, Moshe Berchansky, and Omer Levy. How to train BERT with an academic budget. arXiv
preprint arXiv:2104.07705, 2021.
Stanisław Jastrz˛ebski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio,
and Amos Storkey. Three factors influencing minima in SGD. arXiv preprint arXiv:1711.04623,
2017.
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International
Conference on Learning Representations, 2015.
Alex Krizhevsky. One weird trick for parallelizing convolutional neural networks. arXiv preprint
arXiv:1404.5997, 2014.
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. CIFAR-10 (Canadian Institute for Advanced
Research). URL http://www.cs.toronto.edu/~kriz/cifar.html.
Daniel Kunin, Javier Sagastuy-Brena, Surya Ganguli, Daniel LK Yamins, and Hidenori Tanaka.
Neural mechanics: Symmetry and broken conservation laws in deep learning dynamics. In
International Conference on Learning Representations, 2021.
Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and dynamics of stochastic
gradient algorithms I: Mathematical foundations. Journal of Machine Learning Research, 20(40):
1–47, 2019.
Zhiyuan Li and Sanjeev Arora. An exponential learning rate schedule for deep learning. In International Conference on Learning Representations, 2020.
Zhiyuan Li, Kaifeng Lyu, and Sanjeev Arora. Reconciling modern deep learning with traditional
optimization analyses: The intrinsic learning rate. In H. Larochelle, M. Ranzato, R. Hadsell, M.F.
Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages
14544–14555. Curran Associates, Inc., 2020.
Zhiyuan Li, Sadhika Malladi, and Sanjeev Arora. On the validity of modeling sgd with stochastic
differential equations (sdes). In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and
J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34,
pages 12712–12725. Curran Associates, Inc., 2021.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining
approach. arXiv preprint arXiv:1907.11692, 2019.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.
Liangchen Luo, Yuanhao Xiong, and Yan Liu. Adaptive gradient methods with dynamic bound of
learning rate. In International Conference on Learning Representations, 2019.
Chao Ma, Lei Wu, and Weinan E. A qualitative study of the dynamic behavior for adaptive gradient
algorithms. In Joan Bruna, Jan Hesthaven, and Lenka Zdeborova, editors, Proceedings of the
2nd Mathematical and Scientific Machine Learning Conference, volume 145 of Proceedings of
Machine Learning Research, pages 671–692. PMLR, 16–19 Aug 2022.
11


## Page 12

Jerry Ma and Denis Yarats. Quasi-hyperbolic momentum and Adam for deep learning. In International
Conference on Learning Representations, 2019.
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture
models. In International Conference on Learning Representations, 2017.
Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. arXiv preprint
arXiv:1710.05941, 2017.
Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of Adam and beyond. In
International Conference on Learning Representations, 2018.
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image
recognition. In International Conference on Learning Representations, 2014.
Umut Simsekli, Levent Sagun, and Mert Gurbuzbalaban. A tail-index analysis of stochastic gradient
noise in deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of
Machine Learning Research, pages 5827–5837. PMLR, 09–15 Jun 2019.
Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for
modern deep learning research. Proceedings of the AAAI Conference on Artificial Intelligence, 34
(09):13693–13696, Apr 2020.
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude., 2012. URL https://www.cs.toronto.edu/~tijmen/csc321/
slides/lecture_slides_lec6.pdf.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman.
GLUE: A multi-task benchmark and analysis platform for natural language understanding. In
International Conference on Learning Representations, 2019.
Alexander Wettig, Tianyu Gao, Zexuan Zhong, and Danqi Chen. Should you mask 15% in masked
language modeling? arXiv preprint arXiv:2202.08005, 2022.
Zeke Xie, Issei Sato, and Masashi Sugiyama. A diffusion theory for deep learning dynamics:
Stochastic gradient descent exponentially favors flat minima. In International Conference on
Learning Representations, 2021.
Zeke Xie, Xinrui Wang, Huishuai Zhang, Issei Sato, and Masashi Sugiyama. Adaptive inertia:
Disentangling the effects of adaptive learning rate and momentum. In Kamalika Chaudhuri,
Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of
the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine
Learning Research, pages 24430–24459. PMLR, 17–23 Jul 2022.
Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan
Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep
learning: Training BERT in 76 minutes. In International Conference on Learning Representations,
2020.
Manzil Zaheer, Sashank Reddi, Devendra Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive methods
for nonconvex optimization. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. CesaBianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31.
Curran Associates, Inc., 2018.
Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv
Kumar, and Suvrit Sra. Why are adaptive methods good for attention models? In H. Larochelle,
M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information
Processing Systems, volume 33, pages 15383–15393. Curran Associates, Inc., 2020.
Pan Zhou, Jiashi Feng, Chao Ma, Caiming Xiong, Steven Chu Hong Hoi, and Weinan E. Towards theoretically understanding why SGD generalizes better than Adam in deep learning. In
H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural
Information Processing Systems, volume 33, pages 21285–21296. Curran Associates, Inc., 2020.
12


## Page 13

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and
Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching
movies and reading books. In 2015 IEEE International Conference on Computer Vision (ICCV),
pages 19–27, 2015.
Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar C Tatikonda, Nicha Dvornek, Xenophon Papademetris, and James Duncan. AdaBelief optimizer: Adapting stepsizes by the belief in observed
gradients. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in
Neural Information Processing Systems, volume 33, pages 18795–18806. Curran Associates, Inc.,
2020.

13


## Page 14

Checklist
1. For all authors...
(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s
contributions and scope? [Yes]
(b) Did you describe the limitations of your work? [Yes] Conditions for our results are outlined
in Definitions 2.5 to 2.7. In particular, we discuss the limited applicability of our work to
cases in which the gradient noise is heavy-tailed, though our empirical success suggests that
the gradient noise satisfies our assumptions in many realistic vision and language settings.
(c) Did you discuss any potential negative societal impacts of your work? [Yes] Our work
is primarily theoretical in nature, but we discuss the broader impacts of our work in
Appendix A.2.
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them?
[Yes]
2. If you are including theoretical results...
(a) Did you state the full set of assumptions of all theoretical results? [Yes] Section 2 has a
discussion of the assumptions on the gradient noise and the general requirements needed
to be able to construct a continuous approximation of the discrete algorithms. Moreover,
Section 4.1 motivates the assumption that the gradient noise dominates the gradient. We
validate this assumption experimentally in Appendix G.1. Our assumptions do not stray
from prior works on SDEs (Li et al., 2019, 2021).
(b) Did you include complete proofs of all theoretical results? [Yes] Our proofs are written
clearly in the appendix.
3. If you ran experiments...
(a) Did you include the code, data, and instructions needed to reproduce the main experimental
results (either in the supplemental material or as a URL)? [Yes] We include the code for the
vision experiments in the supplementary material. For the NLP experiments, we use the
code of Wettig et al. (2022).
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were
chosen)? [Yes] Appendix J contains the training details of all the experiments.
(c) Did you report error bars (e.g., with respect to the random seed after running experiments
multiple times)? [No] Due to computational constraints, we didn’t include the error bars for
all the experiments. However, we have made sure that the random seed for different runs in
each experimental setting is the same.
(d) Did you include the total amount of compute and the type of resources used (e.g., type
of GPUs, internal cluster, or cloud provider)? [Yes] We ran our experiments on a cluster
of 34 GPUs, where 24 are RTX 2080 GPUs and 10 are A5000 GPUs. Each experiment
on CIFAR-10 required a single RTX 2080 GPU, each experiment on ImageNet required
a single A5000 GPU, each pretraining experiment on GPT required a set of 4 RTX 2080
GPUs, each pretraining experiment on RoBERTa required a set of 8 RTX 2080 GPUs, and
each finetuning experiment on RoBERTa required a single RTX 2080 GPU.
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
(a) If your work uses existing assets, did you cite the creators? [Yes] We cite Wettig et al.
(2022) for our language model experiments. We also cite the papers of all the deep learning
architectures and datasets that we use for our experiments.
(b) Did you mention the license of the assets? [Yes] The github repository5 of Wettig et al.
(2022) has MIT license.
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes] We
include our code for the vision experiments in the supplementary material.
(d) Did you discuss whether and how consent was obtained from people whose data you’re
using/curating? [N/A] The data we used for our experiments are publicly available.
(e) Did you discuss whether the data you are using/curating contains personally identifiable
information or offensive content? [N/A] The data we used for our experiments are publicly
available.
5

https://github.com/princeton-nlp/DinkyTrain

14


## Page 15

5. If you used crowdsourcing or conducted research with human subjects...
(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]
(b) Did you describe any potential participant risks, with links to Institutional Review Board
(IRB) approvals, if applicable? [N/A]
(c) Did you include the estimated hourly wage paid to participants and the total amount spent
on participant compensation? [N/A]

15


## Page 16

A

Contextualizing our Work

A.1

Additional Recent Works

Variations of Adam have been proposed to improve its speed of convergence, generalization, and
stability during training. Reddi et al. (2018) observed that Adam does not collect long-term memory of
past gradients and therefore the effective learning rate could be increasing in some cases. Hence, they
propose AMSGrad that maintains a maximum over the exponential running average of the squared
gradients. Zaheer et al. (2018) proposed a more controlled increase in the effective learning rate by
switching to additive updates, using a more refined version of AdaGrad (Duchi et al., 2011). Chen
et al. (2021) unified the generalization ability of vanilla SGD and the convergence speed of Adam
by introducing a new adaptive parameter p ∈ (0, 12 ] that can be hypertuned for each setting. Other
variations include (a) Nadam (Dozat, 2016) that uses Nesterov momentum, (b) AdamW (Loshchilov
and Hutter, 2019) that decouples the weight decay from the optimization step, (c) AdaBound (Luo
et al., 2019) that maintains a dynamic upper and lower bound on the step size, (d) AdaBelief (Zhuang
et al., 2020) uses a decaying average of estimated variance in the gradient in place of the running
average of the squared gradients, (e) QHAdam (Ma and Yarats, 2019) that replaces both momentum
estimators in Adam with quasi-hyperbolic terms, etc. LAMB (You et al., 2020) used a layerwise
adaptive version of Adam to pretrain large language models efficiently.
A.2

Broader Impact

Our work is primarily theoretical in nature, but we discuss its broader impacts here. Strubell et al.
(2020) highlighted the environmental impact of training large language models. Formal scaling rules
remove the need to grid search over hyperparameters: in the case of adaptive algorithms, the grid
search is over an even larger space because of the additional adaptivity hyperparameters. Hence, our
work reduces the number of times researchers need to train very large models to find ones that work
just as well as their smaller, slower counterparts. At the same time, we recognize that the presence of
a formal scaling rule may encourage people who were otherwise discouraged by the daunting grid
search to train very large models.

B

SDE Approximation Theorem

In this section, we introduce a theorem in aid of our analysis for approximating Stochastic Gradient
Algorithm (SGA) with Stochastic Differential Equation (SDE).
We consider Stochastic Gradient Algorithm (SGA) of the following form:
xk+1 = xk + ηe hk (xk , ξk , ηe ),

(4)

D

where xk ∈ R is the parameter vector, ηe is the learning rate, hk is a vector-valued function that
can depend on the current step k, the current parameter xk , a random vector ξk , and the learning rate
ηe . The random vector ξk is sampled from a certain distribution Ξ(xk , ηe ) in every step.
We consider Stochastic Differential Equation (SDE) of the following form:
dXt = b(Xt , t)dt + σ(Xt , t)dWt ,
D

D

D

where b : R × R → R is the drift vector function, σ : R × R → R
function.

(5)
D×D

is the diffusion matrix

Let PX (x, s, t) be the probability distribution of Xt in (5) when the initial condition is Xs = x.
Now we define the following random variables to characterize the one-step changes of SGA and
SDE.
∆(x, k) := ηe hk (x, ξ, ηe ),
˜
∆(x,
k) := X(k+1)η − x,
e

where ξ ∼ Ξ(x, ηe ).

(6)

where X(k+1)ηe ∼ PX (x, kηe , (k + 1)ηe ).

(7)

The following regularity condition is needed for our main theorem.
Definition B.1. A function g : Rd → R is said to have polynomial growth if there exist positive
integers κ1 , κ2 > 0 such that
2
|g(x)| ≤ κ1 (1 + ∥x∥2κ
2 ),
16


## Page 17

for all x ∈ Rd . We let G denote the set of all such functions. For each integer α ≥ 1, Gα denotes
the set of α-times continuously differentiable functions Rd → R which, together with its partial
derivatives up to and including order α, belong to G. If g depends on additional parameters, we say
g ∈ Gα uniformly if the constants κ1 , κ2 are independent of those parameters.
Now we are ready to introduce the main theorem for the order of approximation by modeling SGA as
SDE. We follow the same proof strategy as Li et al. (2019). First, we show that the moments of the
one-step difference for the SGA and for the SDE of our interest are close to each other. Then, under
some regularity conditions, we translate the one-step error to an error over a finite interval of time to
get the final result. The following is our main theorem to achieve such translation, and it is adapted
from Theorem 3 in Li et al. (2019) with slightly different conditions to match our need in the analysis
of adaptive gradient methods.
Theorem B.2 (Adaption of Theorem 3 in Li et al. (2019)). Let T > 0, ηe ∈ (0, 1 ∧ T ) and N =
⌊T /ηe ⌋. Consider an SDE with drift vector b(x, ηe ) and diffusion matrix σ(x, ηe ), and a stochastic
gradient algorithm with initial point x0 ∈ RD and update rule xk+1 = xk + ηe h(xk , ξk , ηe ). Let
Xk be the support of the random variable xk given x0 . Assume the following conditions hold:
(1). The drift function b and diffusion function σ are Lipschitz and belong to G4 ;
˜ ∈ RD satisfy the following. There is a function K1 ∈ G
(2). The moments of ∆ ∈ RD and ∆
(independent of ηe ) such that for all x ∈ Xk and 1 ≤ i, j, l ≤ D,
˜ i (x, k)] ≤ K1 (x)ηe2 ,
E[∆i (x, k) − ∆
˜ j (x, k)] ≤ K1 (x)ηe2 ,
E[∆i (x, k)∆j (x, k) − ˜
∆i (x, k)∆
˜ i (x, k)∆
˜ j (x, k)∆
˜ l (x, k)] ≤ K1 (x)ηe2 .
E[∆i (x, k)∆j (x, k)∆l (x, k) − ∆
(3). There exists a subset P of the
index set {1, 2, . . . , D}q
such that the following holds. Below we
qP
P
2
2
use the notations ∥x∥P :=
i∈P xi and ∥x∥R :=
i/
∈P xi .
• There are constants C1 > 0, ω1 > 0 (independent of ηe ) so that for all k ≤ N and x ∈ Xk ,
∥E∆(x, k)∥P ≤ C1 ηe (1 + ∥x∥P ),
1
∥E∆(x, k)∥R ≤ C1 ηe (1 + ∥x∥ω
P )(1 + ∥x∥R ).
• For all m ≥ 1, there are constants C2m , ω2m > 0 (independent of ηe ) so that for all k ≤ N
and x ∈ Xk ,
m
2m
E∥∆(x, k)∥2m
P ≤ C2m ηe (1 + ∥x∥P ),
ω2m
m
2m
E∥∆(x, k)∥2m
R ≤ C2m ηe (1 + ∥x∥P )(1 + ∥x∥R ).

Then for each function g ∈ G4 , there exists a constant C > 0 (independent of ηe ) such that
max |E[g(xk )] − E[g(Xkηe )]| ≤ Cηe ,

0≤k≤N

(8)

when the SDE starts from the same initial point x0 as SGA. That is, the SGA is a first-order weak
approximation of the SDE.
˜ match in moments. The
To apply Theorem B.2, the major condition to verify is that ∆ and ∆
˜
following lemma computes the moments for ∆:
Lemma B.3. For drift function b and diffusion function σ that belong to G4 , there is a function
K2 ∈ G (independent of ηe ) such that for all x ∈ RD and 1 ≤ i, j, l ≤ D,
˜ i (x, k)] − ηe bi (x, kηe ) ≤ K2 (x)ηe2 ,
E[∆
˜ i (x, k)∆
˜ j (x, k)] − ηe2
E[∆

D
X

σi,k (x, kηe )σj,k (x, kηe ) ≤ K2 (x)ηe2 ,

k=1

˜ i (x, k)∆
˜ j (x, k)∆
˜ l (x, k)] ≤ K2 (x)ηe2 .
E[∆
17


## Page 18

B.1

Proof for Theorem B.2

To prove Theorem B.2, we need the following lemma from Li et al. (2019). However, the original
version of the lemma does not apply to time-dependent SDEs, i.e., b and σ cannot change with time
t. By carefully scrutinizing the proof, we find that the proof is indeed applicable to time-dependent
SDEs.
Lemma B.4 (Adaption of Proposition 25, Li et al. (2019)). Suppose that the drift function b and
diffusion function σ are Lipschitz and belong to Gα for some α ≥ 1. Let s ∈ [0, T ] and g ∈ Gα . For
t ∈ [s, T ], define
u(x, s, t) := EXt ∼PX (x,s,t) [g(Xt )],
α
Then u(·, s, t) ∈ G uniformly in s, t.
We need the following lemma to bound the growth of xk .
Lemma B.5. Under Condition (3) of Theorem B.2, given the initial point x0 and any m ≥ 1, there
′
exists a constant C2m
> 0 (depending on x0 but independent of ηe ) such that the parameters {xk }
′
of SGA starting from x0 can be uniformly bounded by E[∥xk ∥2m
2 ] ≤ C2m for all k ≤ N := ⌊T /ηe ⌋.
Proof. It suffices to show that both E[(1 + ∥xk ∥2P )m ] and E[(1 + ∥xk ∥2R )m ] are uniformly bounded.
First, we show that for E[(1 + ∥xk ∥2P )m ]. For all 2 ≤ j ≤ 2m, by Jensen’s inequality,
h
i
j
j

 2m
j
2m j/2
2m .
E ∥∆(xk , k)∥jP | xk ≤ E ∥∆(xk , k)∥2m
≤ C2m
ηe (1 + ∥xk ∥2m
P | xk
P )
Then there exists a constant Ĉ1 > 0 so that
h
i
E ∥∆(xk , k)∥jP | xk ≤ Ĉ1 ηe (1 + ∥xk ∥jP ).
(9)
P
For any two vectors x, y ∈ RD , we use the notation ⟨x, y⟩P := i∈P xi yi to denote the inner
m
product of x and y restricting on coordinates in P . Now we expand 1 + ∥xk+1 ∥2P
using the
update rule. Let δk := 2⟨xk , ∆(xk , k)⟩P + ∥∆(xk , k)∥2P . Then by the binomial theorem,
h
i
h
i
m
m
E 1 + ∥xk+1 ∥2P
| xk = E 1 + ∥xk ∥2P + δk
| xk
= (1 + ∥xk ∥2P )m + mE[δk | xk ](1 + ∥xk ∥2P )m−1
m  
X
m
+
E[δkj | xk ](1 + ∥xk ∥2P )m−j .
j
j=2
By (9), it can be shown that there is a constant Ĉ2 such that E[δkj | xk ] ≤ Ĉ2 ηe (1 + ∥xk ∥2P )j for
j ≥ 2. Then there exists constant Ĉ3 , Ĉ4 such that
h
i
m
E 1 + ∥xk+1 ∥2P
| xk ≤ (1 + ∥xk ∥2P )m + mE[δk | xk ](1 + ∥xk ∥2P )m−1
+ Ĉ3 ηe (1 + ∥xk ∥2P )m
≤ (1 + ∥xk ∥2P )m + 2m ⟨xk , E[∆(xk , k) | xk ]⟩P (1 + ∥xk ∥2P )m−1
+ Ĉ4 ηe (1 + ∥xk ∥2P )m .
Recall that ∥E[∆(xk , k) | xk ]∥P ≤ C1 ηe (1 + ∥xk ∥P ) by Condition (3). Thus, there exists a constant
Ĉ5 (independent of ηe , ∥xk ∥P ) such that
h
i
m
E 1 + ∥xk+1 ∥2P
| xk ≤ (1 + Ĉ5 ηe )(1 + ∥xk ∥2P )m .
Taking the expectation over xk gives
h
m i
E 1 + ∥xk+1 ∥2P
≤ (1 + Ĉ5 ηe )E[(1 + ∥xk ∥2P )m ].
Then taking a telescoping product proves that for all k ≤ N ,
E[(1 + ∥xk ∥2P )m ] ≤ (1 + Ĉ5 ηe )k (1 + ∥x0 ∥2P )m ≤ exp(Ĉ5 kηe )(1 + ∥x0 ∥2P )m
≤ exp(Ĉ5 T )(1 + ∥x0 ∥2P )m .
18


## Page 19

So E[(1 + ∥xk ∥2P )m ] is uniformly bounded (independent of ηe , k).
Now we show that E[(1 + ∥xk ∥2R )m ] is uniformly bounded. We can repeat the argument above to
1
bound E[(1 + ∥xk ∥2R )m ], while utilizing the bounds ∥E∆(x, k)∥R ≤ C1 ηe (1 + ∥x∥ω
P )(1 + ∥x∥R ),
ω2m
2m
m
2m
E∥∆(x, k)∥R ≤ C2m ηe (1 + ∥x∥P )(1 + ∥x∥R ). In the end we can obtain the following for
some real constant Ĉ6 > 0 and some integer constant ω̂ > 0:
h
m i


2 m
E 1 + ∥xk+1 ∥2R
≤ (1 + Ĉ6 ηe E[1 + ∥xk ∥2ω̂
.
P ])E (1 + ∥xk ∥R )
As we have shown, E[1 + ∥xk ∥2ω̂
P ] is uniformly bounded by a constant. Taking a telescoping product
proves that E[(1 + ∥xk ∥2R )m ] is uniformly bounded (independent of ηe , k).
We also need the following lemma adapted from Lemma C.2, Li et al. (2021).
Lemma B.6 (Adaption of Lemma C.2, Li et al. (2021)). Let u1 , . . . , uN be a set of functions that
belong to G4 uniformly. Under Conditions (2), (3) in Theorem B.2, if b and σ are Lipschitz, then
there exists a function K1′ ∈ G (independent of ηe ) such that
˜
E[uj (x + ∆(x, k))] − E[uj (x + ∆(x,
k))] ≤ K1′ (x)ηe2 ,
for all 1 ≤ j ≤ N , 1 ≤ k ≤ N and x ∈ Xk .
Proof. Since u1 , . . . , uN ∈ G4 uniformly, we can find K0 ∈ G such that uj (x) is bounded by
K0 (x) and so are all the partial derivatives of uj up to order 4.
By Taylor’s Theorem with Lagrange Remainder, for all 1 ≤ j ≤ N , 1 ≤ k ≤ N we have
uj (x + ∆(x, k)) − uj (x + ˜
∆(x, k))
!
3
s
s
Y
Y
X 1
X
∂ s uj
˜ i (x, k) +Rj − R̃j ,
∆
∆ir (x, k) −
=
r
∂xi1 ···∂xis (x)
s!
r=1
r=1
s=1
1≤i1 ,...,is ≤D
|
{z
}
=:Mj

where the remainders Rj , R̃j are
Rj :=

R̃j :=

1
4!
1
4!

X
1≤i1 ,...,i4 ≤D

X
1≤i1 ,...,i4 ≤D

∂ 4 uj
∂xi1 ···∂xi4 (x + a∆(x, k))

∂ 4 uj
˜
∂xi1 ···∂xi4 (x + ã∆(x, k))

4
Y

∆ir (x, k).

r=1
4
Y

˜ i (x, k).
∆
r

r=1

for some a, ã ∈ [0, 1].
By Condition (2), the expectation of Mj can be bounded by
E[Mj ] ≤

3
X
1

s!
s=1

X
1≤i1 ,...,is ≤D

∂ s uj
∂xi1 ···∂xis (x)

· K1 (x)ηe2 ≤

3
X
Ds

s!
s=1

K0 (x)K1 (x)ηe2 ,

so η12 E[Mj ] is uniformly bounded by a function in G.
e
Now let κ0 , m be the constants so that K0 (x)2 ≤ κ20 (1 + ∥x∥2m
2 ). For Rj , by Cauchy-Schwarz
inequality we have




1/2
2

 1/2
4
X
Y
2
4
1  X
∂ uj
E[Rj ] ≤
E ∂xi ···∂x
(x + a∆(x, k))  · 
E
∆ir (x, k) 
i4
1
4! i ,...,i
r=1
i ,...,i
1

4




1  X
∂ 4 uj
≤
E ∂xi ···∂x
i4
1
4! i ,...,i
1

1

4


 1/2
2

1/2
(x + a∆(x, k))  · E ∥∆(x, k)∥82

4


1/2
1
≤
D2 · K0 (x + a∆(x, k)) · ηe4 K8 (x)
,
4!
19


## Page 20

where the last line uses Condition (3) and K8 is a function of polynomial growth. For K0 (x +
a∆(x, k)), we can bound its expectation by

1/2
E[K0 (x + a∆(x, k))] ≤ κ0 E 1 + ∥x + a∆(x, k)∥2m
2

2m 1/2
≤ κ0 1 + 22m−1 E[∥x∥2m
2 + E∥∆(x, k)∥2 ]
1/2
2m
2m
≤ κ0 1 + 22m−1 (∥x∥2m
.
2 + C2m ηe (1 + ∥x∥2 ))
Combining this with our bound for E[Rj ] proves that η12 E[Rj ] is uniformly bounded by a function in
e
G:
1/2
1
1/2
2m
2m
E[Rj ] ≤ ηe2 · · D4 · κ0 1 + 22m−1 (∥x∥2m
· K8 (x).
2 + C2m ηe (1 + ∥x∥2 ))
4!
˜ and conclude that 12 E[R̃j ]
Then we can repeat the above argument for Rj while replacing ∆ as ∆,
ηe
is also uniformly bounded by a function in G. To do so, we note that b, σ are Lipschitz, and apply a
similar argument as in Lemma 26 of Li et al. (2019) to show that for all s ≥ 1 there exists a function
K̃ ∈ G such that for all 1 ≤ i1 , . . . , is ≤ D,
" s
#
Y
˜
E
∆ir (x, k) ≤ K̃(x)ηes .
r=1

Finally, we can find a function K1′ ∈ G such that E[Mj ] ≤ 13 K1′ (x)ηe2 , E[Rj ] ≤ 13 K1′ (x)ηe2 ,
E[R̃j ] ≤ 13 K1′ (x)ηe2 . Then E[uj (x + ∆(x, k))] − E[uj (x + ˜
∆(x, k))] ≤ K1′ (x)ηe2 .
Now we are ready to present our proof for Theorem C.2.
Proof for Theorem B.2. For 0 ≤ j ≤ k, let x̂j,k be a random variable that is distributed as the
probability distribution PX (xj , jηe , kηe ) conditioned on xj . By definition, Pr[x̂k,k = xk ] = 1,
x̂0,k ∼ Xkηe . Let u(x, s, t) := EXt ∼PX (x,s,t) [g(Xt )]. Then we can do the following decomposition
for E[g(xk )] − E[g(Xkηe )]:
|E[g(xk )] − E[g(Xkηe )]| =

k−1
X

(E[g(x̂j+1,k )] − E[g(x̂j,k )])

j=0

=

k−1
X

(E[u(x̂j+1,j+1 , (j + 1)ηe , kηe )] − E[u(x̂j,j+1 , (j + 1)ηe , kηe )]) .

j=0

Taking absolute values gives
|E[g(xk )] − E[g(Xkηe )]| ≤

k−1
X

|E[u(x̂j+1,j+1 , (j + 1)ηe , kηe )] − E[u(x̂j,j+1 , (j + 1)ηe , kηe )]| .

j=0

˜ j , j).
Let uj+1 (x) := u(x, (j +1)ηe , kηe ). Note that x̂j+1,j+1 ∼ xj +∆(xj , j), x̂j,j+1 ∼ xj + ∆(x
We can rewrite the above formula as
k−1
X
|E[g(xk )] − E[g(Xkηe )]| ≤
E[uj+1 (xj + ∆(xj , j))] − E[uj+1 (xj + ˜
∆(xj , j))] .
j=0
4

By Lemma B.4, u ∈ G uniformly in s, t, so u1 , . . . , uN ∈ G4 uniformly. Then by Lemma B.6, we
know that there exists a function K1′ (x) = κ1 (1 + ∥x∥2m
2 ) ∈ G such that
˜ j , j))] ≤ E[K ′ (xj )η 2 ],
E[uj+1 (xj + ∆(xj , j))] − E[uj+1 (xj + ∆(x
1

e

for all 0 ≤ j < N . Combining this with Lemma B.5, we can bound |E[g(xk )] − E[g(Xkηe )]| by
|E[g(xk )] − E[g(Xkηe )]| ≤

k−1
X

E[K1′ (xj )ηe2 ] ≤ ηe2

j=0

k−1
X

E[κ1 (1 + ∥xj ∥2m
2 )]

j=0

≤ ηe2

k−1
X

′
′
κ1 (1 + C2m
) ≤ κ1 (1 + C2m
)T ηe .

j=0
′
We can complete the proof by noting that κ1 , C2m , T are independent of ηe .

20


## Page 21

B.2

Proof for Lemma B.3

To prove Lemma B.3, we only need to verify the following lemma using Itô-Taylor expansion.
Lemma B.7. Let ψ : RD → R be a function in G4 . Define


X
X
X
1
2

At ψ(x) :=
bi (x, t)∂i ψ(x) +
σi,l (x, t)σl,j (x, t) ∂i,j
ψ(x).
2
i,j∈[D]

i∈[D]

l∈[D]

Then there exists a function K̂ ∈ G such that
h
i
˜
E ψ(x + ∆(x,
k)) − ψ(x) − ηe Akηe ψ(x) ≤ K̂(x)ηe2 .

(10)

Proof
applying Lemma B.7 with ψ(x) being ψ(x̃) =
Qs for Lemma B.3. We can prove the lemma by
s
(x̃
−
x
)
for
any
tuple
(i
,
.
.
.
,
i
)
∈
[D]
with
s ≤ 3 elements.
i
i
1
s
r
r
r=1
Proof for Lemma B.7. WLOG we prove the case of k = 0, then all the other cases can be proved by
shifting the time. Let Λt ψ(x) := σ(x, t)⊤ ∇ψ(x).
Z ηe
Z ηe
ψ(Xη ) = ψ(x) +
As ψ(Xs )ds +
⟨Λs ψ(Xs ), dWs ⟩.
0

0

Now we further apply the above formula to As ψ(Xs ). Then we have

Z ηe 
Z s
Z s
ψ(Xη ) = ψ(x) +
As ψ(x) +
Ar As ψ(Xr )dr +
⟨Λr As ψ(Xr ), dWr ⟩ ds
0
0
Z ηe 0
+
⟨Λs ψ(Xs ), dWs ⟩
0
Z ηe
Z ηe Z s
= ψ(x) +
As ψ(x)ds +
Ar As ψ(Xr )drds
0
0
0
Z ηe Z
Z ηe
s
+
⟨Λr As ψ(Xr ), dWr ⟩ds +
⟨Λs ψ(Xs ), dWs ⟩.
0

0

0

Taking expectation, the last two integrals vanish. So we have
Z ηe
Z ηe Z s
Eψ(Xη ) = ψ(x) +
As ψ(x)ds +
E[Ar As ψ(Xr )]drds.
0
0
0
1
By Lipschitzness of b and σ, ηe (As ψ(x) − A0 ψ(x)) is bounded by a function of x with polynomial

growth. Also, Ar As ψ(·) is in G uniformly, then Theorem 19 in (Li et al., 2019) implies that
E[Ar As ψ(Xr )] is also in G uniformly. Then we know that there exists two functions K̂1 , K̂2 ∈ G
such that
Z ηe
Z ηe Z s
|E [ψ(Xηe )] − ψ(x) − ηe A0 ψ(x)| ≤
ηe K̂1 (x)ds +
K̂2 (x)drds
0

0

0

≤ ηe2 (K̂1 (x) + K̂2 (x)).
We can conclude the proof by setting K̂(x) := K̂1 (x) + K̂2 (x).

C

RMSProp SDE Proof

In this section, we prove the theorem for the SDE approximation of RMSprop.
Definition C.1 (SDE for RMSprop, matrix form). For constants σ0 , ϵ0 , and c2 , define the SDE as
dXt = b(Xt )dt + σ(Xt )dWt , where Xt ∈ Rd × Rd , and b and σ are defined by
bi (θ, u) := − σ0 √u1i +ϵ0 · ∂i f (θ),


1/2
1
σi,j (θ, u) := √ui +ϵ
·
Σ
(θ)
0 /σ0

bd+i (θ, u) := c2 (Σ(θ)i,i − ui ).
,
i,j

σd+i,j (θ, u) := 0,

σi,d+j (θ, u) := 0,
σd+i,d+j (θ, u) := 0,

for all 1 ≤ i, j ≤ d.
21


## Page 22

Theorem C.2. Fix constants σ0 , c2 > 0, ϵ0 ≥ 0. Let T > 0, η 2 ∈ (0, 1 ∧ T ∧ 2c12 ) and set
N = ⌊T /η 2 ⌋. Let uk ≜ vk /σ 2 and xk ≜ (θk , uk ) ∈ R2d . Let {xk : k ≥ 0} be the discrete
RMSprop iterations defined in Definition 2.1, where σ, ϵ, β are set so that σ0 = ση, ϵ0 = ϵη and
c2 = (1 − β)/η 2 . For well-behaved NGOS that satisfies the bounded moments and low skewness
conditions, the SDE as defined in Definition C.1 is an order-1 weak approximation (Definition 2.4) of
discrete RMSprop, if they start with X0 = x0 .
The basic idea is to apply the general theorem (Theorem B.2) with ηe := η 2 . However,√the SDE
above does not satisfy Condition (1) in Theorem B.2 because the denominators such as σ0 ui + ϵ0
can be unbounded. To solve this issue, the first step is to reduce Theorem C.2 to proving the order-1
weak approximation for the following auxiliary SDE:
Definition C.3. Define τ : R → R to be the following smooth transition function:


if z ≥ 1,
1
−1/z
e
τ (z) = e−1/z +e−1/(1−z)
if z ∈ (0, 1),

0
if z ≤ 0.

(11)

Definition C.4 (Auxiliary SDE for RMSprop, matrix form). For constants σ0 , ϵ0 , c2 and umin , define
µ(u) as the following function
− 1) · (u − 12 umin ),
µ(u) := 12 umin + τ ( u2u
min

(12)

and define the SDE as dXt = b(Xt )dt + σ(Xt )dWt , where Xt ∈ Rd × Rd , and b and σ are
defined by
bi (θ, u) := − √ 1
σ0

σi,j (θ, u) := √

µ(ui )+ϵ0

1
µ(ui )+ϵ0 /σ0

· ∂i f (θ),


· Σ(θ)1/2

bd+i (θ, u) := c2 (Σ(θ)i,i − ui ).
,
i,j

σd+i,j (θ, u) := 0,

σi,d+j (θ, u) := 0,
σd+i,d+j (θ, u) := 0.

for all 1 ≤ i, j ≤ d.
Theorem C.5. In the setting of Theorem C.2, let umin = 2−c2 T mini∈[d] u0,i then the SDE defined by
Definition C.4 is an order-1 weak approximation (Definition 2.4) of discrete RMSprop (Definition 2.1),
if they start with X0 = x0 .
Proof for Theorem C.2. Given Theorem C.5, we only need to show that Xt has the same distribution
in the original and auxiliary SDEs for all t ∈ [0, T ], when X0 = (θ0 , u0 ). To see this, we only need
to note that in the original SDE
dut,i
= c2 (Σ(θt )i,i − ut,i ) ≥ −c2 ut,i .
dt
Thus, ut,i ≥ exp(−c2 t)u0,i ≥ umin , which means Pr[ut,i = µ(ut,i )] = 1 for all t ∈ [0, T ].
It remains to prove Theorem C.5 by applying Theorem B.2. In the rest of this section, we verify the
three conditions in Theorem B.2 respectively.
Below we use the notations xk , Xt , b, σ defined as in Theorem C.5. Let D = 2d. Every xk ∈ RD is
a concatenation of two Rd -vectors θk and uk . According to the update rule of RMSprop, xk can
be seen as SGA xk+1 = xk − ηe hk (xk , zk , ηe ), where ηe = η 2 , zk ∼ Zσ (θk ), and hk is defined
below:


√
−(∇f (θ) + σz) ⊙ (σ0 u + ϵ0 )−1
hk (θ, u, z, ηe ) :=
.
c2 (∇f (θ)/σ + z)2 − u
˜ as in (6) and (7). Fix x0 = (θ0 , u0 ) with u0,j > 0 for all j ∈ [d]. Define umin
We define ∆ and ∆
as in Theorem C.5. Let Xk be the support of the random variable xk given x0 , then it is easy to show
that Xk is a subset of {(θ, u) : uj ≥ umin for all j ∈ [d]}.
22


## Page 23

C.1

Verifying Condition (1)

Lemma C.6. The drift function b and diffusion function σ are Lipschitz and belong to G4 .
Proof. Σ1/2 (θ) is bounded and Lipschitz, so Σ(θ) is Lipschitz. Note that the denominators in the
fractions in the formulas of b and σ are always lower bounded by a constant. Then the Lipschitz
property of b and σ can be implied by the Lipschitz property of ∇f (θ), Σ(θ), Σ1/2 (θ), and
b, σ ∈ G4 can be implied by ∇f, Σ1/2 , µ ∈ G4 .
C.2

Verifying Condition (2)

˜ for the discrete RMSprop
To verify Condition (2), we only need to compute the moments of ∆ and ∆
and the auxiliary SDE, and show that they are close to each other. We compute them by the following
two lemmas.
Lemma C.7. For x = (θ, u) ∈ Xk , if the NGOS is well-behaved and Zσ satisfies the bounded
moments and low skewness condition, then the moments of ∆ := ∆(x, k) − x can be written as
below.
1. For 1 ≤ i ≤ d, the following holds for the first moments:
E[∆i ] = −

σ0

√

η2
∂i f (θ),
ui + ϵ0

E[∆d+i ] = η 2 c2 (Σii (θ) − ui ) +

η 4 c2
(∂i f (θ))2
σ02

= η 2 c2 (Σii (θ) − ui ) + O(η 4 ).
2. For 1 ≤ i, j ≤ d, the following holds for the second moments:
η 2 Σij (θ)
E[∆i ∆j ] = √
+ O(η 4 )
√
( ui + ϵ0 /σ0 )( uj + ϵ0 /σ0 )
E[∆d+i ∆j ] = O(η 4 )

E[∆i ∆d+j ] = O(η 4 )
E[∆d+i ∆d+j ] = O(η 4 ).

for all i, j ∈ [d].
3. The third moments are bounded by E[∆⊗3 ] = O(η 4 ).
Here the big-O notation O( · ) is used in a way that O(1) hides constants (independent of η and x)
and values that are bounded by a function of x with polynomial growth.
Proof. We note that
∆i = −

σ0

√

η2
(∂i f (θ) + σzi ) ,
ui + ϵ0


∆d+i = (1 − β) (∂i f (θ)/σ + zi )2 − ui .

1
2
Let νi := σ0 √u1i +ϵ0 . Since x ∈ Xk , νi ≤ σ0 √umin
+ϵ0 = O(1). Writing 1 − β as c2 η , we have

∆i = −νi η 2 (∂i f (θ) + σzi ) ,
∆d+i = c2 η 2 (∂i f (θ)/σ + zi )2 − ui .
(13)

We can now compute the first moments:
E[∆i ] = −νi η 2 ∂i f (θ)

(E[zi ] = 0)

E[∆d+i ] = c2 η

2

2

+ E[zi2 ] − E[ui ]

E[(∂i f (θ)/σ)

= c2 η

2


(∂i f (θ)/σ)2 + Σii − ui .



Now we observe that 1/σ = O(η), so we can write
E[∆d+i ] = η 2 c2 (Σii − ui ) + O(η 4 ).
Let δ := ∆ − E[∆]. That is,
δi = −νi η 2 σzi
= −νi σ0 ηzi .

δd+i = c2 η 2 (zi2 − Σii + 2zi ∂i f (θ)/σ)
= c2 η 2 (zi2 − Σii ) + 2c2 (∂i f (θ)/σ0 )η 3 zi .
23


## Page 24

For convenience we also define wi = (zi2 − Σii ) + 2(∂i f (θ)/σ0 )ηzi and write δd+i = c2 η 2 wi .
For the second moments we have
E[∆p ∆q ] = E[δp δq ] + E[∆p ]E[∆q ] = E[δp δq ] + O(η 4 )

for all 1 ≤ p, q ≤ 2d.

Then it suffices to compute the second moments for δ. For E[δi δj ] we have
E[δi δj ] = νi νj η 4 σ 2 E[zi zj ] = νi νj σ02 Σij .
For E[δi δd+j ] we have
E[δi δd+j ] = −c2 νi σ0 η 3 E [zi wj ]
= −c2 νi σ0 η 3 E[zi zj2 ] + O(η 4 )
= E[zi zj2 ] · O(η 3 ) + O(η 4 ).
Similarly, we have E[∆d+i ∆j ] = E[zi2 zj ] · O(η 3 ) + O(η 4 ).
For E[δd+i δd+j ], we note that z ∼ Zσ (θ) has bounded 4th-order moments, so E[g(z)] = O(1) for
any polynomial g of degree at most 4, if the coefficients of g are bounded by O(1). Then we have
E[δd+i δd+j ] = c22 η 4 E[wi wj ] = O(η 4 ).
Now we can check the third moments.
E[∆p ∆q ∆r ] = E[δp δq δr ] + (E[δp δq ]E[∆r ] + E[δp δr ]E[∆q ] + E[δq δr ]E[∆p ]) + E[∆p ]E[∆q ]E[∆r ]
= E[δp δq δr ] + O(η 2 ) · O(η 2 ) + O(η 6 )
= E[δp δq δr ] + O(η 4 ).
Note that δp δq δr is a polynomial of z. For p = i, q = j, r = k, by the low skewness condition for
Zσ we have
E[δi δj δk ] = −νi3 σ03 η 3 E[zi zj zk ] = K3 (θ)/σ · O(η 3 ) = O(η 4 ).
Except the above case, it can be shown that δp δq δr is a polynomial with coefficients bounded by
O(η 4 ). Combining this with the fact that z ∼ Zσ (θ) has bounded moments of any order, we have
E[δp δq δr ] = O(η 4 ).
˜ := ˜
Lemma C.8. For x = (θ, u) ∈ Xk , if the NGOS is well-behaved, then the moments of ∆
∆(x, k)
can be written as below.
1. The first moments are given by
˜ i] = −
E[∆

η2
σ0

p

µ(ui ) + ϵ0

˜ d+i ] = η 2 c2 (Σii (θ) − ui ) + O(η 4 ).
E[∆

∂i f (θ),

for all i ∈ [d].
2. The second moments are given by
η 2 Σij (θ)
˜ i∆
˜ j] = p
˜ i∆
˜ d+j ] = O(η 4 )
p
E[∆
+ O(η 4 )
E[∆
( µ(ui ) + ϵ0 /σ0 )( µ(uj ) + ϵ0 /σ0 )
˜ d+i ∆
˜ j ] = O(η 4 )
˜ d+i ∆
˜ d+j ] = O(η 4 ).
E[∆
E[∆
for all i, j ∈ [d].
˜ ⊗3 ] = O(η 4 ).
3. The third moments are bounded by E[∆
Here the big-O notation O( · ) is used in a way that O(1) hides constants (independent of η and x)
and values that are bounded by a function of x with polynomial growth.
Proof. Applying Lemma B.3 gives
E[ ˜
∆] = η 2 b(x) + O(η 4 ),

˜∆
˜ ⊤ ] = η 2 σ(x)σ(x)⊤ + O(η 4 ),
E[∆

Splitting up the formula by indices proves the claim.
24

˜ ⊗3 ] = O(η 4 ).
E[∆


## Page 25

C.3

Verifying Condition (3)

Lemma C.9. Let P := {1, 2, . . . , d}. Then
1. There is a constant C1 > 0 (independent of ηe ) so that for all k ≤ N and x ∈ Xk ,
∥E∆(x, k)∥P ≤ C1 ηe (1 + ∥x∥P ),
∥E∆(x, k)∥R ≤ C1 ηe (1 + ∥x∥2P )(1 + ∥x∥R ),
2. For all m ≥ 1, there is a constant C2m (independent of ηe ) so that for all k ≤ N and x ∈ Xk ,
m
2m
E∥∆(x, k)∥2m
P ≤ C2m ηe (1 + ∥x∥P ),
m
4m
2m
E∥∆(x, k)∥2m
R ≤ C2m ηe (1 + ∥x∥P )(1 + ∥x∥R ),

Proof. By Lemma C.7, for all i ∈ [d],
E[∆i ] = −

σ0

√

η2
∂i f (θ),
ui + ϵ0

E[∆d+i ] = η 2 c2 (Σii (θ) − ui ) +

η 4 c2
(∂i f (θ))2 .
σ02

Combining this with the Lipschitzness of ∇f (θ) and the boundedness of Σ(θ) proves Item 1.
By (13), for all i ∈ [d],
|∆i | = νi η 2 |∂i f (θ) + σzi | ≤ η 2 νi (1 + |∂i f (θ)|)(1 + σ|zi |) ≤ ηνi (1 + |∂i f (θ)|)(1 + σ0 |zi |)
|∆d+i | = c2 η 2 (∂i f (θ)/σ + zi )2 − ui ≤ c2 η 2 (1 + (∂i f (θ)/σ + zi )2 )(1 + ui )
By the Lipschitzness of ∇f (θ) and the bounded moments condition for Zσ , we can prove Item 2 by
taking powers and expectations on both sides of the above inequalities.

D

Adam SDE Proof

In this section, we prove the theorem for the SDE approximation of Adam.
Definition D.1 (SDE for Adam, matrix form). For constants σ0 , ϵ0 , c1 and c2 , define γ1 (t) :=
1 − e−c1 t , γ2 (t) := 1 − e−c2 t and define the SDE as dXt = b(Xt )dt + σ(Xt )dWt , where
Xt ∈ Rd × Rd × Rd , b is defined by
√
γ2 (t)
bi (x, t) := − γ1 (t)
,
· √ mi √
σ0

ui +ϵ0

γ2 (t)

bd+i (x, t) := c1 (∂i f (θ) − mi ),
b2d+i (x, t) := c2 (Σ(θ)i,i − ui ),
for all 1 ≤ i ≤ d, and σd+i,d+j for all 1 ≤ i, j ≤ d is given by


σd+i,d+j (x, t) := σ0 c1 Σ1/2 (θ)

,
i,j

and all the other entries of σ are zero.
Theorem D.2. Fix σ0 , c1 , c2 > 0, ϵ0 ≥ 0. Let T > 0, η 2 ∈ (0, 1 ∧ T ∧ 2c12 ) and set N = ⌊T /η 2 ⌋.
Let uk ≜ vk /σ 2 and xk ≜ (θk , mk , uk ) ∈ R3d . Let {xk : k ≥ 0} be the discrete Adam
iterations defined in Definition 2.2. Set σ, ϵ, β1 , β2 so that σ0 = ση, ϵ0 = ϵη, c1 = (1 − β1 )/η 2
and c2 = (1 − β2 )/η 2 . For well-behaved NGOS that satisfies the bounded moments and low
skewness conditions, for any constant t0 > 0, the solution Xt (t ∈ [t0 , T ]) of the SDE defined as in
Definition D.1 is an order-1 weak approximation (Definition 2.4) of the sequence of Adam iterates xk
starting from k0 = ⌈t0 /η 2 ⌉, if the initial condition of the SDE is set to Xt0 = xk0 .
The proof strategy is essentially the same as that for RMSprop. Similar to what we have done for
RMSprop, we turn to prove the approximation order for the following auxiliary SDE.
25


## Page 26

Definition D.3 (Auxiliary SDE for Adam, matrix form). For constants σ0 , ϵ0 , c1 , c2 and umin , define
− 1) · (u − 12 umin ), and define the
γ1 (t) := 1 − e−c1 t , γ2 (t) := 1 − e−c2 t , µ(u) := 21 umin + τ ( u2u
min
d
d
SDE as dXt = b(Xt )dt + σ(Xt )dWt , where Xt ∈ R × R × Rd , b is defined by
√
γ2 (t)
mi √
,
bi (x, t) := − γ1 (t)
· √
σ0

µ(ui )+ϵ0

γ2 (t)

bd+i (x, t) := c1 (∂i f (θ) − mi ),
b2d+i (x, t) := c2 (Σ(θ)i,i − ui ),
for all 1 ≤ i ≤ d, and σd+i,d+j for all 1 ≤ i, j ≤ d is given by


σd+i,d+j (x, t) := σ0 c1 Σ1/2 (θ)

,
i,j

and all the other entries of σ are zero.
Theorem D.4. In the setting of Theorem D.2, let umin = 2−(T −t0 ) mini∈[d] uk0 ,i , For any constant
t0 > 0, the solution Xt (t ∈ [t0 , T ]) of the SDE defined as in Definition D.3 is an order-1 weak
approximation (Definition 2.4) of the sequence of Adam iterates xk starting from k0 = ⌈t0 /η 2 ⌉, if
the initial condition of the SDE is set to Xt0 = xk0 .
Proof for Theorem D.2. Given Theorem D.4, we only need to show that Xt has the same distribution
in the original and auxiliary SDEs for all t ∈ [0, T ], when X0 = (θ0 , m0 , u0 ). To see this, we only
need to note that in the original SDE
dut,i
= c2 (Σ(θt )i,i − ut,i ) ≥ −c2 ut,i .
dt
Thus, ut,i ≥ exp(−c2 t)u0,i ≥ umin , which means Pr[ut,i = µ(ut,i )] = 1 for all t ∈ [0, T ].
It remains to prove Theorem D.4 by applying Theorem B.2. In the rest of this section, we verify the
three conditions in Theorem B.2 respectively.
Below we use the notations xk , Xt , b, σ defined as in Theorem D.4. Let D = 3d. Every xk ∈ RD is
a concatenation of three Rd -vectors θk , mk and uk . According to the update rule of Adam, xk can
be seen as SGA xk+1 = xk − ηe hk (xk , zk , ηe ), where ηe = η 2 , zk ∼ Zσ (θk ), and hk is defined
below:
 √
 √
−1 
p
1−β k
− 1−β k+12 m ⊙ σ0 u + ϵ0 1 − β2k


1
hk (θ, u, z, ηe ) := 
.
c1 (∇f (θ) + σz − m) 
2
c2 (∇f (θ)/σ + z) − u
˜ as in (6) and (7).
We define ∆ and ∆
Fix x0 = (θ0 , m0 , u0 ) with u0,j > 0 for all j ∈ [d]. Define umin as in Theorem D.4. Let Xk
be the support of the random variable xk given x0 , then it is easy to show that Xk is a subset of
{(θ, m, u) : uj ≥ umin for all j ∈ [d]}.
D.1

Verifying Condition (1)

Lemma D.5. The drift function b and diffusion function σ are Lipschitz and belong to G4 .
Proof. Same argument as for Lemma C.6.
D.2

Verifying Condition (2)

Let x = (θ, m, u) ∈ R3d . For ease of notation, let γ̂1 = 1 − β1k+1 and γ̂2 = 1 − β2k . These are not
constants across time steps like the other constants, but they are deterministic and upper bounded by
constants for k ≥ t0 /η 2 .
˜ for the discrete Adam and
To verify Condition 2, we only need to compute the moments of ∆ and ∆
the auxiliary SDE, and show that they are close to each other. We compute them by the following
two lemmas.
26


## Page 27

Lemma D.6. For x = (θ, u) ∈ Xk , if the NGOS is well-behaved and Zσ satisfies the bounded
moments and low skewness condition, then the moments of ∆ := ∆(x, k) can be written as below.
1. The first moments are given by
√
γ̂2
η2
√ (mi + c1 η 2 (∂i f (θ) − mi ))
E[∆i ] = −
· √
γ̂1 σ0 ui + ϵ0 γ̂2
√
γ̂2
η 2 mi
√ + O(η 4 )
=−
· √
γ̂1 σ0 ui + ϵ0 γ̂2
E[∆d+i ] = c1 η 2 (∂i f (θ) − mi )
E[∆2d+i ] = c2 η 2 ((∂i f (θ)/σ)2 + Σii − ui )
= c2 η 2 (Σii − ui ) + O(η 4 ).
for all i ∈ [d].
2. The second moments are given by
 2 2 2
c σ η Σ + O(η 4 )
E[∆p ∆q ] = 1 04 ij
O(η )

if p = d + i, q = d + j
otherwise.

for some i, j ∈ [d]

for all p, q ∈ [3d].
3. The third moments are bounded by E[∆⊗3 ] = O(η 4 ).
Here the big-O notation O( · ) is used in a way that O(1) hides constants (independent of η and x)
and values that are bounded by a function of x with polynomial growth.
Proof. For notational convenience, we write ∆, Σ instead of ∆(x), Σ(θ). We note that
√
γ̂2 η 2 mi + η 2 (1 − β1 )(∂i f (θ) + σzi − mi )
√
∆i = −
·
√
γ̂1
σ0 ui + ϵ0 γ̂2
∆d+i = (1 − β1 )(∂i f (θ) + σzi − mi )
∆2d+i = (1 − β2 )((∂i f (θ)/σ + zi )2 − ui )
√

Let νi :=

γ̂2
2
√ 1 √
γ̂1 · σ0 ui +ϵ0 γ̂2 . And we write 1 − β = c2 η . Then we have

∆i = −νi η 2 (mi + c1 η 2 (∂i f (θ) + σzi − mi ))
∆d+i = c1 η 2 (∂i f (θ) + σzi − mi )
2

(14)

2

∆2d+i = c2 η ((∂i f (θ)/σ + zi ) − ui )
The first moments follow directly:
E[∆i ] = −νi η 2 (mi + c1 η 2 (∂i f (θ) − mi )) = −νi η 2 mi + O(η 4 )
E[∆d+i ] = c1 η 2 (∂i f (θ) − mi )
E[∆2d+i ] = c2 η 2 ((∂i f (θ)/σ)2 + Σii − ui ) = c2 η 2 (Σii − ui ) + O(η 4 ).
Let δ := ∆ − E[∆]. That is,
δi = −νi c1 η 4 σzi = −νi c1 σ0 η 3 zi
δd+i = c1 η 2 σzi = c1 σ0 ηzi
δ2d+i = c2 η 2 (zi2 − Σii + 2zi ∂i f (θ)/σ)
= c2 η 2 (zi2 − Σii ) + 2c2 (∂i f (θ)/σ0 )η 3 zi .
For convenience we also define wi = (zi2 − Σii ) + 2(∂i f (θ)/σ0 )ηzi and write δd+i = c2 η 2 wi .
Similar as the proof for Lemma C.7, for the second moments we have
E[∆p ∆q ] = E[δp δq ] + E[∆p ]E[∆q ] = E[δp δq ] + O(η 4 )
27

for all 1 ≤ p, q ≤ 2d.


## Page 28

Then it suffices to compute the second order moments for δ. A direct computation gives the following:
E[δi δj ] = νi νj c21 σ02 η 6 E[zi zj ] = O(η 6 ).
E[δi δd+j ] = −νi c21 σ02 η 4 E[zi zj ] = O(η 4 ).
E[δi δ2d+j ] = −νi c1 c2 σ0 η 5 E[zi wj ] = O(η 5 ).
E[δd+i δd+j ] = c21 σ02 η 2 E[zi zj ] = c21 σ02 η 2 Σij .
E[δd+i δ2d+j ] = c1 c2 σ0 η 3 E[zi wj ] = c1 c2 σ0 η 3 E[zi zj2 ] + O(η 4 ).
E[δ2d+i δ2d+j ] = c22 η 4 E[wi wj ] = O(η 4 ).
Now we check the third moments. Similar as the proof for Lemma C.7, E[∆p ∆q ∆r ] = E[δp δq δr ] +
O(η 4 ). Note that δp δq δr is a polynomial of z. For p = d + i, q = d + j, r = d + k, by the low
skewness condition for Zσ we have
E[δd+i δd+j δd+k ] = c31 σ03 η 3 E[zi zj zk ] = K3 (θ)/σ · O(η 3 ) = O(η 4 ).
Except the above case, it can be shown that δp δq δr is a polynomial with coefficients bounded by
O(η 4 ). Combining this with the fact that z ∼ Zσ (θ) has bounded moments of any order, we have
E[δp δq δr ] = O(η 4 ).
˜ := ˜
∆(x, k)
Lemma D.7. For x = (θ, u) ∈ Xk , if the NGOS is well-behaved, then the moments of ∆
can be written as below.
1. The first moments are given by
E[∆i ] = −

√

γ̂2
η 2 mi
√ + O(η 4 )
· √
γ̂1 σ0 ui + ϵ0 γ̂2

E[∆d+i ] = c1 η 2 (∂i f (θ) − mi ) + O(η 4 )
E[∆2d+i ] = c2 η 2 (Σii − ui ) + O(η 4 ).
for all i ∈ [d].
2. The second moments are given by
 2 2 2
c σ η Σ + O(η 4 )
E[∆p ∆q ] = 1 04 ij
O(η )

if p = d + i, q = d + j
otherwise.

for some i, j ∈ [d]

for all p, q ∈ [3d].
3. The third moments are bounded by E[∆⊗3 ] = O(η 4 ).
Here the big-O notation O( · ) is used in a way that O(1) hides constants (independent of η and x)
and values that are bounded by a function of x with polynomial growth.
Proof. Applying Lemma B.3 gives
˜ = η 2 b(x) + O(η 4 ),
˜∆
˜ ⊤ ] = η 2 σ(x)σ(x)⊤ + O(η 4 ),
E[∆]
E[∆

˜ ⊗3 ] = O(η 4 ).
E[∆

Noting that γ1 (kηe ) = γ̂1 + O(η 2 ) and γ2 (kηe ) = γ̂2 + O(η 2 ), we can prove the claim.
D.3

Verifying Condition (3)

Lemma D.8. Let P := {1, 2, . . . , 2d}. Then
1. There is a constant C1 > 0 (independent of ηe ) so that for all k ≤ N and x ∈ Xk ,
∥E∆(x, k)∥P ≤ C1 ηe (1 + ∥x∥P ),
∥E∆(x, k)∥R ≤ C1 ηe (1 + ∥x∥2P )(1 + ∥x∥R ).
2. For all m ≥ 1, there is a constant C2m (independent of ηe ) so that for all k ≤ N and x ∈ Xk ,
m
2m
E∥∆(x, k)∥2m
P ≤ C2m ηe (1 + ∥x∥P ),
m
4m
2m
E∥∆(x, k)∥2m
R ≤ C2m ηe (1 + ∥x∥P )(1 + ∥x∥R ).

28


## Page 29

Proof. By Lemma D.6, for all i ∈ [d],
E[∆i ] = −νi η 2 (mi + c1 η 2 (∂i f (θ) − mi ))
E[∆d+i ] = c1 η 2 (∂i f (θ) − mi )
E[∆2d+i ] = c2 η 2 ((∂i f (θ)/σ)2 + Σii − ui ).
Combining this with the Lipschitzness of ∇f (θ) and the boundedness of Σ(θ) proves Item 1.
By (14), for all i ∈ [d], one can show that there exists a constant Ĉ such that
|∆i | ≤ η 2 Ĉ(1 + |mi | + |∂i f (θ)|)(1 + |zi |)
|∆d+i | ≤ η 2 Ĉ(1 + |mi | + |∂i f (θ)|)(1 + |zi |)
|∆2d+i | ≤ η 2 Ĉ(1 + |∂i f (θ)|2 + zi2 )(1 + ui )
By the Lipschitzness of ∇f (θ) and the bounded moments condition for Zσ , we can prove Item 2 by
taking powers and expectations on both sides of the above inequalities.

E

Analysis of SVAG Operator

Lemma E.1. Let Gσ = (f, Σ, Zσ ) be a NGOS and Gbℓσ = (f, Σ, Zbℓσ ) be the NGOS after applying
the SVAG operator with hyperparameter ℓ > 0. Then Gbℓσ is indeed an NGOS. That is, Zbℓσ (θ) is
well-defined and has mean zero, covariance Σ(θ).
Proof. Let Zbℓσ (θ) be the distribution of ẑ := 1ℓ (r1 (ℓ)z1 + r2 (ℓ)z2 ) when z1 , z2 ∼ Zσ (θ). Then it
is easy to check that ĝ has the same distribution as ∇f (θ) + ℓσ ẑ, since
ĝ = r1 (ℓ)g1 + r2 (ℓ)g2 ∼ r1 (ℓ)(∇f (θ) + σz1 ) + r2 (ℓ)(∇f (θ) + σz2 )
= (r1 (ℓ) + r2 (ℓ))∇f (θ) + σ(r1 (ℓ)z1 + r2 (ℓ)z2 )
= ∇f (θ) + ℓσ ẑ,
where the last equality uses the fact that r1 (ℓ) + r2 (ℓ) = 1. Hence Zbℓσ (θ) is well-defined.
Now we check the mean and covariance of ẑ ∼ Zbℓσ (θ). By linearity of expectation and linearity of
variance (for independent variables), we have
1
1
E[ẑ] = (r1 (ℓ)E[z1 ] + r2 (ℓ)E[z2 ]) = (0 + 0) = 0,
ℓ
ℓ

1 2
1
2
Cov(ẑ) = 2 r1 (ℓ)Cov(z1 ) + r2 (ℓ)Cov(z2 ) = 2 (r12 (ℓ) + r22 (ℓ))Σ(θ) = Σ(θ),
ℓ
ℓ
 √
2  √
2
2
2
2 −1
1+
2ℓ
−1
where the last equality is due to r12 (ℓ)+r22 (ℓ) =
+ 1− 2ℓ
= 1+2ℓ2 −1 = ℓ2 .
2
2
Lemma E.2. Let Gσ = (f, Σ, Zσ ) be a NGOS with scale σ. Applying the SVAG operator with
hyperparameter ℓ ≥ 1, we obtain Gbσ̂ = (f, Σ, Zbσ̂ ) with scale σ̂ = ℓσ. Fixing σ and changing ℓ
produces Gbσ̂ for all scales σ̂ ≥ σ. If Gσ is well-behaved and satisfies the bounded moments condition,
then Gbσ̂ is also well-behaved and satisfies the bounded moments condition. Furthermore, Gbσ̂ satisfies
the low-skewness condition.
Proof. The loss function f and covariance function Σ are not changed after applying the SVAG
operator, so Gbσ̂ is well-behaved.
Now we verify the bounded moments condition. Let ẑ = 1ℓ (r1 (ℓ)z1 + r2 (ℓ)z2 ), where z1 , z2 ∼
Zσ (θ). Then we have

1
1
1
1
2m ≤
2m + |r (ℓ)|E[∥z ∥2m ] 2m
E[∥ẑ∥2m
|r1 (ℓ)|E[∥z1 ∥2m
2
2 2
2 ]
2 ]
ℓ
1
1
2m
≤ (|r1 (ℓ)| + |r2 (ℓ)|) · E[∥z1 ∥2m
2 ]
ℓ
√
1
2m .
≤ (1 + 2)E[∥z1 ∥2m
2 ]
29


## Page 30

By the bounded moments condition for Gσ , there exists a constant√ C2m such that
1
d
2m 1
2m ≤ C
Ez∼Zσ (θ) [∥z∥2m
2m (1+∥θ∥2 ) for all θ ∈ R . So E[∥ẑ∥2 ] 2m ≤ (1+ 2)C2m (1+∥θ∥2 )
2 ]
for all θ ∈ Rd .
We now verify the low skewness condition by showing that third moment of ẑ is O(1/ℓ). By the
bounded moments condition with m = 2 and Jensen’s inequality,
E[z1⊗3 ] ≤ E[∥z1 ∥32 ] ≤ E[∥z1 ∥42 ]3/4 ≤ (C4 (1 + ∥θ∥2 ))

3/4

.

3/4

So E[z1⊗3 ] is bounded by K̃3 (θ) := (C4 (1 + ∥θ∥2 ))
of polynomial growth.
√
Let r = 2ℓ2 − 1. Then r1 (ℓ) = 12 (1 + r), r2 (ℓ) = 12 (1 − r). Since the third moments of two
independent random vectors are additive,
i
1 h
⊗3
E[ẑ ⊗3 ] = 3 E ((1 + r)z1 + (1 − r)z2 )
8ℓ

 

1
= 3 E (1 + r)3 z1⊗3 + (1 − r)3 z2⊗3
8ℓ
 

1
= 3 (1 + r)3 + (1 − r)3 E z1⊗3
8ℓ
1
= 3 (2 + 6r2 )E[z1⊗3 ]
8ℓ
1
1
= 3 (12ℓ2 − 4)E[z1⊗3 ] ≤ K̃3 (θ),
8ℓ
ℓ
3
3
where the 4th equality is due to (1+r) +(1−r) = (1+3r+3r2 +r3 )+(1−3r+3r2 −r3 ) = 2+6r2 .
Therefore, the low skewness condition is verified.

F

Miscellaneous Theoretical Arguments

F.1

How does the noise scale change with batch size?

We discuss how the noise scale√ σ in the NGOS (Definition 2.3) changes when the batch size
changes: in particular, σ ∼ 1/ B. The argument follows from the linearity of covariance and
is an already well-known result - we reproduce it here for clarity. We first fix a parameter vector
θ. Let g (1) , . . . , g (B) be the gradients evaluated at data points from a batch of size B, where
g (b) = ∇f (θ) + z (b) , and every z (b) is a gradient noise vector drawn i.i.d. from a distribution with
mean 0 and covariance Σ(θ). For sampling with replacement on a finite dataset of size n, where
f1 (θ), . . . , fn (θ) are the loss functions for the n data points (and the average of these n functions is
f (θ)), this covariance matrix can be explicitly written as:
n

1X
(∇fi (θ) − ∇f (θ))(∇fi (θ) − ∇f (θ))⊤ .
n i=1
PB
PB
The average gradient over the batch is g := B1 b=1 g (b) = ∇f (θ) + B1 b=1 z (b) . As
P
B
z (1) , . . . , z (B) are sampled i.i.d., their average B1 b=1 z (b) has mean 0 and covariance Σ(θ) by
linearity of expectation and covariance. We can set Zσ (θ) to be the distribution of the random
PB
variable √1B b=1 zb , where σ = √1B , then g has the same distribution as the stochastic gradient
produced by the NGOS Gσ = (f, Σ, Zσ ).
Σ(θ) =

F.2

What happens when the noise does not dominate the gradient?

We discuss the linear warm-up setting described in Section 4.1. Recall that when ignoring the effect
of ϵ, the RMSprop update can be written as
θk+1 ≈ θk − ηgk ⊙ (ḡ 2 + σ 2 1)−1/2 .
From the above equation, it is clear that the dynamics of θ depends on the relationship between the
noise scale σ and the gradient ∥ḡ∥. In Section 4.1, we discuss the case where σ ≫ ∥ḡ∥, which is the
regime where the SDE approximation can exist.
30


## Page 31

||Egk||2/E||zk||2
21

B=256
B=1024
B=4096 2 1
B=16384

B=256
B=1024
B=4096
B=16384

24

24
0.000 0.025 0.050 0.075

0.000 0.025 0.050 0.075

Continuous time = k * 2

Continuous time = k * 2

(a) ResNet-50 with ϵ order 10−30
22

||Egk||2/E||zk||2

22

22

||Egk||2/E||zk||2

21
24

||Egk||2/E||zk||2

B=256
21
B=1024
B=4096 2 1
B=16384 2 3

B=256
B=1024
B=4096
B=16384

25
0.000 0.025 0.050 0.075

0.000 0.025 0.050 0.075

Continuous time = k * 2
(c) VGG-16 with ϵ order 10

(b) ResNet-50 with ϵ order 10−8

Continuous time = k * 2

−30

(d) VGG-16 with ϵ order 10−8

Figure 4: We compare the norm of the average gradient with the noise scale for different batch sizes
during training of ResNet-50 and VGG-16 model with RMSprop on the CIFAR-10 dataset. Here,
(η, β) = (10−3 , 0.999) for batch size 256 and scaled with our proposed square root scaling rule
(Definition 5.1) for the other batch sizes. We show the results for ϵ at both small (of order 10−30 ) and
large scale (of order 10−8 ). We observe that for small batches, the noise in the gradient dominates
the signal in the gradient, supporting our hypothesis. For larger batches, the hypothesis seems to hold
true towards the end of training.

Here, we argue that when σ ≪ ∥ḡ∥, no SDE approximation can exist for the discrete trajectory.
p In
−1
this case, the RMSprop update would instead be θk+1 ≈ θk − ηU gk , where U = diag( ḡ 2 ).
Combining this with gk ∼ N (ḡ, σI) yields that θk+1 − θk ∼ N (ηU −1 ḡ, η 2 σ 2 U −2 ) approximately. We can again take
 a telescoping sum to obtain the marginal distribution of θk :
θk ∼ N kηU −1 ḡ, kη 2 σ 2 U −2 approximately.
However, it is impossible to make the above distribution fixed even as σ changes, so no SDE
approximation exists. In particular, we need to make both kη and kη 2 σ 2 fixed, so ησ must be a
constant. If σ ≪ ∥ḡ∥, then η1 ≪ ∥ḡ∥ too. This requirement on η implies that no SDE approximation
exists when σ ≪ ∥ḡ∥, and hence motivates us to study the case of σ ≫ ∥ḡ∥.

G

Experimental Verification of Assumptions

In this section, we take measurements and perform experiments to verify that the various assumptions
made in our theory do not harm the applicability of our findings to realistic settings.

G.1

Noise Dominates the Gradient

Our analysis in Section 4.1 suggests that an SDE approximation cannot exist when the the gradient ḡ
dominates the noise scale σ. Note that Section 4.1 performs a rough analysis under the assumption
of a linear loss function (i.e., fixed gradient throughout training), which is far from practice. In the
more general setting, we require that for every step k, E∥zk ∥2 (i.e., the gradient variance) dominates
∥Egk ∥ (i.e., the norm of the average gradient), where the expectations are taken over sampling seeds,
in order for the SDE approximation to exist. Figure 4 shows that our assumption holds for small
batches and for large batches near the end of training.
31


## Page 32

29

||Egk||2/E||zk||2

25
21

29

B=256
B=1024
B=4096
B=16384

25
21

B=256
B=1024
B=4096
B=16384

23

23
0.000 0.025 0.050 0.075

0.000 0.025 0.050 0.075

Continuous time = k * 2

Continuous time = k * 2

(a) ResNet-50 with ϵ order 10−30
25

||Egk||2/E||zk||2

||Egk||2/E||zk||2

(b) ResNet-50 with ϵ order 10−8
25

21

B=256
B=1024
22
B=4096
B=16384 2 1

24

24

22

0.000 0.025 0.050 0.075

B=256
B=1024
B=4096
B=16384

0.000 0.025 0.050 0.075

Continuous time = k * 2
(c) VGG-16 with ϵ order 10

||Egk||2/E||zk||2

Continuous time = k * 2

−30

(d) VGG-16 with ϵ order 10−8

Figure 5: We compare the norm of the average gradient with the noise scale for different
batch sizes during training of ResNet-50 model with Adam on the CIFAR-10 dataset. Here,
(η, β1 , β2 ) = (10−3 , 0.999, 0.999) for batch size 256 and scaled with our proposed square root
scaling rule (Definition 5.1) for the other batch sizes. We show the results for ϵ at both small (of order
10−30 ) and large scale (of order 10−8 ). We observe that for small batches, the noise in the gradient
dominates the signal in the gradient, supporting our hypothesis. For larger batches, the hypothesis
seems to hold true towards the end of training.

G.2

Using vk instead of vk+1 in the update rule

In Definitions 2.1 and 2.2, we slightly modify the standard implementation of RMSprop and Adam
by using vk in the update rule instead of vk+1 . Here, we verify for Adam that this modification of
the optimization algorithms does not significantly harm performance. Figure 6 shows the behavior
of ResNet-50 and VGG-16 trained with Adam on CIFAR-10 with the above modification of the
optimization algorithm. We observe a small drop (≈ 1%) in test accuracies. However, the behavior
of the trajectories across different batch sizes for the proposed scaling rule stays the same, i.e. we
observe a maximum of 3% test accuracy gap between training batch size 256 and 8192. Moreover,
the behavior of the test functions match across the trajectories of different batch sizes.

H

SVAG Experiments

Recall that the SVAG algorithm (Definition 6.2) is a computationally efficient simulation of the
SDEs corresponding to RMSprop and Adam. The SVAG algorithm requires a hyperparameter ℓ, and
the resulting parameters after kℓ2 steps should match the parameters on the corresponding discrete
optimization trajectory after k steps. In particular, Theorem 6.3 shows that the SVAG algorithm
is an order-1 weak approximation (Definition 2.4) of the SDE, and the approximation error scales
as 1/ℓ. One may be initially concerned that realistic deep learning settings require 1/ℓ to be very
small, which would make ℓ large and hence computationally intractable. Li et al. (2021) showed that
the SVAG trajectories appear to converge to the SDE trajectory for computationally tractable small
values of ℓ. We similarly find that our proposed SVAG-like algorithms in Definition 6.2 appear to
converge for small ℓ in various settings.
CIFAR-10. Figures 7 and 8 show that SVAG converges and closely tracks RMSprop at smaller
ϵ (=10−30 ), and at larger ϵ (=10−8 ) respectively. Figures 3 and 9 show that SVAG converges and
closely tracks Adam at smaller ϵ (=10−30 ), and at larger ϵ (=10−8 ) respectively. All experiments
follow the setting in Appendix J.2 for batch size 256.
32


## Page 33

Train Accuracy
100
100
90
90
80
80
70
70
B=256
60
60
B=512
B=1024 50
50
B=2048
40
B=4096 40
30
B=8192 30
200.00
0.05
0.10 20 0

Test Accuracy

Train Accuracy
100
100
90
90
80
80
70
70
B=256
60
60
B=512
B=1024
50
50
B=2048
40
B=4096 40
30
B=8192 30
200.00
0.05
0.10 20 0

Test Accuracy

||zk(1)||2 Mean (Log Scale)

||gk(1)|| Mean (Log Scale)

211
210

25

29
28
200

400

24

0.00

Continuous time = k * 2

0.05

0.00

0.05

(a) ResNet-50

200

||zk(1)||2 Mean (Log Scale)
214

27

212

26

210

25

28

24

26

23

24

22

400

0.00

Continuous time = k * 2

||gk(1)|| Mean (Log Scale)

21
0.00

0.05

0.05

(b) VGG-16
Figure 6: We repeat the Square Root Scaling experiments with Adam for ResNet-50 and VGG-16
on CIFAR-10 dataset, where we slightly modify the standard implementation of Adam by using vk
in the update rule instead of vk+1 . For batch size 256, (η, ϵ, β1 , β2 ) = (10−3 , 10−8 , 0.999, 0.999)
and the hyperparameters are scaled according to the square root scaling rule for other batch sizes.
We observe a small drop (≈ 1%) in test accuracies. However, the behavior of the trajectories across
different batch sizes stays the same.

Train Accuracy
100
95
90
85
80
75
70
65
600.00

0.02

95
90
85
80
= 1 75
=2
= 4 70
= 8 65
0.04 0.06 600.00

||zk(1)||2 Mean (Log Scale)

Test Accuracy

||gk(1)|| Mean (Log Scale)

211
210

25

29
28
0.02

0.04

0.06

0.00

24
0.02

Continuous Time = k * 2/ 2

0.04

0.00

0.02

0.04

Figure 7: SVAG experiments on ResNet-50 trained on CIFAR-10 with RMSprop. We use batch size
256 and the hyperparameters η = 10−3 , β = 0.999, ϵ = 10−30 and a weight decay of 10−4 . Since
SVAG takes ℓ smaller steps to simulate the continuous dynamics in η time, we plot accuracy against
continuous time defined as k × η 2 /ℓ2 .

Wikipedia + Books (Academic BERT). Figure 10 shows that SVAG converges and closely
tracks Adam. We use the experimental setting for batch size 1024 in Appendix J.4 except the
hyperparameters β1 and β2 are fixed at 0.9 and 0.98 respectively.
WikiText-103 (GPT). Figure 23 shows that SVAG converges and closely tracks Adam. We use the
experimental setting for batch size 1024 in Appendix J.4 except the hyperparameters β1 and β2 are
fixed at 0.9 and 0.98 respectively. Additionally, for computational reasons, we pretrain on sequences
of length 64.

I

Square Root Scaling Experiments

We experimentally evaluate the scaling rules proposed in Definitions 5.1 and 5.2 by training models
with different batch sizes and modifying the optimization hyperparameters accordingly. The number
of gradient steps were modified to keep the total amount of continuous time same across all the
33


## Page 34

Train Accuracy
100
95
90
85
80
75
70
65
600.00

0.02

||zk(1)||2 Mean (Log Scale)

Test Accuracy

||gk(1)|| Mean (Log Scale)

211

95
90
85
80
= 1 75
=2
= 4 70
= 8 65
0.04 0.06 600.00

210

25

29
28
0.02

0.04

0.06

0.00

24
0.02

Continuous Time = k * 2/ 2

0.04

0.00

0.02

0.04

Figure 8: SVAG experiments on ResNet-50 trained on CIFAR-10 with RMSprop. We use batch size
256 and the hyperparameters η = 10−3 , β = 0.999, ϵ = 10−8 , and a weight decay of 10−4 . Since
SVAG takes ℓ smaller steps to simulate the continuous dynamics in η time, we plot accuracy against
continuous time defined as k × η 2 /ℓ2 .

Train Accuracy
100
95
90
85
80
75
70
65
600.00

0.02

0.04

Test Accuracy
95
90
85
80
75
70
65
0.06 600.00

212

||zk(1)||2 Mean (Log Scale)

26

||gk(1)|| Mean (Log Scale)

211

0.02

0.04

25

210

=1
=2
=4
=8

29
28

0.06

0.00
0.02
Continuous Time = k * 2/ 2

24
0.04

0.00

0.02

0.04

Figure 9: SVAG experiments on ResNet-50 trained on CIFAR-10 with Adam. We use batch size 256
and the hyperparameters η = 10−3 , β1 = 0.9, β2 = 0.999, ϵ = 10−30 , and a weight decay of 10−4 .
Since SVAG takes ℓ smaller steps to simulate the continuous dynamics in η time, we plot accuracy
against continuous time defined as k × η 2 /ℓ2 .

12

Train Log Perplexity

=1
=2
=4

10
8
6
4
2
00.000

0.002

0.004

Gradient Norm
12 Validation Log Perplexity 2.00
1.75
10
1.50
8
1.25
6
1.00
0.75
4
0.50
2
0.25
00.000
0.002
0.004 0.000.000
0.002
0.004

Continuous Time = k * 2/ 2

Figure 10: SVAG experiments on RoBERTa pretrained on Bookcorpus + Wikipedia dataset with
Adam. We use batch size 1024 and the hyperparameters η = 10−3 , β1 = 0.9, β2 = 0.98, ϵ =
2 × 10−6 . Since SVAG takes ℓ smaller steps to simulate the continuous dynamics in η time, we plot
accuracy against continuous time defined as k × η 2 /ℓ2 .

34


## Page 35

12

Train Log Perplexity

=1
=2
=4

10
8
6
4
2
00.000

0.001

0.002

Gradient Norm
12 Validation Log Perplexity 2.00
1.75
10
1.50
8
1.25
6
1.00
0.75
4
0.50
2
0.25
00.000
0.001
0.002 0.000.000
0.001
0.002

Continuous Time = k * 2/ 2

Figure 11: SVAG experiments on GPT pretrained on WikiText-103 with Adam. We use batch size
1024 and the hyperparameters η = 10−3 , β1 = 0.9, β2 = 0.98, ϵ = 2 × 10−8 . Since SVAG takes ℓ
smaller steps to simulate the continuous dynamics in η time, we plot accuracy against continuous
time defined as k × η 2 /ℓ2 . For computational constraints, we train on sequences of length 64.

batches. The warmup schedule and the learning rate decay schedule were kept the same. See
Appendix J for details about the baseline runs for each dataset.
We perform two types of ablation studies. In Appendix I.1, we compare our proposed scaling rule to
variants that only scale some subset of the optimization hyperparameters. We find that our proposed
scaling rule is the best at preserving the validation accuracy and the other test functions across
different batch sizes. In Appendix I.2, we compare the proposed square root scaling rule to a linear
one and find that they perform comparably on CIFAR-10, though the square root scaling significantly
outperforms linear scaling on the ImageNet dataset. We hypothesize that for simpler datasets, like
CIFAR-10, the differences between the two scaling rules is not reflected clearly in the validation
accuracies because the task is too easy to learn.
CIFAR-10. Figures 12 and 14 show the performance of VGG-16 when trained with different
batch sizes with RMSprop at smaller ϵ (=10−30 ), and at larger ϵ (=10−8 ) respectively. Figures 13
and 15 show the performance of VGG-16 when trained with different batch sizes with Adam at
smaller ϵ (=10−30 ), and at larger ϵ (=10−8 ) respectively. For the corresponding experiments on
ResNet-50, please refer to Figures 16 to 19. For the details on the experimental setting, please refer
to Appendix J.2.
We observe that in all settings the test accuracies vary by at most 3% across batch sizes when using
the proposed square root scaling rule. Moreover, the test functions stay close across multiple batch
sizes, signifying that the trajectories stay close across the batch sizes using the Square root scaling
rule.
ImageNet. Figures 20 and 21 show the performance of ResNet-50 when trained with different
batch sizes with RMSprop at smaller ϵ (=10−30 ), and at larger ϵ (=10−8 ) respectively. For the details
on the experimental setting, please refer to Appendix J.3.
We observe that on the validation set, the loss and accuracy behavior for the model is very similar,
when trained with different batch sizes. Moreover, the difference between the validation accuracies is
atmost 3% between the batch sizes 256, 1024, 4096, and 16384 for smaller ϵ. The difference between
the validation accuracies is atmost 1.5% between the batch sizes 256, 1024, 4096, 16384, and 32768
for larger ϵ.
Wikipedia + Books (Academic BERT). Figure 22 shows the performance of a RoBERTa model
when pretrained with different batch sizes with ADAM. The scaling rule is applied to modify the
peak values of the optimization hyperparameters. We also evaluate the pretrained models on several
downstream tasks, and show the results in Table 1. For the details on the setting for both pretraining
and downstream experiments, please refer to Appendix J.4.
35


## Page 36

Pretrain batch size B
1024
2048
4096
8192

CoLA
0.585
0.563
0.581
0.626

SST-2
0.92
0.928
0.921
0.929

MRPC
0.73
0.803
0.778
0.778

STS-B
0.866
0.869
0.869
0.884

QQP
0.873
0.875
0.875
0.877

MNLI
0.836
0.826
0.839
-

QNLI
0.906
0.897
0.892
0.9

RTE
0.682
0.653
0.675
0.675

Table 1: Performance of the pretrained RoBERTa models when finetuned on different downstream
tasks in GLUE Wang et al. (2019). F1 scores are reported for QQP and MRPC, Spearman correlations
are reported for STS-B, Matthews correlations for CoLA, and accuracy scores are reported for the
other tasks. Here, B denotes the batch size used for pretraining. We run an extensive grid search
(Table 4) to find the best performance of each pretrained model on each of the downstream tasks.

We observe that the log perplexity on the training and validation datasets matches across different
batch sizes throughout pretraining. Moreover, we also observe that the gradient norms across different
batch sizes remain close throughout the pretraining. Furthermore, the models pretrained across
different batch sizes can achieve very similar performance in the downstream tasks.
WikiText-103 (Academic GPT). Figure 23 shows the log perplexity behavior of GPT on training
and validation datasets, across different batch sizes of pretraining . We observe that the log perplexity
matches across different batch sizes. Moreover, we also observe that the gradient norms across
different batch sizes remain close throughout the training.
Train Accuracy
100
100
90
90
80
80
70
70
60
60
B=256
B=512
50
50
B=1024 40
40
B=2048
30
B=8192 30
200.00
0.05
0.10 20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy
213
210

25

27

23

24

400

||gk(1)|| Mean (Log Scale)

21

21
200

27

0.00

Continuous time = k * 2

0.05

0.00

0.05

Figure 12: VGG-16 trained on CIFAR-10 using RMSprop are close for different batch sizes when
the optimization hyperparameters are varied according to the proposed scaling rule for RMSprop
(Definition 5.2). We use a baseline setting of η = 10−3 , ϵ = 10−8 , and β = 0.999 for batch size 256.
We use a weight decay factor of 10−4 . We observe a gap of at most 3% among the different batch
sizes under consideration.

Train Accuracy
100
100
90
90
80
80
B=256
70
70
B=512
60
60
B=1024
B=2048
50
50
B=4096
40
40
B=8192
30
B=16384 30
200.00
0.05
0.10 20 0

Test Accuracy

200

215
213
211
29
27
25
23

400

||zk(1)||2 Mean (Log Scale)

0.00

Continuous time = k * 2

0.05

||gk(1)|| Mean (Log Scale)

27
26
25
24
23
22
21
0.00

0.05

Figure 13: VGG-16 trained on CIFAR-10 using Adam are close for different batch sizes when the
optimization hyperparameters are varied according to the proposed scaling rule for Adam (Definition 5.2). We use a baseline setting of η = 10−3 , ϵ = 10−8 , and (β1 , β2 ) = (0.999, 0.999) for batch
size 256. We use a weight decay factor of 10−4 . We observe a gap of at most 3% among the different
batch sizes under consideration.

36


## Page 37

Train Accuracy
100
100
90
90
80
80
70
70
B=256
60
60
B=512
B=1024 50
50
B=2048 40
40
B=4096
30
B=8192 30
200.00
0.05
0.10 20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy
213
27

23

24

21

21
400

||gk(1)|| Mean (Log Scale)

25

210

200

27

0.00

Continuous time = k * 2

0.05

0.00

0.05

Figure 14: VGG-16 trained on CIFAR-10 using RMSprop are close for different batch sizes when
the optimization hyperparameters are varied according to the proposed scaling rule for RMSprop
(Definition 5.2). We use a baseline setting of η = 10−3 , ϵ = 10−30 , and β = 0.999 for batch size
256. We use a weight decay factor of 10−4 . We observe a gap of at most 3% among the different
batch sizes under consideration.

Train Accuracy
100
100
90
90
80
80
B=256
70
70
B=512
60
60
B=1024
B=2048
50
50
B=4096
40
40
B=8192
30
B=16384 30
200.00
0.05
0.10 20 0

Test Accuracy

200

215
213
211
29
27
25
23

400

||zk(1)||2 Mean (Log Scale)

0.00

Continuous time = k * 2

0.05

||gk(1)|| Mean (Log Scale)

27
26
25
24
23
22
21
0.00

0.05

Figure 15: VGG-16 trained on CIFAR-10 using Adam are close for different batch sizes when the
optimization hyperparameters are varied according to the proposed scaling rule for Adam (Definition 5.2). We use a baseline setting of η = 10−3 , ϵ = 10−30 , and (β1 , β2 ) = (0.999, 0.999) for batch
size 256. We use a weight decay factor of 10−4 . We observe a gap of at most 3% among the different
batch sizes under consideration.

Train Accuracy
100
100
90
90
80
80
70
70
B=256
60
60
B=512
B=1024 50
50
B=2048 40
40
B=4096
30
B=8192 30
200.00
0.05
0.10 20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy

||gk(1)|| Mean (Log Scale)

211
210

25

29
28

24

27
200

400

0.00

Continuous time = k * 2

0.05

0.00

0.05

Figure 16: ResNet-50 trained on CIFAR-10 using RMSprop are close for different batch sizes when
the optimization hyperparameters are varied according to the proposed scaling rule for RMSprop
(Definition 5.2). We use a baseline setting of η = 10−3 , ϵ = 10−8 , and β = 0.999 for batch size 256.
ϵ = 10−30 ≈ 0 for all experiments. We use a weight decay factor of 10−4 . We observe a gap of at
most 3% among the different batch sizes under consideration.

37


## Page 38

Train Accuracy
100
100
90
90
80
80
B=256
70
70
B=512
60
60
B=1024
B=2048
50
50
B=4096
40
40
B=8192
30
B=16384 30
200.00
0.05
0.10 20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy

||gk(1)|| Mean (Log Scale)

211
210

25

29
28
200

400

24

0.00

Continuous time = k * 2

0.05

0.00

0.05

Figure 17: ResNet-50 trained on CIFAR-10 using Adam are close for different batch sizes when the
optimization hyperparameters are varied according to the proposed scaling rule for Adam (Definition 5.2). We use a baseline setting of η = 10−3 and (β1 , β2 ) = (0.999, 0.999) for batch size 256.
ϵ = 10−30 ≈ 0 for all experiments. We use a weight decay factor of 10−4 . We observe a gap of at
most 3% among the different batch sizes under consideration.

Train Accuracy
100
100
90
90
80
80
70
70
B=256
60
60
B=512
B=1024 50
50
B=2048
40
B=4096 40
30
B=8192 30
200.00
0.05
0.10 20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy

||gk(1)|| Mean (Log Scale)

211
210

25

29
28

24

27
200

400

0.00

Continuous time = k * 2

0.05

0.00

0.05

Figure 18: ResNet-50 trained on CIFAR-10 using RMSprop are close for different batch sizes when
the optimization hyperparameters are varied according to the proposed scaling rule for RMSprop
(Definition 5.2). We use a baseline setting of η = 10−3 , ϵ = 10−8 , and β = 0.999 for batch size 256.
We use a weight decay factor of 10−4 . We observe a gap of at most 3% among the different batch
sizes under consideration.

Train Accuracy
100
100
90
90
80
80
B=256
70
70
B=512
60
60
B=1024
B=2048
50
50
B=4096
40
40
B=8192
30
B=16384 30
200.00
0.05
0.10 20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy

||gk(1)|| Mean (Log Scale)

211
210

25

29
28
200

400

0.00

Continuous time = k * 2

24
0.05

0.00

0.05

Figure 19: ResNet-50 trained on CIFAR-10 using Adam are close for different batch sizes when the
optimization hyperparameters are varied according to the proposed scaling rule for Adam (Definition 5.2). We use a baseline setting of η = 10−3 , ϵ = 10−8 , and (β1 , β2 ) = (0.999, 0.999) for batch
size 256. We use a weight decay factor of 10−4 . We observe a gap of at most 3% among the different
batch sizes under consideration.

38


## Page 39

80 Top-1 Accuracy(Train)
70
60
50
40
B=2048
30 B=32768
B=1024
20 B=16384
B=8192
B=512
10 B=4096
B=256
0 0.00 0.01 0.02 0.03 0.04

Gradient norm
80 Top-1 Accuracy(Validation)
25
70
24
60
23
50
22
40
21
20
30
1
2
20
22
10
23
00.00 0.01 0.02 0.03 0.04
0.00 0.01 0.02 0.03 0.04
2
Continuous Time=k *

Figure 20: ResNet-50 trained on ImageNet using Adam are close for different batch sizes when the
optimization hyperparameters are varied according to the proposed scaling rule for Adam (Definition 5.2). We use a baseline setting of η = 3 × 10−4 , ϵ = 10−8 , and (β1 , β2 ) = (0.999, 0.999) for
batch size 256. We use a weight decay factor of 10−4 . We achieve around 74% validation accuracy
with batch size 256 and the accuracy drops by at most 1.5% at batch size 32768.

Top-1 Accuracy(Train)
80
70
60
50
40
B=16384
30
B=4096
20
B=1024
10
B=256
0 0.00 0.01 0.02 0.03 0.04

80 Top-1 Accuracy(Validation)
70
60
50
40
30
20
10
0 0.00 0.01 0.02 0.03 0.04

Continuous Time=k * 2

Gradient norm
29
28
27
26
25
24
23
22
21

0.00

0.01

0.02

0.03

0.04

Figure 21: ResNet-50 trained on ImageNet using Adam are close for different batch sizes when the
optimization hyperparameters are varied according to the proposed scaling rule for Adam (Definition 5.2). We use a baseline setting of η = 3 × 10−4 and (β1 , β2 ) = (0.999, 0.999) for batch size
256. ϵ = 10−30 ≈ 0 for all experiments. We use a weight decay factor of 10−4 . We achieve around
74% validation accuracy with batch size 1024 and the accuracy drops by at most 3% at batch size
16384.

I.1

Ablation study on the proposed scaling rule

CIFAR-10. We conduct an ablation study on whether all the parameters η, ϵ, β1 , β2 need to be
scaled in our proposed scaling rule. To do so, we compare the performance of a ResNet-50 model
trained with batch size 256 and hypeparameters (ϵ = 10−8 , β1 = 0.999, β2 = 0.999) with the
performance at a larger batch size, across 5 runs representing 5 different scaling rules: (a) Scale
η, keeping others fixed, (b) Scale η, ϵ, keeping others fixed, (c) Scale η, ϵ, β1 , keeping others fixed,
(d) Scale η, ϵ, β2 , keeping others fixed, and (e) Scale η, ϵ, β1 , β2 . Please check the behaviors of the
different scaling rules at batch size 2048, 4096, 8192 and 16384 in Appendix I.2. We found that (e)
consistently beats others in terms of the test functions and the validation accuracies at all batch sizes.
The closest scaling rule (c) involved scaling only η, ϵ, and β1 while keeping β2 fixed.
I.2

Ablation against linear scaling rules

CIFAR-10. We compared the proposed scaling rule against possible linear scaling rules, that scale
the hyperparameters linearly with the increase in training batch size. We focused on ResNet-50
training with Adam. The linear scaling rules that we tried were: (a) Scale η linearly, keeping β1 , β2 , ϵ
39


## Page 40

12

Train Log Perplexity
B 256
B 1024
B 2048
B 4096
B 8192

12 Validation Log Perplexity

Gradient Norm

10

25

8

23

6

21

4

4

21

2

2

23

10
8
6

00.00

0.02

0.04

00.00

0.02

0.04

Continuous time = k * 2

0.00

0.02

0.04

Figure 22: The train and validation log perplexities of RoBERTa-large trained on the Wiki+Books
corpus using Adam (for 48 hours) are close for moderate batch sizes using the Square Root Scaling
Rule on Adam. η = 10−3 and (β1 , β2 ) = (0.99375, 0.996) for batch size 1024, ϵ = 2 × 10−6 for
batch size 1024 and scaled likewise for other batch sizes. We achieve a validation log perplexity of
2.1 ± 0.1 for batch size 1024, 2048, 4096 and 8192. Training with batch size 256 is computationally
inefficient, but follows the same behavior during its 48-hour trajectory.

12

Train Log Perplexity
B 512
B 1024
B 2048
B 4096
B 8192

10
8
6

12 Validation Log Perplexity
10
8
6

4

4

2

2

00.000

0.005

0.010

0
0.000

0.005

0.010

Continuous time = k * 2

Gradient Norm
23
22
21
20
21
22
23
24
0.000

0.005

0.010

Figure 23: The train and validation perplexities of 12 layer GPT trained on the Wikitext corpus
using Adam (for 48 hours) are close for moderate batch sizes Square Root Scaling Rule on Adam on
RoBERTa-large. η = 10−3 and (β1 , β2 ) = (0.9875, 0.996) for batch size 1024, ϵ = 2 × 10−8 for
batch size 1024 and scaled likewise for other batch sizes. We achieve a validation log perplexity of
5 ± 0.1 for all the batch sizes under consideration. Moreover, we observe an alignment in the behavior
of the trajectories across the different batch sizes (except 8192 in the first half of the training).

fixed, (b) Scale η, 1 − β1 linearly, keeping β2 , ϵ fixed, (c) Scale η, 1 − β2 linearly, keeping β1 , ϵ fixed,
and (d) Scale η, 1 − β1 , 1 − β2 linearly, keeping ϵ fixed. Fig. 25 shows the behavior of these scaling
rules at batch size 8192 and 16384, at different values of ϵ.
The linear scaling rule in (d) seems to perform as well as the proposed Square Root scaling rule
(Definition 5.2) in terms of validation accuracies. However, with a closer look on the train accuracy
plots, we observe that the Square Root scaling rule tracks the smaller batch training trajectory better
than the linear scaling rule. The linear scaling rules seem to catch up, only after the learning rate is
decayed. Our hypothesis is that the CIFAR-10 dataset is simple enough for different scaling rules to
work well.
ImageNet. We conduct ablation experiments on ResNet-50, trained with Adam on ImageNet, where
we follow a linear scaling rule to scale the hyperparameters across batch sizes. Due to computational
issues, we didn’t conduct extensive experiments, as was done for CIFAR-10. The linear scaling rule
for Adam is as follows: the hyperparameters η, 1 − β1 and 1 − β2 are scaled by κ, when the batch size
is scaled by κ, and ϵ isn’t scaled. As was noted earlier, the definition of continuous time will change
to η × #{gradient steps}. We keep the number of training epochs equal to 90 as before, follow the
40


## Page 41

same learning rate schedule, and show the performance of the models in Figure 26. We observe that
scaling the hyperparameters to larger batch training with the proposed LSR results in a big drop in
validation accuracies.

J

Experiment Configuration Details

J.1

A note on learning rate schedule and warm-up

We have used a learning rate schedule and a warm-up phase in all our experiments. We have to
admit that our current theorems do not directly apply to time-varying learning rates or batch sizes.
But our experiments demonstrate that our scaling rules continue to hold for learning rate schedulers
with a special warm-up, even if they go beyond the scope of our theoretical setting. Technically, the
extensions of our theorems to time-varying learning rates or batch sizes are interesting, and we believe
they can indeed be shown following the same proof strategy. The corresponding SDE approximations
should have hyperparameters changing with time.
J.2

CIFAR-10

There are 50000 images in the training set and 10000 images in the validation set of CIFAR-10
Krizhevsky et al..
Architecture. We used the architecture of ResNet-56 from He et al. (2015a) without modification.
We used the same architecture of VGG-16 with batch normalization from Simonyan and Zisserman
(2014). However, we kept the final layer of the architecture fixed throughout training, to make the
model 1-homogenous and avoid the optimization difficulties of 2-homogenous networks (Li and
Arora (2020)).
RMSprop. To fix a baseline to compare against, we first trained the models with batches of size
256, sampled with replacement, at peak learning rate η = 10−3 and β = 0.999. The model was
trained for 500 × ⌊(50000/256)⌋ = 97500 gradient steps (or 500 epochs). We followed an initial
warmup for the first 2% of the total gradient steps. The learning rate schedule during the warmup
phase is given by η × 10−3 × (103 )# epochs /10 . We also followed a learning rate decay schedule, the
learning rate was decayed by 0.1 when the model reaches 80% (400 epoch) and 90% (450 epoch) of
the total continuous time respectively. We did experiments at two values of ϵ, small (= 10−30 ) and
large (= 10−8 ).
We then made multiple runs of the same model with batches of varying sizes in {1024, 4096, 16384},
with the hyperparameters η, ϵ and β modified appropriately according to the scaling rule. The number
of gradient steps were modified to keep the total amount of continuous time same across all the
batches (which amounted to 500 epochs by the equivalence between continuous time and the number
of training epochs). The warmup schedule and the learning rate decay schedule were kept the same.
Adam. We first trained the models with batches of size 256, sampled with replacement, at peak
learning rate η = 10−3 and (β1 , β2 ) = (0.999, 0.999). The total continuous time of training (or
number of epochs), the amount of continuous time in the warmup phase, and the learning rate
schedule in the warmup phase were same as RMSprop. The only difference was the learning rate
schedule after the warmup phase, the learning rate was decayed by 0.1 when the model reaches 60%
of the total continuous time (or 300 epochs).
We then made multiple runs of the same model with batches of varying sizes in {1024, 4096, 16384},
with the hyperparameters η, ϵ and β1 , β2 modified appropriately according to the scaling rule. The
number of gradient steps were modified to keep the total amount of continuous time same across all
the batches. The warmup schedule and the learning rate decay schedule were kept the same.
J.3

ImageNet

There are 1281167 images in the training set and 50000 images in the validation set of ImageNet
(Deng et al., 2009).
41


## Page 42

||zk(1)||2 Mean (Log Scale) ||gk(1)|| Mean (Log Scale)
Train Accuracy
Test Accuracy
100
100
31
2
213
90
90
26
2
80
80
211
70
Baseline B=256 70
221
29
Scale
60
60
16
27
2
Scale ,
50
50
11
Scale
,
,
25
2
2
40
40
Scale , , 1
6
23
2
30
30
Scale , , 1, 2
200.00
0.05
0.10 200.00
0.05
0.10 0.00
0.05
0.00
0.05
Continuous time = k * 2

(a) B = 2048
100
90
80
70
60
50
40
30
200.00

Train Accuracy

100
90
80
Baseline B=256 70
Scale
60
Scale ,
50
Scale , , 2
40
Scale , , 1
Scale , , 1, 2 30
0.05
0.10 200.00

||zk(1)||2 Mean (Log Scale)

Test Accuracy
235
229
223
217
211
25
0.05

0.10

217

||gk(1)|| Mean (Log Scale)

214
211
28
25
22

0.00

Continuous time = k * 2

0.05

0.00

0.05

(b) B = 4096
100
90
80
70
60
50
40
30
200.00

Train Accuracy

100
90
80
Baseline B=256 70
Scale
60
Scale ,
50
Scale , , 2
40
Scale , , 1
Scale , , 1, 2 30
0.05
0.10 200.00

Test Accuracy

0.05

239
233
227
221
215
29
23
0.10

||zk(1)||2 Mean (Log Scale)

217

||gk(1)|| Mean (Log Scale)

214
211
28
25
22

0.00

Continuous time = k * 2

0.05

0.00

0.05

(c) B = 8192
100
90
80
70
60
50
40
30
200.00

Train Accuracy

100
90
80
Baseline B=256 70
Scale
60
Scale ,
50
Scale , , 2
40
Scale , , 1
Scale , , 1, 2 30
0.05
0.10 200.00

||zk(1)||2 Mean (Log Scale)

Test Accuracy
239
233
227
221
215
29
23
0.05

0.10

0.00

Continuous time = k * 2

217

||gk(1)|| Mean (Log Scale)

214
211
28
25
22
0.05

0.00

0.05

(d) B = 16384
Figure 24: Ablation study for the square root scaling rule on Resnet-50 trained with Adam on
CIFAR-10. We compare the performance of a model trained with batch size 256 and hypeparameters
(ϵ = 10−8 , β1 = 0.999, β2 = 0.999) with the performance at a larger batch size, across 5 runs
representing 5 variations of the square root scaling rule: (a) Scale η, keeping others fixed, (b) Scale
η, ϵ, keeping others fixed, (c) Scale η, ϵ, β1 , keeping others fixed, (d) Scale η, ϵ, β2 , keeping others
fixed, and (e) Scale η, ϵ, β1 , β2 . We use a weight decay of 10−4 in all the experiments. We observe
that scaling all the hyperparameters consistently gives better performance at higher batch size. Scaling
rule (c) is close second.

42


## Page 43

100
90
80
70
60
50
40
30
20 0

Train Accuracy

Baseline B=256
Scale
Scale , 2
Scale , 1
Scale , 1, 2
Sqrt scaling
200

400

100
90
80
70
60
50
40
30
20 0

Test Accuracy

200

400

238
233
228
223
218
213
28

Epochs

||zk(1)||2 Mean (Log Scale)

||gk(1)|| Mean (Log Scale)
216
213
210
27
24

0

200

400

0

200

400

(a) B = 8192, ϵ of order 10−30
100
90
80
70
60
50
40
30
20 0

Train Accuracy

Baseline B=256
Scale
Scale , 2
Scale , 1
Scale , 1, 2
Sqrt scaling
200

400

100
90
80
70
60
50
40
30
20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy
232
228
224
220
216
212
28
200

400

Epochs

||gk(1)|| Mean (Log Scale)
214
212
210
28
26
24

0

200

400

0

200

400

(b) B = 16384, ϵ of order 10−30
100
90
80
70
60
50
40
30
20 0

Train Accuracy

Baseline B=256
Scale
Scale , 2
Scale , 1
Scale , 1, 2
Sqrt scaling
200

400

100
90
80
70
60
50
40
30
20 0

Test Accuracy

233

||zk(1)||2 Mean (Log Scale)

228
223
218
213
28
200

400

Epochs

0

200

215
213
211
29
27
25
23

400

||gk(1)|| Mean (Log Scale)

0

200

400

(c) B = 8192, ϵ of order 10−8
100
90
80
70
60
50
40
30
20 0

Train Accuracy

Baseline B=256
Scale
Scale , 2
Scale , 1
Scale , 1, 2
Sqrt scaling
200

400

100
90
80
70
60
50
40
30
20 0

||zk(1)||2 Mean (Log Scale)

Test Accuracy

215

229

212

224

29

219

200

400

||gk(1)|| Mean (Log Scale)

234

214

26

29

23

Epochs

0

200

400

0

200

400

(d) B = 16384, ϵ of order 10−8
Figure 25: Ablation study against (possible) linear scaling rules on Resnet-50 trained with Adam on
CIFAR-10. We compare the performance of a model trained with batch size 256 and hypeparameters
(ϵ = 10−8 /10−30 , β1 = 0.999, β2 = 0.999) with the performance at a larger batch size, across 5
runs representing 5 possible linear scaling rules: (a) Scale η linearly, keeping β1 , β2 , ϵ fixed, (b) Scale
η, 1 − β1 linearly, keeping β2 , ϵ fixed, (c) Scale η, 1 − β2 linearly, keeping β1 , ϵ fixed, and (d) Scale
η, 1 − β1 , 1 − β2 linearly, keeping ϵ fixed. We use a weight decay of 10−4 in all the experiments.
We also compare the behavior of the linear scaling rules against the square root scaling rule. Since
the continuous time definition varies across the scaling rules, we plot against the number of epochs
trained. A closer look at the training accuracy plots shows that the square root scaling rule tracks the
smaller batch training trajectory better than the linear scaling rule.

43


## Page 44

80
70
60
50
40
30
20
10
0 0

Top-1 Accuracy(Train)

B=16384
B=4096
B=1024
B=256
25

50

75 100 125

80
70
60
50
40
30
20
10
0 0

Top-1 Accuracy(Test)

25

50

75 100 125

Continuous Time=k *

25
24
23
22
21
20
21
22
23

Gradient norm

0

25

50

75 100 125

Figure 26: Ablation study against a (possible) linear scaling rule on Resnet-50 trained with Adam on
Imagenet. We compare the performance of a model trained with batch size 256 and hypeparameters
(η = 3 × 10−4 , ϵ = 10−8 , and (β1 , β2 ) = (0.999, 0.999)) with the performance at a larger batch
size, when the hyperparameters are scaled as follows: scale η, 1 − β1 , 1 − β2 by κ, if the batch size
is scaled by κ, keeping ϵ fixed. We use a weight decay factor of 10−4 . We clearly observe a decrease
in performance at larger batch size, in comparison to a model trained with square root scaling rule
(see Figure 20).

Architecture. We trained a ResNet-50 (He et al., 2015a) model without modification.
Adam. To fix a baseline to compare against, we first trained the models with batches of size 256,
sampled with replacement, at learning rate η = 3 × 10−4 and (β1 , β2 ) = (0.999, 0.999). The
model was trained for a total of 90 × ⌊(1281167/256)⌋ = 450360 gradient steps (or 90 epochs).
1
We followed an initial warmup for the first 18
fraction of the total continuous time (or the first 5
epochs). The learning rate schedule during the warmup phase is increased linearly with epoch, i.e.
the learning rate is given by η × 10−3 × (103 )# epochs /5 . We then followed a learning rate decay
5
schedule, where the learning rate was decayed by 0.1 when the model reaches 18
fraction (or 50
8
epoch ) and 9 fraction (or 80 epoch) of the total continuous time respectively. We use two different
values for ϵ, small ϵ (= 10−30 ≈ 0) and a larger ϵ (= 10−8 ).
J.4

Books and Wikipedia (Academic BERT)

We use a combination of Bookcorpus (Zhu et al., 2015) plus English Wikipedia, which totals 16 GB
of uncompressed text. We split the data uniformly with a ratio 9 : 1, to create training and validation
datasets for pretraining.
Architecture. We pretrain a 24-layer RoBERTa Liu et al. (2019) model. We pretrain on sequences
of length 128.
Pre-training with Adam. We use the code from Wettig et al. (2022). We follow the optimization
recipe from Izsak et al. (2021) for efficient pre-training. To fix a baseline, we first trained our model
with batch size 1024, with the optimization parameters given in table 2. In the warmup phase, the
learning rate is increased linearly over the interval, i.e. the learning rate at step k in the warmup phase
k
is given by kwarmup
η, where kwarmup denotes the total number of warmup steps and η denotes the
peak learning rate. Moreover, after the warmup phase, the learning rate is decayed linearly to 0, i.e.
k−kwarmup
the learning rate at step k after the warmup phase is given by kmax −k
η, where η denotes the
warmup
peak learning rate and kmax denotes the maximum number of gradient steps intended for pretraining.
Fine-tuning with Adam: We also validate the performance of the pretrained models from the
previous section on the GLUE(Wang et al., 2019) datasets. For each downstream task, we run an
extensive grid search on the hyperparameters for finetuning each pretrained model. We focused on
the following hyperparameters for grid search: batch size, the peak learning rate and the total number
of training epochs. Please see Table 4 for the hyperparameter grid. The rest of the hyperparameters
44


## Page 45

Hyperparameter
Dropout
Attention Dropout
Warmup Steps
Peak Learning Rate
Batch Size
Weight Decay
Max Steps
Learning Rate Decay
Adam β1
Adam β2
Adam ϵ
Gradient Clipping
Position embeddings

Value
0.1
0.1
5520
10−3
1024
10−4
92000
Linear
0.99375
0.996
2 × 10−6
0.0
128

Table 2: Optimization hyperparameters of baseline RoBERTa model during pretraining.

Hyperparameter
Dropout
Attention Dropout
Warmup Steps
Weight Decay
Learning Rate Decay
Adam β1
Adam β2
Adam ϵ
Gradient Clipping

Value
0.1
0.1
6% of total
0.1
Linear
0.9
0.98
10−6
0.0

Table 3: Optimization hyperparameters of all pretrained RoBERTa models during finetuning.

are fixed for all the runs and are given in Table 3. We follow a similar learning rate schedule during
the warmup phase as was used for pretraining: the learning rate at step k in the warmup phase is
k
η, where kwarmup denotes the total number of warmup steps and η denotes the peak
given by kwarmup
learning rate. Moreover, after the warmup phase, the learning rate is decayed linearly to 0, i.e. the
k−kwarmup
learning rate at step k after the warmup phase is given by kmax −k
η, where η denotes the peak
warmup
learning rate and kmax denotes the maximum number of gradient steps intended for finetuning.

Dataset
CoLA
SST-2
MRPC
STS-B
RTE
QQP
MNLI
QNLI

Finetune batch size
{16, 32}
{16, 32}
{16, 32}
{16, 32}
{16, 32}
{32}
{32}
{32}

Peak Learning rate
{10−5 , 3 × 10−5 , 5 × 10−5 , 8 × 10−5 }
{10−5 , 3 × 10−5 , 5 × 10−5 , 8 × 10−5 }
{10−5 , 3 × 10−5 , 5 × 10−5 , 8 × 10−5 }
{10−5 , 3 × 10−5 , 5 × 10−5 , 8 × 10−5 }
{10−5 , 3 × 10−5 , 5 × 10−5 , 8 × 10−5 }
{5 × 10−5 , 8 × 10−5 }
{5 × 10−5 , 8 × 10−5 }
{5 × 10−5 , 8 × 10−5 }

Total training epochs
{3, 5, 10}
{3, 5, 10}
{3, 5, 10}
{3, 5, 10}
{3, 5, 10}
{3, 5}
{3, 5}
{3, 5}

Table 4: Hyperparameter grid for pretrained RoBERTaLARGE on the downstream tasks.

45


## Page 46

Hyperparameter
Dropout
Attention Dropout
Warmup Steps
Peak Learning Rate
Batch Size
Weight Decay
Max Steps
Learning Rate Decay
Adam β1
Adam β2
Adam ϵ
Gradient Clipping
Position embeddings

Value
0.1
0.1
1000
10−3
1024
10−4
12500
Linear
0.9875
0.996
2 × 10−8
0.0
128

Table 5: Optimization hyperparameters of baseline GPT model during pretraining.

J.5

WikiText-103 (GPT)

WikiText-103 (Merity et al., 2017) is a dataset with 103 million tokens extracted from Wikipedia. We
split the data uniformly with a ratio 9 : 1, to create training and validation datasets for pretraining.
We use Adam (Kingma and Ba, 2015) optimization algorithm.
Architecture. We pretrain a 12-layer GPT (Brown et al., 2020) model without modification. The
model has 12 layers with hidden dimension 768, feedforward network dimension 3072, 12 attention
heads in each attention layer, and attention head size 64. We pretrain on sequences of length 128
(unless stated otherwise).
Adam. We use the code from Wettig et al. (2022). To fix a baseline, we first trained our model
with batch size 1024, with the optimization parameters given in Table 5. In the warmup phase, the
learning rate is increased linearly over the interval, i.e. the learning rate at step k in the warmup phase
k
is given by kwarmup
η, where kwarmup denotes the total number of warmup steps and η denotes the
peak learning rate. Moreover, after the warmup phase, the learning rate is decayed linearly to 0, i.e.
k−kwarmup
the learning rate at step k after the warmup phase is given by kmax −k
η, where η denotes the
warmup
peak learning rate and kmax denotes the maximum number of gradient steps intended for pretraining.

46


