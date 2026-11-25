## Page 1

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Yao Zhao 1 Connor James Stephens 2 Csaba Szepesvári 2 Kwang-Sung Jun 1

Abstract

After T time steps, the learner is required to output an arm
JT that is estimated to have the largest mean reward. The
budget T may or may not be given to the algorithm as input.
Such a problem was formalized by Even-Dar et al. (2006)
and Bubeck et al. (2009), but similar problems were considered much earlier (Bechhofer, 1958; Chernoff, 1959).
Recently, pure exploration has found many applications
such as efficient crowdsourcing for cartoon caption contests (Tanczos et al., 2017), hyperparameter optimization (Li
et al., 2018; 2020), and improving the time complexity of
clustering algorithms (Baharav and Tse, 2019). Hereafter,
we take σ 2 = 1 for ease of exposition, but all our results can
easily be extended to σ 2 -sub-Gaussian noise.

Simple regret is a natural and parameter-free performance criterion for pure exploration in multiarmed bandits. However, possibly due to the challenge of characterizing it, in the theory community, it has been largely overlooked in favor of
studying the probability of selecting the best, or
an ϵ-good arm. In this paper, we make significant
progress on minimizing simple regret in both datarich (T ≥ n) and data-poor regime (T ≤ n) where
n is the number of arms, and T is the number of
samples. At its heart is our improved instancedependent analysis of the well-known Sequential Halving (SH) algorithm, where we bound the
probability of returning an arm whose mean reward is not within ϵ from the best (i.e., not ϵ-good)
for any choice of ϵ > 0, despite ϵ not being input
to SH. Our bound not only leads to√an optimal
worst-case simple regret bound of n/T up to
logarithmic factors but also essentially matches
the instance-dependent lower bound for returning an ϵ-good arm reported by Katz-Samuels and
Jamieson (2020). For the more challenging datapoor regime, we propose Bracketing SH (BSH)
that enjoys the same improvement even without
sampling each arm at least once. Our empirical
study shows that BSH outperforms existing methods on real-world tasks.

Among many performance criteria, simple regret proposed
by Bubeck et al. (2009) is a natural measure of the quality
of the estimated best arm JT :
SRegT ∶= E [µ1 − µJT ] .

(1)

Simple regret is an unverifiable performance measure, meaning that the algorithm need not verify its performance to a
prescribed level. This is in stark contrast to the fixed confidence setting (Even-Dar et al., 2006) such as (ϵ,δ)-PAC that
requires the algorithm to verify the quality of the estimated
best arm to the prescribed target accuracy ϵ and confidence
level δ. While certain applications do need such verification,
there are many applications where the sampling budget is
too limited to show meaningful guarantees, such as cartoon
caption contests (Jain et al., 2020) and biological experiments (Jun et al., 2016). In these cases, algorithms with
verifiable guarantees may not be meaningful. Furthermore,
existing algorithms with verifiable guarantees are vacuous
in the data-poor regime of T ≤ n as it requires each arm to
be pulled at least once (Katz-Samuels and Jamieson, 2020,
Section C.1). The fixed budget setting considers an unverifiable measure of P(µJT ≠ µ1 ) (Audibert et al., 2010),
but this quantity inevitably depends on the smallest gap
µ1 − µ2 (Carpentier and Locatelli, 2016), which is usually
not meaningful until the sample size is very large.

1. Introduction
We consider the pure exploration problem in multi-armed
bandits. In this problem, given n arms, the learner sequentially chooses an arm It ∈ [n] ∶= {1, . . . , n} at time t to
observe a reward rt = µIt + ηt where µi is the mean reward
of arm i and ηt is stochastic σ 2 -sub-Gaussian noise. Without loss of generality, we assume that 1 ≥ µ1 ≥ ⋯ ≥ µn ≥ 0.
1
Department of Computer Science, University of Arizona, Tucson, USA 2 Department of Computing Science, University of Alberta, Edmonton, Canada. Correspondence to: Kwang-Sung Jun
<kjun@cs.arizona.edu>.

Despite being attractive, the analysis of simple regret has
been elusive. Existing studies focus on either achieving minimax simple regret bound (Lattimore et al., 2016) or characterization of how simple regret is different from the cumulative regret (Bubeck et al., 2009). It is not known whether
simple algorithms like Sequential Halving (SH) (Karnin

Proceedings of the 40 th International Conference on Machine
Learning, Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright
2023 by the author(s).

1

## Page 2

Revisiting Simple Regret: Fast Rates for Returning a Good Arm
Table 1. Sample complexity comparison with ME (Even-Dar et al., 2006), P3 (Chaudhuri and Kalyanakrishnan, 2019), and BUCB (KatzSamuels and Jamieson, 2020). The lower bound for Polynomial(α) instances (Theorem 3, Lemma 3.1) is a contribution of this paper which
improves on Katz-Samuels and Jamieson (2020, Theorem 1) to include the correct dependence on δ. The lower bound for EqualGap(m)
was first shown by Chaudhuri and Kalyanakrishnan (2017), which works for the fixed budget setting as well. We omit constant and
logarithmic factors in n and m. Methods with asterisk consider the fixed confidence setting, and those with dagger have an in expectation
bound rather than the high probability one. Anytime algorithms do not require the knowledge of the budget T as input, and parameter-free
ones do not require any of {α, m, ϵ, δ} as input.

∗

ME

P3∗
BUCB

†

DSH (ours)
BSH (ours)
Lower bound

Polynomial(α) with α > 12

EqualGap(m) with m ≤ n/4

Data-poor

Anytime

Parameter-free

n
log ( 1δ )
ϵ2
1
log( 1δ ) ∨ ϵ12 log2 ( 1δ )
ϵ2+(1/α)
n
log ( 1δ )
ϵ2−(1/α)
n
1
log ( 1δ ) ∨ ϵ1/α
ϵ2
1
log ( 1δ )
ϵ2
1
log ( 1δ )
ϵ2

n
log ( 1δ )
ϵ2
1
( n log ( 1δ ) + log2 ( 1δ ))
ϵ2 m
n
log ( 1δ )
mϵ2
n
log ( 1δ ) ∨ ϵn2
mϵ2
n
log ( 1δ )
mϵ2
n
log ( 1δ )
mϵ2

✗

✗

✗

✓

✗

✗

✓

✓

✗

✗

✓

✓

✓

✓

✓

et al., 2013) enjoy the optimal simple regret or not.

is never worse than the original bound of SH presented
in Karnin et al. (2013): P(µJT ≠ µ1 ) ≤ exp(−Θ̃( HT2 )). Furthermore, our bound essentially matches the lower bound
presented in Katz-Samuels and Jamieson (2020). We
also derive a minimax-style bound on the ϵ-error probability, which√
can be easily integrated out over ϵ to obtain
SRegT = Õ( n/T ), and further show that a variant of SH
results in removing logarithmic
factors from the bound and
√
achieve SRegT = O( n/T ). Finally, we derive an anytime
version of SH that we call Doubling SH (DSH), which enjoys the same guarantees without requiring T as input. We
present details of these claims in Section 3.

In this paper, we revisit simple regret and make two main
contributions. First, we provide a novel and tight analysis of Sequential Halving (SH), one of the state-of-the-art
algorithms for pure exploration. We analyze a strong performance criterion that we call ϵ-error probability:
P(µ1 − µJT > ϵ)

(2)

where ϵ is not given to the algorithm as input. This is precisely characterizing the distribution of µJT , which was
mentioned as an interesting direction of research in Lattimore and Szepesvári (2020, Section 33.4 Note 9). If an
algorithm A does not take ϵ as input yet has a guarantee for
any ϵ, we say A enjoys a uniform ϵ-error probability bound.
The ϵ-error probability is stronger than simple regret since

As the second contribution, we propose an improved pure
exploration algorithm for the data-poor regime. This is motivated by the fact that the guarantee (4) becomes vacuous
in the data-poor regime (T ≤ maxi≥g(ϵ)+1 i∆−2
i to be precise) as it does not even allow us to pull every arm once.
Inspired by Katz-Samuels and Jamieson (2020), we propose Bracketing SH (BSH) that progressively subsamples a
larger set of arms (i.e., brackets) for which we invoke SH
to find a good arm. Our analysis shows that BSH enjoys a
bound that is essentially the same as (4) but works even for
T ≤ maxi≥g(ϵ)+1 i∆−2
i . Compared to BUCB (Katz-Samuels
and Jamieson, 2020), the state-of-the-art algorithm for the
data-poor regime, BSH achieves three improvements: (i)
BSH is parameter-free whereas BUCB requires the target
error rate δ as input, (ii) BSH has a high probability sample
complexity bound as opposed to the expected sample complexity bound of BUCB, (iii) BSH shows a much stronger
bound than BUCB. We provide details on BSH in Section 4.

∞

SRegT = E[µ1 − µJT ] = ∫

0

P(µ1 − µJT > ϵ) dϵ . (3)

Thus, a uniform ϵ-error probability bound implies a simple
regret bound.
We call an arm i to be ϵ-good if µi ≥ µ1 −ϵ. Let ∆i ∶= µ1 −µi
be the gap of arm i and g(ϵ) ∶= ∣{i ∈ [n] ∣ µi ≥ µ1 − ϵ}∣ be
the number of ϵ-good arms. We prove that SH (Karnin et al.,
2013) has the following bound for any ϵ > 0 (Theorem 1):
⎛
⎞
T
P(µJT < µ1 − ϵ) ≤ exp −Θ̃(
) ,
H2 (ϵ) ⎠
⎝

(4)

where Θ̃(⋅) hides logarithmic factors and
H2 (ϵ) ∶=

We compare our bounds with existing bounds in various
setups in Table 1. For ease of comparison, we consider
special instances called Polynomial(α) and EqualGap(m).
α
Polynomial(α) is an instance with gap structure ∆i = ( ni )
for 2 ≤ i ≤ n, where α > 0.5 (Jamieson et al., 2013), and

i
1
max
.
g(ϵ/2) i≥g(ϵ)+1 ∆2i

is the sample complexity function. By setting ϵ = 0, the complexity becomes H2 (0) = maxi≥2 i∆−2
i =∶ H2 , so our bound
2

## Page 3

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

2010), the learner is given a budget T ≥ 1 as input and is
required to return an estimated best arm JT ∈ [n] at the
end of T -th time step. In the anytime setting, there is no
prescribed T , and the learner is required to output Jt ∈ [n]
for every time step t ≥ 1. In these settings, unlike the fixed
confidence setting, the learner is not required to verify the
quality of the returned arm JT . We collectively call these
settings the unverifiable setting and refer to the fixed confidence setting as the verifiable setting. Note that for any
verifiable setting, one can consider an unverifiable version.
For example, the verifiable (m, ϵ)-good arm identification
problem (Chaudhuri and Kalyanakrishnan, 2017) requires
that the learner, given (m, ϵ, δ), identify an arm i that is
(m, ϵ)-good (i.e., µi ≥ µm − ϵ) w.p. at least 1 − δ. One
can consider the unverifiable (m, ϵ)-good arm identification
problem where the goal is to minimize P(µJT ≥ µm − ϵ)
after T time steps.

Figure 1. The polynomial fitting for the best 150 arms of the New
Yorker Cartoon Caption contest 788.

Sample Complexity. When comparing bounds for simple
regret or ϵ-error probability, it is often easier to consider the
sample complexity, which measures the number of samples
required to achieve the target performance level. For example, the sample complexity for the ϵ-error probability is the
smallest time step τ such that ∀t ≥ τ, P(µ1 − µt > ϵ) ≤ δ.

EqualGap(m) is an instance that has m − 1 arms with the
gap ϵ/4 and n−m arms with the gap 54 ϵ. Both α and m have
the role of adjusting the number of good arms. Our bound
for BSH matches the lower bounds for the first time to our
knowledge, thereby establishing the optimality for these
two instances. Note that many real-world instances follow
Polynomial(α); e.g., see Figure 1. One interesting aspect
of Polynomial(α) is that, although α affects the fraction of
good arms, the optimal rate 1/ϵ2 is independent of α. This
is because the hardness of returning an ϵ-good arm depends
not just on how many good arms we have but also on how
far they are from bad arms as we explain in Section 3.

Notations. Throughout the paper, “const” is a universal
and positive constant, which may have different values for
different expressions. Both Θ̃(⋅) and ˜
O(⋅) omits logarithmic
factors on m, n, but not 1/δ.

3. Improved Analyses of Sequential Halving
Sequential Halving (SH) (Karnin et al., 2013) is a simple
and parameter-free algorithm that has had an outsize impact
on pure exploration since its introduction. Many recent practical algorithms for pure exploration (Aziz et al., 2022; Li
et al., 2018; 2020; De Heide et al., 2021) make use of this
algorithm, either with slight modifications or as a subroutine. These works have demonstrated that the algorithm’s
practical and theoretical significance is greater than its original presentation. Even beyond the basic bandit setting, SH
also has been used for planning in reinforcement learning
(Danihelka et al., 2022), where SH is observed to be easier
to tune compared to some UCB style algorithms as it does
not have problem-dependent parameters.

Our result also resolves the open problem of having a factor
of ln2 (1/δ) rather than ln(1/δ) when obtaining accelerated
sample complexities under the presence of multiple good
arms. Such an issue appeared repeatedly in the literature;
e.g., Chaudhuri and Kalyanakrishnan (2019), Katz-Samuels
and Jamieson (2020, Section E), and Aziz et al. (2018, Section 5). They all rely on the subsampling device or its
variants. Our improved bound can be attributed to the fact
that an accelerated sample complexity can be obtained even
without the subsampling device – we are the first to achieve
such a result, to our knowledge.
We evaluate our algorithm on the cartoon caption contest
dataset in Section 5 and conclude our paper with future work
in Section 6. Due to space constraints, we have the related
work section in the appendix, but directly-related works are
discussed throughout the paper. All the proofs are deferred
to the appendix.

We present SH in Algorithm 3.1. SH consists of ⌈log2 (n)⌉
stages. In each stage ℓ, the algorithm performs a uniform
sampling over the surviving arms Sℓ followed by eliminating the bottom half of Sℓ w.r.t. empirical means and sets
Sℓ+1 as the resulting arm set.
SH is known to achieve a nearly optimal instance-dependent
bound on the (ϵ = 0)-error probability, which we detail in
Table 2 in the appendix. However, it is unknown if SH is
a good algorithm for returning an ϵ-good arm. For ϵ > 0,

2. Preliminaries
We focus on the fixed budget and anytime setting in pure exploration. In the fixed budget setting (Audibert and Bubeck,
3

## Page 4

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Corollary 1.1. For Polynomial(α) with α > 21 , we have

though in the fixed confidence setting, Median Elimination
(ME) achieves the sample complexity of ϵn2 log( 1δ ) (EvenDar et al., 2006). Does SH achieve a similar sample complexity for ϵ-error probability? Despite the similarity between SH and ME, the answer is not clear because the
sampling scheme of SH is different from ME. In light of
the ubiquity of this algorithm and recent empirical success,
a stronger characterization of the performance of SH is of
great interest to the community.

H2 (ϵ) =

1
i
4
max
< .
g( 2ϵ ) i≥g(ϵ)+1 ∆2i ϵ2

The proof is in Appendix B.6. While this result initially
seems surprising, the lack of dependence on n stems from
the ratio between the number of ‘well-separated’ ϵ-good
arms and the sample complexity of filtering out ϵ-bad arms
(i.e., non-ϵ-good arms) on these instances. As n increases,
due to the polynomial gap structure, the number of wellseparated ϵ-good arms (e.g., the collection of ϵ/2-good arms)
increases linearly with n, which balances the linear growth
in n of the sum over the ϵ-bad arm gaps ∑ni=g(ϵ)+1 ∆−2
i .
This second term is effectively the sample complexity of
discarding all ϵ-bad arms. Since we only require one of the
good arms to be returned, we get to divide ∑ni=g(ϵ)+1 ∆−2
i
by g(ϵ/2).

In this section, via a powerful analysis of the ϵ-error probability incurred by SH, we show that this algorithm achieves
an optimal bound for the unverifiable (m, ϵ)-good arm
identification problem that further implies a minimax optimal simple regret bound (Lattimore and Szepesvári, 2020,
Exercise 33.1). We also provide a near-optimal instancedependent uniform ϵ-error probability bound for the first
time.
Our main theorem shows the following instance-dependent
uniform ϵ-error probability bound for SH.

Lower bounds. We compare our upper bound in Theorem 1
with lower bounds. For brevity, we present an informal
version here and defer the full version to Appendix C.3
along with a slightly stronger version.

Theorem 1. For any ϵ ∈ (0, 1), the ϵ-error probability of
SH satisfies

Theorem 2. (Informal version of Katz-Samuels and
Jamieson (2020, Theorem 1)) For any algorithm, there exists
an instance where the expected unverifiable sample complexity for returning an ϵ-good arm with probability greater
than 15/16 satisfies

⎛
⎞
T
P(µJT < µ1 − ϵ) ≤ exp −Θ̃(
) ,
H2 (ϵ) ⎠
⎝

for T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ).

n
⎛
1
1
1 ⎞
⎟.
Ω ⎜H low (ϵ) ∶=
∑ ( 2)− 2
g(ϵ) i=g(ϵ)+1 ∆i
∆g(ϵ)+1 ⎠
⎝

i

The proof is in Appendix B.5. Note the requirement of
Corollary 2.1 (Comparison with previous lower bound).

T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ) is not a weakness of our result.

g( ϵ )

2
In the case that g(ϵ) and g( 2ϵ ) have the same order ( g(ϵ)
is irrelevant to ϵ), the sample complexity measure H2 (ϵ)
satisfies

i

One can easily see that if ∆i = ϵ, ∀i ≥ 2, the standard
result of SH (Karnin et al., 2013) becomes vacuous for
T = o( ϵn2 ). We stress that, unlike all existing work that
we are aware of, Theorem 1 bounds the ϵ-error probability
for an algorithm that does not require ϵ as input. That
is to say, ϵ is a parameter for analysis only. This upper
bound of ϵ-error probability implies a sample complexity of
Õ(H2 (ϵ) ln(1/δ)).

H low (ϵ) ≲ H2 (ϵ) ≲ H low (ϵ) +

2
∆2g(ϵ)+1

where ≲ hides constants and logarithms of n and 1/δ.
The sample complexity implied by Theorem 2 for returning
an ϵ-good arm with probability 1 − δ, with δ ∈ (0, 1/16), is
Ω(H low (ϵ)), which does not have the factor ln(1/δ). We
contribute a new lower bound that suggests a sample complexity of Ω(H̃(ϵ) log(1/δ)), with a complexity term H̃(ϵ)
that we describe next, and with which we show that Theorem 1 is essentially tight on polynomial gap instances with
α > 1/2.

Polynomial gap instances. The class of polynomial gap
instances was introduced by Jamieson et al. (2013), who
referred to them as ‘α-parameterized’ instances. The authors studied these instances in the best-arm identification
setting, showing that for α > 1/2, the sample complexity
2α
H = ∑ni=2 ∆−2
i scales like Ω(n ), with the increase in sample complexity corresponding to the growing number of
close-to-optimal arms as α increases. In one of the highlight
results of this work, we show that when ϵ > 0 is not pathologically small, SH returns ϵ-good arms on these instances
with an optimal sample complexity that is independent of
the number of arms n.

For an n-armed bandit instance ν, let
n/g(ϵ)

H̃(ν, ϵ) = ∑ ∆−2
i⋅g(ϵ) ,
i=2

4

## Page 5

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Algorithm 1 Sequential Halving (SH)
Input: budget: T , arms: [n]
Initialize: S1 = [n]

satisfies
P(µJT < µm − ϵ) ≤
⎛
⎛
⎞⎞
ϵ2 T
log2 n⋅ exp⎜−const⋅m⋅
− ln(4e) ⎟.
2
⎝ 4n log2 (2m) log2 n
⎠⎠
⎝

for ℓ = 1, . . . , ⌈log2 n⌉ do
Sample each arm i ∈ Sℓ for Tℓ times where

Thus, there exist an absolute constant c1 > 0 s.t. if T ≥

⎢
⎥
⎢
⎥
T
⎢
⎥
Tℓ = ⎢
⎥
⎢ ∣Sℓ ∣⌈log2 n⌉ ⎥
⎣
⎦

c1

⎛
ϵ2 T ⎞
P(µJT < µm − ϵ) ≤ exp −Θ̃(m
) .
n ⎠
⎝

Let Sℓ+1 be the set of ⌈Sℓ /2⌉ arms in Sℓ with the largest
empirical rewards.
Set JT as the only arm in S⌈log2 n⌉ .
Output: JT

The proof is in Appendix B.3.
Remark 1. Note that our bound implies a verifiable sample
complexity bound for the (m, ϵ)-good arm identification
setup as well because, given (m, ϵ, δ), one can work out the
sufficient samples size T ′ to control the RHS of Theorem 4
to be at most δ and then run SH with T ′ .

where for simplicity we assume that n is an integer multiple
of g(ϵ).
Theorem 3. Fix n > 2, ϵ > 0 and an n-armed unitvariance Gaussian instance ν. With a ∶= H̃(ν, ϵ) we define
Bn (a, ϵ), the collection of n-armed bandit instances with
unit-variance gaussian arms for which H̃(ν ′ , ϵ) ≤ a for
each ν ′ ∈ Bn (a, ϵ). Then for any algorithm π and sample
budget T ∈ N,
sup

ν ′ ∈Bn (a,ϵ)

(ln(4e)+ln ln n)n log22 (2m) log2 n
= Θ̃( ϵn2 ), we have
ϵ2

Implications for m = 1. The worst-case upper bound of
Theorem 4 for m = 1 corresponds to the sample complexity
˜ n2 log 1 ). This is the same as the classic lower bound
of O(
ϵ
δ
result of the (ϵ, δ)-PAC problem (Mannor and Tsitsiklis,
2004) up to logarithmic factors. While numerous algorithms
achieve a matching upper bound, including Even-Dar et al.
(2006) and Hassidim et al. (2020), our result in Theorem 1
indicates a tighter bound when there are many good arms.

Pν ′ ,π (µJT < µ1 − ϵ)

We now apply (3) to convert a uniform ϵ-error probability
bound into a simple regret bound.
√
Corollary 4.1. SH satisfies SRegT ≤ ˜
O( n ).

1 1
≥ max { ,
exp(−24T /a)} .
2 36

T

The proof is in Appendix B.2. Note that our bound is
minimax optimal up to logarithmic factors (Lattimore and
Szepesvári, 2020, Exercise 33.1). To our knowledge, the
only algorithm that we are aware of that achieves the minimax simple regret is MOSS (Audibert et al., 2009; Lattimore
et al., 2016). However, as MOSS is designed to minimize
cumulative regret, it cannot enjoy low instance-dependent
bounds for pure exploration tasks (Bubeck et al., 2011).
While uniform sampling can also achieve a near-optimal
simple regret, one can show that uniform sampling can be
arbitrarily worse than SH w.r.t. the (ϵ = 0)-error probability as its sample assignments over the arms are inherently
non-adaptive.
Remark 2. In Appendix B.12, we show that one can adjust
Tℓ in Algorithm 3.1 to achieve the minimax optimality up to
a constant factor (i.e., no logarithmic factors), but it is not
clear if this variant achieves a similar instance-dependent
sample complexity as Algorithm 3.1.

The complexity term H̃(ν, ϵ) plays the same role in our result as that of H(ν) = ∑ni=2 ∆−2
i in Carpentier and Locatelli
(2016, Theorem 1). The proof of Theorem 3 can be found
in Appendix B.8.
Lemma 3.1. Consider a Polynomial(α) instances with fixed
α > 1/2. Then for any ϵ ≤ 1/2, H̃ ≥ c(α)
where c(α) is a
ϵ2
problem-dependent constant which does not depend on ϵ.
The proof is in Appendix B.9. Combining Corollary 1.1
and Lemma 3.1, we see that Theorem 1 is optimal up to
logarithmic factors in the sample complexity for polynomial
gap instances with α > 1/2.
3.1. Unverifiable (m, ϵ)-Good Arm Identification
We show the error probability of SH for unverifiably identifying an (m, ϵ)-good arm (defined in Section 2) and discuss
its implications.

Implications for m > 1. We now show that SH enjoys
the optimal worst-case guarantee for the unverifiable (m, ϵ)good arm identification problem up to logarithmic factors.

Theorem 4. For any m ≤ n and any ϵ ∈ (0, 1), the error probability of SH for identifying an (m, ϵ)-good arm
5

## Page 6

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Algorithm 2 Doubling Sequential Halving (DSH)
Input arms: [n]
Initialize T1 = ⌈n log2 n⌉, ĵ = j for some arbitrarily chosen
j ∈ [n], µ̂∗ = −∞. Define the time blocks T1 = {1, . . . , T1 }
and Tk = {T1 (2k−1 − 1) + 1, . . . , T1 (2k − 1)}, ∀k ≥ 2, which
satisfies ∣Tk ∣ = T1 2k−1 . k = 1. An instance of SH denoted
by S1 with budget ∣T1 ∣.
for t = 1, 2, . . . do
Pull arm It according to the recommendation from Sk
Receive reward Rt and send it to Sk
if t = max Tk then
Set (ĵ, µ̂∗ ) as the output arm and its empirical mean
from the last stage of Sk
k ←k+1
Initialize a new instance of SH denoted by Sk with
budget ∣Tk ∣

the following ϵ-error probability
⎞
⎛
T
) ,
P(µJT < µ1 − ϵ) ≤ exp −Θ̃(
H2 (ϵ) ⎠
⎝
simultaneously for all T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 )
i

This is a direct consequence of the fact that at time t, the
latest finished SH was run with a budget of at least t/4. The
full proof is in Appendix C.2.

4. Simple Regret in the Data-Poor Regime
Despite the optimality of SH shown in the last section, it
requires at least Θ̃(maxi≥g(ϵ)+1 ∆i2 ) samples, which in the
i

worst case could be Θ̃(n/ϵ2 ). This requirement is unacceptable when the number of arms n is large (or even infinite) while the fraction of the ϵ-good arms is kept constant.
This setup is commonly referred to as the data-poor regime,
where one would like to return a good arm even before the
algorithm samples every arm once.

Output: Jt = ĵ and Mt = µ̂∗ .

Our result is especially attractive since the target number of
good arms m is not an input to SH.
Our upper bound of Theorem 4 corresponds to a sample
˜ n 2 log( 1 )), which matches the lower
complexity of O(
mϵ
δ
bound in Chaudhuri and Kalyanakrishnan (2017), thereby
closing the gap between the upper and lower bounds from
previous work; the algorithms therein have suboptimal upper bounds that scale with log2 ( 1δ ). Furthermore, these
algorithms all require (m, ϵ, δ) as input, which is natural
for obtaining verifiable sample complexities but becomes
a limitation in other settings. In contrast, Theorem 4 allows (m, ϵ) to be chosen freely to measure the algorithm’s
performance since the parameters for measuring the performance are no longer necessary for executing the algorithm.
Therefore, one can rewrite Theorem 4 as follows:

P(µJT < µ1 − ϵ) ≤

To cope with the data-poor regime, we take inspiration from
BUCB (Katz-Samuels and Jamieson, 2020) and propose
Bracketing SH (BSH) that enjoys a similar guarantee as
DSH but enjoys non-vacuous sample complexity in the datapoor regime. To understand the bracketing trick, suppose
that we uniformly sample a subset of size n/m (0 < m ≤
n) from the entire arm set [n]. With constant probability,
this subset includes at least one of the top m arms. By
applying any pure exploration algorithm to this subset, we
expect to find the best arm within the subset with the sample
complexity that scales with n/m rather than n. Such a
subset is called a bracket. Note that there is a natural tradeoff here with the bracket size. If the size is large, we are
likely to include many good arms, but the sample complexity
of identifying a good arm becomes large because we have
a lot of arms to pull at least once. On the other hand, if
the size is small, we are not likely to include any good arm,
although the sample complexity of identifying an arm that
is good relative to the bracket is small. As we show later,
one can precisely work out such a tradeoff mathematically
and find the best bracket size. The challenge is, however,
that the best bracket size typically requires knowledge of
the number of good arms.

⎛ ⎛ ϵ′ 2 T ⎞⎞
⎟.
exp⎜−Θ̃ m′
n ⎠⎠
{(m′ ,ϵ′ )∶∆m′ +ϵ′ ≤ϵ}
⎝ ⎝
min

3.2. Anytime version of SH
To support anytime simple regret minimization, we combine
SH with the doubling trick (Jun and Nowak, 2016), which
we call Doubling SH (DSH). Specifically, DSH repeatedly
runs SH. The k-th SH receives a doubled budget compared
with the (k − 1)-th SH. Before it finishes k-th SH, it always
returns the output of (k − 1)-th SH. The pseudocode can
be found in Algorithm 3.1. The following theorem shows
that DSH enjoys the same guarantee as SH up to a constant
factor.

To avoid requiring such knowledge, BSH adopts the bracketing technique of Katz-Samuels and Jamieson (2020) that
progressively creates a larger and larger bracket size as the
time step t gets larger while invoking a base algorithm for
each bracket by cycling through them (i.e., Round Robin).
Specifically, BSH uses DSH as the base algorithm. At each
time step, BSH takes in the estimated best arms and their
empirical means from all the brackets and outputs the one

Theorem 5. Let ϵ ∈ (0, 1). A single run of DSH satisfies
6

## Page 7

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Algorithm 3 Bracketing SH (BSH)
Input: arms: [n]
Initialize: t = 1, B = 0, b1 = 1.
Define
(I(D), J(D), M (D)) to be the arm to be pulled next, the
current estimated best arm, and its empirical mean from an
algorithm D, respectively.
for t = 1, 2, 3 . . . do
if t ≥ B2B then
B ←B+1
Sample a bracket AB = U ([n], n ∨ 2B )
Initialize a new instance of DSH, denoted by DB ,
with the bracket AB .
Pull arm It = I(Dbt )
Receive a reward Rt from the environment and send Rt
to Dbt
Output Jt = J(Db̂ ) where b̂ = arg maxb∈[B] M (Db )
⎧
⎪
⎪bt + 1 if bt < B
bt+1 = ⎨
⎪
1
otherwise
⎪
⎩

To show the effectiveness of Theorem 6, we now discuss
the implication of our result and compare it with BUCB
by Katz-Samuels and Jamieson (2020) using their performance measure called (ϵ, δ)-unverifiable sample complexity.
Definition 1 ((ϵ, δ)-unverifiable sample complexity). (KatzSamuels and Jamieson, 2020). For an algorithm π and an
instance ρ. Let τϵ,δ be a stopping time such that
P(∀t ≥ τϵ,δ ∶ µJt > µ1 − ϵ) ≥ 1 − δ.
Then τϵ,δ is called (ϵ, δ)-unverifiable sample complexity of
the algorithm with respect to ρ.
The (ϵ, δ)-unverifiable sample complexity indicates how
many samples are sufficient for an algorithm to begin to
output an ϵ-good arm with high probability. Compared with
the verifiable sample complexity, it does not require the algorithm to verify the output is ϵ-good. More discussion can
be found in Katz-Samuels and Jamieson (2020). While the
verbatim requirement of (ϵ, δ)-unverifiable sample complexity is stronger than the uniform ϵ-error probability bound,
we show that the latter implies the former under a mild
assumption in Appendix C.8.

with the largest empirical mean. We summarize BSH in
Algorithm 4 where the operator U ([n], k) samples k items
with replacement from [n] (uniformly at random). If k ≥ n,
U ([n], k) returns [n]. To define terminology, we say a new
bracket is opened when a new bracket is sampled. We set
the bracket-opening schedule such that the B-th bracket is
opened at time step (B − 1) ⋅ 2B−1 . The arm pulls between
the time step (B − 1) ⋅ 2B−1 and B ⋅ 2B are equally allocated
to the opened brackets (total B of them). We say an arm represents bracket AB at time step t if it is the output returned
by the DSH on bracket AB at time step t.

To make explicit comparisons, we turn to specific problem
instances and show the (ϵ, δ)-unverifiable sample complexity bounds for BUCB (Katz-Samuels and Jamieson, 2020,
Theorem 7) and BSH.
Corollary 6.1. Consider the EqualGap(m) instance.
BUCB achieves an expected (ϵ, δ)-unverifiable sample complexity as
˜
E[τϵ,δ ] ≤ O

Let us first show the properties of the bracketing technique.
The bracketing design ensures the diversity of the size of
opened brackets. Specifically, the smallest bracket has 2
arms, and the largest bracket has Θ̃(t) arms. Moreover, it
also ensures that all the opened brackets have received an
(order-wise) equal amount of sampling budget at any time.

⎛ n
1 ⎞
log( ) .
δ ⎠
⎝ ϵ2 m

BSH achieves an (ϵ, δ)-unverifiable sample complexity as,
with probability 1 − δ,
˜ ⎛ n log( 1 )⎞.
τϵ,δ ≤ O
δ ⎠
⎝ ϵ2 m

We now present the ϵ-error probability bound of BSH.
Theorem 6. Let ϵ ∈ (0, 1). A single run of BSH satisfies the
following ϵ-error probability

The proof is in Appendix C.5. In this instance, BUCB
and BSH have nearly the same performance guarantees
except that BUCB’s (ϵ, δ)-unverifiable sample complexity
is stated as an expected sample complexity. Interestingly,
Katz-Samuels and Jamieson (2020, Section E) remarks that
their attempt to derive a high-probability bound resulted
in the factor of ln2 (1/δ), which we believe is due to the
suboptimal bound (i.e., no accelerated rates) of their choice
of the base algorithm.

⎛ ⎛
⎞⎞
t
⎜
⎟⎟
⎟,
P(µ1 − µJt > ϵ) ≤ exp⎜−Θ̃⎜
⎜ ⎜ max {H ( ϵ ), 1 } ⎟⎟
2
2
⎠⎠
2
ϵ
⎝ ⎝
simultaneously for all t ≥ Θ̃(H2 ( 2ϵ )).

Scaling with n/m instead of n reveals the merit of the (ϵ, δ)unverifiable sample complexity that was not achieved by the
algorithms designed to achieve verifiable sample complexity
such as Median Elimination (Even-Dar et al., 2006).

The proof is in Appendix C.4. The key difference between
Theorem 5 and Theorem 6 is the latter does not require the
number of samples to be at least T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ).
i

7

## Page 8

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Corollary 6.2. Consider the Polynomial(α) instance; i.e.,
α
∆i = ( ni ) with α > 0.5. For any ϵ ∈ (0, 1), let τ̂ϵ,δ be
the upper bound of the expected (ϵ, δ)-unverifiable sample
complexity reported in Katz-Samuels and Jamieson (2020,
Theorem 7) for BUCB. Then, BUCB satisfies

# of arms: 6509

0.6

BSH
BUCB, = 0.01
BUCB, = 0.05
BUCB, = 0.1
BUCB, = 0.2

simple regret

0.5

⎛ 2α−1
1 ⎞
E[τϵ,δ ] ≤ τ̂ϵ,δ = Θ̃ ϵ− α n log( ) .
δ ⎠
⎝

0.4
0.3
0.2
0.1

0

2000 4000 6000 8000 10000
t

# of arms: 5969

On the other hand, BSH satisfies, with probability 1 − δ,

BSH
BUCB, = 0.01
BUCB, = 0.05
BUCB, = 0.1
BUCB, = 0.2

simple regret

0.5

˜ ⎛ϵ−2 log( 1 )⎞ .
τϵ,δ ≤ O
δ ⎠
⎝

0.4
0.3
0.2
0.1

The proof is in Appendix C.6. The (ϵ, δ)-unverifiable sample complexity of BSH for the Polynomial(α) instance does
not scale with the number of arms n polynomially, unlike
BUCB. For large-scale instances, BSH provides a much
stronger guarantee than BUCB. Even when n is not too
1
large, as long as ∆2 ≤ ϵ (which implies ϵ α ≤ n), BSH still
has a better sample complexity bound than BUCB.

0

2000 4000 6000 8000 10000
t

# of arms: 4389
BSH
BUCB, = 0.01
BUCB, = 0.05
BUCB, = 0.1
BUCB, = 0.2

simple regret

0.5

5. Experiments

0.4
0.3
0.2
0.1

We test Bracketing SH on a real-world dataset called The
New Yorker Cartoon Caption Contest (Jain et al., 2020).
For each cartoon, the editors of The New Yorker collect
the evaluation score of n captions from the participants.
Specifically, upon arrival of a participant, the algorithm sequentially shows a number of captions and receives the evaluation score of “unfunny”, “somewhat funny”, or “funny”.
Following Katz-Samuels and Jamieson (2020), we use the
proportion of times the score was “somewhat funny” or
“funny” as the ground truth mean reward for each caption.
We then set the reward distribution as the Bernoulli. We
choose contests 780, 781, and 782, which contain 6509
arms, 5969 arms, and 4389 arms, respectively. As we are
especially interested in the data-poor regime, we report the
performance of the algorithms up to time step 10,000, which
is only about twice larger than the number of arms in our
dataset. We ran BSH and BUCB with various δ choices to
see the best version of BUCB, which was repeated 500 times
with a different random seed. We used a desktop with AMD
Ryzen 5 PRO 4650GE CPU and 16GB RAM to conduct
the experiment, which took two hours to produce each plot.
We summarize the results in Figure 2, where BSH is a clear
winner over BUCB. Specifically, in contest 781, at time step
4, 000, Bracketing UCB achieves simple regret about 0.18
for δ = 0.2, while BSH achieves simple regret about 0.11,
amounting to 39% improvement. We explain more on the
implementation in Appendix D.

0

2000 4000 6000 8000 10000
t

Figure 2. The simple regret comparison between Bracketing UCB
and Bracketing SH for the New Yorker Cartoon Caption contest
780, 781, 782.

6. Conclusion
Obtaining a uniform ϵ-error probability bound is precisely a
way to characterize the distribution function of ∆JT induced
by a particular algorithm, based on which we believe that a
uniform ϵ-error probability bound is a fundamental quantity
for any measures for returning an ϵ-good arm.
As such, we believe our paper opens up numerous exciting open problems. First, SH does not achieve an optimal
worst-case sample complexity of Θ( ϵn2 ln(1/δ)) (due to
logarithmic factors). While adjusting its sample allocation
scheme does achieve the optimal rate as we report in the
appendix, it does not seem to achieve the usual instancedependent sample complexity of Θ̃(∑i (ϵ∨∆i )−2 ) achieved
by many existing algorithms. We wonder if a modified SH
or other algorithms can simultaneously be optimal for both
sample complexity measures. Second, another practical
pure exploration algorithm is track-and-stop by Garivier
and Kaufmann (2016) and their variants. They are also
8

## Page 9

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

parameter-free if we only take their sampling strategy and
not the stopping strategy. It would be interesting to investigate if track-and-stop achieves similar near-optimal
guarantees as DSH.

tional conference on Algorithmic learning theory, pages
23–37. Springer, 2009.
Sébastien Bubeck, Rémi Munos, and Gilles Stoltz. Pure
exploration in finitely-armed and continuous-armed
bandits.
Theoretical Computer Science, 412(19):
1832–1852, apr 2011. URL https://hal-hec.
archives-ouvertes.fr/hal-00609550.

Acknowledgements
We thank Jaehyeok Shin for providing help in the initial development of the improved bound for SH. Kwang-Sung Jun
and Yao Zhao were supported by the Eighteenth Mile TRIF
Funding from Research, Innovation & Impact at University
of Arizona. Csaba Szepesvári gratefully acknowledges funding from the Canada CIFAR AI Chairs Program, through
Amii and NSERC.

Alexandra Carpentier and Andrea Locatelli. Tight (lower)
bounds for the fixed budget best arm identification bandit
problem. Journal of Machine Learning Research, 49
(June):590–604, 2016.
Arghya Roy Chaudhuri and Shivaram Kalyanakrishnan.
PAC identification of a bandit arm relative to a reward
quantile. In Thirty-First AAAI Conference on Artificial
Intelligence, pages 1777–1783, 2017.

References
Jean-Yves Audibert and Sébastien Bubeck. Best arm identification in multi-armed bandits. In Proceedings of the
Conference on Learning Theory (COLT), pages 41–66,
2010.

Arghya Roy Chaudhuri and Shivaram Kalyanakrishnan.
PAC identification of many good arms in stochastic multiarmed bandits. In International Conference on Machine
Learning (ICML), pages 991–1000. PMLR, 2019.

Jean-Yves Audibert, Sébastien Bubeck, et al. Minimax
policies for adversarial and stochastic bandits. In Annual
Conference on Computational Learning Theory (COLT),
volume 7, pages 1–122, 2009.

Herman Chernoff. Sequential design of experiments. The
Annals of Mathematical Statistics, 30(3):755–770, 1959.

Jean-Yves Audibert, Sébastien Bubeck, and Rémi Munos.
Best Arm Identification in Multi-Armed Bandits. In Proceedings of the Conference on Learning Theory (COLT),
2010.

Ivo Danihelka, Arthur Guez, Julian Schrittwieser, and David
Silver. Policy improvement by planning with gumbel. In
International Conference on Learning Representations,
2022.

Maryam Aziz, Jesse Anderton, Emilie Kaufmann, and Javed
Aslam. Pure exploration in infinitely-armed bandit models with fixed-confidence. In Algorithmic Learning Theory, pages 3–24. PMLR, 2018.

Eyal Even-Dar, Shie Mannor, Yishay Mansour, and Sridhar
Mahadevan. Action elimination and stopping conditions
for the multi-armed bandit and reinforcement learning
problems. Journal of machine learning research, 7(6),
2006.

Maryam Aziz, Jesse Anderton, Kevin Jamieson, Alice Wang,
Hugues Bouchard, and Javed Aslam. Identifying New
Podcasts with High General Appeal Using a Pure Exploration Infinitely-Armed Bandit Strategy. In Proceedings
of the 16th ACM Conference on Recommender Systems,
pages 134–144, 2022.

Aurélien Garivier and Emilie Kaufmann. Optimal best arm
identification with fixed confidence. In Proceedings of
the Conference on Learning Theory (COLT), pages 998–
1027, 2016.
Avinatan Hassidim, Ron Kupfer, and Yaron Singer. An
optimal elimination algorithm for learning a best arm.
Advances in Neural Information Processing Systems
(NeurIPS), 33:10788–10798, 2020.

Tavor Baharav and David Tse. Ultra fast medoid identification via correlated sequential halving. Advances in
Neural Information Processing Systems (NeurIPS), 32,
2019.

Rianne de Heide, James Cheshire, Pierre Ménard, and
Alexandra Carpentier. Bandits with many optimal arms.
Advances in Neural Information Processing Systems
(NeurIPS), 34:22457–22469, 2021.

Robert E Bechhofer. A Sequential Multiple-Decision Procedure for Selecting the Best One of Several Normal
Populations with a Common Unknown Variance, and Its
Use with Various Experimental Designs. Biometrics, 14
(3):408–429, 1958.

Lalit Jain, Kevin Jamieson, Robert Mankoff, Robert Nowak,
and Scott Sievert. The new yorker cartoon caption contest dataset. 2020. URL https://nextml.github.
io/caption-contest-data/.

Sébastien Bubeck, Rémi Munos, and Gilles Stoltz. Pure
exploration in multi-armed bandits problems. In Interna9

## Page 10

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Kevin Jamieson, Matthew Malloy, Robert Nowak, and Sebastien Bubeck. On finding the largest mean among many.
arXiv preprint arXiv:1306.3917, 2013.
K.-S. Jun and R. Nowak. Anytime exploration for multiarmed bandits using confidence information. In Proceedings of the International Conference on Machine Learning (ICML), volume 48, pages 974–982, 2016. ISBN
9781510829008.
K.-S. Jun, K. Jamieson, R. Nowak, and X. Zhu. Top arm
identification in multi-armed bandits with batch arm pulls.
In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS), 2016.
Zohar Karnin, Tomer Koren, and Oren Somekh. Almost
optimal exploration in multi-armed bandits. In International Conference on Machine Learning (ICML), pages
1238–1246. PMLR, 2013.
Julian Katz-Samuels and Kevin Jamieson. The true sample
complexity of identifying good arms. In International
Conference on Artificial Intelligence and Statistics (AISTATS), volume 108, pages 1781–1791, 2020.
Finnian Lattimore, Tor Lattimore, and Mark D Reid. Causal
bandits: Learning good interventions via causal inference. Advances in Neural Information Processing Systems (NeurIPS), 2016.
Tor Lattimore and Csaba Szepesvári. Bandit algorithms.
Cambridge University Press, 2020.
Liam Li, Kevin Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Jonathan Ben-tzur, Moritz Hardt, Benjamin
Recht, and Ameet Talwalkar. A System for Massively
Parallel Hyperparameter Tuning. In Proceedings of Machine Learning and Systems (MLSys), pages 230–246.
2020.
Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: A Novel
Bandit-Based Approach to Hyperparameter Optimization.
Journal of Machine Learning Research, 18(185):1–52,
2018.
Shie Mannor and John N Tsitsiklis. The sample complexity
of exploration in the multi-armed bandit problem. Journal
of Machine Learning Research, 5(Jun):623–648, 2004.
Ervin Tanczos, Robert Nowak, and Bob Mankoff. A
KL-LUCB algorithm for Large-Scale Crowdsourcing.
In Advances in Neural Information Processing Systems
(NeurIPS), pages 5894–5903. 2017.

10

## Page 11

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Appendix
A. Related work
Pure exploration. Pure exploration has a broad position in several closely related research directions, including multi-armed
bandits (Audibert and Bubeck, 2010; Karnin et al., 2013; Carpentier and Locatelli, 2016; Aziz et al., 2018; Karnin et al.,
2013; Jamieson et al., 2014; Garivier and Kaufmann, 2016; Chen et al., 2017b; Simchowitz et al., 2017; Hassidim et al.,
2020) and reinforcement learning (Even-Dar et al., 2006; Azar et al., 2017; Kaufmann et al., 2021). Even more, the
Monte-Carlo tree search with tree depth 1 is also a pure exploration problem (Kocsis and Szepesvári, 2006). Even if we only
consider multi-armed bandits, there are various types of problems that have been formulated. Our investigation focuses on
the standard K-armed bandits model. But we note here pure exploration has also been studied in structured bandits like
kernel bandits (Camilleri et al., 2021) and linear bandits (Soare et al., 2014; Zhu et al., 2022).
Started from Even-Dar et al. (2006); Mannor and Tsitsiklis (2004), pure exploration in K-armed bandits is studied with the
celebrated (ϵ, δ)-PAC framework. Mannor and Tsitsiklis (2004) show a lower bound, which matches the upper bound of
Median Elimination in Even-Dar et al. (2006). The recent work by Hassidim et al. (2020) further shows that it is possible to
achieve 2ϵn2 log 1δ asymptotically, where the constant factor of 1/2 is the optimal one. Aziz et al. (2018) study (ϵ, δ)-PAC
for the infinitely-armed bandits model, where they model the infinitely-armed bandits setting as the arm reservoir. Among
them, Even-Dar et al. (2006); Hassidim et al. (2020) focus on the worst-case guarantee. Aziz et al. (2018) provide more
instance-dependent bounds.
Best arm identification. The best arm identification is the factually dominating setting for studying the pure exploration
problem. Specifically, there are two problem setups of the best arm identification, fixed budget (Audibert and Bubeck, 2010;
Karnin et al., 2013; Carpentier and Locatelli, 2016) and fixed confidence (Karnin et al., 2013; Jamieson et al., 2014; Garivier
and Kaufmann, 2016; Chen et al., 2017b; Simchowitz et al., 2017; Degenne and Koolen, 2019). In the fixed budget setting,
the algorithm takes a budget as the input and is required to return an output before exhausting all the sample budget. The
paper by Audibert and Bubeck (2010) is the first one to open up this direction and proposes Successive Rejects algorithm,
which achieves the optimality up to logarithmic factors. The follow-up work of Karnin et al. (2013) proposes SH algorithm,
which is also an elimination-based algorithm as Successive Reject, but it eliminates empirically bad arms more aggressively.
It was not until Carpentier and Locatelli (2016) SH was proven to be optimal up to doubly-logarithmic factors.
Many meaningful results have been developed for the fixed confidence setting. To our knowledge, the paper by Garivier and
Kaufmann (2016) is the first one that claims asymptotic optimality. They propose a non-asymptotic lower bound for the
sample complexity and also an algorithm called Track-and-Stop that matches the lower bound asymptotically as δ goes to 0.
This result is further extended to the setting where the best arm is not unique (Degenne and Koolen, 2019).
Simple regret. In our opinion, simple regret is the least understood measure for pure exploration. Though many works claim
simple regret as their target, their performance bounds are either on best arm misidentification or (ϵ, δ)-PAC identification.
Of course, by Audibert and Bubeck (2010), one can always use the probability of best arm misidentification to upper bound
simple regret. However, in the data-poor regime, i.e., t < n, the probability of best arm misidentification is vacuous since
we cannot even guarantee each arm has been sampled at least once. For (ϵ, δ)-PAC identification, the algorithm usually
requires ϵ as the input of the algorithm. However, simple regret itself is a parameter-free performance measure. Perhaps
only Carpentier and Valko (2015) truly deals with simple regret directly. The algorithm of Carpentier and Valko (2015)
works for the simple regret minimization problem since it does not take such a predefined threshold ϵ as the input. There
is another different problem setup in which the algorithm is required to output one of the top-m arms with ϵ slackness
(Chaudhuri and Kalyanakrishnan, 2017; 2019). Though such a problem setup also characterizes how close the output is to
the best one, it still needs m and ϵ as the input of the algorithm. Recent work by de2 proposes an algorithm that guarantees
that it returns one of the m best arms with the same mean reward without knowledge of m as input to the algorithm. The
algorithm of de2 works for both finite and infinite arms settings. However, in light of its limited problem setting that requires
the top m arms to have the same mean reward, it is unclear to us if the result can be extended to guarantee simple regret
or ϵ-error probability bound as we do. The model of our paper can be regarded as a generalization of de2 because all the
results of de2 can be matched with our results up to logarithmic factors if we consider their setting, but our paper does not
require the identity among the optimal arms. Furthermore, we provide analysis on ϵ-error probability, a generalization of
11

## Page 12

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

misidentification probability, and analysis on simple regret.
(ϵ, δ)-Verifiable and unverifiable sample complexity. The fixed confidence best arm identification shares some similarities
with the (ϵ, δ)-PAC identification. For both settings, the algorithm is required to verify the output can meet the corresponding
criteria, simple regret of 0 or simple regret of ϵ, with error probability δ. Verifiability requires the algorithm to pull each
arm at least once. Therefore the (ϵ, δ)-PAC is inherently impossible for the data-poor regime, e.g., t < n. Katz-Samuels
and Jamieson (2020) argue (ϵ, δ)-PAC is not aligned with practical usage. They propose the (ϵ, δ)-unverifiable sample
complexity, which does not require the algorithm to verify the output is ϵ-good. Instead, the (ϵ, δ)-unverifiable sample
complexity represents the number of samplings that the algorithm minimally needs such that it has the ability to output an
ϵ-good arm. Note the sample complexity of a fixed budget algorithm (or, to be more accurate, the doubling trick version
of the fixed budget algorithm) also captures the nature of the (ϵ, δ)-unverifiable sample complexity. In the reinforcement
learning setting, Dann et al. (2017) also discuss the limitation of (ϵ, δ)-PAC and propose a notion called Uniform-PAC.
Top-k arm identification. The top-k arm identification problem requires the algorithm to return k arms with the highest
mean reward instead of only the best one (Kalyanakrishnan and Stone, 2010; Kalyanakrishnan et al., 2012; Chen et al.,
2017a; Simchowitz et al., 2017). The seminal work is Kalyanakrishnan and Stone (2010). Their problem formulation is
a direct extension of the (ϵ, δ)-PAC identification problem. The goal is to return k arms that have mean rewards no less
than µk − ϵ. The worst-case lower bound is shown in Kalyanakrishnan et al. (2012). The first instance-dependent lower
bound for the k-identification problem is given by Simchowitz et al. (2017) and Chen et al. (2017a) almost at the same time,
together with the near-optimal algorithms. Zhou et al. (2014) proposes the aggregate regret to measure the performance of
the algorithms for top-k arm identification, which is defined as the gap between the average reward of the top-k arms and
that of the output. The best arm identification can be regarded as a special case of the top-k arm identification problem. The
similarity is that we are both interested in good arms, but we are measuring the likelihood of identifying one arm out of the
top-k rather than the set of top-k arms.

B. Improved Analyses of Sequential Halving
B.1. Implication of Theorem 4 for m = 1
The following theorem corresponds to the implication of m = 1 of Theorem 4. However, we prove it independently here.
Theorem 7. For any ϵ ∈ (0, 1), the error probability of SH for identifying an ϵ-good arm satisfies,
P(µJT < µ1 − ϵ) ≤ 3 log2 n⋅ exp(−

T
ϵ2
).
32 n log2 n

Proof. To avoid redundancy and for the sake of readability, we assume n is of a power of 2. It is easy to verify the result for
any n. Let ϵ1 = ϵ/4, T ′ ∶= logT n . And define ϵℓ+1 = 43 ⋅ ϵℓ . For each stage ℓ, define the event Gℓ as
2

Gℓ ∶= { max µi ≥ max µi − ϵℓ } .
i∈Sℓ+1

i∈Sℓ

log n

Thus as long as ⋂ℓ=12 Gℓ happens, we have that the arm returned after the final stage is an ϵ-good arm, because
log2 n

∞
3
∑ ϵℓ < ∑ ( )
4
ℓ=1
ℓ=1

ℓ−1

⋅ ϵ1 =

ϵ ∞ 3
∑( )
4 ℓ=1 4

ℓ−1

≤

ϵ
1 − (3/4)n
lim
= ϵ.
4 n→∞ 1 − 3/4

Thus, by a union bound,
c

⎛⎛log2 n ⎞ ⎞
P(µJT < µ1 − ϵ) ≤ P⎜ ⋂ Gℓ ⎟
⎠ ⎠
⎝⎝ ℓ=1
log2 n

≤ ∑ P(Gcℓ ).
ℓ=1

12

(5)

## Page 13

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Let aℓ be the best arm in Sℓ ,
P(Gcℓ ) = P(Gcℓ , µ̂aℓ < µaℓ − ϵℓ /2) + P(Gcℓ , µ̂aℓ ≥ µaℓ − ϵℓ /2)
≤ P(µ̂aℓ < µaℓ − ϵℓ /2) + P(Gcℓ ∣ µ̂aℓ ≥ µaℓ − ϵℓ /2).
ϵ2 T ′
≤ exp(− ℓ
) + P(Gcℓ ∣ µ̂aℓ ≥ µaℓ − ϵℓ /2).
2 ∣Sℓ ∣
For the second term,
P(Gcℓ ∣ µ̂aℓ ≥ µaℓ − ϵℓ /2) ≤ P(∣{i ∈ Sℓ ∣ µ̂i > µi + ϵℓ /2}∣ ≥ ∣Sℓ ∣/2)
(a1 ) E[∣{i ∈ Sℓ ∣ µ̂i > µi + ϵℓ /2}∣]

≤
∣Sℓ ∣/2
ϵ2ℓ

′

∣Sℓ ∣ exp(− 2 ∣STℓ ∣ )
≤
∣Sℓ ∣/2
ϵ2 T ′
).
= 2 exp(− ℓ
2 ∣Sℓ ∣
For (a1 ), we use Markov’s inequality. Then,
ℓ−1 2
⎛ 9 ℓ−1 ϵ2 T ′ ⎞
⎛
ϵ2 T ′
ϵ
T′ ⎞
9
⎟
⎜−( )
⎟.
P(Gcℓ ) ≤ 3 exp(− ℓ
=
3
exp
) = 3 exp⎜−( )
2 ∣Sℓ ∣
32 2−(ℓ−1) n ⎠
32 n ⎠
⎝ 8
⎝ 16

Taking the above into (5), we have
log2 n

P(µJT < µ1 − ϵ) ≤ ∑ P(Gcℓ )
ℓ=1

⎛ 9 ℓ−1 ϵ2 T ′ ⎞
⎟
≤ ∑ 3 exp⎜−( )
32 n ⎠
ℓ=1
⎝ 8
log2 n

≤3 log2 n⋅ exp(−

T
ϵ2
).
32 n log2 n

B.2. Proof of Corollary 4.1
We state the following result that includes two different budget allocation schemes for SH. The appendix B.12 should be
read first where we define the two budget allocation schemes:
⎢
⎥
⎢
⎥
T
⎢
⎥,
Option 1: Tℓ = ⎢
⎥
⎢ ∣Sℓ ∣⌈log2 n⌉ ⎥
⎣
⎦

⎥
⎢
ℓ−1
⎥
⎢ T
16
⎢
⋅ ( ) ⋅ ℓ⎥⎥.
Option 2: Tℓ = ⎢
81n
9
⎥
⎢
⎣
⎦

Corollary 4.1. The simple regret of SH satisfies
√
⎛
Option 1: SRegT ≤ O⎜
⎝

n log3 n ⎞
⎟,
T
⎠

√
Option 2: SRegT ≤ O(

n
).
T

Proof. The simple regret can be calculated by considering the following integral with respect to ϵ. We use SH with Option
13

## Page 14

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

2. For Option 1, the same analysis holds.
∞

SRegT = ∫

0

P(µ1 − µJT > ϵ) dϵ
∞⎛

=∫

0

1 ∧ exp(−ϵ2

⎝
√n
T

≤∫

0

√
≤

∞

1 dϵ + ∫

n
+∫
T
0

∞

T ⎞
) dϵ
n ⎠
exp(−ϵ2

0

exp(−ϵ2

T
) dϵ
n

T
) dϵ.
n

We can borrow the result from Gaussian distribution with mean 0 and variance σ 2 ,
∞

∫

0

1
x2
1
√ exp(− 2 ) dx = .
2σ
2
σ 2π

Taking 2σ1 2 = Tn , we have
∞

∫

0

√
√ √
π n
σ 2π
exp(−ϵ ) dϵ =
=
.
n
2
2
T
2T

Thus, we have
√
n
SRegT ≤ O(
).
T

B.3. Proof of Theorem 4
Theorem 4 is an equivalent result to Theorem 8, so the proof of one implies the other. While Theorem 8 takes a form that is
easier to instantiate the bound for specific instances, Theorem 4 takes a form that is easier to prove. Thus, we choose to
prove Theorem 4 directly, leaving the proof of Theorem 8 as a consequence of Theorem 4.
Let us first provide an intuitive explanation. Imagine one possible ‘typical’ scenario where, in each stage, the set of surviving
arms for the next stage happens to maintain the fraction of good arms as at least m/n. This means that, after Θ(log2 m)
stages, we expect to have at least one good arm in the surviving arm set. The number of surviving arms at that time is around
n/m. At this point, the rest of the procedure can be analyzed by the standard analysis of SH where the goal is to find the
arm that is the best in the current surviving arm set. Thus, it remains to bound how likely it is to have the ‘typical’ event
above. That is, we would like to bound the failure probability of this event at each stage as tightly as possible. Bounding this
failure probability is our key technical innovation (Proposition 9), which promotes the fast rate of the failure probability that
improves as a function of the number of good arms. Another key technical step of the proof is a careful design of events
that would guarantee the chosen arm’s suboptimality in the last stage to be at most ϵ, which requires splitting ϵ into smaller
pieces to be distributed over the stages.
Theorem 4. For any m ≤ n and any ϵ ∈ (0, 1), the error probability of SH for identifying an (m, ϵ)-good arm satisfies
⎛
⎛
⎞⎞
ϵ2 T
⎟.
P(µJT < µm − ϵ) ≤ log2 n⋅ exp⎜−const⋅m
−
ln(4e)
⎝ 4n log22 (2m) log2 n
⎠⎠
⎝
Thus, there exist a positive constant c1 such that for T ≥ c1

(ln(4e)+ln ln n)n log22 (2m) log2 n
= Θ̃( ϵn2 ),
ϵ2

⎛
ϵ2 T ⎞
) ,
P(µJT < µm − ϵ) ≤ exp −Θ̃(m
n ⎠
⎝
where, Θ̃ means ignoring the logarithmic factors of m, n and constants.
14

## Page 15

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Proof. Let us consider the case of m ≤ n/2 first.
Let ℓ∗ = ⌈log2 m⌉, ϵ′ = 2 logϵ m , T ′ ∶= ⌈ logT n ⌉. And gℓ denotes the number of (m, ℓ⋅ϵ′ )-good arms after finishing stage
2

2

ℓ ∈ [⌈log2 m⌉]. For stage ℓ, we define the event Gℓ as,

Gℓ ∶= {gℓ ≥ 2−ℓ ⋅m} .

Specifically, we have Gℓ∗ to be the event that the number of (m, ϵ/2)-good arms after finishing stage ℓ∗ is at least 1. We
define Gℓ∗ +1 as the algorithm succeeds to return an arm in Topm (ϵ),

Gℓ∗ +1 ∶= {µJT ≥ µm − ϵ} .

∗

Then the event ⋂ℓℓ=1+1 Gℓ is a possible path the algorithm returns an arm in Topm (ϵ) in the end. Thus the probability of
c
c
missing all of the (m, ϵ)-good arms can be upper bounded as follows. Define Fℓ = (⋂ℓ−1
i=1 Gi ) ∩ Gℓ for ℓ > 1, F1 = G1 .
∗

∗

c

ℓ +1
Since ⋂ℓ=1
Gℓ ⊂ Gℓ∗ +1 implies Gcℓ∗ +1 = {µJT < µm − ϵ} ⊂ (⋂ℓℓ=1+1 Gℓ ) ,

c

⎛⎛ℓ∗ +1 ⎞ ⎞
P(µJT < µm − ϵ) ≤ P⎜ ⋂ Gℓ ⎟
⎠ ⎠
⎝⎝ ℓ=1
∗

⎛ℓ +1 ⎞
= P ⋃ Fℓ
⎝ ℓ=1 ⎠
∗

(a1 ) ℓ +1

ℓ−1
⎛
⎞
≤ ∑ P Gcℓ ∣ ⋂ Gi
⎝
⎠
i=1
ℓ=1
∗

∗

ℓ
ℓ−1
ℓ
⎛
⎞
⎛
⎞
= ∑ P Gcℓ ∣ ⋂ Gi + P Gcℓ∗ +1 ∣ ⋂ Gi .
⎠
⎝
⎠
i=1
i=1
ℓ=1 ⎝

(6)

For (a1 ), we use P(A, B) ≤ P(A∣B).
For the first term of (6), we apply the result of Proposition 9 to each stage of SH with the parameters therein as k = 2−ℓ ⋅m, s =
15

## Page 16

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

2−ℓ ⋅n, m′ = 2−ℓ+1 ⋅m, ϵ = ϵ′ , and T = T ′ . Thus, for stage ℓ,
P(Gcℓ ∣ Gℓ−1 )
2−ℓ+1 ⋅n
ϵ′2 T ′
2−ℓ+1 ⋅m
ϵ′2 T ′
) exp(−2−ℓ ⋅(n − m) −(ℓ−1) )
≤ ( −ℓ
) exp(−2−ℓ ⋅m −(ℓ−1) ) + ( −ℓ
2 ⋅(n − m)
2 ⋅m
2⋅2
⋅n
2⋅2
⋅n
(a3 )

≤ hℓ,1 exp(−2−ℓ ⋅m

≤ hℓ,1 exp(−

ϵ′2 T ′
ϵ′2 T ′
−ℓ
)
+
h
exp(−2
⋅(n
−
m)
)
ℓ,2
2⋅2−(ℓ−1) ⋅n
2⋅2−(ℓ−1) ⋅n

m ϵ′2 T ′
n − m ϵ′2 T ′
) + hℓ,2 exp(−
)
2 2n
2
2n

(a4 )

≤ hℓ,1 exp(−

m ϵ′2 T ′
n ϵ′2 T ′
) + hℓ,2 exp(−
)
2 2n
4 2n

(7)

⎛ m
⎞
ϵ2 T
≤ exp −
+ ln(2e)2−ℓ ⋅m
2
⎝ 2 2n log2 m log2 n
⎠
⎛ n
⎞
ϵ2 T
+ exp −
+ ln(4e)2−ℓ ⋅(n − m)
2
⎝ 4 2n log2 m log2 n
⎠
⎛ ⎛
⎞
ϵ2 T
−ℓ ⎞
⎟
−
2
ln(4e)2
≤ exp⎜−m
2
⎠⎠
⎝ ⎝ 4n log2 m log2 n
⎛ n⎛
⎞⎞
ϵ2 T
+ exp⎜−
− 2 ln(4e)2−ℓ ⎟
2
⎠⎠
⎝ 2 ⎝ 4n log2 m log2 n
⎛ ⎛
⎞⎞
ϵ2 T
≤ 2 exp⎜−m
− 2 ln(4e)2−ℓ ⎟
2
⎠⎠
⎝ ⎝ 4n log2 m log2 n
−ℓ+1

⋅m
) ≤ (2e)2
where (a3 ) is (22−ℓ ⋅m

(4e)

2−ℓ ⋅n

−ℓ

−ℓ+1

y

⋅m

−ℓ

⋅n
) ≤ (2en/(n − m))2
=∶ hℓ,1 (Sterling’s formula (xy) ≤ ( ex
) ) and (2−ℓ2 ⋅(n−m)
y

⋅(n−m)

≤

=∶ hℓ,2 and (a4 ) is by the assumption m ≤ n/2.

Then we have,
ℓ∗
⎛ ⎛
⎞⎞
ϵ2 T
c
− ln(4e) ⎟.
∑ P(Gℓ ∣ Gℓ−1 ) ≤ const⋅ log2 m⋅ exp⎜−m
2
⎠⎠
ℓ=1
⎝ ⎝ 4n log2 m log2 n

(8)

For the second term of (6), Gcℓ∗ +1 ∣ Gℓ∗ indicates the events where SH fails to return an (m, ϵ)-good arm given the event
that there is at least one (m, ϵ/2)-good arm after finishing stage ℓ∗ . Note the stages from ℓ∗ to the last stage can be regarded
n
) logT(n) . The initial arms are the surviving arms
as a whole process of SH with a budget (⌈log2 n⌉ − ℓ∗ )T ′ ≥ const⋅ log2 ( m
2

∗

after finishing stage ℓ∗ . Note we have 2−ℓ ⋅n = n/m arms surviving, denoted by Sℓ∗ , after finishing stage ℓ∗ . Let c be the
index of the true best arm in Sℓ∗ . Due to the event Gℓ∗ , we have µc ≥ µm − 2ϵ . Then, by applying the result of Theorem 7,
P(Gcℓ∗ +1 ∣ Gℓ∗ ) = P(µJT < µm − ϵ ∣ Gℓ∗ )
ϵ
≤ P(µJT < µc − ∣ Gℓ∗ )
2
2
∗
′⎞
⎛
n
m ϵ (⌈log2 n⌉ − ℓ )T ⎟
⎜
≤ 3 log2 ( )⋅ exp⎜−
⎟
n
m
128
)
n log2 ( m
⎝
⎠
≤ log2 (

n
ϵ2 T
)⋅ exp(−const⋅m
).
m
n log2 n
16

(9)

## Page 17

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Bringing (8),(9) into (6), we have,
⎛
⎛
⎞⎞
ϵ2 T
⎟.
P(µJT < µm − ϵ) ≤ log2 n⋅ exp⎜−const⋅m
−
ln(4e)
⎝ 4n log22 (m) log2 n
⎠⎠
⎝
Note the above analysis is for m > 1. To incorporate the result of Theorem 7 for m = 1, we rewrite the final formula as
⎛
⎞⎞
⎛
ϵ2 T
⎟.
−
ln(4e)
P(µJT < µm − ϵ) ≤ log2 n⋅ exp⎜−const⋅m
⎠⎠
⎝ 4n log22 (2m) log2 n
⎝
Thus, there exist a positive constant c1 such that for T ≥ c1

(ln(4e)+ln ln n)n log22 (2m) log2 n
= Θ̃( ϵn2 ),
ϵ2

⎛
ϵ2 T ⎞
P(µJT < µm − ϵ) ≤ exp −Θ̃(m
) .
n ⎠
⎝
For the case of m > n/2, we consider
P(µJT < µm − ϵ) ≤ P(µJT < µn/2 − ϵ).
Thus, one can repeat the same analysis as above with m replaced by n/2. Using m/2 ≤ n/2 ≤ m, the statement of this
theorem holds.
2 1
n
The result of Theorem 4 directly improves the previous works including O( mϵ
2 log ( δ )) of Chaudhuri and Kalyanakrishnan
n
(2017), and O( ϵ12 ( m
log( 1δ ) + log2 ( 1δ ))) of Chaudhuri and Kalyanakrishnan (2019). They share a similar strategy that,
n
assuming the knowledge of m, ϵ and δ, uniformly samples Θ( m
log( 1δ )) arms first and then runs the Median-Elimination
algorithm of Even-Dar et al. (2006) or LUCB of Kaufmann and Kalyanakrishnan (2013) for this subset.

B.4. Proof of Theorem 8
Theorem 8. For any ϵ ∈ (0, 1), the error probability of SH for identifying an ϵ-good arm satisfies,
P(µJT < µ1 − ϵ) ≤

⎛
⎞⎞
⎛
ϵ′2 T
log2 n⋅ exp⎜−const⋅m′
− ln(4e) ⎟.
2
⎝ 4n log2 (2m′ ) log2 n
⎠⎠
{(m′ ,ϵ′ )∶∆m′ +ϵ′ ≤ϵ}
⎝
min

Proof. Note the result we proved in Theorem 4 holds for any m ≤ n and any ϵ ∈ (0, 1). Thus P(µJT < µ1 − ϵ) can be upper
bounded for any (m′ , ϵ′ ) ∈ {(m′ , ϵ′ ) ∶ ∆m′ + ϵ′ ≤ ϵ}. The result is therefore implied.
B.5. Proof of Theorem 1
Theorem 1. For any ϵ ∈ [0, 1), the ϵ-error probability of SH satisfies
⎛ ⎛
⎞⎞
T
⎜
⎟⎟
P(µJT < µ1 − ϵ) ≤ exp⎜
−
Θ̃
⎜
⎟,
maxi≥g(ϵ)+1 ∆i2 ⋅ g(1ϵ ) ⎠
⎝
⎝
⎠
i
2
for T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ).
i

Proof. To avoid redundancy and for the sake of readability, we assume n is of a power of 2. It is easy to verify the result for
any n. Recall g( 2ϵ ) is the number of 2ϵ -good arms in the instance, and gℓ ( 2ϵ ) is the number of 2ϵ -good arms at the beginning
n
of stage ℓ. Thus g1 ( 2ϵ ) = g( 2ϵ ). Let ℓ′ = max {1 ≤ ℓ ≤ log2 (n) − 2 ∣ ∆ ℓ+1
> ϵ}. There are three cases we need to consider:
2

17

## Page 18

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

• ℓ′ = log2 n − 2.
• 1 ≤ ℓ′ < log2 n − 2.
• Such an ℓ′ does not exist.
The last case that such a ℓ′ does not exit means ϵ ≥ ∆ n4 , which means that the instance has Θ(n) amount of ϵ-good arms,
therefore an easy instance. We address this case at the end of the proof.
Case 1. ℓ′ = log2 n − 2
In this case, we have ∆2 > ϵ, then the problem is a fixed budget best arm identification problem rather than an ϵ-good arm
identification problem. Since g(ϵ) = g( 2ϵ ) = 1 for ∆2 > ϵ, the theorem statement reduces to
⎛ ⎛
⎞⎞
T
⎟⎟
P(µ1 − µJT > ϵ) ≤ exp⎜
⎟,
⎜−Θ̃⎜
maxi≥2 ∆i2 ⎠
⎝
⎠
⎝
i
which is a trivial consequence of (Karnin et al., 2013, Theorem 4.1) under the theorem’s condition on T .
Case 2. 1 ≤ ℓ′ < log2 n − 2
n
. One can show that
For 1 ≤ ℓ′ < log2 n − 2, recall that ∣Aℓ ∣ = 2ℓ−1
∣Aℓ′ +1 ∣/4 ≤ g(ϵ) ≤ ∣Aℓ′ ∣/4 and ∆∣Aℓ′ +1 ∣/4 ≤ ∆g(ϵ) ≤ ϵ < ∆∣Aℓ′ ∣/4 .

(10)

For ℓ ∈ [ℓ′ + 1], define the good event for stage ℓ as
⎧
⎡
⎤ ⎫
⎪
⎪
⎪
⎢
g( 2ϵ ) ⎥ ⎪
ϵ
ϵ
⎪
⎢
⎥ + 1⎪
Gℓ ∶= ⎨gℓ ( ) ≥ g( ) − ⎢(ℓ − 1)
⎬.
⎥
⎪
2
2
2 log2 n ⎥ ⎪
⎢
⎪
⎪
⎪
⎢
⎥ ⎪
⎩
⎭
g( ϵ )

Note G1 holds with probability 1. Define m′ ∶= ⌈ 2 log2 n ⌉ = Θ̃(g( 2ϵ )). Intuitively, we allow at most m′ of 2ϵ -good arms to
2

be eliminated per stage in the initial phases, i.e., ℓ ∈ [ℓ′ + 1] of running a Sequential Halving. The ϵ-error probability can be
expressed as
′

′

c

+1
+1
P(µ1 − µJT > ϵ) = P(µ1 − µJT > ϵ, ∩ℓℓ=2
Gℓ ) + P(µ1 − µJT > ϵ, (∩ℓℓ=2
Gℓ ) ).

Since
′

′

+1
+1
P(µ1 − µJT > ϵ, ∩ℓℓ=2
Gℓ ) ≤ P(µ1 − µJT > ϵ ∣ ∩ℓℓ=2
Gℓ ),

and
′

′

c

c

ℓ′

+1
+1
P(µ1 − µJT > ϵ, (∩ℓℓ=2
Gℓ ) ) ≤ P((∩ℓℓ=2
Gℓ ) ) ≤ ∑ P(Gcℓ+1 ∣ ∩ℓi=1 Gi ),
ℓ=1

we have
′

ℓ′

+1
P(µ1 − µJT > ϵ) ≤ P(µ1 − µJT > ϵ ∣ ∩ℓℓ=2
Gℓ ) + ∑ P(Gcℓ+1 ∣ ∩ℓi=1 Gi ).

(11)

ℓ=1

We bound the two terms in (11) respectively.
We start with the second term. Let ℓ ∈ [ℓ′ ]. To bound P(Gcℓ+1 ∣ ∩ℓi=1 Gi ) with Lemma 9.1, we need to relate the quantities in
stage ℓ with those in Lemma 9.1.
We first claim that, given ∩ℓi=1 Gi , the event Gcℓ+1 implies that there are at least m′ of 2ϵ -good arms that are eliminated during
the stage ℓ, i.e., gℓ ( 2ϵ ) − gℓ+1 ( 2ϵ ) ≥ m′ .
18

## Page 19

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

To see why, we have by ∩ℓi=1 Gi ,
⎡
⎤
⎢
g( 2ϵ ) ⎥
ϵ
ϵ
⎥ + 1,
gℓ ( ) ≥ g( ) − ⎢⎢(ℓ − 1)
2
2
2 log2 n ⎥⎥
⎢
⎢
⎥
and by Gcℓ+1 ,
⎡
⎤
⎢ g( 2ϵ ) ⎥
ϵ
ϵ
⎥.
gℓ+1 ( ) ≤ g( ) − ⎢⎢ℓ
⎥
2
2
⎢ 2 log2 n ⎥
⎢
⎥
Therefore, using ⌈x + y⌉ − ⌈x⌉ + 1 ≥ ⌈y⌉, we have
⎡
⎤
⎤ ⎡
⎤
⎡
⎢ g( 2ϵ ) ⎥
⎢
g( 2ϵ ) ⎥ ⎢ g( 2ϵ ) ⎥
ϵ
ϵ
⎥ + 1 − ⎢(ℓ − 1)
⎥≥⎢
⎥ = m′ ,
gℓ ( ) − gℓ+1 ( ) = ⎢⎢ℓ
⎥
⎢
2
2
2 log2 n ⎥⎥ ⎢⎢ 2 log2 n ⎥⎥
⎢ 2 log2 n ⎥
⎢
⎢
⎥
⎥ ⎢
⎥
⎢
which proves the claim.
Define Aℓ as the set of surviving arms at the beginning of stage ℓ, Aℓ,i as i-th best arm in Aℓ (ties are broken arbitrarily),
and Aℓ [L + 1 ∶ R] ∶= {Aℓ,i , i ∈ [R] / [L]}. Let dℓ (ϵ) be the smallest gap of the mean rewards between any two arms chosen
n
n
] and Aℓ [1 ∶ gℓ ( 2ϵ )] respectively, i.e.,
from Aℓ [ 2ℓ+1
+ 1 ∶ 2ℓ−1

⎧
⎫
⎪
n
n
ϵ ⎪
⎪
⎪
dℓ (ϵ) = min ⎨µi − µj ∣ j ∈ Aℓ [ ℓ+1 + 1 ∶ ℓ−1 ] , i ∈ Aℓ [1 ∶ gℓ ( )]⎬ .
⎪
2
2
2 ⎪
⎪
⎪
⎩
⎭
∆2 n

2ℓ+1
n . Define S (ϵ) ∶=
.
The definition of dℓ (ϵ) is same as dq of Lemma 9.1. By Lemma 8.1, we have dℓ (ϵ) ≥ 12 ∆ ℓ+1
ℓ
n⋅2−ℓ+1
2

n
as follows:
We can now apply Lemma 9.1 with T = T , s = 2nℓ , k = m′ , and q = 2ℓ+1
′

P(Gcℓ+1 ∣ ∩ℓi=1 Gi )
3
⎛ 1
n⋅2−ℓ+1
d2 (ϵ)T ′ ⎞
d2 (ϵ)T ′
gℓ ( ϵ )
≤( 2′ ) exp(−m′ ℓ −ℓ+1 ) + ( 1 4 −ℓ+1
) exp −( n⋅2−ℓ+1 + m′ ) ℓ −ℓ+1
′
m
8n⋅2
8n⋅2
⎠
⎝ 4
n⋅2
+m
4

d2 (ϵ)T ′
gℓ ( ϵ )
n⋅2−ℓ+1
n d2 (ϵ)T ′
≤( 2′ ) exp(−m′ ℓ −ℓ+1 ) + (
) exp(− ℓ+1 ℓ −ℓ+1 )
−ℓ
m
8n⋅2
2 8n⋅2
n⋅2

b1

⎛
⎛ egℓ ( 2ϵ ) ⎞⎞
d2 (ϵ)T ′
n d2ℓ (ϵ)T ′
n
⎟
≤ exp⎜−m′ ℓ −ℓ+1 + m′ ln
+
exp(−
+ ℓ ln(2e))
′
ℓ+1
−ℓ+1
8n⋅2
2 8n⋅2
2
⎝ m ⎠⎠
⎝

b2

′
∆2 n T ′
n T
⎛
⎛ n ∆2 ℓ+1
⎞
egℓ ( 2ϵ ) ⎞⎞
n
′
′ ⎛
2ℓ+1
2
⎜
⎟
⎜
⎟
≤ exp −m
+
m
ln
+
exp
−
+
ln(2e)
ℓ+1
−ℓ+1
32n⋅2−ℓ+1
2ℓ
⎝ m ′ ⎠⎠
⎝
⎠
⎝ 2 32n⋅2

n
≤ exp(−Θ̃(m′ Sℓ (ϵ)T )) + exp(−Θ̃( ℓ+1 Sℓ (ϵ)T )),
2

b3

n )
(dℓ (ϵ) ≥ 21 ∆ ℓ+1
2

(T ≥ Θ̃( Sℓ1(ϵ) ))
(12)

where,
′

x
x
) = (⌈x/2⌉
),
• b1 is by (xy) < (xy ) for x′ > x and maxy (xy) = (⌊x/2⌋
y

• b2 is by upper bounding the binomial coefficients by Sterling’s formula (xy) ≤ ( ex
) and moving the coefficients into
y
the exponent.
19

## Page 20

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

• b3 is by T ≥ Θ̃( Sℓ1(ϵ) ) which is implied by T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ) in the theorem statement.
i

By the definition of ℓ′ and the fact of ℓ ∈ [ℓ′ ], we have
n
∆ ℓ+1
> ϵ.
2

By the definition of g( 2ϵ ), we have
ϵ
∆g( ϵ ) ≤ .
2
2
Thus
ϵ
n
− ∆g( ϵ ) > ,
∆ ℓ+1
2
2
2
n
which means 2ℓ+1
> g( 2ϵ ) ≥ m′ . We continue to upper bound (12). For T ≥ Θ̃( Sℓ1(ϵ) ), we have

n
P(Gcℓ+1 ∣ ∩ℓi=1 Gi ) ≤ exp(−Θ̃(m′ Sℓ (ϵ)T )) + exp(−Θ̃( ℓ+1 Sℓ (ϵ)T ))
2
≤ exp(−Θ̃(m′ Sℓ (ϵ)T )) + exp(−Θ̃(m′ Sℓ (ϵ)T ))
⎛
⎞
ϵ
≤ exp −Θ̃(g( )Sℓ (ϵ)T ) .
2
⎝
⎠

(m′ = Θ̃(g( 2ϵ )))
(13)

Next, we bound the first term in (11). The stages ℓ ≥ ℓ′ + 1 can be viewed as a new Sequential Halving game with Aℓ′ +1 as
′
+1
the initial arm set, gℓ′ +1 ( 2ϵ ) of which are 2ϵ -good. When ∩ℓi=1
Gi happens, we have, as ℓ′ < log2 n,
⎡
⎤
⎡ ϵ ⎤
⎢ ′ g( 2ϵ ) ⎥
⎢ g( ) ⎥
g( ϵ )
ϵ
ϵ
ϵ
ϵ
⎢
⎥
+ 1 ≥ g( ) − ⎢⎢ 2 ⎥⎥ + 1 ≥ 2 = Θ̃(g( )).
gℓ′ +1 ( ) ≥ g( ) − ⎢ℓ
⎥
2
2
2
log
n
2
2
2
2
⎢
⎢
⎥
2 ⎥
⎢
⎥
⎢
⎥
∆

By Lemma 8.2, a (gℓ′ +1 ( 2ϵ ), g(ϵ)+1
)-good arm in Aℓ′ +1 returned after finishing the final stage of Sequential Halving is an
2
ϵ-good arm. We define µm (A) as the m-th largest value in {µj ∶ j ∈ A}. We apply Theorem 4 and set m, n, ϵ of Theorem 4
∆
as m = gℓ′ +1 ( 2ϵ ), n = ∣Aℓ′ +1 ∣, ϵ = g(ϵ)+1
. To apply Theorem 4, we need the condition of T ≥ Θ̃( ϵn2 ) in Theorem 4 to be
2
∣A ′

true, which means we need T ≥ Θ̃( ∆2ℓ +1 ). Note by (10), we have 2ℓn′ +2 ≤ g(ϵ), thus ∣Aℓ′ +1 ∣ = 2nℓ′ ≤ 4g(ϵ) ≤ 4(g(ϵ) + 1).
∣

g(ϵ)+1

Using this bound and the condition on T from the theorem statement, we have T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ) ≥ Θ̃( ∆g(ϵ)+1
)≥
2
i

g(ϵ)+1

∣A ′ ∣
Θ̃( ∆2ℓ +1 ) So, the condition of Theorem 4 is satisfied. Then, we have
g(ϵ)+1
′

+1
P(µ1 − µJT > ϵ ∣ ∩ℓi=1
Gi )
′
1
+1
≤ P(µgℓ′ +1 ( ϵ ) (Aℓ′ +1 ) − µJT > ∆g(ϵ)+1 ∣ ∩ℓi=1
Gi )
2
2

2
′
⎛ ⎛
ϵ ∆g(ϵ)+1 T ⎞⎞
⎟
≤ exp⎜−Θ̃ gℓ′ +1 ( )⋅
2
∣Aℓ′ +1 ∣ ⎠⎠
⎝ ⎝

⎛ ⎛ ϵ ∆2g(ϵ)+1 T ′ ⎞⎞
⎟.
≤ exp⎜−Θ̃ g( )⋅
⎝ ⎝ 2 4(g(ϵ) + 1) ⎠⎠

(Lemma 8.2)
(T ≥ Θ̃( ∆g(ϵ)+1
))
2
g(ϵ)+1

(gℓ′ +1 ( 2ϵ ) = Θ̃(g( 2ϵ )))
(14)

20

## Page 21

Revisiting Simple Regret: Fast Rates for Returning a Good Arm
′

+1
The role of considering ℓ′ is that as long as the Sequential Halving goes well (∩ℓi=1
Gi happens) the ratio of 2ϵ -good arms to
ϵ
′
the all surviving arms is increasing to g( 2 )/g(ϵ) after finishing stage ℓ . Specifically, for the Polynomial(α) instance of our
α
interest where ∆i = ( ni ) with α > 0.5, g( 2ϵ )/g(ϵ) can be lower bounded by a constant. The last step is to combine (13)

and (14). For T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ), we have
i

ℓ′
ℓ′ +1
P(µ1 − µJT > ϵ) ≤ P(µ1 − µJT > ϵ ∣ ∩ℓ=2 Gℓ ) + ∑ P(Gcℓ+1 ∣ ∩ℓi=1 Gi )
ℓ=1

⎛ ⎛ ϵ ∆2g(ϵ)+1 T ⎞⎞
⎛
⎞
ϵ
⎟ + log(n) exp −Θ̃(g( ) min′ Sℓ (ϵ)T )
≤ exp⎜−Θ̃ g( )⋅
ℓ≤ℓ
2
2
⎝
⎠
⎝
⎠
4(g(ϵ) + 1) ⎠
⎝
⎛ ⎛
⎞
⎫
⎧
∆2g(ϵ)+1
⎪
⎪
ϵ
⎪ ⎞⎟
⎪ ϵ
⎜
⎟
)⋅
,
g(
)
min
≤ exp⎜
S
(ϵ)
−
Θ̃
min
⎬
T
⎨
g(
ℓ
⎜
⎟
⎪
⎪
2 4(g(ϵ) + 1)
2 ℓ≤ℓ′
⎪
⎪
⎭ ⎠⎠
⎩
⎝ ⎝

(T ≥ Θ̃(maxi≥g(ϵ)+1 ∆i2 ))
i

⎛ ⎛
⎞
⎧
2
⎪
⎪
∆2 n ⎫
⎪ ⎞⎟
⎪
⎪ ∆g(ϵ)+1
⎜ ⎜ ϵ
2ℓ+1 ⎪
⎟
≤ exp⎜−Θ̃⎜g( )⋅ min ⎨
, min
⎬ T ⎟⎟
⎜
⎟
⎪
⎪
2
4(g(ϵ) + 1) ℓ≤ℓ′ 24n
⎪
ℓ+1 ⎪
⎪
⎪
⎝
⎭ ⎠⎠
⎩
⎝
⎛
⎞
∆2i
ϵ
≤ exp −Θ̃(g( )⋅ min
T)
2 i≥g(ϵ)+1 i
⎝
⎠
⎛ ⎛
⎞⎞
T
⎟⎟
= exp⎜
⎜−Θ̃⎜
⎟.
maxi≥g(ϵ)+1 ∆i2 ⋅ g(1ϵ ) ⎠
⎝
⎝
⎠
i
2
Case 3. ℓ′ does not exist
This case condition implies that ∆ n4 ≤ ϵ, which means the number of ϵ-good arms is Θ(n), so we can use (14) solely to
claim our bound.
Lemma 8.1. In the case of 1 ≤ ℓ′ < log2 n − 2, for ℓ ∈ [ℓ′ ], define dℓ (ϵ) as the minimum gap of the mean rewards between
n
n
] and Aℓ [1 ∶ gℓ ( 2ϵ )] respectively,
any two arms chosen from Aℓ [ 2ℓ+1
+ 1 ∶ 2ℓ−1
⎫
⎧
⎪
n
ϵ ⎪
n
⎪
⎪
dℓ (ϵ) = min ⎨µi − µj ∣ ∀j ∈ Aℓ [ ℓ+1 + 1 ∶ ℓ−1 ] , ∀i ∈ Aℓ [1 ∶ gℓ ( )]⎬ .
⎪
⎪
2
2
2
⎪
⎪
⎭
⎩
Then dℓ (ϵ) satisfies
1
n .
dℓ (ϵ) ≥ ∆ ℓ+1
2 2
Proof. By definition, we have Aℓ ⊂ A1 , which implies the i-th best arm in Aℓ has an mean reward lower than or equal to the
i-th best arm in A1 . Then we have
n
n
n
n
max {µi ∣ i ∈ Aℓ [ ℓ+1 + 1 ∶ ℓ−1 ]} ≤ max {µi ∣ i ∈ A1 [ ℓ+1 + 1 ∶ ℓ−1 ]}
2
2
2
2
n
≤ µ ℓ+1
2

n .
= µ1 − ∆ ℓ+1
2

Also,
⎧
⎫
⎪
ϵ ⎪
ϵ
⎪
⎪
min ⎨µi , i ∈ Aℓ [1 ∶ gℓ ( )]⎬ ≥ µ1 − .
⎪
2 ⎪
2
⎪
⎪
⎩
⎭
21

## Page 22

Revisiting Simple Regret: Fast Rates for Returning a Good Arm
n
n
] and ∀i ∈ Aℓ [1 ∶ gℓ ( 2ϵ )],
Therefore for ∀j ∈ Aℓ [ 2ℓ+1
+ 1 ∶ 2ℓ−1

µi − µj ≥ µ1 −

ϵ
ϵ
n ) = ∆ n
− (µ1 − ∆ ℓ+1
− .
2
2ℓ+1
2
2

Since ℓ ∈ [ℓ′ ], by the definition of ℓ′ we have
n
∆ ℓ+1
> ϵ.
2

Thus
n
−
µi − µj ≥ ∆ ℓ+1
2

Lemma 8.2. A (gℓ′ +1 ( 2ϵ ),
ϵ-good arm.

n
n
n
∆ ℓ+1
ϵ ∆ 2ℓ+1
ϵ ∆ ℓ+1
=
+ 2
− > 2 .
2
2
2
2
2

∆g(ϵ)+1
)-good arm in Aℓ′ +1 returned after finishing the final stage of Sequential Halving is an
2

Proof. We first claim that
∆JT > ϵ ⇐⇒ ∆JT > ∆g(ϵ) .
To see this, for the forward direction, we have ∆JT > ϵ ≥ ∆g(ϵ) . For the other direction, if ∆JT > ∆g(ϵ) is true, then
JT > g(ϵ). This means that ∆JT > ϵ. Then, we can also show that
∆JT > ϵ ⇐⇒ ∆JT ≥ ∆g(ϵ)+1 ,
which can be shown easily by showing
∆JT ≥ ∆g(ϵ)+1 ⇐⇒ ∆JT > ∆g(ϵ) .
Define ∆m (A) to be the m-th smallest value in {∆i ∶ i ∈ A}. Then, we have ∆gℓ′ +1 ( ϵ ) (Aℓ′ +1 ) ≤ 12 ϵ. Using the fact that
2
ϵ < ∆g(ϵ)+1 , we have
∆JT ≥ ∆g(ϵ)+1 = ∆gℓ′ +1 ( ϵ ) (Aℓ′ +1 ) + ∆g(ϵ)+1 − ∆gℓ′ +1 ( ϵ ) (Aℓ′ +1 )
2

2

1
≥ ∆gℓ′ +1 ( ϵ ) (Aℓ′ +1 ) + ∆g(ϵ)+1 − ϵ
2
2
1
> ∆gℓ′ +1 ( ϵ ) (Aℓ′ +1 ) + ∆g(ϵ)+1 .
2
2

(ϵ < ∆g(ϵ)+1 )

By defining µm (A) as the m-th largest value in {µj ∶ j ∈ A}, the above equivalently means
1
µJT < µgℓ′ +1 ( ϵ ) (Aℓ′ +1 ) − ∆g(ϵ)+1 .
2
2
Therefore the output is not a (gℓ′ +1 ( 2ϵ ),

∆g(ϵ)+1
)-good arm in Aℓ′ +1 . Our claim holds by contraposition.
2

B.6. Proof of Corollary 1.1
α

Corollary 1.1. For the Polynomial(α) instance where ∆i = ( ni ) with α > 0.5, we have
i
4
1
⋅ ϵ < 2.
2
i≥g(ϵ)+1 ∆i g( )
ϵ
2

H2 (ϵ) = max
Proof.

1−2α
1
i
1
1
4
⋅ ϵ = max i1−2α n2α
< (nϵ1/α )
n2α
< 2.
2
1/α
1/α
ϵ
ϵ
i≥g(ϵ)+1 ∆i g( )
i≥g(ϵ)+1
ϵ
n( )
n( )
2

max

2

where both the last inequality and the last equality are by α > 0.5.
22

2

## Page 23

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

B.7. Proof of Corollary 2.1
Corollary 2.1. Katz-Samuels and Jamieson (2020) shows a lower bound of the sample complexity for identifying an ϵ-good
arm that scales as
n
⎛
1
1 ⎞
1
⎟.
Ω ⎜H low (ϵ) ∶=
∑ ( 2)− 2
g(ϵ) i=g(ϵ)+1 ∆i
∆g(ϵ)+1 ⎠
⎝
g( ϵ )

2
In the case that g(ϵ) and g( 2ϵ ) have the same order ( g(ϵ)
is irrelevant to ϵ), the sample complexity measure H2 (ϵ) satisfies

H low (ϵ) ≲ H2 (ϵ) ≲ H low (ϵ) +

2
∆2g(ϵ)+1

,

where ≲ hides logarithms of n and constants.
Proof.
H low (ϵ) =

n
1
1
1
− 2
∑
2
g(ϵ) i=g(ϵ)+1 ∆i ∆g(ϵ)+1

=

n
1 i
1
1
−
∑
g(ϵ) i=g(ϵ)+1 i ∆2i ∆2g(ϵ)+1

≤

n
1
i
1
1
max
⋅ ∑
−
2
g(ϵ) i≥g(ϵ)+1 ∆i i=g(ϵ)+1 i ∆2g(ϵ)+1

(a1 )

≤

1
i
1
⋅ log(2n) − 2
max
g(ϵ) i≥g(ϵ)+1 ∆2i
∆g(ϵ)+1

=

g( 2ϵ )

≤

g( 2ϵ )

g(ϵ)
g(ϵ)

H2 (ϵ)⋅ log(2n) −

1
∆2g(ϵ)+1

H2 (ϵ)⋅ log(2n),

for (a1 ), we use ∑ni=g(ϵ) 1i ≤ ∑i∈[n] 1i ≤ log(2n), which can be found in Audibert and Bubeck (2010). On the other hand,
H low (ϵ) =

n
1
1
1
−
∑
g(ϵ) i=g(ϵ)+1 ∆2i ∆2g(ϵ)+1

≥

i − g(ϵ)
1
1
max
− 2
2
g(ϵ) i≥g(ϵ)+1 ∆i
∆g(ϵ)+1

=

1
i
g(ϵ)
1
( max
− 2 )− 2
2
g(ϵ) i≥g(ϵ)+1 ∆i
∆i
∆g(ϵ)+1

≥

1 ⎛
i
g(ϵ) ⎞
1
⎜ max
⎟− 2
− 2
2
g(ϵ) ⎝i≥g(ϵ)+1 ∆i ∆g(ϵ)+1 ⎠ ∆g(ϵ)+1

=

i
1
2
max
− 2
2
g(ϵ) i≥g(ϵ)+1 ∆i ∆g(ϵ)+1

=

g( 2ϵ )
g(ϵ)

H2 (ϵ) −

2
.
∆2g(ϵ)+1

23

## Page 24

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

B.8. Proof of Theorem 3
For an n-armed bandit instance ν and a fixed value of ϵ > 0, consider a partition of the arms of ν, A =
{[g(ϵ)], [g(ϵ) + 1 ∶ 2⋅g(ϵ)], . . . , [n − g(ϵ) + 1 ∶ n]}, where we avoid rounding by making the assumption that n is an
integer multiple of g(ϵ). We use Ai ∶= [(i − 1)⋅g(ϵ) ∶ i⋅g(ϵ)] to denote the subsets of the partition A. Making use of this
partition, we define
n/g(ϵ)

˜ −2
H̃(ν, ϵ) ∶= ∑ ∆
Ai ,
i=2

˜ A ∶= maxj∈A ∆j , the largest gap among the arms in A.
where for an arbitrary set of arms A ⊆ [n] we define ∆
Proof of Theorem 3. The proof structure follows Carpentier and Locatelli (2016, Theorem 1). Fix n > 2, ϵ > 0 an n-armed
unit-variance gaussian instance ν and an arbitrary policy π. Let Pν,π denote the probability measure over interaction
histories induced by the interaction of π with instance ν. Without loss of generality, we take the arm means of ν to be
ordered such that µ1 ≥ µ2 ≥ . . . ≥ µn . Understanding g(ϵ) to refer to the number of ϵ-good arms on ν, we define instances
ν i for i ∈ [2 ∶ n/g(ϵ)] from ν by setting νji = νj for j ∉ Ai , and setting µij ∶= µj + 2∆j for j ∈ Ai . Each instance ν i has g(ϵ)
ϵ-good arms with indices in Ai , and H̃(ν i , ϵ) ≤ H̃(ν, ϵ) = a for all i ∈ [2 ∶ n/g(ϵ)].
We will make use of a change of measure between ν and these alternate instances, restricted on an event on which the
interaction histories on ν and ν i are hard to distinguish from one another:
⎧
⎫
⎪
⎪
1
⎪T n
⎪
Ei = ⎨ ∑ ∑ 1 {It = j} [Λit (j) − (1 + ρ)KL(νji , νj )] ≤ log(1/δ)⎬ ,
⎪
⎪
ρ
⎪
⎪
⎩t=1 j=1
⎭
where Λit (j) = log

⎛ dνj (Xt ) ⎞
for i ∈ [2 ∶ n/g(ϵ)], and ρ > 0 is a parameter to be chosen later.
⎝ dνji (Xt ) ⎠

Lemma 8.3 (Jun and Zhang (2022, Lemma 10)). For all i ∈ [2 ∶ n/g(ϵ)] and arbitrary π, ρ > 0, and δ ∈ (0, 1),
Pν,π (Ei ) ≥ 1 − δ.
We can now prove the result. Denoting the number of times arm j ∈ [n] has been pulled in T rounds of interaction by
Nj (T ), let ti = Eν,π [∑j∈Ai Nj (T )] be the expected number of pulls that arm group Ai receives, for i ∈ [2 ∶ n/g(ϵ)]. By
Markov’s inequality, for all i ∈ [2 ∶ n/g(ϵ)] we have
⎛
⎞
Pν,π ∑ Nj (T ) ≥ 6ti ≤ 1/6.
⎝j∈Ai
⎠
Defining the event
⎧
⎫
⎪
⎪
⎪
⎪
Ei = {Jt ∈ A1 } ∩ {Ei } ∩ ⎨ ∑ Nj (T ) ≤ 6ti ⎬
⎪
⎪
⎪
⎪
⎩j∈Ai
⎭
for i ∈ [2 ∶ n/g(ϵ)] and setting δ = 1/6 in Lemma 8.3, for all i ∈ [2 ∶ n/g(ϵ)] and arbitrary π
Pν,π (Ei ) ≥ 1 − (1/6 + 1/6 + ξ) = 2/3 − ξ,
where ξ ∶= Pν,π (Jt ∉ A1 ).
It follows that for i ∈ [2 ∶ n/g(ϵ)]
Pν i ,π (Jt ∉ Ai ) ≥ Pν i ,π (Jt ∈ A1 )
≥ Pν i ,π (Ei ).
24

## Page 25

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Making use of a standard change of measure technique for bandits (Audibert et al., 2010), we have
⎡
⎤
⎢
⎞⎥⎥
⎛ T
i
⎢
Pν i ,π (Ei ) = Eν,π ⎢1 {Ei } exp − ∑ ∑ 1 {It = a} Λt (j) ⎥
⎠⎥
⎝ t=1 j∈Ai
⎢
⎣
⎦
⎡
⎤
T
⎢
⎛
⎞⎥
≥ Eν,π ⎢⎢1 {Ei } exp − ∑ ∑ 1 {It = j} 2KL(νji , νj ) − log(6) ⎥⎥
(on Ei , taking ρ = 1 and δ = 1/6)
⎝ t=1 j∈Ai
⎠⎥
⎢
⎣
⎦
⎡
⎤
⎢
⎥
⎞
⎛
≥ Eν,π ⎢⎢1 {Ei } exp − ∑ 4Nj (T )⋅ max ∆2k − log(6) ⎥⎥
(evaluating the KL-divergence)
k∈Ai
⎠⎥
⎝ j∈Ai
⎢
⎣
⎦
⎡
⎤
⎢
⎥
⎛
⎞
1
˜ 2A ⎥
˜A )
(definition of ∆
= Eν,π ⎢⎢1 {Ei } exp − ∑ 4Nj (T )∆
i
i ⎥
6
⎝ a∈Ai
⎠⎥
⎢
⎣
⎦
1
˜ 2A )]
(on Ei )
≥ Eν,π [1 {Ei } exp(−24ti ∆
i
6
1
˜ 2A ) Pν,π (Ei ).
= exp(−24ti ∆
i
6
If ξ ≥ 1/2, the result follows. Otherwise if ξ ≤ 1/2, then Pν (Ei ) ≥ 1/6 and it follows that Pν i (Jt ∉ Ai ) ≥
1
˜ 2 ).
exp(−24ti ∆
Ai
36
n/m
n/g(ϵ) ˜ −2
∆Ai , and ∑i=1 ti = T , there exists some i ∈ [2 ∶ n/g(ϵ)] such that

Observing that H̃(ϵ, ν) = ∑i=2

ti ≤

T
T
,
=
2
˜
H̃(ϵ, ν)∆Ai a∆˜2A
i

which can easily be shown by way of contradiction. For this i, by our display above, we have
max {Pν i ,π (Jt ∉ Ai ), Pν,π (Jt ∉ A1 )} ≥ max {1/2,

1
exp(−24T /a)} .
36

B.9. Proof of Lemma 3.1
Proof of Lemma 3.1. To see this, for a given value of n and α > 1/2, fix ϵ ≤ 1/2. We have
n/g(ϵ)

˜ −2
H̃(ϵ) = ∑ ∆
Ai
i=2
2α

n/g(ϵ)

n
)
i⋅g(ϵ)
i=2
⎤
2α ⎡n/g(ϵ)
i+1
⎢
⎥
n
≥(
) ⎢⎢ ∑ ∫
x−2α dx⎥⎥
g(ϵ)
⎢ i=2 i
⎥
⎣
⎦
2α
n/g(ϵ)+1
n
) ∫
x−2α dx
=(
g(ϵ)
2

(Polynomial(α) instance)

= ∑ (

≥(

n
)
g(ϵ)

(x−2α decreases monotonically on x > 0, for α > 0)

ϵ−1/α +1

2α

∫

α

(( g(ϵ)
) < ϵ, by definition of g(ϵ).)
n

x−2α dx
2
−2α+1

−2α+1
− (ϵ−1/α + 1)
1 2
= 2
ϵ
2α − 1
⎡ −2α+1
−2α+1 ⎤
⎢2
⎥
− (21/α + 1)
⎥
1 ⎢⎢
⎥.
≥ 2⎢
⎥
ϵ ⎢
2α − 1
⎥
⎢
⎥
⎣
⎦

(assuming ϵ < 1/2)

25

## Page 26

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

B.10. Analysis of the uniform sampling
Uniform sampling means the algorithm that takes a budget T as the input and equally allocates the budget to all the arms.
Define Topm (ϵ) ∶= {i ∶ µi ≥ µm − ϵ} and a shortcut of Topm ∶= Topm (0).
Proposition 9. Suppose we run the uniform sampling on a n-armed bandit with a budget T and output the s arms with the
largest empirical rewards. Let CT (s, m′ , ϵ) be the (m′ , ϵ)-good arms of the output. Then,

P(∣CT (s, m′ , ϵ)∣ ≤ k) ≤ (

n − ∣Topm′ (ϵ)∣
m′
ϵ2 T
ϵ2 T
′
)
exp(−(m
−
k)
)
+
(
)
exp(−(s
−
k)
).
m′ − k
8n
8n
s−k

This proposition is where our key contribution is rooted at. We show that our bound is tight for the uniform sampling in
Section B.11.

Proof. Step 1
We claim that the intersection of the two conditions below is a sufficient condition for ∣CT (s, m′ , ϵ)∣ > k.

• ∣{i ∈ Topm′ ∶ µ̂i > µi − 2ϵ }∣ ≥ k + 1.

• ∣{i ∈ Topcm′ (ϵ) ∶ µ̂i < µi + 2ϵ }∣ ≥ ∣Topcm′ (ϵ)∣ − s + k + 1.

If all arms in {i ∈ Topm′ ∶ µ̂i > µi − 2ϵ } are included in the output, the claim naturally holds. Thus we focus on the situation
where at least one of the arms in {i ∈ Topm′ ∶ µ̂i > µi − 2ϵ } is not included in the output. We prove the claim by contradiction.
Suppose ∣CT (s, m′ , ϵ)∣ ≤ k. Then the output includes no less than s − k non-(m′ , ϵ)-good arms. Since

s − k + ∣Topcm′ (ϵ)∣ − s + k + 1 = ∣Topcm′ (ϵ)∣ + 1 > ∣Topcm′ (ϵ)∣,

at least one arm in {i ∈ Topcm′ (ϵ) ∶ µ̂i < µi + 2ϵ } is included in the output. To be more specific, there must exist overlap
between {i ∈ Topcm′ (ϵ) ∶ µ̂i < µi + 2ϵ } and the set of the non-(m′ , ϵ)-good arms included in the output; otherwise the union
of these two sets is even larger than the set of all the non-(m′ , ϵ)-good arms we have in the entire arm set. Additionally,
all arms in {i ∈ Topcm′ (ϵ) ∶ µ̂i < µi + 2ϵ } have less empirical rewards than any arms in {i ∈ Topm′ ∶ µ̂i > µi − 2ϵ }. Thus the
output includes all arms in {i ∈ Topm′ ∶ µ̂i > µi − 2ϵ } as well. This leads to the contradiction with our supposition that at
least one of the arms in {i ∈ Topm′ ∶ µ̂i > µi − 2ϵ } are not included in the output.
Step 2
26

## Page 27

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

We apply union bound and Hoeffding’s inequality for the event that at least one of the two conditions does not hold.
⎛
⎞
ϵ
P(∣CT (s, m′ , ϵ)∣ ≤ k) ≤ P ∣{i ∈ Topm′ ∶ µ̂i > µi − }∣ < k + 1
2
⎝
⎠
⎞
⎛
ϵ
+ P ∣{i ∈ Topcm′ (ϵ) ∶ µ̂i < µi + }∣ < ∣Topcm′ (ϵ)∣ − s + k + 1
2
⎠
⎝
⎞
⎛
ϵ
= P ∣{i ∈ Topm′ ∶ µ̂i ≤ µi − }∣ ≥ m′ − k
2
⎠
⎝
⎛
⎞
ϵ
+ P ∣{i ∈ Topcm′ (ϵ) ∶ µ̂i ≥ µi + }∣ ≥ s − k
2
⎝
⎠
ϵ
≤ P(∃A ⊂ Topm′ , s.t. ∣A∣ = m′ − k and ∀i ∈ A, µ̂i ≤ µi − )
2
ϵ
+ P(∃A ⊂ Topcm′ (ϵ), s.t. ∣A∣ = s − k and ∀i ∈ A, µ̂i ≥ µi + )
2
2
∣Top
(ϵ)∣
n
−
m′
ϵ
ϵ2 T
T
′
m
≤( ′
) exp(−(m′ − k)
)+(
) exp(−(s − k)
).
m −k
8n
8n
s−k

Lemma 9.1. Suppose we run the uniform sampling on a n-armed bandit with a budget of T and output the s arms with the
largest empirical rewards. Let CT (s, ϵ) be the ϵ-good arms in the output. Assume s > g(ϵ) with g(ϵ) being the number of
ϵ-good arms. Let q be a parameter satisfying g(ϵ) < q < s and dq = µg(ϵ) − µq+1 , then
d2q T ⎞
⎛ d2q T ⎞
⎛
g(ϵ)
n−q
P(∣CT (s, ϵ)∣ ≤ g(ϵ) − k) ≤ (
) exp −k
+(
.
) exp −(s − q + k)
k
8n ⎠
8n ⎠
s−q+k
⎝
⎝
Especially for Sequential Halving where s = n2 , choosing q = n4 we have
3n
d2q T ⎞
⎞
⎛
⎛ d2q T ⎞
⎛ n
g(ϵ)
n
+ ( n 4 ) exp −( + k)
.
P ∣CT ( , ϵ)∣ ≤ g(ϵ) − k ≤ (
) exp −k
2
8n ⎠
8n ⎠
k
+k
⎠
⎝
⎝
⎝ 4
4

Proof. The event
{∣CT (s, ϵ)∣ ≤ g(ϵ) − k}
implies that there are at least k ϵ-good arms eliminated. Since g(ϵ) < q < s, the s output arms include at most q − k arms
that are from the best q arms {i ∣ µi ≥ µq }. Such that, the s output arms include at least s − q + k arms that are from the
worst n − q arms {i ∣ µi < µq }; otherwise, the number of outputs is even less than s. Denote the set of eliminated arms that
are ϵ-good as A′ , and the set of output arms that are from the worst n − q arms as B ′ , we have ∣A′ ∣ ≥ k, ∣B ′ ∣ ≥ s − q + k. We
then have the existence of A, B satisfying
{∃A ⊂ [g(ϵ)] , ∣A∣ = k, ∃B ⊂ [n] / [q] , ∣B∣ = s − q + k, s.t., ∀i ∈ A, j ∈ B, µ̂i ≤ µ̂j }.
The above implication can also be seen as a generalization of the technique of Karnin et al. (2013). We then have
{∃A ⊂ [g(ϵ)] , ∣A∣ = k, ∃B ⊂ [n] / [q] , ∣B∣ = s − q + k, s.t., ∀i ∈ A, j ∈ B, µ̂i ≤ µi −
By the distributive law of Boolean algebra, we have at least one of the following two holds,
{∃A ⊂ [g(ϵ)] , ∣A∣ = k, s.t., ∀i ∈ A, µ̂i ≤ µi −

27

dq
},
2

dq
dq
or µ̂j ≥ µj + }.
2
2

## Page 28

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

or
{∃B ⊂ [n] / [q] , ∣B∣ = s − q + k, s.t., j ∈ B, µ̂j ≥ µj +

dq
}.
2

Thereby the probability of {∣CT (s, ϵ)∣ ≤ g(ϵ) − k} can be upper bounded as
P(∣CT (s, ϵ)∣ ≤ g(ϵ) − k)
≤ P(∃A ⊂ [g(ϵ)] , ∣A∣ = k, s.t., ∀i ∈ A, µ̂i ≤ µi −

dq
)
2

+ P(∃B ⊂ [n] / [q] , ∣B∣ = s − q + k, s.t., j ∈ B, µ̂j ≥ µj +

dq
)
2

⎛
⎛
dq 2 ⎞
dq 2 ⎞
)
T
) T⎟
(
(
g(ϵ)
n
−
q
2
2
⎜
⎟
⎜
⎟+(
⎟
≤(
) exp⎜−k
) exp⎜−(s − q + k)
⎜
⎜
k
2n ⎟
2n ⎟
s−q+k
⎝
⎠
⎠
⎝
d2q T ⎞
⎛
⎛ d2q T ⎞
n−q
g(ϵ)
+(
.
) exp −(s − q + k)
=(
) exp −k
8n ⎠
8n ⎠
s−q+k
k
⎝
⎝

B.11. Lower bound for uniform sampling
Theorem 10. Consider a family E(n, m, ϵ) of all two-level univariate Gaussian bandit instances with n arms, of which m
arms have mean ϵ > 0 and all other arms have to mean 0. We consider the problem in which samples are drawn from some
instance θ ∈ E(n, m, ϵ) in an off-line, uniform fashion with B samples from each arm (i.e., with a total sampling budget of T
samples, B = T /n ignoring integer effects) and a subset of arms Ŝ ⊆ [n] of size s is then chosen based only on the observed
samples (i.e., without knowledge of θ) in a symmetric fashion. Here ‘symmetric’ is taken to mean that for any subset S ⊆ [n],
Pθ (Ŝ = S) = Pσ(θ) (Ŝ = σ(S)) where σ ∈ Σn is an element of the permutation group on sets of size n, and σ(S) is taken
to mean the element-wise application of σ to the elements of S.1 That is to say that the choice of Ŝ is independent of the
specific indices of the arms chosen.
6
Suppose m ≤ s ∧ n/3 and s < 11
n. Let M̃θ be the number of false negatives (i.e. ∣∣ ≥) Then, there exists θ ∈ E(n, m, ϵ) s.t.
3
⎛
16 m
⎛ ⎛
⎞⎞
1
6
n
−
s
ln(en/m) ⎞
1
2 ⎟
⎜
⎟⎟
,
2
(
⋅
)
exp
−
1
+
16⋅
Pθ (M̃θ ≥ m) ≥ min ⎜
mBϵ
⎜2
4
5 s
)⎠
ln ( 65 ⋅ n−s
⎝ ⎝
⎠⎠
s
⎝

where Pθ is the probability measure induced by sampling from instance θ.
Proof. Let θ = (ϵ, . . . , ϵ, 0, . . . , 0) ∈ Rn with ϵ > 0 be a vector of mean rewards for each arm where the first m coordinates
are nonzero. Let Ŝ ∈ [n] be the output of the algorithm (recall ∣Ŝ∣ = s). Let M̃θ be the number of false negatives when
taking Ŝ as the prediction for the true support [s] of θ. Let Q̃a ⊂ 2[n] be the collection of all subsets of [n] of size s such
that M̃θ = a. For convenience, let γ = 18 m. Let k = 34 s.
k

ξ ∶= Pθ (M̃θ ≥ m − k) = ∑ Pθ (M̃θ = a)
a=m−k
5γ

≥ ∑ Pθ (M̃θ = a).
a=3γ
1
Note that the assumption of a symmetric selection Ŝ is equivalent to considering a lower bound for general selections Ŝ in the face of
a uniform mixture over all permutations of θ (see Simchowitz et al. (2017) for a more in-depth discussion)

28

## Page 29

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Let first(Q̃a ) be the first member of Q̃a in lexicographic order. For example, first(tQa ) = [a + 1 ∶ a + s] where [A ∶ B] ∶=
{A, A + 1, . . . , B}. Then, by symmetry, one can see that Pθ (M̃θ = a) = ∣Q̃a ∣ Pθ (Ŝ = first(Q̃a )).
We consider the event on which the ‘empirical KL-divergence’ concentrates, which can be shown to be true with probability
at least 1 − δ by Jun and Zhang (2020, Lemma 5):
⎫
⎧
n
⎪
⎪
pθ′ (Xi,t ∣At = i)
1
⎪n B
⎪
) − (1 + ρ)B ∑ KL(θi′ , θi ) ≤ ln(1/δ)⎬.
conc(θ′ , θ) ∶= ⎨∑ ∑ ln(
⎪
⎪
p
(X
∣A
=
i)
ρ
θ
i,t
t
⎪
⎪
i=1
⎩i=1 t=1
⎭
We now employ a change of measure argument to Pθ (Ŝ = first(Q̃a )) and switch θ with θ′ that would result in M̃θ′ =
a − 3γ =∶ b. For a ∈ [3γ ∶ 5γ], this can be achieved with the choice of θ′ = (0, . . . , 0, ϵ, . . . , ϵ, 0, . . . , 0) that is supported on
[3γ + 1 ∶ m + 3γ]. Let Θ be the set of permutations of θ. Then,
Pθ (M̃θ = a)
= ∣Q̃a ∣ Pθ (Ŝ = first(Q̃a ))
≥ ∣Q̃a ∣ Pθ′ (Ŝ = first(Q̃a ), conc(θ′ , θ)) exp(−(1 + ρ)2mB⋅(ϵ2 /2) − ρ−1 ln(1/δ))

(change of measure; ρ > 0)

≥ ∣Q̃a ∣ Pθ′ (Ŝ = first(Q̃a ), ∩σ∈Σ conc(θ′ , σ(θ′ ))) exp(−(1 + ρ)mBϵ2 − ρ−1 ln(1/δ))

(Σ: symmetric group of [n])

2

≥ ∣Q̃a ∣ Pθ (Ŝ = first(Q̃b ), ∩σ∈Σ conc(θ, σ(θ))) exp(−(1 + ρ)mBϵ − ρ
=
≥

−1

ln(1/δ))

∣Q̃a ∣
Pθ (M̃θ = b, ∩σ∈Σ conc(θ, σ(θ))) exp(−(1 + ρ)mBϵ2 − ρ−1 ln(1/δ))
∣Q̃b ∣

(symmetry)
(symmetry)

∣Q̃a ∣
(Pθ (M̃θ = b) − ∣Θ∣δ) exp (−(1 + ρ)mBϵ2 − ρ−1 ln(1/δ)) .
∣Q̃b ∣
(P(A, B) ≥ P(A) − P(B c ); union bound over {σ(θ) ∶ σ ∈ Σ})

Thus,
5γ

∑ Pθ (M̃θ = a)
a=3γ

⎞
⎛
⎟
∣Q̃a ∣ ⎜
⎜Pθ (M̃θ ∈ [0 ∶ 2γ]) −(2γ + 1)∣Θ∣δ ⎟ exp (−(1 + ρ)mBϵ2 − ρ−1 ln(1/δ)) .
≥ min
⎟
⎜
a∈[3γ∶5γ] ∣Q̃a−3γ ∣ ⎜´¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¸ ¹ ¹¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¶
⎟
⎠
´¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¸ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹ ¹¹ ¹ ¹ ¹ ¶ ⎝
≥1−ξ
=∶ Y
1−ξ
Let us choose δ = 21 ⋅ (2γ+1)∣Θ∣
. Since the LHS above is at most ξ, we have

ξ≥Y

⎛
1−ξ
2(2γ + 1)∣Θ∣ ⎞
exp −(1 + ρ)mBϵ2 − ρ−1 ln (
)
2
1−ξ
⎝
⎠

One can consider two cases, namely ξ ≥ 12 and ξ < 21 , to arrive at
1
ξ ≥ min ( , exp (−(1 + ρ)mBϵ2 − ρ−1 ln (4(2γ + 1)∣Θ∣) + ln(4Y )))
2
It remains to find an appropriate value of ρ. One simple choice is
ρ−1 =

1
ln(4Y )
,
2 ln (4(2γ + 1)∣Θ∣)

which satisfies that ρ−1 > 0 as show later. Thus, we have
⎛
⎛ ⎛
⎞⎞
√
ln (4(2γ + 1)∣Θ∣) ⎞
1
ξ ≥ min ⎜
mBϵ2 ⎟⎟
⎜ 2 , 2 Y exp ⎜− ⎝1 + 2
⎟
ln(4Y )
⎠
⎝
⎠⎠
⎝
29

## Page 30

Revisiting Simple Regret: Fast Rates for Returning a Good Arm
m
n−m
)(s−(m−a)
). So, for a ∈ [3γ ∶ 5γ] and b = a − 3γ,
It remains to figure out bounds for Y and ∣Θ∣. For Y , note that ∣Q̃a ∣ = (m−a

min
a

m
n−m
∣Q̃a ∣ (m−a)(s−m+a)
= m
n−m
)
∣Q̃b ∣ (m−b)(s−m+b

=

(m − b)(m − b − 1)⋯(m − a + 1) (n − s − b)(n − s − b − 1)⋯(n − s − a + 1)
⋅
a(a − 1)⋯(b + 1)
(s − m + a)(s − m + a − 1)⋯(s − m + b + 1)

m−b
≥ (
)
a

(a)

3γ

6
≥( )
5
Ô⇒ Y =

a−b

n−s−b
⋅(
)
s−m+a
3γ

⋅(

n − s − 2γ
)
s − 3γ

a−b

3γ

6
≥( )
5

6
∣Q̃a ∣
≥( )
a∈[3γ∶5γ] ∣Q̃a−2γ ∣
5

3γ

min

⋅(

⋅(

n − s 3γ
)
s

n − s 3γ
)
s

> 1 implies (m − b − i)/(a − i) ≥ (m − b)/a for i ∈ [0 ∶ a − b − 1] and, with a similar
where (a) is due to the fact that m−b
a
n−s−a+3γ
reasoning, n ≥ s/2 Ô⇒ s−m+a > 1 Ô⇒ (n − s − b − i)/(s − m + a − i) ≥ (n − s − b)/(s − m + a). Then, using
m
n
) ≤ ( en
) , we have
∣Θ∣ = (m
m
ρ=2

ln (4(2γ + 1)∣Θ∣)
ln(4Y )

≤2

)
ln(m + 4) + m ln ( en
m
)
ln(4) + m
ln ( 56 ⋅ n−s
4
s
)
m ln ( en
m

(a)

≤4

)
ln(4) + m
ln ( 56 ⋅ n−s
4
s

≤ 16

ln(en/m)
)
ln ( 65 ⋅ n−s
s

where (a) is by m ≤ n/3 Ô⇒ ln(m + 4) ≤ m ln(en/m).
Altogether,
3
m
⎛
⎛ ⎛
⎞⎞
1
1
6 n − s 16
ln(en/m) ⎞
2 ⎟
⎜
Pθ (M̃ ≥ m) ≥ min ⎜ , 2 ( ⋅
)
exp ⎜− 1 + 16⋅
mBϵ ⎟⎟
4
2
5 s
)⎠
ln ( 65 ⋅ n−s
⎝ ⎝
⎠⎠
s
⎝

6
> 1, which is true for s < 11
n.
To verify that our choice of ρ is nonnegative, it suffices to show that 56 ⋅ n−s
s

B.12. A minimax optimal budget allocation scheme for SH
T
)
We show that with a different budget allocation scheme Tℓ = ⌊ 81n
⋅ ( 16
9

ℓ−1

⋅ ℓ⌋, SH can achieve a sample complexity of

O( ϵn2 log 1δ ) (without any logarithmic terms of n). We call this budget allocation scheme Option 2 and the original one of
Karnin et al. (2013) Option 1. The price of achieving this exact minimax optimality is the potential loss of the near instance
optimality (or at least the standard proof technique does not work). We summarize the comparison in Table 2. It is not clear
to us whether there exists a different value of Tℓ in SH that can achieve both the minimax and instance-dependent optimality.
It is also not clear whether Option 2 can achieve a similar bound as Theorem 1. Note that for the fixed budget setting, there
is still a log log(n) gap between the instance-dependent upper and lower bounds.
Theorem 11. For any ϵ ∈ (0, 1), with Option 2, there exists an absolute constant c1 , such that the error probability of SH
for identifying an ϵ-good arm satisfies,
⎛
ϵ2 T ⎞
P(µJT < µ1 − ϵ) ≤ exp −Θ(
) ,
n ⎠
⎝
for T > c1 ⋅ ϵn2 .
30

## Page 31

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Proof. Still let ϵ1 = ϵ/4. And define ϵℓ+1 = 34 ⋅ ϵℓ . For each stage ℓ, define the event Gℓ as
Gℓ ∶= { max µi ≥ max µi − ϵℓ } .
i∈Sℓ+1

i∈Sℓ

We have
log n

∞
3
∑ ϵℓ < ∑ ( )
4
ℓ=1
ℓ=1

ℓ−1

ℓ−1

⋅ ϵ1 =

ϵ ∞ 3
∑( )
4 ℓ=1 4
ℓ−1

∑ Tℓ ⋅ n2
ℓ=1

T
⋅ ℓ ⋅ n2−(ℓ−1) = 81
⋅ ( 89 )

ℓ−1

8
T
⋅( )
≤∑
81
9
ℓ=1
∞

−(ℓ−1)

ϵ
1 − (3/4)n
lim
= ϵ.
4 n→∞ 1 − 3/4
ℓ−1

T
)
For stage ℓ, we assign budget Tℓ ⋅ n2−(ℓ−1) ∶= 81n
⋅ ( 16
9
scheme does not exceed the budget as follows,
log n

≤

⋅ ℓ. We can verify this allocation

ℓ

1 − (8/9)
T
⋅ℓ≤
lim
= T.
9 ℓ→∞ 1 − 8/9

Let aℓ be the best arm in Sℓ ,
P(Gcℓ ) = P(Gcℓ , µ̂aℓ < µaℓ − ϵℓ /2) + P(Gcℓ , µ̂aℓ ≥ µaℓ − ϵℓ /2)
≤ P(µ̂aℓ < µaℓ − ϵℓ /2) + P(Gcℓ ∣ µ̂aℓ ≥ µaℓ − ϵℓ /2).
For the first term
ϵ2 Tℓ
P(µ̂aℓ < µaℓ − ϵℓ /2) ≤ exp(− ℓ
)
2 ∣Sℓ ∣
⎛ ϵ21 ( 9 )ℓ−1 T ⋅ ( 16 )ℓ−1 ⋅ ℓ ⋅ n2−(ℓ−1) ⎞
16
81n
9
⎟
≤ exp⎜−
2
n2−(ℓ−1)
⎝
⎠
≤ exp(−

ϵ2 T
ℓ).
32 ⋅ 81n

For the second term,
P(Gcℓ ∣ µ̂aℓ ≥ µaℓ − ϵℓ /2) ≤ P(∣{i ∈ Sℓ ∣ µ̂i > µi + ϵℓ /2}∣ ≥ ∣Sℓ ∣/2)
E[∣{i ∈ Sℓ ∣ µ̂i > µi + ϵℓ /2}∣]
≤
∣Sℓ ∣/2
2

ϵ T
∣Sℓ ∣ exp(− 32⋅81n
ℓ)

≤
∣Sℓ ∣/2
= 2 exp(−

ϵ2 T
ℓ).
32 ⋅ 81n

By contraposition,
log n

P(µJT < µ1 − ϵ) ≤ ∑ P(Gcℓ )
ℓ=1
∞

≤ ∑ 3 exp(−
ℓ=1

ϵ2 T
ℓ).
32 ⋅ 81n

2

ϵ T
Let X ∶= 32⋅81n
> 0,
∞

P(µJT < µ1 − ϵ) ≤ ∑ 3 exp(−ℓX)
ℓ=1

1 − exp(−ℓX)
1 − exp(−X)
1
≤ 3 exp(−X)
.
1 − exp(−X)

≤ 3 exp(−X)

31

## Page 32

Revisiting Simple Regret: Fast Rates for Returning a Good Arm
Table 2. The ϵ-error probability of SH (worst-case) and (ϵ = 0)-error probability (instance-dependent) for Option 1 and 2 where
H2 ∶= maxi ∆i2 is the problem hardness parameter. Here, we omit constant factors. Check marks for the worst-case means that it is
i

optimal, and those for the instance-dependent means that it is optimal up to log log(n) factors in the sample complexity (Mannor and
Tsitsiklis, 2004; Carpentier and Locatelli, 2016).

Worst-case

Instance-dependent
2

T
Option 1: Tℓ = ⌊ ∣S ∣⌈log
⌋
n⌉

ϵ T
✗ log2 n⋅ exp(− n log
)
n

✓ log2 n⋅ exp(− log TnH2 )

ℓ−1
T
) ⋅ ℓ⌋
⋅ ( 16
Option 2: Tℓ = ⌊ 81n
9

2
✓ exp(− ϵ nT )

✗ log2 n⋅ exp(− n logT nH2 )

ℓ

2

2

2

2

1
is a monotonically decreasing function, there exists an absolute constant c1 , such that
As 1−exp(−X)

⎛
ϵ2 T ⎞
P(µJT < µ1 − ϵ) ≤ exp −Θ(
) ,
n ⎠
⎝
holds for T > c1 ⋅ ϵn2 .

C. Simple Regret in the Data-Poor Regime
C.1. The number of opened brackets
Lemma 11.1. The number of opened brackets till round t, denoted by Lt , satisfies
0.63⋅ log2 (1 + ln(2)t) < Lt ≤ 1 + log2 (1 + ln(2)t).
The size of the largest bracket satisfies
2Lt ≥ Θ̃(t).
Proof. Recall that whenever t ≥ B2B is true, we open a new bracket and increase B by 1. B = 0 before the game starts.
First, note that for t = 1 we have Lt = 1. Then, let us focus on t ≥ 2. It is not hard to verify that
Lt = max{B ∶ t ≥ (B − 1)2B−1 }.
Fix t ≥ 2. Then, we can define
t′ ∶= (Lt − 1)2Lt −1 ,
the first time point that the bracket Lt was opened. Clearly t′ ≤ t. Solve (15) for Lt .
t′ = (Lt − 1) exp(ln(2)(Lt − 1)),
ln(2)t′ = ln(2)(Lt − 1) exp(ln(2)(Lt − 1))
=∶ X exp(X).
Solving it for X is exactly the Lambert W function. By Lemma 17 of Orabona and Pal (2016), we have
0.6321 ⋅ ln(1 + ln(2)t′ ) ≤ X ≤ ln(1 + ln(2)t′ ).
This implies that
X = ln(2)(Lt − 1) ≤ ln(1 + ln(2)t′ ) ≤ ln(1 + ln(2)t) Ô⇒ Lt ≤ 1 + log2 (1 + ln(2)t).
To obtain a lower bound, define
t′′ ∶= Lt 2Lt ,
32

(15)

## Page 33

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

which is strictly larger than t. Using similar techniques as above, we can derive
Lt ≥ 0.6321 ⋅ log2 (1 + ln(2)t′′ ) > 0.63 ⋅ log2 (1 + ln(2)t).
Together, we have
0.63 ⋅ log2 (1 + ln(2)t) < Lt ≤ 1 + log2 (1 + ln(2)t).
Note that we have
Lt 2Lt ≤ t ≤ (Lt + 1)2Lt +1 .
Thus
2Lt ≥

t
t
t
≥
≥
= Θ̃(t).
2(Lt + 1) 2(2 + log (1 + ln(2)t)) 4 log2 t

(t ≥ 4)

2

C.2. The number of pulls for the representative arms
Lemma 11.2. At round t, the number of times the representative arm of bracket B has been pulled in the latest finished SH,
denoted as DB,t , satisfies,
DB,t ≥ const⋅

t
.
ln(t)⋅ log2 (n)

Proof. We call the round interval (Lt − 1)2Lt −1 ≤ t < Lt 2Lt as phase Lt , whose length is Lt 2Lt − (Lt − 1)2Lt −1 =
Lt −1
(Lt + 1)2Lt −1 . Correspondingly, sampling budget of (Lt +1)2
is assigned to each bracket. For each opened bracket
Lt
AB , B ∈ [Lt − 1], the sampling budget for it is accumulated from phase B to phase Lt , while phase Lt is not finished yet.
The sampling budget bracket AB receives satisfies
Lt −1

2i−1 (i + 1) t − 2Lt −1 (Lt − 1)
+
i
Lt
i=B

TB,t = ∑

2Lt −2 Lt t − 2Lt −1 (Lt − 1)
+
Lt − 1
Lt
t
>
4Lt
t
t
>
>
.
4 + 4 log(1 + ln(2)t) 4 ln t

>

(t > 16)

Recall we run the SH with the doubling trick on each bracket individually. We claim that the latest finished SH receives
a budget of at least TB,t /4. To see this, note that we initialize Algorithm 3.1 with budget starting from ⌈n log2 n⌉. Then
restart the SH algorithm with budget 2⋅⌈n log2 n⌉, and so on. Before finishing the k-th doubling trick, the best arm from
(k − 1)-th doubling trick is output. Thus the portion of the lasted finished SH ranges in
⎡
⎢ ∣Tk−1 ∣
∣Tk−1 ∣ ⎞
⎢
, k−1
,
⎢ k
⎢ ∑i=1 ∣Ti ∣ ∑i=1 ∣Ti ∣ ⎠
⎣
which, by some simple calculations, is
[

br−1
br−1
,
).
4br−1 − b0 2br−1 − b0

(16)

That is to say, the most recently finished SH, which gives the current output, has a budget of at least TB,t /4. We notice that
TB,t is essentially irrelevant to B. This means each bracket receives an order-wise equal amount of budget to query the SH
33

## Page 34

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

algorithm. In the SH algorithm, the output arm is pulled a fixed number of times when the number of arms and the sampling
budget is fixed. It is pulled Θ( logT n ) times for budget T . We have all we need to determine the lower bound for DB,t .
2

DB,t ≥ const⋅

t
.
ln(t)⋅ log2 (n)

C.3. Lower bounds and their implications in special instances
While Katz-Samuels and Jamieson (2020) shows a lower bound for the (ϵ, δ)-unverifiable sample complexity that can match
the upper bound in special instances, the final result they presented has a negative term, which results in looseness, and
the bound can even go to 0 in certain instances. Recall the definition of τϵ,δ , the (ϵ, δ)-unverifiable sample complexity
(Definition 1).
Theorem 12 (Theorem 1 of Katz-Samuels and Jamieson (2020)). Fix ϵ > 0, δ ∈ (0, 1/16), and a vector µ ∈ Rn . Consider n
arms where rewards from the i-th arm are distributed according to N (µi , 1). For every permutation π ∈ Sn , let (Ftπ )t∈N be
the filtration generated by the algorithm playing on instance π(µ). Then the (ϵ, δ)-unverifiable sample complexity satisfies,

E[τϵ,δ ] ≥

n
⎞
1⎛
1
−2
−2
⎜−(µ1 − µg(ϵ)+1 ) +
∑ (µ1 − µi ) ⎟,
64 ⎝
g(ϵ) i=g(ϵ)+1
⎠

where the expectation is with respect to π ∈ Sn and π(µ).
In fact, the proof of Katz-Samuels and Jamieson (2020, Theorem 1) shows a stronger bound, which is more useful for
deriving lower bounds for specific instances.
Theorem 13 (A stronger version of Theorem 1 of Katz-Samuels and Jamieson (2020)). Fix ϵ > 0, δ ∈ (0, 1/16), and a vector
µ ∈ Rn . Consider n arms where rewards from the i-th arm are distributed according to N (µi , 1). For every permutation
π ∈ Sn , let (Ftπ )t∈N be the filtration generated by the algorithm playing on instance π(µ). Then the (ϵ, δ)-unverifiable
sample complexity satisfies,
E[τϵ,δ ] ≥

n
1 1
−2
∑ (µ1 − µi ) ,
64 g(ϵ) i=2g(ϵ)+1

where the expectation is with respect to π ∈ Sn and π(µ).
Corollary 13.1. For the EqualGap(m) instance, Proposition 13 shows a lower bound of the (ϵ, δ)-unverifiable sample
complexity as
E[τπ ] ≥

1 n 1
2
(
− ).
100 m ϵ2 ϵ2

Proof.
n
n
1 1
1 1
16
−2
∑ (µ1 − µi ) =
∑
64 g(ϵ) i=2g(ϵ)+1
64 m i=2m+1 25ϵ2

1 n − 2m 16
64 m 25ϵ2
1 n 1
2
=
(
− ).
100 m ϵ2 ϵ2
=

Corollary 13.2. For the Polynomial(α) instance, Proposition 13 shows a lower bound of the (ϵ, δ)-unverifiable sample
complexity as
E[τπ ] ≥

1
1
1
(21−2α ϵ−2 − ϵ− α ).
128 2α − 1

34

## Page 35

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Proof. For the Polynomial(α) instance,
−2α

n
n
1 1
i
1 1
−2
∑ (µ1 − µi ) =
∑ ( )
64 g(ϵ) i=2g(ϵ)+1
64 g(ϵ) i=2g(ϵ)+1 n

2α

=

n
1 n2α
1
( )
∑
1/α
64 nϵ
i=2g(ϵ)+1 i

2α

n
1 n2α−1
1
=
( ) .
∑
64 ϵ1/α i=2g(ϵ)+1 i

The summation of ∑ni=2g(ϵ)+1 ( 1i )

2α

n

can be bounded as

1
( )
i=2g(ϵ)+1 i

2α

∑

2α

n

(a1 ) 1

1
( ) di
∫
2 2g(ϵ) i

≥

1
= (f (n) − f (2g(ϵ)))
2
1
1
1
= (
n−2α+1 −
(2g(ϵ))−2α+1 )
2 −2α + 1
−2α + 1

1
x−2α+1 )
(f (x) = −2α+1

1
1
1
n−2α+1 −
(2nϵ1/α )−2α+1 ),
= (
2 −2α + 1
−2α + 1
where (a1 ) is by the geometric meaning of integral and
1 1
( )
2 i

2α

1
=(

1

2α

)

2α

≤(

2 2α i

1
)
2i

2α

≤(

1
) .
i+1

Thus
2α

n
n
1 1
1
1 n2α−1
−2
( )
∑
∑ (µ1 − µi ) =
1/α
64 g(ϵ) i=2g(ϵ)+1
64 ϵ
i=2g(ϵ)+1 i

≥

1
1 n2α−1
1
n−2α+1 −
(2nϵ1/α )−2α+1 )
(
1/α
128 ϵ
−2α + 1
−2α + 1

1 1
1
21−2α −2α+1
(
−
ϵ α )
1/α
128 ϵ
−2α + 1 −2α + 1
−2α+1
1 − α1 1
=
ϵ
(21−2α ϵ α − 1)
128
2α − 1
1
1
1
=
(21−2α ϵ−2 − ϵ− α ).
128 2α − 1
=

C.4. Proof of Theorem 6
Theorem 6. For any ϵ ∈ (0, 1), the ϵ-error probability of BSH satisfies
⎛ ⎛
⎞⎞
t
⎜
⎟⎟
P(µ1 − µJt > ϵ) ≤ exp⎜
−
Θ̃
⎜
⎟,
1
+ maxi≥g( ϵ )+1 ∆i2 ⋅ g(1ϵ ) ⎠
⎝ ⎝ ϵ2
⎠
2
i
4
for t ≥ Θ̃(H2 ( 2ϵ )).
35

(i ≥ 1, α > 0.5)

## Page 36

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Proof. Due to the doubling nature of the bracketing scheme, it is not hard to see that each bracket receives an order-wise
equal amount of sampling budget Θ̃(t) (details in Appendix C.2). Our analysis is centered around finding the ideal bracket
whose representative arm is expected to be of the highest quality, which we call the best bracket. Intuitively, on the one
hand, if the best bracket is too large, the per-arm sampling budget will be too small to guarantee a meaningful output. On
the other hand, if the best bracket is too small, it may not well represent the entire instance. The core idea of the proof is
balancing this trade-off.
We define the following notations.
• A∗ : the best bracket, which is a multiset satisfying ∀i ∈ A∗ , i ∈ [n]. We have ∣A∗ ∣ = L with L being specified later.
• r= L
: the subsampling ratio of the best bracket A∗ .
n
• ϵk ∶= 2k−1 ⋅ 4ϵ , for k ∈ [K], where K = ⌈log2 ( 8ϵ )⌉. Also K satisfies ϵK−1 ≤ 1, ϵK ≥ 1.
• Zk = {i ∈ [n] ∣ ∆i ∈ [0, min {ϵk , 1}]}.
• g(ϵk ) = ∣Zk ∣.
• Zk′ = {i ∈ A∗ ∣ ∆i ∈ [0, min {ϵk , 1}]}.
• g ′ (ϵk ) = ∣Zk′ ∣.
• µi (A): the mean reward of the i-th best arm in A.
• ∆′i = max {µj (A∗ ) ∣ j ∈ A∗ } − µi (A∗ ): the suboptimality gap of i-th best arm in A∗ with respect to the best arm in
A∗ .
• ∆′′i = µ1 − µi (A∗ ): the suboptimality gap of i-th best arm in A∗ with respect to the best arm in the entire instance.
• U1 ( 2ϵ ) = maxi≥g(ϵ2 )+1 ∆i2 , and U2 ( 2ϵ ) = maxi≥g′ (ϵ2 )+1 ∆i′ 2 . We call this the un-accelerated sample complexity on the
i

i

entire instance and the best bracket for identifying an 2ϵ -good arm.
We define the following events:
′

k)
k)
− g(ϵnk ) ∣ < g(ϵ
.
• E1 : ∀k ∈ [K], ∣ g (ϵ
L
2n

• E2 : an 2ϵ -good arm represents the best bracket.
The event E1 ensures that the best bracket can well represent the entire arm set in the sense that, for each peeling piece k, the
′
k)
ratio of the arms therein to all the arms in the best bracket g (ϵ
is close to its original ratio g(ϵnk ) . The ϵ-error probability
L
can now be bounded as
P(µJt < µ1 − ϵ) = P(µJt < µ1 − ϵ, E1c ) + P(µJt < µ1 − ϵ, E1 , E2c ) + P(µJt < µ1 − ϵ, E1 , E2 )
≤ P(E1c ) + P(E1 , E2c ) + P(µJt < µ1 − ϵ, E2 )
≤ P(E1c ) + P(E2c ∣ E1 ) + P(µJt < µ1 − ϵ ∣ E2 ).
36

(17)

## Page 37

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

We bound each term separately. For the first term.
⎛
g ′ (ϵk ) g(ϵk )
g(ϵk ) ⎞
P(E1c ) = P ∃k, ∣
−
∣>
L
n
2n ⎠
⎝
K
⎛ g ′ (ϵk ) g(ϵk )
g(ϵk ) ⎞
≤∑P ∣
−
∣>
L
n
2n ⎠
k=1 ⎝
K

≤ ∑ P(
k=1
K

≤ ∑ P(
k=1

g ′ (ϵk ) 3g(ϵk )
g ′ (ϵk ) g(ϵk )
<
or
>
)
L
2n
L
2n
K
g ′ (ϵk ) g(ϵk )
g ′ (ϵk ) 3g(ϵk )
<
) + ∑ P(
>
).
L
2n
L
2n
k=1

Let KL(p, q) be the Kullback-Leibler divergence between two Bernoulli distributions with parameters p, q. The first
summation can be upper bounded by the additive Chernoff bound for binomial distribution (Arratia and Gordon, 1989),
K

∑ P(
k=1

K
⎛
g ′ (ϵk ) g(ϵk )
g(ϵk ) g(ϵk ) ⎞
<
) ≤ ∑ exp −LKL(
,
)
L
2n
2n
n
⎝
⎠
k=1
2

K
⎛ 1 g (ϵ2k ) ⎞
n
⎟
≤ ∑ exp⎜− L g(ϵ
k)
8
k=1
⎠
⎝
n

2

(p−q)
(KL(p, q) ≥ 2 max{p,q}
)

K
1 g(ϵk )
= ∑ exp(− L
).
8
n
k=1

The second summation can be upper bounded by
K

∑ P(
k=1

K
⎛
3g(ϵk )
g(ϵk ) ⎞
g ′ (ϵk ) 3g(ϵk )
>
) ≤ ∑ exp −LKL(1 −
,1 −
)
L
2n
2n
n
⎠
⎝
k=1
(a1 ) K

⎛
3g(ϵk ) g(ϵk ) ⎞
≤ ∑ exp −LKL(
,
)
2n
n
⎝
⎠
k=1
2

⎛ 1 g (ϵ2k ) ⎞
n
⎟
≤ ∑ exp⎜− L g(ϵ
k)
12
k=1
⎝
⎠
n
K

K

≤ ∑ exp(−
k=1

2

(p−q)
(KL(p, q) ≥ 2 max{p,q}
)

1 g(ϵk )
L
),
12
n

where (a1 ) is by the definition of KL divergence for Bernoulli distribution. Since g(ϵk ) is a non-decreasing function of k,
we have
K
K
1 g(ϵk )
1 g(ϵk )
P(E1c ) ≤ ∑ exp(− L
) + ∑ exp(− L
)
8
n
12
n
k=1
k=1
⎡
1 g(ϵ1 )
8 ⎤⎥
⎢
≤2⎢log2 ( )⎥ exp(− L
).
⎢
⎥
ϵ
12
n
⎢
⎥
nt
We choose L = Θ̃( g(ϵ )H
). Then
(ϵ)
1

2

2

⎡
⎛ ⎛ t ⎞⎞
8 ⎤⎥
⎢
⎟
P(E1c ) ≤2⎢log2 ( )⎥ exp⎜−Θ̃
ϵ
⎢
ϵ ⎥⎥
⎝ ⎝ H2 ( 2 ) ⎠ ⎠
⎢
⎛ ⎛ t ⎞⎞
⎟.
≤ exp⎜−Θ̃
ϵ
⎝ ⎝ H2 ( 2 ) ⎠⎠

(t ≥ Θ̃(H2 ( 2ϵ )))
(18)

37

## Page 38

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Our choice of L here also ensures that a bracket of size L is opened because the largest bracket is in the size of Θ̃(t) by
Lemma 11.1. Furthermore,
⎛
⎞
⎞
⎛
nt
nt
nt
⎟ ≤ Θ̃(
= Θ̃⎜
L = Θ̃
) = Θ̃(∆2n t) ≤ Θ̃(t).
i
ϵ
2
n/∆
⎝ g(ϵ1 )H2 ( 2 ) ⎠
max
2
i≥g(ϵ2 )+1 ∆ ⎠
n
⎝
i

The second term of (17) means the probability that SH fails to return an 2ϵ -good arm from bracket A∗ given that bracket A∗
can well represent the entire instance. Denote H2′ ( 2ϵ ) as the sample complexity of returning an 2ϵ -good from the bracket A∗ ,
i.e.,
1
ϵ
i
H2′ ( ) = max
.
′2 ⋅
′
′
i≥g (ϵ2 )+1 ∆i g (ϵ1 )
2
Let H2 ( 2ϵ ) be the sample complexity of returning an 2ϵ -good arm from the entire instance, i.e.,
ϵ
1
i
H2 ( ) = max
⋅
.
2
i≥g(ϵ2 )+1 ∆i g(ϵ1 )
2
We use Theorem 1 to upper bound the second term of (17). To do so, we need to verify that the best bracket is allocated to
enough budget, i.e., the budget allocated to the best bracket should be no less than U2 ( 2ϵ ). By event E1 , we have
1
rg(ϵk ) ≤ g ′ (ϵk ) ≤ 2rg(ϵk ).
2

(19)

Then, by Lemma 13.1 and our choice of L, we have
⎛ H2′ ( 2ϵ ) ⎞
ϵ
ϵ
L
ϵ
ϵ
t = Θ̃(t).
U2 ( ) = g ′ (ϵ1 )H2′ ( ) ≤ 2rg(ϵ1 )H2′ ( ) = 2 g(ϵ1 )H2′ ( ) = Θ̃
2
2
2
n
2
⎝ H2 ( 2ϵ ) ⎠
Meanwhile, each bracket receives Θ̃(t) budget by the bracketing design. Thus the best bracket is allocated to enough budget.
By applying Theorem 1 to (a1 ) and Lemma 13.1 to (a2 ) below, we have
⎛ ⎛ t ⎞⎞ (a2 )
⎛ ⎛ t ⎞⎞
(a1 )
⎟ ≤ exp⎜−Θ̃
⎟.
P(E2c ∣ E1 ) ≤ exp⎜−Θ̃
ϵ
ϵ
′
⎝ ⎝ H2 ( 2 ) ⎠⎠
⎝ ⎝ H2 ( 2 ) ⎠ ⎠

(20)

The third term of (17) means an arm whose mean reward is less than µ1 − ϵ has a larger empirical reward than an arm whose
mean reward is larger than µ1 − ϵ/2. Note the recently opened brackets may not have finished their first SH yet. In this case,
the DSH always returns an empirical reward of negative infinity. Thus the arms returned from these brackets will never be
t
selected by BSH. Denote Ȧ(t) ∶= {a ∈ ⋃L
k=1,Ak ≠A∗ Ak , µa < µ1 − ϵ}. We have
Lt

∣Ȧ(t)∣ ≤ ∑ 2k ≤ 2⋅(21+log2 (1+ln(2)t) − 1) ≤ 8⋅(1 + ln(2)t).
k=1

By the number of arm pulls in Lemma 11.2,
P(µJt < µ1 − ϵ ∣ E2 ) ≤ P(∃a1 ∈ Ari∗ (t) , ∃a2 ∈ Ȧ(t), s.t., µa1 ≥ µ1 − ϵ/2, µ̂a1 < µ̂a2 )
≤∣Ȧ(t)∣∣Ari∗ (t) ∣ exp(−Θ̃(ϵ2 t))
≤ exp(−Θ̃(ϵ2 t) + ln(∣Ȧ(t)∣∣Ari∗ (t) ∣))
≤ exp(−Θ̃(ϵ2 t) + ln(8⋅(1 + ln(2)t)t))
≤ exp(−Θ̃(ϵ2 t) + O(ln t))
≤ exp(−Θ̃(ϵ2 t)).

(21)

Given (18), (20) and (21), we have the result proved.
38

## Page 39

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Lemma 13.1. Let H2′ ( 2ϵ ) be the complexity measure of returning an 2ϵ -good from the bracket A∗ , i.e.,
1
ϵ
i
.
H2′ ( ) = max
′2 ⋅
′
′
i≥g (ϵ2 )+1 ∆i g (ϵ1 )
2
Let H2 ( 2ϵ ) be the complexity measure of returning an 2ϵ -good from the entire instance, i.e.,
ϵ
1
i
H2 ( ) = max
⋅
.
i≥g(ϵ2 )+1 ∆2
2
g(ϵ
1)
i
Given the event E1 happens, we have
ϵ
ϵ
H2′ ( ) ≤ 64H2 ( ).
2
2
Proof. For the entire instance, by Lemma 13.2,
g(ϵk )
ϵ
4g(ϵk )
≤ U1 ( ) ≤ max
.
2
k∈{3,⋯,K} ϵk
k∈{3,⋯,K}
2
ϵ2k
max

(22)

For the best bracket, by Lemma 13.3, we have
max

i

′′

i≥g ′ (ϵ2 )+1 ∆i 2

≤

max

i

′

i≥g ′ (ϵ2 )+1 ∆i2

≤4

max

i

′′

i≥g ′ (ϵ2 )+1 ∆i 2

.

Note the term maxi≥g′ (ϵ2 )+1 ∆i′′ 2 can be considered the un-accelerated sample complexity on a virtual instance where all the
i

arms are the same as the ones of A∗ , except the best arm of A∗ is replaced as the best arm in the entire instance (if the best
arm of A∗ is same as the best arm in the entire instance, then the virtual instance is same as A∗ ). The above inequalities
essentially show that on event E1 , setting the best arm in A∗ to have the same mean reward as the best arm of the entire
instance does not change its un-accelerated sample complexity (up to a constant of 4). Denote
ϵ
i
U2′ ( ) = max
′′ .
i≥g ′ (ϵ2 )+1 ∆i 2
2
Thus,
ϵ
ϵ
ϵ
U2′ ( ) ≤ U2 ( ) ≤ 4U2′ ( ).
2
2
2

(23)

By Lemma 13.2,
g ′ (ϵk )
ϵ
4g ′ (ϵk )
≤ U2′ ( ) ≤ max
.
2
k∈{3,⋯,K}
k∈{3,⋯,K}
ϵk
2
ϵ2k
max

By event E1 , we have for k ∈ [K]
1
rg(ϵk ) ≤ g ′ (ϵk ) ≤ 2rg(ϵk ).
2

(24)

rg(ϵk )
ϵ
8rg(ϵk )
≤ U2′ ( ) ≤ max
.
2
k∈{3,⋯,K} 2ϵk
k∈{3,⋯,K}
2
ϵ2k

(25)

rg(ϵk )
ϵ
32rg(ϵk )
≤ U2 ( ) ≤ max
.
k∈{3,⋯,K}
2ϵ2k
2
ϵ2k

(26)

Thus
max

By (23) and (25),
max
k∈{3,⋯,K}

39

## Page 40

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Therefore, by (22) and (26), we have
U1 ( 2ϵ ) 8
1
≤
≤ .
32r U2 ( 2ϵ ) r

(27)

By U1 ( 2ϵ ) = g(ϵ1 )H2 ( 2ϵ ), U2 ( 2ϵ ) = g ′ (ϵ1 )H2′ ( 2ϵ ), (24) and (27), we have
g(ϵ1 )U2 ( 2ϵ )
2
ϵ
ϵ
ϵ
ϵ
H2′ ( ) =
H2 ( ) ≤ ⋅32r⋅H2 ( ) = 64H2 ( ).
ϵ
′
2
2
r
2
2
g (ϵ1 )U1 ( 2 )
Lemma 13.2. Let U ( 2ϵ ) be the un-accelerated sample complexity on any bandit instance for identifying an 2ϵ good arm
(note ϵ2 = 2ϵ ), i.e.,
ϵ
i
U ( ) = max
.
i≥g(ϵ2 )+1 ∆2
2
i
k)
Then up to a constant of 4, the un-accelerated sample complexity can be approximated by maxk∈{3,⋯,K} g(ϵ
, i.e.,
ϵ2
k

max
k∈{3,⋯,K}

g(ϵk )
ϵ
4g(ϵk )
≤ U ( ) ≤ max
.
k∈{3,⋯,K}
ϵ2k
2
ϵ2k

Proof. Let i0 ≥ g(ϵ2 ) + 1 satisfy
ϵ
i
i0
U ( ) = max
= 2 .
i≥g(ϵ2 )+1 ∆2
2
∆
i
i0
Denote k0 as the smallest index of the peeling interval to which i0 belongs, i.e., ϵk0 −1 < ∆i0 ≤ ϵk0 . Since i0 ≥ g(ϵ2 ) + 1, we
ϵ
ϵ
know k0 ≥ 3. By ϵk0 −1 = k20 , we have k20 < ∆i0 . Thus,
4g(ϵk )
4i0 4g(ϵk0 )
i0
ϵ
≤ max
,
U( ) = 2 < 2 ≤
k∈{3,⋯,K}
2
∆i0 ϵk0
ϵ2k0
ϵ2k
where the second inequality is by i0 ≤ g(ϵk0 ). By the definition of U ( 2ϵ ), we have
g(ϵk )
ϵ
g(ϵk )
U ( ) ≥ max
≥ max
k∈{3,⋯,K} ϵ2
k∈{3,⋯,K} ∆2
2
k
g(ϵk )
Thus,
max
k∈{3,⋯,K}

ϵ
g(ϵk )
4g(ϵk )
≤ U ( ) ≤ max
.
k∈{3,⋯,K}
ϵ2k
2
ϵ2k

Lemma 13.3. Given that the event E1 holds, for i ≥ g ′ (ϵ2 ) + 1, ∆′i the suboptimality gap of i-th best arm in the best bracket
with respect to the best arm in the best bracket can be approximated by ∆′′i , the suboptimality gap of i-th best arm in the
best bracket with respect to the best arm in the entire instance, i.e.,
1
4
1
≤
≤ ′′ .
′′
∆i 2 ∆′i 2 ∆i 2
Proof. The first inequality holds trivially, and the equality holds when the best arm out of the entire arm set is included in
the best bracket. For the second inequality, as the best arm in the best bracket is 4ϵ -good, we have
∆′i −

ϵ
ϵ ∆′′ ϵ
≥ ∆′′i − ≥ i − ,
4
2
2
4

which results in the second inequality.
40

## Page 41

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

C.5. Proof of Corollary 6.1
Corollary 6.1. Consider the EqualGap(m) instance. BUCB achieves an expected (ϵ, δ)-unverifiable sample complexity as
˜ ⎛ n log( 1 )⎞.
E[τϵ,δ ] ≤ O
δ ⎠
⎝ ϵ2 m
BSH achieves an (ϵ, δ)-unverifiable sample complexity as, with probability 1 − δ,
˜
τϵ,δ ≤ O

⎛ n
1 ⎞
log( ) .
δ ⎠
⎝ ϵ2 m

Proof. For BUCB, Theorem 7 of Katz-Samuels and Jamieson (2020) gives the following instance dependent upper bound
for the (ϵ, δ)-unverifiable sample complexity,
n
1 ⎞
n
1 ⎛g(ϵ)
−2
⎜ ∑ (∆i∨j,g(ϵ)+1 ) ln( ) + ∑ ∆−2
ln(
)⎟.
j,i
j∈[g(ϵ)] j ⎝ i=1
jδ
δ ⎠
i=g(ϵ)+1

E[τϵ,δ ] ≤ min
Plug the instance

5
µ1 = ϵ, µi = ϵ, ∀i ∈ {2, . . . , m}, µi = 0, ∀i ≥ m + 1 for some m ≥ 2, ϵ > 0 .
4
We easily have
˜
E[τϵ,δ ] ≤ O

⎛ n
1 ⎞
log( ) .
2
δ ⎠
⎝ϵ m

For BSH,
i
1
16n
⋅ ϵ =
.
2
i≥g(ϵ/2)+1 ∆i g( )
25mϵ2
4
max

C.6. Proof of Corollary 6.2
α

Corollary 6.2. Consider the Polynomial(α) instance; i.e., ∆i = ( ni ) with α > 0.5. For any ϵ ∈ (0, 1), let τ̂ϵ,δ be the upper
bound of the expected (ϵ, δ)-unverifiable sample complexity reported in Katz-Samuels and Jamieson (2020, Theorem 7) for
BUCB. Then, BUCB satisfies
⎛ 2α−1
1 ⎞
E[τϵ,δ ] ≤ τ̂ϵ,δ = Θ̃ ϵ− α n log( ) .
δ ⎠
⎝
On the other hand, BSH satisfies, with probability 1 − δ,
˜ ⎛ϵ−2 log( 1 )⎞ .
τϵ,δ ≤ O
δ ⎠
⎝
Proof. We first show the result for Bracketing UCB. Theorem 7 of Katz-Samuels and Jamieson (2020) gives the following
instance dependent upper bound for the (ϵ, δ)-unverifiable sample complexity,
n
1 ⎛g(ϵ)
n
1 ⎞
−2
⎜ ∑ (∆i∨j,g(ϵ)+1 ) ln( ) + ∑ ∆−2
j,i ln( )⎟.
j∈[g(ϵ)] j ⎝ i=1
jδ
δ ⎠
i=g(ϵ)+1

τ̂ϵ,δ = min

41

(28)

## Page 42

Revisiting Simple Regret: Fast Rates for Returning a Good Arm
α

Plug the instance ∆i = ( ni ) into the above,
n
n
1 ⎞
1 ⎛g(ϵ)
−2
⎜ ∑ (∆i∨j,g(ϵ)+1 ) ln( ) + ∑ ∆−2
j,i ln( )⎟
j∈[g(ϵ)] j ⎝ i=1
jδ
δ ⎠
i=g(ϵ)+1

τ̂ϵ,δ = min

n ⎞
1 ⎛g(ϵ)
−2
⎜ ∑ (∆i∨j,g(ϵ)+1 ) ln( )⎟
j∈[g(ϵ)] j ⎝ i=1
jδ ⎠

> min

⎞
1 ⎛g(ϵ)
1
−2
ln( )⎜ ∑ (∆i∨j,g(ϵ)+1 ) ⎟
j∈[g(ϵ)] j
δ ⎝ i=1
⎠

> min

1
1
−2
ln( )((∆g(ϵ),g(ϵ)+1 ) )
j∈[g(ϵ)] j
δ

> min

⎞
n2α
1 ⎛
1
⎟
ln( )⎜
j∈[g(ϵ)] j
δ ⎝ (g(ϵ) + 1)α − (g(ϵ))α )2 ⎠

> min

⎛
⎞
⎟
n2α
1
1 ⎜
⎟
> min
ln( )⎜
⎜
⎟
2
j∈[g(ϵ)] j
δ ⎜
α−1
⎟
(α(g(ϵ)
+
1)
)
⎝
⎠

(a1 )

⎞
n2α
1 ⎛
1
⎟
ln( )⎜
2α−2
j∈[g(ϵ)] j
δ ⎝ α2 (2g(ϵ))
⎠

> min

2−2α+2 α−2 n2α
=

2α−1

(g(ϵ))
(a2 ) 2

α n

−2α+2 −2

=
ϵ

2α−1
α

1
ln( )
δ

1
ln( ).
δ

Note here we lower bound the upper bound in Theorem 7 of Katz-Samuels and Jamieson (2020). The inequality (a1 ) results
by
(g + 1)α − g α = ∫

g+1

αxα−1 dx ≤ α(g + 1)α−1 .

g

The equation (a2 ) is because of g(ϵ) = nϵ1/α by the definition of the instance.
We also upper bound (28). Note that for i > j, we have
iα − j α = ∫

i

αxα−1 dx ≥ α(i − j)⋅j α−1 .

j

With this, we have
n
⎞
⎛
1 g(ϵ)
1
1
1
1
⎟
+
+
τ̂ϵ,δ ≤ n2α ⎜ min
∑
∑
2 2α−2
j i=j+1 (g(ϵ) + 1 − i)2 i2α−2 j i=g(ϵ)+1 (i − j)2 j 2α−2 ⎠
⎝j≤g(ϵ) (g(ϵ) + 1 − j) j

We can bound the third sum by
n
1
1
1
1
≲ 2α−1 ⋅
∑
2
2α−2
j i=g(ϵ)+1 (i − j) j
j
g(ϵ) + 1 − j

42

## Page 43

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

√
Take j = g(ϵ) −

√
g(ϵ). This means that j = Θ(g(ϵ)) and g(ϵ) − j =

g(ϵ). With this choice,

⎞
⎛
⎞
g(ϵ)
1 ⎜
1
1
1
1
1
⎟
⎟
⎟
⎜
√
+
⋅
⋅
⋅
+
∑
2 g(ϵ)2α−2 ⎟
2α−1
⎟
⎜ g(ϵ)⋅g(ϵ)2α−2 g(ϵ) ⎜
√
(g(ϵ)
+
1
−
i)
g(ϵ)
1
+
g(ϵ)
i=g(ϵ)−
g(ϵ)+1
⎝
⎠
⎠
⎝
⎛

τ̂ϵ,δ ≲ n

2α ⎜

≤ n2α

⎞
⎛
1
π2
1
1
1
√
+
⋅ +
⋅
2α−1
2α−1
2α−1
g(ϵ)
6
g(ϵ)
⎝ g(ϵ)
1 + g(ϵ) ⎠

≲ n2α ⋅

1
1
n
= n2α
.
=
g(ϵ)2α−1
(nϵ1/α )2α−1 ϵ 2α−1
α

For Bracketing SH,
1−2α

⎛ ϵ 1/α ⎞
1
i
1
1−2α 2α
<
n( )
max
⋅
=
max
i
n
ϵ
1/α
i≥g(ϵ/2)+1 ∆2
⎝ 2
⎠
i≥g(ϵ/2)+1
i g( 4 )
n( ϵ )

n2α

4

1
1/α
n( 4ϵ )

<

16
.
ϵ2

where both the last inequality and the last equality are by α > 0.5.

C.7. An alternative upper bound for BSH
The following bound of BSH is similar to the bound of Theorem 6. Though the bound of Theorem 14 is minimax in nature,
we show, in Corollary 14.1 and 14.2, that it still achieves the same upper bound as BUCB for the EqualGap(m) instance and
a better upper bound for Polynomial(α) than BUCB when n is large enough because it does not scale with n polynomially.
Since this bound involves an optimization problem, we speculate it could be tighter than the bound of Theorem 6 for certain
instances.
Theorem 14. For any ϵ ∈ (0, 1), the error probability of Bracketing SH satisfies
⎛ ⎛
⎞
⎧
i∆2i,g(ϵ/2)+1 2 ⎫
⎪
⎪
⎪ ′ ⎞⎟
⎪
⎜
⎟
P(µJt < µ1 − ϵ) ≤ exp⎜
−
Θ̃
min
⎨
max
,
ϵ
⎬
t
⎜
⎟,
⎪
⎪
i∈[g(ϵ/2)]
n
⎪
⎪
⎩
⎭ ⎠⎠
⎝ ⎝
t
where t′ = 4 ln
. Accordingly, ˜
O(max {mini∈[g(ϵ/2)] i∆2 n
t

i,g(ϵ/2)+1

log( 1δ ), ϵ12 log( 1δ )}) samples are sufficient for Bracketing

SH to output an ϵ-good arm with probability 1 − δ.
Proof. The proof idea is similar to the proof of Theorem 6. The core idea is to find the best bracket whose size is wellbalanced. At round t, define the best bracket as bracket ri∗ (t) = ⌈log2 (c∆2i,g(ϵ/2)+1 t′ )⌉, where i ∈ [g(ϵ/2)] is a free
parameter and c is a logarithmic term for shorthand c = (16 ln(4e) log22 (

i∆2i,g(ϵ/2)+1 t′
) log2 (2∆2i,g(ϵ/2)+1 t′ ))
n

−1

. Thus the

size of bracket ri∗ (t) satisfies,
log2 (c∆2i,g(ϵ/2)+1 t′ )+1

c∆2i,g(ϵ/2)+1 t′ ≤ ∣Ari∗ (t) ∣ ≤ 2

= 2c∆2i,g(ϵ/2)+1 t′ .

Define the following events:
• E1 : the number of arms with mean reward at least µi included in bracket ri∗ (t) is at least jt , where jt = ⌊
• E2 : µar∗ (t) ≥ µ1 − ϵ/2, where ari∗ (t) is the arm representing bracket ri∗ (t) at round t.
i

43

ci∆2i,g(ϵ/2)+1 t′
⌋.
2n

## Page 44

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Then the error probability of Bracketing SH can be expressed as follows,

P(µJt < µ1 − ϵ) = P(µJt < µ1 − ϵ, E1c ) + P(µJt < µ1 − ϵ, E1 , E2c ) + P(µJt < µ1 − ϵ, E1 , E2 )
≤ P(E1c ) + P(E1 , E2c ) + P(µJt < µ1 − ϵ, E2 ).

(29)

We bound the three terms respectively. For the first term, we use the same technique as in the proof sketch.

⎛

⎞⎞
i ⎟⎟
, ⎟⎟
n ⎟
⎝ ∣Ari∗ (t) ∣ ⎠⎠
⎛

jt

⎜
P(E1c ) ≤ exp⎜−∣Ari∗ (t) ∣⋅KL⎜
⎜
⎜
⎝

2

⎛
⎛
⎞ ⎞
∗
i⎟ ⎟
⎜ ∣Ari (t) ∣n ⎜ jt
− ⎟ ⎟
≤ exp⎜−
⎜
⎜
2i
∣Ari∗ (t) ∣ n ⎠ ⎟
⎝
⎠
⎝

(a1 )

2

2
′
⎛
2
′ ⎛ i∆i,g(ϵ/2)+1 t
⎞ ⎞
i⎟ ⎟
⎜ ∆i,g(ϵ/2)+1 t n ⎜
2n
⎟
≤ exp⎜−
−
⎜ ∆2
⎜
2i
t′ n ⎟ ⎟
i,g(ϵ/2)+1
⎝
⎠
⎝
⎠

2

⎛ ci∆i,g(ϵ/2)+1 t ⎞
≤ exp −
8n
⎝
⎠
′

⎛ ⎛ i∆2i,g(ϵ/2)+1 t′ ⎞⎞
⎟.
≤ exp⎜−Θ̃
n
⎠⎠
⎝ ⎝

(30)

2

(p−q)
The inequality (a1 ) is due to KL(p, q) ≥ 2 max{p,q}
. For the second term of (29), we bound the event that SH returns a
non-(jt , ∆i,g(ϵ/2)+1 )-good arm, where jt is with respect to the best bracket. We use µ(jt ) for the mean reward of the jt -th
best arm in bracket ri∗ (t). Thus µ(jt ) − ∆i,g(ϵ/2)+1 ≥ µ1 − ϵ/2. By Theorem 4,

P(E1 , E2c ) ≤ P(µar∗ (t) < µ1 − ϵ/2 ∣ E1 )
i

≤ P(µar∗ (t) < µ(jt ) − ∆i,g(ϵ/2)+1 ∣ E1 )
i

⎛
⎛
⎞⎞
∆2i,g(ϵ/2)+1 t′
⎜
⎟
⎜
≤ log2 ∣Ari∗ (t) ∣⋅ exp⎜−const⋅jt ⎜
− ln(4e)⎟
⎟⎟
⎜
⎟
2
∗
∗
4∣A
∣
log
(2j
)
log
∣A
∣
t
ri (t)
ri (t)
2
2
⎝
⎠⎠
⎝

(a1 )

2

ci∆i,g(ϵ/2)+1 t ⎞
⎛
≤ log2 ∣Ari∗ (t) ∣⋅ exp −const⋅
n
⎝
⎠
⎛ ⎛ i∆2i,g(ϵ/2)+1 t′ ⎞⎞
⎟.
≤ exp⎜−Θ̃
n
⎠⎠
⎝ ⎝

′

(31)

44

## Page 45

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

For the inequality (a1 ), we bound the term in parenthesis as a constant,
∆2i,g(ϵ/2)+1 t′
4∣Ari∗ (t) ∣ log22 (2jt ) log2 ∣Ari∗ (t) ∣

− ln(4e)

∆2i,g(ϵ/2)+1 t′
>
8c∆2i,g(ϵ/2)+1 t′ log22 (2c

i∆2i,g(ϵ/2)+1 t′
2n

1
>
8c log22 (

i∆2i,g(ϵ/2)+1 t′
n

− ln(4e)

) log2 (2c∆2i,g(ϵ/2)+1 t′ )
− ln(4e)

) log2 (2∆2i,g(ϵ/2)+1 t′ )

>2 ln(4e) − ln(4e)
= ln(4e).
The third term of (29) means an arm whose mean reward is less than µ1 − ϵ has a larger empirical reward than an arm whose
mean reward is larger than µ1 − ϵ/2. Note the recently opened brackets may not have finished their first SH yet. In this case,
the DSH always returns an empirical reward of negative infinity. Thus the arms returned from these brackets will never be
t
selected by BSH. Denote Ȧ(t) ∶= {a ∈ ⋃L
A , µa < µ1 − ϵ}. We have
k=1,k≠r ∗ (t) k
i

Lt

∣Ȧ(t)∣ ≤ ∑ 2k ≤ 2⋅(21+log2 (1+ln(2)t) − 1) ≤ 8⋅(1 + ln(2)t).
k=1

By the number of arm pulls in Lemma 11.2,
P(µJt < µ1 − ϵ, E2 ) ≤ P(∃a1 ∈ Ari∗ (t) , ∃a2 ∈ Ȧ(t), s.t., µa1 ≥ µ1 − ϵ/2, µ̂a1 < µ̂a2 )
ϵ 2
t′
)
≤∣Ȧ(t)∣∣Ari∗ (t) ∣ exp(−( ) ⋅const⋅
2
log2 n
≤ exp(−const⋅ϵ2

t′
+ ln(∣Ȧ(t)∣∣Ari∗ (t) ∣))
log2 n

≤ exp(−const⋅ϵ2

t′
+ ln(8⋅(1 + ln(2)t)2∆2i,g(ϵ/2)+1 t′ ))
log2 n

≤ exp(−const⋅ϵ2

t′
+ O(ln t))
log2 n

≤ exp(−Θ̃(ϵ2 t′ )).

(32)

Combine (30)(31)(32) and the fact i ∈ [g(ϵ/2)] is a free parameter, then we have
⎛ ⎛
⎞
⎧
i∆2i,g(ϵ/2)+1 2 ⎫
⎪
⎪
⎪
⎪ ′ ⎞⎟
⎜
P(µJt < µ1 − ϵ) ≤ exp⎜−Θ̃⎜min ⎨ max
, ϵ ⎬ t ⎟⎟.
⎪
⎪
i∈[g(ϵ/2)]
n
⎪
⎪
⎩
⎭ ⎠⎠
⎝ ⎝

Corollary 14.1. Consider the EqualGap(m) instance. BUCB achieves an expected (ϵ, δ)-unverifiable sample complexity as
˜
E[τϵ,δ ] ≤ O

⎛ n
1 ⎞
log( ) .
δ ⎠
⎝ ϵ2 m

For BSH, the bound of Theorem 14 shows that BSH achieves an (ϵ, δ)-unverifiable sample complexity as
˜
τϵ,δ ≤ O

⎛ n
1 ⎞
log( )
δ ⎠
⎝ ϵ2 m

with probability 1 − δ.
45

## Page 46

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

Proof. For BUCB, the proof is presented in the proof of Corollary 6.1. For BSH, the result is straightforward by plugging
the instance,
⎧
⎫
⎪
⎪
⎪
1
1
n
1 ⎪
n
1
⎪ n
⎪
log(
)
≤
min
⎨
log(
),
log(
)
⎬=
log( ).
2
2
2
2
⎪
⎪
i∈[g(ϵ/2)] i∆
δ
δ
mϵ
δ
mϵ
δ
⎪
⎪
(ϵ/2)
i,g(ϵ/2)+1
⎪
⎪
⎩
⎭
min

n

The following corollary shows that even if we use the upper bound of Theorem 14 that is minimax in nature, we can still
achieve an upper bound for the Polynomial(α) instance that does not scale with the instance size n, and for moderately
large ϵ or large enough n, the upper bound of Theorem 14 is always better than the upper bound of BUCB.
α

Corollary 14.2. Consider the Polynomial(α) instance ∆i = ( ni ) . BUCB achieves an expected (ϵ, δ)-unverifiable sample
complexity as
⎛ 2α−1
1 ⎞
E[τϵ,δ ] ≤ τ̂ϵ,δ = Θ̃ ϵ− α n log( ) .
δ ⎠
⎝
For BSH, the bound of Theorem 14 shows that BSH achieves an (ϵ, δ)-unverifiable sample complexity as
1 ⎞
˜ ⎛ϵ− 2α+1
α
τϵ,δ ≤ O
log( ) .
δ ⎠
⎝
with probability 1 − δ.
Proof. For BUCB, the proof is presented in the proof of Corollary 6.2. For BSH, first, notice that
α
α 2
⎛ j
i ⎞
2
2
∆i,j+1 > ∆i,j = ( ) − ( )

⎝ n

j
=( )
n

n

2α

⎠

2α

i
+( )
n

α

− 2(

ij
) .
n2

The sample complexity satisfies,
min

n

i∈[g(ϵ/2)] i∆2
i,g(ϵ/2)+1

1
n
1
log( ) ≤ min
log( )
2α
α
i∈[g(ϵ/2)]
g(ϵ/2)
ig(ϵ/2)
δ
δ
i 2α
i((
) + ( ) − 2(
) )
n

=

(a1 )

≤

2α

i((g(ϵ/2))

α
+ i2α − 2(ig(ϵ/2)) )

2⋅n2α+1
1
log( )
2α+1
−2α
−α+1
(g(ϵ/2))
(1 + 2
−2
)
δ
2⋅n2α+1

=

n2

n⋅n2α

min
i∈[g(ϵ/2)]

n

(n(ϵ/2)1/α )2α+1 (1 + 2−2α − 2−α+1 )

1
log( )
δ

2α+1
1
−1 2α+1
=2 α +1 (1 + 2−2α − 2−α+1 ) ϵ− α log( ).
δ

The inequality (a1 ) is by taking i = g(ϵ/2)/2.
46

1
log( )
δ

## Page 47

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

C.8. (ϵ, δ)-unverifiable sample complexity of algorithms with ϵ-error probability bound
Definition 1 ((ϵ, δ)-unverifiable sample complexity (Katz-Samuels and Jamieson, 2020)). For an algorithm π and an
instance ρ. Let τϵ,δ be a stopping time such that
P(∀t ≥ τϵ,δ ∶ µJt > µ1 − ϵ) ≥ 1 − δ.
Then, τϵ,δ is called (ϵ, δ)-unverifiable sample complexity of the algorithm with respect to ρ.
The (ϵ, δ)-unverifiable sample complexity requires all the outputs at and after t = τϵ,δ are ϵ-good. This is slightly different
from the sample complexity of BSH. However, we show they are order-wise equivalent. To see this, let us consider an
anytime algorithm that achieves an exponentially decreasing error probability,
P(µJt < µ1 − ϵ) ≤ exp(−c⋅t).
Let tϵ,δ be the first time step that satisfies P(µJtϵ,δ < µ1 − ϵ) ≤ δ. Note this only guarantees the output at round tϵ,δ is ϵ-good
with probability 1 − δ, instead of all the outputs at and after tϵ,δ . We prove our claim by showing the probability that there is
a non-ϵ-good output at or after tϵ,δ has an exponentially decreasing rate with the same parameter.
∞

P(∃t ≥ tϵ,δ ∶ µJt < µ1 − ϵ) ≤ ∑ P(µJt < µ1 − ϵ)
t=tϵ,δ
∞

≤ ∑ exp(−c⋅t)
t=tϵ,δ

1 − exp(−c⋅t)
1 − exp(−c)
1
< exp(−c⋅tϵ,δ )
1 − exp(−c)
= lim exp(−c⋅tϵ,δ )
t→∞

= exp(−Θ(c⋅tϵ,δ )).

D. Discussion on the Practical Algorithm Implementation
The design of DSH (Algorithm 3.1) and BSH (Algorithm 4) involves two key ideas, bracketing and doubling trick. In
practice, both two techniques could be implemented in a more efficient way. In addition, the base algorithm SH (Algorithm
3.1) has a commonly used implementation for reusing samples (Baharav and Tse, 2019; Jun and Nowak, 2016). We
summarize these strategies as follows for the practitioner’s consideration. However, they do not make any order-wise
difference (not more than logarithmic factors) in terms of the theoretical guarantee.
• SH: The whole procedure of SH is divided into log2 n stages. All the stages are independent of each other since the
samples of prior stages are abandoned when a new stage starts, as described in Algorithm 3.1. In practical usage, one
can keep all the samples since the first stage for the surviving arms.
• DSH: The common doubling trick does not require keeping the samples after finishing one invocation of the base
algorithm. Not requiring to keep the samples is convenient to implement as we only need to repeatedly initialize a new
instance of the base algorithm with the doubled budget parameter. In fact, for multi-armed bandit problems, keeping all
the samples of prior invocations is beneficial. For the implementation, one can create a class for the base algorithm SH,
and we initialize a new instance of the class with initial empirical rewards equal to the empirical rewards saved in the
prior instance.
• BSH: The bracketing technique does not promise to avoid overlap with the already-opened brackets. A more practical
implementation of BSH is to share the samples of the same arm across different brackets. Such that the empirical
reward is more accurate. However, reusing sampling across different brackets is meaningless if we consider the
infinitely-armed bandit models. Because, for the infinitely-armed bandit models, we will not draw exactly the same
arm more than once. Thus it is barely possible to have overlap among the opened brackets.
Note our implementation reported in section 5 only uses the first reusing strategy for the base algorithm SH.
47

## Page 48

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

References
Richard Arratia and Louis Gordon. Tutorial on large deviations for the binomial distribution. Bulletin of mathematical
biology, 51(1):125–131, 1989.
Jean-Yves Audibert and Sébastien Bubeck. Best arm identification in multi-armed bandits. In Proceedings of the Conference
on Learning Theory (COLT), pages 41–66, 2010.
Jean-Yves Audibert, Sébastien Bubeck, and Rémi Munos. Best Arm Identification in Multi-Armed Bandits. In Proceedings
of the Conference on Learning Theory (COLT), 2010.
Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In
International Conference on Machine Learning (ICML), pages 263–272. PMLR, 2017.
Maryam Aziz, Jesse Anderton, Emilie Kaufmann, and Javed Aslam. Pure exploration in infinitely-armed bandit models
with fixed-confidence. In Algorithmic Learning Theory, pages 3–24. PMLR, 2018.
Tavor Baharav and David Tse. Ultra fast medoid identification via correlated sequential halving. Advances in Neural
Information Processing Systems (NeurIPS), 32, 2019.
Romain Camilleri, Kevin Jamieson, and Julian Katz-Samuels. High-dimensional experimental design and kernel bandits. In
International Conference on Machine Learning (ICML), pages 1227–1237. PMLR, 2021.
Alexandra Carpentier and Andrea Locatelli. Tight (lower) bounds for the fixed budget best arm identification bandit problem.
Journal of Machine Learning Research, 49(June):590–604, 2016.
Alexandra Carpentier and Michal Valko. Simple regret for infinitely many armed bandits. In International Conference on
Machine Learning (ICML), pages 1133–1141. PMLR, 2015.
Arghya Roy Chaudhuri and Shivaram Kalyanakrishnan. PAC identification of a bandit arm relative to a reward quantile. In
Thirty-First AAAI Conference on Artificial Intelligence, pages 1777–1783, 2017.
Arghya Roy Chaudhuri and Shivaram Kalyanakrishnan. PAC identification of many good arms in stochastic multi-armed
bandits. In International Conference on Machine Learning (ICML), pages 991–1000. PMLR, 2019.
Lijie Chen, Jian Li, and Mingda Qiao. Nearly instance optimal sample complexity bounds for top-k arm selection. In
International Conference on Artificial Intelligence and Statistics (AISTATS), pages 101–110. PMLR, 2017a.
Lijie Chen, Jian Li, and Mingda Qiao. Towards instance optimal bounds for best arm identification. In Conference on
Learning Theory (COLT), pages 535–592. PMLR, 2017b.
Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017.
Rémy Degenne and Wouter M Koolen. Pure exploration with multiple correct answers. Advances in Neural Information
Processing Systems, 32, 2019.
Eyal Even-Dar, Shie Mannor, Yishay Mansour, and Sridhar Mahadevan. Action elimination and stopping conditions for the
multi-armed bandit and reinforcement learning problems. Journal of machine learning research, 7(6), 2006.
Aurélien Garivier and Emilie Kaufmann. Optimal best arm identification with fixed confidence. In Proceedings of the
Conference on Learning Theory (COLT), pages 998–1027, 2016.
Avinatan Hassidim, Ron Kupfer, and Yaron Singer. An optimal elimination algorithm for learning a best arm. Advances in
Neural Information Processing Systems (NeurIPS), 33:10788–10798, 2020.
Kevin Jamieson, Matthew Malloy, Robert Nowak, and Sébastien Bubeck. lil’UCB: An optimal exploration algorithm for
multi-armed bandits. In Proceedings of the Conference on Learning Theory (COLT), pages 423–439, 2014.
48

## Page 49

Revisiting Simple Regret: Fast Rates for Returning a Good Arm

K.-S. Jun and R. Nowak. Anytime exploration for multi-armed bandits using confidence information. In Proceedings of the
International Conference on Machine Learning (ICML), volume 48, pages 974–982, 2016. ISBN 9781510829008.
Kwang-Sung Jun and Chicheng Zhang. Crush Optimism with Pessimism: Structured Bandits Beyond Asymptotic Optimality.
ICML Workshop on Theoretical Foundations of Reinforcement Learning (arXiv:2006.08754), 2020.
Kwang Sung Jun and Chicheng Zhang. PopArt: Efficient Sparse Regression and Experimental Design for Optimal Sparse
Linear Bandits. Advances in Neural Information Processing Systems (NeurIPS), 2022-December, 2022.
Shivaram Kalyanakrishnan and Peter Stone. Efficient selection of multiple bandit arms: Theory and practice. In International
Conference on Machine Learning (ICML), 2010.
Shivaram Kalyanakrishnan, Ambuj Tewari, Peter Auer, and Peter Stone. PAC subset selection in stochastic multi-armed
bandits. In International Conference on Machine Learning (ICML), volume 12, pages 655–662, 2012.
Zohar Karnin, Tomer Koren, and Oren Somekh. Almost optimal exploration in multi-armed bandits. In International
Conference on Machine Learning (ICML), pages 1238–1246. PMLR, 2013.
Julian Katz-Samuels and Kevin Jamieson. The true sample complexity of identifying good arms. In International Conference
on Artificial Intelligence and Statistics (AISTATS), volume 108, pages 1781–1791, 2020.
Emilie Kaufmann and Shivaram Kalyanakrishnan. Information complexity in bandit subset selection. In Conference on
Learning Theory (COLT), pages 228–251. PMLR, 2013.
Emilie Kaufmann, Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Edouard Leurent, and Michal Valko.
Adaptive reward-free exploration. In Algorithmic Learning Theory, pages 865–891. PMLR, 2021.
Levente Kocsis and Csaba Szepesvári. Bandit based monte-carlo planning. In European conference on machine learning,
pages 282–293. Springer, 2006.
Shie Mannor and John N Tsitsiklis. The sample complexity of exploration in the multi-armed bandit problem. Journal of
Machine Learning Research, 5(Jun):623–648, 2004.
Francesco Orabona and David Pal. Coin betting and parameter-free online learning. In Advances in Neural Information
Processing Systems (NeurIPS), volume 29, pages 577–585, 2016.
Max Simchowitz, Kevin Jamieson, and Benjamin Recht. The Simulator: Understanding Adaptive Sampling in the
Moderate-Confidence Regime. Proceedings of Machine Learning Research, 65:1–41, 2017.
Marta Soare, Alessandro Lazaric, and Rémi Munos. Best-arm identification in linear bandits. Advances in Neural Information
Processing Systems (NeurIPS), 27, 2014.
Yuan Zhou, Xi Chen, and Jian Li. Optimal PAC multiple arm identification with applications to crowdsourcing. In
International Conference on Machine Learning, pages 217–225. PMLR, 2014.
Yinglun Zhu, Julian Katz-Samuels, and Robert Nowak. Near instance optimal model selection for pure exploration linear
bandits. In International Conference on Artificial Intelligence and Statistics (AISTATS), pages 6735–6769. PMLR, 2022.

49
