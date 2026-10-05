# Experiment results — 2026-10-05

Training pilots started from random weights; existing checkpoints were used only for evaluation and frozen-weight diagnostics. Strength conclusions use complete, balanced arenas; throughput is not evidence of stronger play. Local experiment artifacts were removed after this summary was written.

## 1. Early CPU architecture/control runs — excluded

**Takeaway:** These exceeded the subsequently clarified CPU scope and were stopped/excluded from architecture selection.  
**Evidence:** Original-curriculum legacy/residual controls tied 40–40 over 80 games, entirely through Mysore wins; partial graph and faithful-original-trainer controls established no reliable advantage.

## 2. Rule-only bootstrap — excluded CPU pilot

**Takeaway:** A cheap rule teacher supplied extremely imbalanced outcomes and did not establish useful competitive British play.  
**Evidence:** Its 200 games produced 11,799 Mysore-labelled versus 117 British-labelled samples; bootstrapped legacy beat random initialization 29/50, including only four British wins. Excluded from recommendations.

## 3. Bootstrap followed by curriculum training — excluded CPU pilot

**Takeaway:** A small single-seed result was insufficient to recommend bootstrapping or the graph architecture.  
**Evidence:** After 320 curriculum games per candidate, graph beat bootstrapped legacy 55/100, with six British wins; this CPU research run was excluded from the authorized GPU findings.

## 4. Balanced replay and bootstrap/replay combination — excluded CPU pilots

**Takeaway:** Reweighting positive outcomes had no completed controlled strength result supporting adoption.  
**Evidence:** With positive fraction 0.25, legacy/residual completed 320 games but graph stopped at 288; the combined bootstrap/reweighting run completed only legacy, so no valid architecture comparison resulted.

## 5. GPU architecture screening: residual and graph

**Takeaway:** Neither replacement demonstrated a reliable improvement; retain the legacy architecture as the control.  
**Evidence:** Two seeds, 320 scratch games/candidate at 48 nominal simulations: residual won 78/160 versus legacy; graph won 84/160, but its seed results reversed (45/80, 39/80). Graph was smaller yet slower in these pilots.

## 6. GPU factorized-residual refinement

**Takeaway:** Preserving the factorized policy and adding a small residual/LayerNorm block did not establish stronger play.  
**Evidence:** At 240 scratch games/candidate, rank-0 residual tied legacy 32/64 across two seeds; rank-4 tied 16/32 in the completed seed. Rank 0 had 396,720 parameters versus legacy's 395,696.

## 7. GPU no-PCR check

**Takeaway:** Removing PCR did not expose an advantage for the residual candidate in the small pilot.  
**Evidence:** Original curriculum, 80 scratch games, 16 full simulations at every move: rank-0 residual tied legacy 8/16. This is undertraining evidence, not a full-budget activation/architecture verdict.

## 8. Evaluation against the established checkpoint

**Takeaway:** None of the pilot models was competitive with the established model; no improved model was produced.  
**Evidence:** Seed-17 legacy and both refinement candidates each lost 0–32 against `ckpt_027100.pt`; no-PCR legacy/residual each lost 0–16. Opponent games never supplied training data; these matches were against that checkpoint, not a separately reported v13 arena.

## 9. Arena validity correction

**Takeaway:** Partial or one-sided arenas must not enter automatic architecture selection.  
**Evidence:** Screening accidentally scored a 40-game one-sided arena; corrected conclusions exclude it and the seven-game rank-4 refinement arena. Complete, balanced comparisons supplied the reported win counts.

## 10. Export and implementation smoke checks

**Takeaway:** Candidate plumbing was functional, but passing correctness checks did not demonstrate a strength improvement.  
**Evidence:** Fourteen bounded unit checks covered phase/value signs, batching, RNG isolation, policy weights, checkpoints and gradients; random-weight Torch/ONNX parity passed for all candidate architectures, including rank-4 factorization.

## 11. Queued full-budget GPU training — cancelled

**Takeaway:** No full-budget residual model or final strength comparison was completed.  
**Evidence:** The checked-in 26,000-game run was queued from scratch, then cancelled at the user's request after an approximately 100-GPU-hour projection with the then-unoptimized implementation; cancellation was confirmed.

## 12. v13 input encoding and initialization

**Takeaway:** All-zero strength bits were an initialization inconsistency, not a consistent “no battle” representation; the constructor fix was warranted.  
**Evidence:** Semantically equivalent opening inputs `0000`/`1000` changed value −0.1975→−0.2070 and legal-policy total variation by 0.2733; Python clearing and browser initialization already wrote the one-hot zero bit.

## 13. v13 self-play strength-bit trace

**Takeaway:** Legal play confirmed the inconsistent initialization convention and separated it from actual battle presence.  
**Evidence:** Seed 42, 400 simulations/decision: the opening and first two moves retained `0000`; `HL:pdc` changed it to `1000` without a battle. Battle presence comes from the attacker sentinel, not those bits.

## 14. v13 synthetic tactical probe

**Takeaway:** Raw policy can miss an immediate win in a synthetic position even when value is high; this alone does not prove searched play fails.  
**Evidence:** With four occupied keys and Coimbatore empty, winning Princely States received 3.00% legal probability versus 60.19% for Highlanders at Goa; value was +0.7593, and MCTS checks exact terminal outcomes.

## 15. v13 factorized-head weight inspection

**Takeaway:** Some raw policy outputs are structurally unused; removing them would be a small efficiency cleanup, not an established strength improvement.  
**Evidence:** Thirty noncoastal naval source/destination output rows are never selected during policy expansion and have negligible weight norms; norms were inspected in float64 to avoid underflow-based claims of exact zeros.

## 16. v13 replay sensitivities

**Takeaway:** Most controlled resource/deadline effects were strategically coherent; no new sparse-reward or curriculum failure was established.  
**Evidence:** Across 67 recorded positions and 669 paired interventions, advancing turn decreased value in all 36 pairs; tiring armies decreased it in 95/98, removing British cards in 137/145, and removing Mysore cards increased it in 130/135.

## 17. v13 fatigue-sensitive winning tactic

**Takeaway:** v13 already represents a useful interaction between army freshness, location, remaining cards and winning opportunities.  
**Evidence:** Refreshing only tired Mangalore changed −0.993537→+0.970611; engine-verified `mlrxsrp`, `pass`, `HL:x` captures the fifth key, with legal policy probabilities 69.4%, 100%, and 97.9%.

## 18. White-box activation patching

**Takeaway:** The fatigue-sensitive value signal is distributed through the trunk and concentrated into opposing value-head groups.  
**Evidence:** Patching 20 selected FC3 units recovered 92% of the value-logit change versus 14% for random controls; removing positive head units 17/39/36 changed +0.970611→+0.063612. This is a circuit case study, not unique neuron semantics.

## 19. White-box policy/value separation and legal masking

**Takeaway:** Masked tactical probabilities must not be interpreted as proof that the raw policy learned legality.  
**Evidence:** Value-selected and policy-selected top-ten FC3 sets overlapped in one unit; the attack logit was already high while the army was tired and the attack illegal. The engine mask enforces legality, and illegal logits are not directly supervised by the original loss.

## 20. Dead-ReLU value-head positions

**Takeaway:** Real played positions can have an output-bias plateau that blocks value gradients while retaining distinct trunk representations.  
**Evidence:** Before recorded moves 4/5, all 64 value ReLUs were off, value equalled −0.08790155, and value gradients into the trunk/head weights were zero; FC3 vectors differed by L2=1.92083 and policy gradients remained nonzero. The encoding fix did not remove this plateau.

## 21. Gate reactivation and legal successors

**Takeaway:** The plateau was local and recoverable, not proof that those value neurons were permanently dead or search could not distinguish descendants.  
**Evidence:** Two donor FC3 coordinates reopened a gate at move 4, one at move 5; only 1/11 legal-action successors of move 4 and 0/35 of move 5 stayed fully off in every chance outcome.

## 22. Frozen-weight activation substitution

**Takeaway:** A nonzero negative activation slope restores local value gradients, but stronger play from SiLU/GELU/LeakyReLU remains untested.  
**Evidence:** Replacing only the value-head ReLU with slope-0.01 LeakyReLU changed the two plateau input-gradient norms from zero to 0.2851/0.3262. No replacement activation was trained or accepted as a strength improvement.

## 23. Three-game activation-frequency survey

**Takeaway:** Many deeper neurons were unused in sampled search; sampled inactivity is not proof of globally dead neurons.  
**Evidence:** Three v13 games, 400 simulations/decision: 308/1,088 hidden units never activated across 60,815 evaluations (layer counts 1/49/81/45/132); the whole value head was off on 142 evaluations (0.2335%). All three games were British losses.

## 24. Conservative weight-bound proof

**Takeaway:** At least some inactivity is certified by the weights, rather than merely absent from the three-game sample.  
**Evidence:** Numerically padded interval bounds over `[0,1]^148` certify 44 inactive units: FC1=1, FC2=19, FC3=24. FC1 unit 207's unpadded maximum is −0.0044; first-layer weights are roughly half negative, and only 92.5/256 units activate per search input on average.

## 25. Initial Kaggle batching benchmark

**Takeaway:** Batching greatly speeds neural prediction, but the unoptimized cooperative CPU loop lost to a persistent original worker pool.  
**Evidence:** T4 batch-32 inference was 28.9× faster per position; eight short games took 6.433 s cooperatively versus 4.847 s with warm original workers (12.901 s cold). Eight fixed roots ×800 simulations improved 11.287→6.877 s versus sequential search.

## 26. Local CPU hotspot profile and fixes

**Takeaway:** Hoist PUCT/turn calculations, reuse parent evaluation for unvisited children, and iterate only positive priors during expansion.  
**Evidence:** Selection consumed about 63% and expansion 18–21% of profiled CPU time; the fixes reduced eight-root search 1.626→0.676 s and eight short games 1.514→0.629 s, with exact full-tree/game-record parity.

## 27. Kaggle batching rerun with CPU fixes

**Takeaway:** CPU fixes reversed the previous cooperative-batching regression; persistence also removes substantial worker startup costs.  
**Evidence:** Eight short games: original warm workers 5.107 s, CPU-fixed workers 3.444 s, old cooperative batching 6.709 s, CPU-fixed cooperative batching 2.937 s. Fixed-root visit counts matched exactly across methods.

## 28. Parallel CPU workers feeding one GPU batcher

**Takeaway:** Four persistent workers plus immediate central-GPU dispatch were the best tested larger-search setup; eight workers added no clear benefit.  
**Evidence:** All variants CPU-fixed: 16/32 roots ×800 simulations took 3.302/5.144 s versus cooperative 4.528/8.601 s and original eight-worker prediction 6.606/13.279 s. Exact visit/evaluation counts matched; the 32-root gains were 1.67× and 2.58×.

## 29. MPS inference batching and CPU policy expansion

**Takeaway:** The same device-selectable batching API works on Apple M2 MPS, but v13 is small enough that CPU inference usually wins locally.  
**Evidence:** Batch-32 standard MPS reached 23,446 positions/s versus 855 singly; CPU reached 125,462/s. CPU-side factorized-policy expansion raised MPS to 29,192/s, but did not consistently improve full-search time.

## 30. MPS full-search worker comparison

**Takeaway:** Keeping one accelerator owner works with MPS; CPU parallelism should not be mistaken for a GPU advantage.  
**Evidence:** For 32 roots ×800 simulations, standard MPS took 4.114 s, four-worker MPS 2.746 s, CPU batching 2.863 s, and four-worker CPU prediction 1.332 s. All backends produced identical root visits and evaluation counts.

## 31. MPS batch-size and collection-wait follow-up

**Takeaway:** Request-collection timing is hardware-dependent, and larger MPS batches do not automatically minimize total search time.  
**Evidence:** A 1 ms wait reduced four-worker MPS's 32-root time to 2.585 s, versus 2.986 s at 0.25 ms; isolated MPS inference only edged ahead of CPU at batch 1,024 (259,411 vs 234,812 positions/s). Short-run variation prevents a precise optimum claim.

## 32. Eight complete scratch games: CPU versus MPS

**Takeaway:** CPU-only was the fastest tested local rollout configuration for the current architecture.  
**Evidence:** Fresh random weights, eight full games at 128 simulations/decision: four-worker CPU took 3.609 s versus MPS 11.244 s; cooperative CPU/MPS took 5.072/12.198 s. All configurations produced identical 390 decisions, 49,920 simulations, and replay samples.

## 33. Matched CPU/MPS learning updates and mixed backend

**Takeaway:** CPU learning was slightly faster after warmup at minibatch 256; CPU inference can still accompany MPS learning by refreshing a CPU weight mirror.  
**Evidence:** Eight matched scratch-data Adam steps: warm CPU/MPS medians 6.11/7.54 ms, first MPS step 1.287 s, mirror refresh 3.35 ms; losses agreed closely. Updates bypassed normal replay warmup solely for timing because eight games supplied 390 samples, below 1,000.

## 34. Full local curriculum projection — estimate only

**Takeaway:** The four-worker CPU throughput projects to roughly one day, with 1–2 days a provisional planning allowance rather than a measured bound.  
**Evidence:** Scaling the 128-simulation eight-game timing linearly gives 19.6 h for 25,000 games ×800 and 3.9 h for 1,000 ×4,000, plus ~3 min of updates: 23.5 h total. Shorter curriculum stages, deeper search and changing game lengths can shift this substantially; worker integration is required.

## Retained conclusions

Keep initialization consistency and the measured MCTS CPU optimizations; retain legacy architecture and the original curriculum as controls. SiLU/GELU, SAE training, alternate input layouts and a full-budget architecture gain were not established by these experiments.

## Remote provenance

Local raw outputs, scratch checkpoints, diagnostic visualizations and benchmark sources were deleted at the user's request; the numerical evidence above is retained for commit. Remote Kaggle runs were not deleted:

- [GPU screening](https://www.kaggle.com/code/krishs1234/tigersday-nn-screening-20261004), [refinement](https://www.kaggle.com/code/krishs1234/tigersday-nn-refinement-20261005), [no-PCR check](https://www.kaggle.com/code/krishs1234/tigersday-nn-nopcr-20261005).
- [Initial batching](https://www.kaggle.com/code/krishs1234/tigersday-batching-benchmark-20261005), [CPU-worker batching](https://www.kaggle.com/code/krishs1234/tigersday-batching-workers-benchmark-20261005), [larger-batch scaling](https://www.kaggle.com/code/krishs1234/tigersday-batching-scale-benchmark-20261005).
