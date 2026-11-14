# Research Log Guidance

Git history review: inspected through `1f785ed` (August 28, 2026, dynamic PUCT); discussed through `98fae57` (May 5, curriculum cycling and factory changes), including the April 19 removal of the extra starting army and the later removal of early stopping. Continue chronologically and skip frontend and backend connection work.

The August opening-tree analysis, dynamic PUCT, playout cap randomization in `8e4c4dd`, and the following curriculum adjustment have also been discussed. Continue after `09d0b5e`.

The user said the block of September commits preceding `5677294` was reverted and should not be considered. That September 27 commit restored the project baseline; continue with the October 5 changes after it.

The August 30 weak-reference fix in `ed9dd25` has also been discussed and recorded under tree retention. The October 5 policy-target fix in `690a938` has been discussed and recorded under exploration.

The October 5 experiments and the switch to SiLU in `2a4d3f8` have been discussed. The user considers the initialization inconsistency a routine bug and does not want it included in the log.

The removal of noise and samples on PCR fast searches in `b7c9145` and the training-credit change in `871ec0b` have also been discussed and recorded. Continue chronologically after those changes.

The v13-guided curriculum rebalance in `fabcb35` and motivation for distillation-based v14 have been discussed. The user confirmed that curriculum-trained v14beta was weaker than v13; the later distilled v14 was stronger. The v14 model card supplies the training recipe and reported evaluations, with its stated provenance limitation.

The user explained the winner-selected policy labels: the game winner was used as an available signal for which teacher was strongest in a position. Later experiments did not establish a benefit from that selection with other teachers.

The rationale for blended value targets, the 75/25 split, outcome recovery by sign, and a possible turn-based schedule have been discussed and recorded. A later TD(λ) distillation experiment was tested, but the choice between TD(λ) and turn-based decay remains future work. The exact prior source for the 75/25 split has not been verified.

The switch from MSE to BCE value loss and its motivation have been discussed and recorded. The v17beta versus v16beta manifest ratings differ by 50 Elo, but other settings changed; the remembered 50 Elo Leela result and its source remain unverified. GoNN's 2018 cross-entropy versus L2 experiment used Leela Zero and ELF games and found more confident predictions but no clear average strength difference. Continue chronologically after the BCE change.

The November 1 reduction in temperature cutoffs has been discussed and recorded: it aimed to reduce the gap between Mysore-favoring opening evaluations and arena results; stronger models later seemed to contribute more to closing the gap. Continue after this change.

The November 1 distillation schedule experiments and v18 results are recorded from DISTILL_RESULTS.md, including early-stopping noise, gradient and cooldown observations, the ten-epoch linear-decay choice, and limits on the match evidence. Continue after those experiments.

The user recalls that learning-rate tuning raised concerns about coupled L2, and that this together with interpretability work led to inspecting constant units and switching to AdamW. The detailed experiment results and limitations are recorded. Continue after the AdamW change.

The distillation-gain investigation, importance of distinct game volume, reuse and relabelling of Elo tournament games, opening/player-strength bias, and TD(λ), Tree-Backup, and filtering experiments are recorded. Fresh distillation data versus direct pretraining and TD(λ) versus turn-based decay remain open future choices. DISTILL_RESULTS.md retains the full tables, setups, and provenance.

The motivation for efficient pretraining is to compare frontier-model research in time-limited VMs against human-developed agents, potentially giving the models more time. The user's recalled sub-hour 1,600–1,700 Elo result is recorded as a recollection; the checked mini card and CPU sweep do not verify that exact time/rating combination. A summary of the simulation, buffer, and training-ratio work is recorded with its limits.

The endgame target/budget pilots and muP width-transfer experiment are summarized. The detailed interpretability research plan and individual throughput measurements remain in their linked reports rather than being fully transcribed into the log.

The motivation for a more rigorous games-versus-search and replay-window study is recorded. Jones 2021 examines train-time versus evaluation-time compute with fixed training search, not the exact games-versus-training-search tradeoff. The 12–13% figure is our initial fixed-buffer finding, not a general KataGo setting; KataGo used a growing sublinear window.

The Elo tournament protocol, v8 anchor, paired factions and uncertainty, shared faction model, and its limitations are summarized from the estimator and tournament report.

The research log should record the decisions we made, the experiments we ran, and the things we tried, including what worked, what didn't, and why things are the way they are. It should capture how we made hypotheses, refuted them, and came up with new explanations.

The log should help us understand the project as it is now and why we chose one approach over another for practical development. It should also record how we do research as people, so that we can look back for research tips and potential employers can read about our research work.

We are working toward an expository paper that takes all of our ideas and composes them so that someone with a reasonable background can understand our thought process from start to finish. Every step should be clear and make sense.

Act as a facilitator: look at the Git history and the current state of the repository, and ask us about things we haven't explained clearly or decisions we seem to have made but haven't explained. We will mostly dictate and discuss. Write down and sensibly organize what we say, cleaning up our words without adding your own ideas or explanations.
