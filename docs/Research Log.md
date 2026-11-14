# Research Log

## AlphaTiger – Research Process

*First-person recollections · Lightly edited from conversation · November 2026*

### Why we wanted to build an agent

One of us came from GamesCrafters, where the research focused on deterministic games in which both players have full information. Their PI had mentioned that games based on chance or incomplete information were different classes of games, where standard techniques might not apply or might look very different.

When Krish introduced Tiger's Day, it was different from most of the games being studied in the lab. Cards are discarded randomly when a battle is resolved. We originally considered hiding the available cards, but that stopped making sense after a while, so they became revealed to some extent. Even if a player's strategy is perfect, luck can still interfere; that doesn't mean optimal strategies cannot be developed. The game initially seemed like a simplified version of other games involving chance, but further research suggested that it was a different class of game because of how it was developed.

The initial challenge was to understand what kinds of algorithms or heuristics could produce a system that learned how to play this different class of game. We converged on building an AI. We also hoped to find vulnerabilities in the game; eventually we found some, changed the game, and learned from that process.

Because Tiger's Day was a new game, it did not have the game-specific techniques available for well-studied games such as chess and Go. Randomness was another challenge, as was our limited compute budget compared with standard game engines: a full neural-network training run might take several days on our laptop. We wanted to see whether building a good bot under those conditions was possible.

The game and its turn order are asymmetric: play does not simply alternate one move per side. British gets two moves per impulse, and from the starting position random play favors Mysore. For British to have a meaningful chance, it has to plan many moves ahead. The two sides have different policies in general, although they can share concepts such as coasts, keys, and adjacency. A model that learns one side does not necessarily know how to play the other. This asymmetry recurs throughout the research.

The turn order also affects value handling in search. We cannot simply multiply the value by minus one on every move, as in a game where the players strictly alternate. Instead, we use an absolute evaluation and account for which side is making the decision.

The project eventually became a question of how to make the AI better and whether we could make it better than ourselves. We reached that goal, and the model has continued to improve. We adapted techniques from other bots, but a new game also called for new techniques or new twists on old ones.

### Testing frontier models on the research problem

Another motivation was to compare frontier AI models with human researchers on an AI research problem like this. We tried this several times, using different handicaps as we thought might be needed, and also tried automating parts of our own engineering loop. We did not intend the agent experiments to be controlled comparisons against one another. In retrospect, clearly more handicaps would be needed for the human–agent comparison.

We allowed the agents to test all of their bots against random bots, but not against the bots they would ultimately be evaluated against. Their bots almost always beat random play convincingly, which gave us confidence that they were doing something correctly and that their code was not buggy, even though they did not match our approach.

As soon as we had implemented the game engine, before attempting our own agent implementation, we told Claude Opus 4.6 about our proposed AlphaZero-style Monte Carlo tree search approach and asked it to implement it. We think it probably implemented the standard method correctly, but after training, its bot achieved effectively zero Elo: it did not win against people and did not play British effectively. It moved British pieces back and forth a few times before the game ended, and we won every time. We saw total value collapse, which the agent did not identify or address.

About half a year later, we tried GPT-6.1 Sol with two hours of GPU time. That was less than we used for our frontier pretraining runs, but probably about as much as we used for our initial model pretraining. It first tried traditional approaches such as normal search, which did not work. It then chose AlphaZero, but ran out of compute after attempting to implement and train it. Our weakest active agent, version 8, beat its bot 1,000–0.

GPT-6.1 Sol identified the value collapse issue but did not find an adequate way to address it. It did not want to author a curriculum, apparently because it considered that methodologically impure. It had access to our current approach, whereas Claude Opus 4.6 and 5.5 were not given access to it. Even informed of our approach, GPT-6.1 Sol chose a different one that did not work.

We also tried Claude Opus 5.5, which thought traditional algorithms could work. It implemented minimax with alpha-beta pruning; the resulting bot achieved zero Elo and did not win a game against our earliest models. We then let it examine version 19's evaluations across a range of positions and use them to devise a handwritten evaluation function, expressed in code in terms such as numbers of forts and weights on armies. Its search did not use the neural evaluator directly at each position; the neural evaluator informed the creation of its evaluation function. With that advantage, it achieved 850 Elo, 150 Elo below our weakest existing AlphaZero-style model.

These tests also served as a kind of ablation. Traditional alpha-beta search and a Monte Carlo-style search produced similarly weak play, around 600–800 Elo, though their results were not identical. We took this as evidence that priors are important: search using the handwritten evaluation informed by version 19, but without priors, was not as effective as having priors too. With neither value nor priors, the approach did not work. We did not test search using version 19's neural value directly without its priors; that struck us as cheating.

Our interpretation is that frontier models have been useful at implementing standard techniques throughout development, but have struggled to understand what went wrong with the networks and to adapt standard methods to the particular problem. Their ability to identify the problem appears to have developed over time, but even identifying it did not necessarily lead to a remedy. In our attempts to automate the engineering loop, the AI did not seem capable of replacing even amateurs like us.

These experiments suggest to us that agents may lack the ability to apply known methods effectively to new contexts outside their training data. For example, if Claude can make a good chess bot, that may be because chess engines are well explored, rather than because it knows how to make a game-playing AI for a novel and difficult game.

We also wonder whether visual intuition is part of the difficulty: how the rules interact may not be clear to the AI even when the rules are explicitly laid out, whereas a human might understand those interactions intuitively. Models can get lost in the numbers. Numbers and proof are important, but some of our most useful insights and Elo improvements came from stepping back, interacting with the model, and noticing where we felt it was weak, rather than where we could programmatically prove it was weak.

### How the experiments were documented

Most of the experiment results so far were written up by agents. When I did research, I mostly just took the takeaway and then applied it, and didn’t write it down in super great detail. I had my agents write things down when they did the experiments I asked for, so I could read the results.

I think it would be a good idea to sit down with my collaborator for a few hours and just free-form write down our entire research process across all the versions.

### The game engine and the first complexity estimates

At first glance, AlphaZero seems kind of overkill when you read the game rules, because you think it’s an easy game. The first thing we did was make a game engine. There was a whole software design architecture for this, which I guess isn’t very important for ML research, but it was very rewarding. We wrote very good code there, and I’m very proud of that code. I haven’t had any of my agents touch it.

We optimized the engine and benchmarked legal move generation early because we expected to run an incredible amount of search. We estimate that AlphaTiger has performed tens of billions of neural network evaluations so far, with probably a similar number of steps through the game engine. Later, we optimized the engine again in C.

We debated writing the engine in Python or C for speed. We chose Python because keeping the engine and game readable was more important for ensuring that it was free of bugs: a bug in the engine could corrupt the entire training stack. The rules originally existed only in natural language, so we manually tested many edge cases to make the environment robust. In doing so, we also found ambiguities that required us to refine the written rules.

From the beginning, we designed the engine with the other parts of the software in mind, rather than as a standalone component. At that time, we did not have agentic tools; we talked to Gemini and pasted code. We extensively reviewed the early files, and most lines were written by us. We consider the game engine handcrafted and distinguish it from later parts of the project, where we incorporated more agentic tools to speed up coding and encountered bugs. Our current [module guidance](AGENTS.md) reflects different levels of caution: `/game` is high risk and written by hand, `/ai` is medium risk and read line by line, and `/public` is low risk.

The game originally had 23 locations; it now has 25, many connections have changed, and we think some card effects changed as well. Early networks exposed degenerate strategies in the original game. We identified how the networks exploited them and changed the game design to fix the problems. Networks earlier than version 8 were trained on the earlier rules, which is why they are no longer included in the game.

#### The southern triangle

One degenerate strategy involved a triangle of forts that we recall as Mahé, Coimbatore, and Palgacherry, a place that is no longer in the game. The triangle had few connections to the rest of the board. We recall connections from Srirangapatnam to Mahé, from Travancore to Palgacherry, and from Coimbatore to Dindigul and Erode.

With a fort on each of the three locations and the cards Mysore had at the time, Mysore could refute all four attacks the British could try, even if those were the only three forts left and the British had a full turn and whatever armies they wanted. We addressed this by reorganizing the southern connections and adding the British card Divide and Rule, a powerful tool for breaking formations that also made gameplay interesting for people and agents.

We discovered this while playing against a model. Both of us enjoy playing the game, and when a new model came out, we wanted to play it as well as test whether it was stronger than previous versions. In one game, a researcher appeared to be winning but realized they could not win because of this formation. We had played the game ourselves before developing the models, but there were so many permutations and interactions on the board that we had not thought of this detail. The model learned it through training, and playing against it let us see what it was doing.

#### Adding and removing a starting army

The models evaluated the game as extremely favorable to Mysore. At the time, we thought they were correct, so we added Poona and another starting British army, hoping it would help. As the models improved, it quickly became apparent that their assessment had not been correct. Human play also showed that starting with five armies was too powerful. We removed the extra starting army; the history records its addition on April 13 and removal on April 19, 2026.

Version 8 was trained before the starting army was removed and retained afterward. In theory, it should be able to evaluate arbitrary positions, but much of its training included positions starting with the army at Poona. It therefore had less training on the later starting position than subsequent models, which may help explain its weakness.

Once we had the engine, we did the obvious thing. We played relatively random games, or I think actually a few human games, and at each position counted the number of legal moves and multiplied those counts. It’s a standard way to estimate the size of the game tree. You might expect something small, but unfortunately it came out around 10^100. That puts the game above checkers and slightly below chess in terms of game-tree complexity.

We also came up with what we thought was a very efficient way to represent the state. We deliberately kept everything one-hot and binary, partly so that if we wanted to use bitboards or something similar later, we could do so effectively. One-hot made sense because, for instance, turns 1, 2, 3, and 4 are numbers in some sense, but they aren’t really scalar quantities. Turn 2 is a different concept from turn 3; it isn’t just turn 1 plus something. The same goes for the pieces. There isn’t necessarily a meaningful scale, so we represented the pieces and other things one-hot. And the binary representation was very fast.

With this, we estimated the state space at about 10^20. So we had something like the state-space size of checkers and the game-tree size of chess. Clearly we weren’t going to brute-force solve it using minimax.

### Chance in the game engine and search

Chance is a difficult problem, and the literature offers a number of ways to tackle it. We saw expectiminimax as the standard, mathematically optimal approach and wanted an approach that would at least converge to its result.

We used two kinds of nodes: decision nodes and luck nodes. At a decision node, MCTS uses the prior and PUCT to choose a decision. We call luck nodes *slippery*: the agent never stops at a position where luck is active. Instead, the game engine fires again and randomly picks an outcome. During search, MCTS similarly slips through luck nodes until it reaches a decision node. By sampling each branch at its true rate, the search converges toward the expected value, although random sampling adds variance.

In this game, search can slip at most twice in succession: at most two ongoing battles can be resolved at the end of an impulse. In other games, an indefinitely long sequence of chance events could pose a problem if it were unclear whether a decision node would ever be reached.

Later, we introduced round-robin sampling to make the sampled distribution match the true distribution more closely. Because all outcomes at a luck node in our game have equal probability, it is cheap to select a least-visited outcome. Always starting with the first outcome would introduce a systematic bias: for example, the first option might discard Iron Rockets, which could be a significantly better card to discard than the average card. We therefore give each luck position a random starting offset and maintain round-robin sampling after that.

The benefit is that the same number of searches can be used more effectively to arrive at the true evaluation. There is a tradeoff in games with unequal probabilities: calculating which outcome best matches the true distribution might cost enough compute that random sampling, despite its variance, allows more search and works better overall. We did not face that tradeoff because the calculation was cheap for our game.

### Retaining the search tree

We retained the relevant subtree after a move and its luck outcomes partly to save compute. It also matched our intuition about human calculation: after choosing a move we have thought through, we do not throw away everything we calculated about what might happen next. For example, after moving an army, information already gathered about possible card plays can still be useful.

Tree retention also exposed a memory problem. Even after we severed the retained subtree, parent and child nodes in the old tree still referred to one another, so Python did not promptly reclaim them. We changed the parent links to weak references, breaking those reference cycles, and memory usage dropped substantially.

Tree retention interacts with Dirichlet noise and other exploration measures. We also found a positive interaction with the playout cap randomization added later: a turn with a low playout cap may retain a large subtree from an earlier, deeper search. This makes effective search more stable than it would be under pure playout cap randomization and gives us additional search for essentially no extra cost. That helps particularly with British, which is harder to play and needs more search to understand.

Chance creates a separate challenge for showing and debugging the model's plans. The top-moves display only shows a line up to the first luck event, because it cannot know which discard will occur. This makes it difficult to tell how the model planned to respond to the different possible discards, or whether a move reflected a mistake. We need a better way to inspect that.

### Early training on Latte and the mutation error

At this point, we ran training on the CSUA's Latte server over SSH. Runs took a long time on our computers and could be interrupted if a machine was turned off or went to sleep, including while we were in class. Latte let us offload that compute.

During a March 17 run, training had progressed through the curriculum into the game before an MCTS mutation error stopped it. The researcher running it was asleep and noticed the crash two or three hours later. They traced through the code with an agent to find the cause.

The issue involved a rare sequence of two luck evaluations. Royal Navy could start another battle, and the model had randomly chosen to start two different battles in the same turn. The original code did not handle this: after skipping past one luck node, it treated the next node as a decision node even though it was also a luck node. MCTS errored out. We fixed the handling, and the training run then ran smoothly.

### What the first models showed

The early models had obvious weaknesses and seemed to think Mysore won almost always. When we put them in endgames we thought were winning or losing, however, their evaluations broadly matched our intuition. We also discovered that the models had learned to exploit the southern triangle strategy described above.

We made a number of rule changes before arriving at a game we no longer considered broken, where we could be confident that a remaining problem was with the model rather than the game.

### Judging play against people

The most direct way for us to judge whether a model had surpassed us was to play against it. We used two signs: whether it could beat us consistently, and whether it found ideas we had not found ourselves. Around versions 13 and 14, it began playing lines that initially looked questionable to us, only for their tactical combinations to become clear several moves later. Some were legal possibilities we had not seen in human games or anticipated in our own analysis. The bots also won most of their games against us, so at the time we considered them superhuman.

In hindsight, that judgment may have been premature. We could still sometimes defeat those models without assistance by exploiting known weaknesses, including their handling of Royal Navy and Sea Trade and their evaluation of luck positions. We addressed these weaknesses with source–destination policy factorization and round-robin sampling at luck nodes, respectively, and the models improved. We recall the last victory against a model being around 1,600 Elo, perhaps version 16, though we are unsure whether that particular game was assisted.

Playing with an evaluation bar changes the comparison. It can be tempting to use one, and human judgment combined with the bot's evaluations has helped us beat models: we could sometimes recognize a move or line the bot had not handled well. This was a period in which the human–bot combination could be stronger than the bot alone. Against versions 19 and 20, we have found it much harder to add value: even when we strongly disagree with a move, our alternative is not necessarily better. We suspect that advantage may disappear around 2,000 Elo, but that is a prediction; our models have not reached it.

Our claim of superhuman play is relative to the people currently playing Tiger's Day. None of us is a grandmaster of the game or has studied it through thousands of games, so a highly skilled human player might still win. Within the existing player pool, however, we are now confident the bots are stronger than us.

### Evaluating known positions

We built tools for working with specific positions, including positions where the model had made a blunder or where we had analyzed the game and believed we knew the correct move. These gave us an early way to test whether the model chose the right move and evaluated the position correctly.

At that point, we did not have Elo ratings: a person could not play enough games against a model to produce an accurate rating, and we did not yet have other models we were confident were good enough to play against it. Checking individual positions where we felt we knew the answers served as an early heuristic for judging progress.

### Arena and disagreement analysis

Reusable tooling became important for analyzing our bots. We built an arena so models could play each other, letting us compare them and decide whether to promote a new bot. By then, we had trained around ten iterations as well as several failed attempts. A standardized arena made it practical to benchmark many models routinely, measure how often each defeated another, and consider statistical significance. It also helped us investigate possible cycles in matchups, where one model might beat a second, the second a third, and the third the first because they had found different strategies.

We also built an AI disagreement visualizer to inspect entire games, compare evaluations, and see where models disagreed. This helped us investigate what a winning model had done to exploit a position. The arena log recorded luck nodes: an evaluation can swing substantially after a random outcome because the prior evaluation averages over possible outcomes. We did not treat such a swing by itself as a problem, but a large swing without a luck event often pointed to a model weakness. This helped us identify several weaknesses in older models, especially their difficulty playing Royal Navy and Sea Trade well.

### Measuring Elo across models

As the collection of models grew, we used tournaments to put them on a shared relative strength scale. The [Elo estimator](ai/elo.py) fits a joint Bradley–Terry model to the match results, fixing v8 at 1,000 Elo as an anchor. That number defines our scale; it is not a rating against an external human population. An Elo difference of 400 corresponds to 10-to-1 expected odds under the model. The fit also estimates a shared faction advantage, since British and Mysore play differently.

In the [October 26 tournament](benchmarks/KAGGLE_ELO_RESULTS.md), models used 400 simulations per decision, no root noise, and temperature for the first six decisions. Each matchup used paired seeds with the factions swapped. Pairing helps account for correlations in the outcomes; it does not guarantee identical luck events once the games take different paths. The tournament planner used existing rating estimates to allocate games, but the estimates were not treated as observations or priors in the final fit. Uncertainty estimates account for seed pairs, and the stopping intervals adjust for repeated checks of whether the desired precision has been reached.

The completed stage included 26,016 games; 11 of 13 non-anchor models reached the targeted adjusted interval half-width of 25 Elo. A later [tournament dataset](benchmarks/elo_results/tournament.json) contains 32,096 games and was also used for distillation. These precision intervals describe uncertainty under the fitted model, not every source of error. In particular, one shared faction effect cannot capture each bot's different strengths as British and Mysore. For example, in the October tournament v16 performed much better as British against v15 than the shared-effect fit predicted. Matchup-specific styles and nontransitive play can also make a single rating an incomplete description, and a model's rating depends on the search and evaluation conditions.

### Factorizing Royal Navy and Sea Trade

The large evaluation swings after Royal Navy and Sea Trade, without any intervening luck, led us to factorize those moves. Each card had around 250 possible source–destination combinations, almost as many as the width of a hidden layer. We thought this made it difficult for the model to form useful representations and made these moves unlike the rest of the move space.

We had deliberately given other node-related moves a 25-location representation, even when that included impossible options such as Princely States on locations where it is never legal. Although that leaves unused weights, we thought it might let the network reuse an internal representation of the board across moves rather than learn separate representations for subsets such as coastal keys. We did not know whether that was actually true. For Royal Navy and Sea Trade, this suggested using separate 25-location source and destination representations, even though some combinations would be illegal.

Our own play suggested the same decomposition. Sometimes the important thing was to get an army to a particular location, with the source a secondary question; at other times, it was to move an army away from danger, with the destination a secondary question. We thought the network could learn these general principles instead of each source–destination pair separately. Before factorization, it appeared to overfit to particular moves, such as Royal Navy from Erode to Pondicherry, without learning the general principle. Factorization also reduced the number of parameters, which we hoped would help with overfitting.

The change worked well in our experience: the model was faster and played Royal Navy and Sea Trade both more often and more effectively. We recall their use rising from about 0.2 to around one per game, closer to human play, although perhaps still low. All subsequent models have been factorized. The [model manifest](ai/models/manifest.json) records v12 at 498,496 parameters and an estimated 1,350 Elo, and factorized v13 at 395,696 parameters and 1,400 Elo. Other training settings also changed, so this comparison does not isolate the effect of factorization.

### Encouraging exploration

We noticed that our models assigned very high priors to particular moves, especially in the opening but also throughout the game, and rarely played other moves. Their play styles were broadly similar. Sometimes a move the model had not considered would cause the evaluation to swing in its favor, suggesting that it was overlooking better moves.

We researched exploration techniques used by AlphaZero and added Dirichlet noise at the root to encourage the model to explore and reinforce moves it otherwise would not consider. The idea was to concentrate enough probability on an explored move for search to assess it, rather than spread a few noisy searches over many unfamiliar moves. We applied the noise only at the root so that subsequent responses could focus on the best way to refute the move rather than being noisy too.

We also added a dynamic temperature cutoff. Our hypothesis was that using temperature throughout the game could encourage the model to concentrate too narrowly on a single correct move: even a small weight on a bad move might lead it to play that move and lose. Poor moves sampled late in the game could also affect game outcomes and therefore position evaluations. Cutting off temperature was intended to address this.

We have carried both changes through every model trained since. We did not ablate them separately because they were well-backed approaches, but models trained afterward with both changes played more dynamically and were much stronger than version 8.

We later found a bug in the policy target: at zero temperature, training received a target containing only the selected move rather than the full distribution of search visits. This could explain some of the extreme overconfidence we saw in earlier networks. We fixed it, but did not test its effect rigorously because we regarded it as a straightforward bug.

Later, we shortened the temperature cutoff across the curriculum: for the full-game stage, for example, from 18 decisions to 12. At the standard starting position, models were evaluating the game at around −0.3, favoring Mysore, while arena games with a cutoff of six were roughly even or slightly favored British. We thought the longer period of temperature in self-play made British sample poor opening moves, producing more Mysore wins and pulling the learned evaluation away from the results seen under arena conditions.

Reducing the cutoff appeared to lessen the Mysore bias somewhat. Our later interpretation, however, is that improving the bots' strength was the larger factor in bringing evaluations closer to even in versions 18 and 19.

### Studying openings

We wanted to know which openings the AI played and whether they matched the opening theory we had developed through human play. We used its opening decisions constantly as a qualitative check on the model: if it made bad decisions there, that suggested something was wrong. We considered the opening the hardest part of the game to get right, because understanding it required searching deeply and correctly enough to understand the game as a whole.

The replay opening tree let us see how often each opening was played and examine what the model was trying to do in particular lines. Having that tool made it easier to develop opening theory, as players do in chess or Go.

### Dynamic PUCT for deeper search

The over-concentration on a few opening moves led us to try dynamic PUCT. At inference time, we noticed that search concentrated heavily on the top move: one with both a strong prior and a good value. Sometimes the model missed another move and only found it after hundreds of thousands of simulations because it had allocated so little search outside its preferred option. We adopted dynamic PUCT, a known approach from AlphaGo, to distribute search more broadly as the simulation count increased.

We initially set the base coefficient to 1.0, then raised it to 1.25 to preserve the existing training behavior. At the 400 simulations we used for training at the time, the effective coefficient was similar to the previous fixed value of 1.5. This was intended to isolate the effect of making PUCT dynamic without generally increasing or decreasing exploration at the training search budget, while encouraging broader search at inference with hundreds of thousands of simulations.

### Scaling search and the deep-game stage

We increased search across the curriculum to improve policy targets. The value head generally made sensible evaluations, but the policy head did not always find the best move. At test time we could increase the simulation count substantially, even to one million, and sometimes saw the model play much better with more search. Later experiments found an improvement of roughly 180 Elo from a tenfold increase in simulations. This suggested an amplify-and-distill approach: use deeper search to find better moves, then train the network on those search results.

We scaled search over several models and added a deep-game stage of full games with many more simulations. We compared checkpoints before and after that stage and observed gains as large as around 200 Elo. Much of the raw gain appeared toward the end of the curriculum, during the last full deep-game games. The earlier stages built the baseline competence needed to train the model. After this period, we held the scale relatively steady because of our compute budget, until a more recent round of scaling.

One way we think about this is that the network learns values and policies for positions throughout the game tree. Earlier stages expose it to different parts of the tree and let it learn locally, but those lessons may not yet be connected across a full game. Repeated deep searches from the start to the end can help connect them and reinforce good behavior, rather than treating a line that happened to win through luck as reliably winning.

Around this time, we noticed a particular advantage over strong human players: the models could plan across a turn boundary more effectively. People might make a plan for one turn and start a new plan when it ends; for the model, a turn boundary is part of the continuing game. We had designed the engine for an AI from the start, so it handled battles, luck discards, and turn refreshes without pausing at those boundaries for an additional player decision. This let play proceed from decision to decision.

### Cycling through the curriculum

We tried further training a completed model by repeatedly cycling through the curriculum. The idea was to keep the model from forgetting how to play positions such as endgames or overfitting to the starting position. In practice, the network tended to collapse and models became weaker. We eventually removed this approach and used deep-game-only fine-tuning instead. It was an interesting experiment, but it did not work.

A later set of [v15 fine-tuning pilots](FINETUNE_RESULTS.md) also found that mixed-curriculum training weakened the model. In 200-game evaluations against unchanged v15, mixed training scored 58–142 at a learning rate of 0.001 and 87–113 at 0.0001. Full-game training at 0.0001 scored 108–92. Each configuration used one training seed, and the full-game run received more updates.

### Playout cap randomization and balanced self-play

From the first few agents onward, we used a heuristic to judge whether a model had learned a curriculum stage: in self-play from positions in that stage, both British and Mysore should win regularly, with win rates reasonably close to even. This helped us watch for value collapse.

After we lost access to Latte, the next models had to be trained on a Mac with much less processing power. Training also effectively prevented other use of the laptop, so speed mattered. We looked to KataGo, which also operated under compute constraints, and adopted its playout cap randomization (PCR). The idea was to obtain value positions from independent games while saving search on most decisions. Initially, one decision in four received a full search; the others used one-tenth of the full budget. We trained the policy on full-search decisions and the value on all decisions.

With PCR, our balanced self-play heuristic revealed a problem in the endgame and late-game stages: search was not deep enough to find British wins reliably when the prior pointed elsewhere. We doubled those stages' full-search budgets from 400 to 800 simulations. This also increased the short searches from 40 to 80, and we hoped that subtree retention would carry some work from the 800-simulation searches into later decisions.

The change allowed the models to learn to play both sides in the endgame and late-game stages, evidenced by our heuristic. We recall that the models trained with 400 simulations played at around 1,200 Elo and were retired, while the models with 800 reached around 1,400 Elo. Both used PCR, and the search-budget increase was the only change, so this was a controlled comparison. PCR had interacted negatively with British learning, but our heuristic helped us identify the stages that needed more search. Later, we reduced the PCR divisor from ten to five so the models would play better generally.

We suspected that root noise on fast searches could also be weakening play in the endgame and late-game stages. We removed it so the limited simulations could focus on choosing a good move. Since fast-search policies were not used as training targets, that direct benefit of exploring with noise was absent as well.

Initially, we kept fast-search positions for value training because they were essentially free samples. Later, we stopped recording them to simplify a new training-credit scheme. We believed we had been training too much on each sample and wanted to set a ratio of eight sample presentations per new training position. Mixing full-search samples with value-only samples made the meaning of that ratio harder to define. There was nothing inherently wrong with using the fast-search positions for value; omitting them made the accounting simpler.

Before training credit, the code performed one update with a batch of 256 after each game once the buffer had warmed up. Even when every decision was recorded, this meant uneven training across stages: an endgame could last fewer than 15 moves, giving more than ten sample presentations per new position, while a longer game might yield 60 positions, giving roughly four. We considered the former overtrained and the latter undertrained. Once we recorded only full searches, PCR reduced the number of samples by about fourfold, making the endgame ratio much worse; we recall it reaching around 50 presentations per new sample. This could lead to overfitting and was not a principled way to set the amount of training. The credit system instead aimed to keep the ratio consistent at eight across stages.

### MCTS early stopping

We added early stopping to save compute: if a move seemed obvious, search could stop and play it. We think the parameters came from research papers, though we do not remember which ones. We recall a speedup of roughly 25%, but ultimately considered the change a mistake.

With subtree reuse, simulations spent on the leading move are not necessarily wasted: the retained subtree lets us use that work later. Early stopping also let the model become overconfident in certain moves and play them without enough thought. We later removed it; Git records the removal in `2786b36` on August 29, 2026.

### Illegal moves, policy loss, and passive play

Early models often played British extremely passively, passing or tiring armies in place. To us, this seemed contrary to the human intuition that a turn should usually accomplish something, similar to the null-move pruning heuristic. As models became stronger, we observed that passing and tiring in place decreased monotonically with Elo, although they did not disappear: a well-timed pass or tire can be important.

Initially, we allowed the policy loss to penalize illegal moves, hoping that learning the rules would give the network useful representations. For example, knowing where Highlanders is legal requires knowing which locations are coastal, a concept that could be useful elsewhere. Perhaps this would have worked with more training, but we came to suspect that the immediate effect was excessive passivity: passing was a safe move in most positions, and tiring an army in place was legal wherever the army was, while a move to an occupied location could be illegal. The models could rely on these safe choices instead of fully learning which more active moves were legal.

We found that masking illegal moves is a standard approach and changed the policy loss to mask them as well. Models trained afterward passed dramatically less and played much better as British. We took this as evidence for our explanation of the earlier passivity and for why masking is useful; it also helped reduce the asymmetry in model performance. Experiencing the pitfall ourselves seemed like a potentially useful lesson for other researchers.

### The first network architecture

We started with a proof-of-concept MLP: three shared dense layers, each 256 units wide, with ReLU activations and separate policy and value heads. It remains our basic architecture because we have not yet seen proof that another is better. The current game has an input of 148 features and an output covering 959 actions; 256 seemed like a reasonable width, slightly wider than the input. The original game had 23 locations and its first state representation had 138 inputs, before the rules changed. We tried 128 units: it still played decently, though not quite as well. A width of 512 was impractical for our compute budget. Scaling depth is a possible next step, for which techniques such as residual connections and normalization could make training effective; later small pilots included a residual block, as noted below.

The shared network with policy and value heads followed the idea used in AlphaGo: the representations needed to judge who is winning and choose the next move are related. That also makes training cheaper than using two separate networks. We kept an extra layer in each head so that policy and value could still specialize. The value head's smaller, 64-wide layer was chosen with overfitting in mind rather than going directly from 256 units to one output. These head sizes have not yet been tuned.

The small network was useful because much of our training ran on CPU without access to strong GPUs. Keeping evaluations on CPU avoided the cost of moving data between CPU and GPU for each evaluation, a constraint we have seen larger systems such as KataGo and AlphaZero address in other ways. Fast evaluations also allowed more search. Our initial network had about half a million parameters; by comparison, we understand KataGo's smallest network to have had one million.

In [local benchmarks and architecture pilots](EXPERIMENT_RESULTS.md), four-worker CPU prediction completed eight scratch games in 3.609 seconds, compared with 11.244 seconds using MPS. Small residual and graph-network pilots did not demonstrate a reliable strength improvement over the existing architecture.

### Inactive neurons and SiLU

We tested the network on many positions to see which neurons activated. A large number were inactive: some were mathematically guaranteed to remain inactive for any input in the tested input bounds, while many more did not activate in any of the positions we tried. In a few positions, all 64 value-head ReLUs were off, so the value head output only its bias, without input from the board representation.

Leaky ReLU was suggested as a remedy, but we chose SiLU instead. Its gradient is nonzero over much more of its range than ReLU's, and we hoped this would prevent inactive neurons from accumulating. In later testing, we found none were dead in after the switch.

The same period included small-scale tests of residual and graph networks. They did not find an improvement, but the experiments were small enough that we still want to try those architectures at a larger scale. We also found CPU inference faster than MPS locally, which we attribute to the network being extremely small.

Our interpretation is that the network is not yet saturated and that, at our compute budget, more data may be more useful than a larger network. We plan to continue with it until increasing the data and curriculum produces diminishing returns. The [model manifest](ai/models/manifest.json) records version 8 at an estimated 1,000 Elo and version 20 at 1,800 Elo. It also records 10,000 games per main curriculum stage for version 19 beta and 20,000 for version 20 beta; other training settings changed as well, so this is not an isolated test of data volume.

### Why search alone seemed insufficient

We also had a sense from playing that the game would be resistant to alpha-beta pruning. There are many positions where you have to make agonizing decisions: you genuinely don’t know what’s best, and lots of moves seem about equally good. Maybe they aren’t, but many probably are. Later, when we had good agents, they supported that intuition. You can put the latest models at the opening, force different legal moves, and evaluate the resulting positions. Basically, at least among the top twenty or so legal moves, the evaluations can all be within about 0.1 of one another.

And with randomness, similar move evaluations really do make alpha-beta pruning less effective, because you have to resolve the chance outcomes and distinguish expected values that are close together. The randomness also naturally suggests some sort of Monte Carlo approach. You could try expectiminimax, but the game tree is just too large.

You can’t even brute-force solve the last turn, no matter what you try, because the branching factor is around twenty, and the proportion of repeated positions between different lines only converges to around fifty percent. So with every move, you can still be multiplying the number of distinct positions you need to account for by something like ten. That’s every move, not even every turn. It becomes basically impossible to solve that way.

We needed some sort of evaluation function, some sort of intuition about which moves were worth searching. That’s what led me to AlphaZero.

### Why uninformed MCTS was not attractive

The main objection to UCT was that there’s a very large move space containing bad moves. Not that most moves are necessarily bad in every position, but Royal Navy and Sea Trade create a lot of moves that aren’t useful. This was a problem we would confront again even while training AlphaZero.

And ordinary Monte Carlo search didn’t seem practical for us with a tree of around 10^100 and the compute we had. More importantly, random play produces almost no useful British victories. If you play randomly, British will win something like one in a million games, largely because Mysore makes an obvious blunder. If you prevent Mysore from making that one blunder – moving the fort in Mahé away – then British wins dropped to zero in all the testing we had.

We didn’t have the Force or a supercomputer, so we needed a learned evaluation. Actually, this is also the justification for curriculum generation, because it’s telling you how hard it will be to get a useful training signal from random play.

### What I anticipated before training

I anticipated these problems in advance. Up to this point, this was basically my starting idea for how to tackle the game. The problems with conventional search and with ordinary self-play weren’t things we had to discover through a long series of failed experiments. I had thought through them beforehand.

Later on, contributors and agents suggested approaches like those mentioned above.They thought would work, but which I thought were unlikely to work. We eventually tried some of them, and it was nice to have experimental evidence, but we initially avoided those experiments because we didn’t want to waste compute.

### Why we needed a curriculum

Random play almost never wins as British, and a randomly initialized network initially plays randomly. It therefore learns that Mysore wins everything and evaluates every position as approximately minus one. Search then cannot distinguish between moves using those values and continues to play essentially random moves, reinforcing the same problem. This self-reinforcing loop leads to total value collapse and eventually policy collapse.

The networks could not find the end of the game through search, and they could not reach useful British wins through random play. We therefore started them from positions much closer to the end.

### Constructing the synthetic curriculum

The curriculum positions were constructed directly by hand and then semi-randomized. You could think of the hand-constructed positions as pseudo-sampling from human agents, or from already good players, because the people constructing them knew the game.

We found that models tended not to play well in positions they probably had not encountered during training. Our theory was that the state factories needed to generate a broader variety of positions, so we changed them to produce more varied positions.

A few of the positions came from the idea: why don’t we start with a winning position for the British and perturb it slightly? Those were some of the last positions we added. The midgame and endgame positions were semi-hand-constructed, and of course the standard setup was defined by the game itself.

I also thought about the difficulty of reverse curriculum generation. If you try to step backward through every legal move that could have led to a position, lots of those moves are terrible or would never actually occur in good play. Only a small subset of predecessors will be informative. And the further back you go, the more you face the same branching-tree problem as brute-force search. You could train a turn at a time, and that might work in a general case, but we had something better for this game.

### Direct perturbations and legal move exploration

Luck isn’t inherently reversible, and the game engine isn’t reversible in general. If a fort was destroyed last turn, for example, you don’t necessarily know what the previous position was. But actually we did use something like Florensa-style forward exploration, without knowing about that work at the time. We came up with it independently. It doesn’t strictly work as a reverse process, but it was the final stage of our curriculum.

Over short horizons, approximate reversibility is a decent enough assumption. Some things, like FA, kind of reverse destruction even though the game isn’t actually reversible. And our state factories used a bit of both: actual game moves and direct perturbations of the state.

### What the early networks revealed, and refining the factories

The earliest networks actually showed us problems with the game design.

Later, once we had decent bots, we went back and used them to tune our state factories so they would include more balanced positions, rather than positions that were already clearly decided. We just used the bots’ value estimates to do that, instead of playing out matches from each candidate position, because we wanted to save compute.

Version 13 was the first model whose evaluations we trusted to be reasonably good, which is why we rebalanced the curriculum with its estimates rather than doing so earlier. We wanted the factories to produce positions across the spectrum of British and Mysore advantages, without too many positions that were unwinnable or forced wins for either side. We did not eliminate forced wins entirely: recognizing them was important too.

Later testing showed a discontinuity in playing strength at version 13, but not in evaluation. Earlier models' value estimates predicted game outcomes about as well as version 13's. We think the improved play that led to our trust was probably driven by masking illegal moves, which the policies show began with version 12 beta, and by factorizing Royal Navy and Sea Trade, which began with version 13. Training without masking pushes the probability of illegal moves toward zero, while a masked loss never sees them: in positions from tournament games, versions 8 and 11 put a median of 1.6% and 3.3% of their policy on illegal moves, against 98–99.6% for version 12 beta and every later model. Our distrust of the earlier evaluations therefore did not make much sense. We may have reasoned, somewhat illogically, that because their play was so bad, their evaluations could not be trusted either.

### Distilling version 14

The curriculum-trained v14beta was weaker than v13. We think PCR's reduction in effective search budget may have contributed. The run already took several days on a laptop, and we could not afford to give it the budget we had used for earlier models. But v14beta and v13 played very differently, each showing strengths and winning in different positions. This prompted us to try distillation in the hope of combining their strengths.

We thought it would be useful to take the policy of whichever model was strongest in a given position, but needed a way to decide which model that was. The clearest signal we could use was which model won the game, so we used the winner to label its positions. That seemed sensible for these teachers, though later experiments with other models did not establish a benefit from winner-selected labels.

As documented in the [v14 model card](ai/models/v14%20Model%20Card.md), the two frozen models played 20,000 games at 40 new simulations per decision. For every decision position in a game, including the opponent's turns, the winning model supplied its raw policy over legal moves; the game's outcome supplied the value target. A freshly initialized student learned from these positions without subsequent self-play updates.

The reported student, released as v14, won 119 of 200 games against v13 and 84 of 100 against v14beta. These are historical reports; the model card notes that the underlying release records are not available in this checkout. Combining complementary teachers was one possible explanation for the gain, but the original experiments did not isolate that contribution.

### Blending value targets

The outcome of a single game is a noisy label for an early position. Many moves and luck outcomes still lie between that position and the end, so the eventual winner may not represent how favorable the position was. Near the end, the outcome should generally be a better label. We hoped to reduce this variance by blending the outcome with a value estimate.

In self-play we use the search's value estimate: we already run the search to choose a move, so the estimate costs no additional search. In distillation we use a single forward pass from the teacher instead, saving the cost of searching each recorded position again. The blend gives 75% weight to the game outcome and 25% to the estimate. The larger outcome weight keeps the target anchored to what actually happened, while the estimate is intended to reduce variance. The choice was loosely inspired by prior work, although we have not verified a source for this exact split.

The split has another practical benefit. With outcomes of +1 or −1 and value estimates bounded between −1 and +1, the blended target always has the same sign as the outcome. We can therefore recover the outcome from a saved target and relabel the position with another network without replaying the game or validating positions again.

The mixed-target student retained as v15 scored 211–189 against v14. We regarded the observed result as encouraging and thought the idea likely to work given the broader precedent for value bootstrapping, though the match alone does not establish an improvement. The [experiment notes](DISTILL_RESULTS.md) report an estimated +19 Elo with a 95% interval of −14 to +53; the candidates differed in other training choices as well.

The v15 experiments also investigated whether outcome-only students needed more emphasis on value learning. Changing the relative policy and value loss weights, or balancing their shared-layer gradient norms, did not fix their weak play against v14. A head-swap test with one such student implicated its value outputs under that search setup, but a small branch-ranking pilot did not establish a general value-ranking deficit. We also tried different shuffle schemes, but the comparison changed the validation holdout as well. The mixed-target candidate had the best observed final record, not a controlled proof that the blend alone caused the difference.

We have considered varying the blend over the course of a game, giving early positions more weight on the estimate and later positions more weight on the outcome. A turn-based schedule would need to account for variable game length while avoiding a target based entirely on estimates at the start. Later distillation experiments tested TD(λ), another way to vary the outcome weight. We have not yet decided between TD(λ) and turn-based decay; further work on that choice remains.

### Cross-entropy for the value head

We switched the value loss from mean squared error (MSE) to binary cross-entropy (BCE), motivated by prior work we remember as coming from Leela. We had noticed that even in endgames the network sometimes stopped around 0.99 rather than converging closer to 1, and that it could occasionally be confidently wrong, predicting near one extreme when the target was near the other.

Our concern was the gradient through the bounded value output. With MSE and a saturated tanh, the gradient with respect to the pre-activation can shrink even as a prediction becomes more confidently wrong. BCE on the corresponding win-probability logit gives a gradient proportional to prediction error instead. We hoped that would give a more useful correction for confident mistakes and help the network learn more decisive values where appropriate.

We did not run many experiments isolating this change because the prior work and the reasoning made us fairly confident in it. The model manifest estimates v17beta at 1,500 Elo versus 1,450 for v16beta, which seemed encouraging. Other training settings also changed, including the temperature schedule, so that difference does not establish a gain from BCE alone. We remember a roughly 50 Elo improvement being suggested in the Leela work, but have not verified that figure or the exact source.

An August 2018 [GoNN experiment by KataGo's author](https://github.com/lightvector/GoNN#cross-entropy-vs-l2-value-head-loss-aug-2018) compared cross-entropy and L2 value losses using networks trained on Leela Zero and ELF games. The cross-entropy networks gave more confident predictions, but the author reported no statistically clear average strength difference and noted substantial variation between independently trained networks. A separate [Leela Zero discussion](https://github.com/leela-zero/leela-zero/issues/1610) also reported an experiment with mixed observations. Neither source establishes the remembered 50 Elo gain.

### Choosing a distillation schedule

We investigated whether early stopping and the learning-rate schedule were limiting distillation. The [November 1 experiments](DISTILL_RESULTS.md) trained students on 20,000 v16-versus-v15 games and evaluated checkpoints against v15. Validation score tracked playing strength across the two runs, but it fluctuated more from epoch to epoch than the late training improvements. Replaying early-stopping rules on the logs showed that short patience could stop one run at epoch 8, even though later checkpoints generally played better. We therefore moved away from selecting a checkpoint with short-patience early stopping.

Gradient measurements suggested that a constant learning rate of 0.001 oscillated late in training. Reducing it to 0.0001 made updates less prone to overshooting, but learned too slowly: after 25 epochs, that run had worse policy imitation loss and scored 85–115 against its 0.001 counterpart. A three-epoch cooldown from 0.001 reduced validation policy KL by 7–10% compared with continuing at the constant rate, though the matches did not establish a strength gain from the cooldown alone.

We then tested schedules that decayed linearly to zero over ten epochs. Peak rates around 0.002–0.003 produced similar validation losses, and the 0.002 run matched the observed strength of the longer constant-rate run on the v15 benchmark. A 40-epoch decaying run scored 99–101 against the ten-epoch 0.002 model, suggesting that the shorter schedule was sufficient at this dataset size, within the precision of that match. We adopted ten epochs with a peak rate of 0.002 and no early stopping; these comparisons were limited by noisy match results and a small number of training seeds.

Validation loss is not the same as playing strength. In self-play, it is affected by the data the model generates, which itself changes with the model's strength. Even in distillation, where the dataset is fixed, a better loss did not necessarily mean stronger play: the cooldown improved imitation loss without an established playing-strength gain. The research lesson for us is not to rely too heavily on a proxy; measure the outcome we actually want as well.

We used that recipe on 20,000 v16-versus-v17 games to train v18. In 200-game evaluations it scored 119–81 against v17 and 111–89 against v16. The first result favored v18; the interval for the second included no difference. These comparisons assess the resulting student, not the schedule in isolation.

The same experiments compared a deeper plain MLP and two residual architectures with the existing three-layer student. Residual students fit the policy targets better, but none established better play in 200-game comparisons. Swapping policy and value heads between the baseline and one residual student did not identify a clear source for the residual student's observed deficit. We also tried targets consisting entirely of the teacher's evaluation: they fit the teacher values more closely, but did not improve play over the outcome blend. Local training was limited by random disk reads; keeping the dataset on a Kaggle GPU reduced an epoch from roughly 120 seconds on local CPU to 5.8 seconds, while reproducing the local losses.

### Constant units and AdamW

We think concerns about coupled L2 regularization arose while tuning the distillation learning rate. Together with interpretability work inspecting activations, this led us to look at neurons and notice that many had become nearly constant across positions. Our theory was that L2 weight decay was responsible, so we switched to AdamW, which decouples weight decay from Adam's adaptive gradient scaling.

The [later investigation](DISTILL_RESULTS.md) defined a constant unit as one whose SiLU output had a standard deviation below 0.01 across 1.21 million positions. These units were not necessarily zero: their outputs could be close to a constant determined by their biases, so checking only for zero activation would miss them. Freezing v18's constant units at their means changed no top moves on 61,569 positions and changed values by at most 0.00009.

The records support our regularization explanation. In a v18-recipe rerun with Adam and coupled L2, the number of constant units in the second shared layer grew from zero after one epoch to 167 after ten. With AdamW and weight decay 0.1, none were constant by that measure. The initial AdamW setting of 0.0001 was found to be effectively unregularized over a distillation run, so we increased it to 0.1.

The AdamW student fit the targets better, but scored essentially even against v18 over 1,000 games, with an estimated difference of 0 Elo and a 95% interval of −21 to +21. Thus the experiment showed that AdamW avoided the constant units and improved the measured fit, but did not establish a playing-strength gain. As with the learning-rate experiments, better loss was not sufficient evidence of stronger play.

### What made distillation work?

We investigated why earlier distilled students gained strength over their teachers, and why a student trained on v19beta-versus-v18 games seemed to stop gaining. The [experiments](DISTILL_RESULTS.md) tested several explanations, including complementary teachers, selecting the winner's labels, starting from fresh weights, and the number of distinct games. In those comparisons, neither two teachers, winner-selected labels, nor fresh initialization produced a measurable advantage. A student trained on 20,000 v16 self-play games matched the estimated strength of v18, which had two teachers. These tests used individual training seeds, so they do not rule out smaller effects.

Data volume was the main gain we could identify. Students trained on 5,000, 10,000, and 20,000 distinct v16 self-play games improved as the dataset grew. Training for 40 epochs on 5,000 games, matching the number of updates in the 20,000-game run, did not recover the difference: it rated 87 Elo lower in that comparison. This suggested that seeing more distinct games mattered more than repeatedly training on the smaller set. The experiments also found that a blended value target outperformed an outcome-only target in one controlled comparison, although that result could not distinguish the teacher's information from the effect of softer, less variable targets.

We also considered whether students merely exploited their teachers, or whether a gap between teachers prevented improvement. Tournament comparisons suggested the earlier students' gains extended beyond their own teachers, and v14 and v15 had gained despite a large gap between their teachers. Another hypothesis was that a fresh network needed to reorganize knowledge across the whole game. An initial warm-start test was weak because the high learning rate substantially changed its policy. A later low-rate warm start retained 98% of v16's top moves on sampled positions and gained about as much strength as a fresh student, arguing against the need for that reorganization in this setup. The value-target comparison found the outcome-only warm student about 34 Elo below its blended counterpart; proposed controls for target softness, shuffled labels, and different labeler strengths were not run.

### Reusing the Elo tournament

We found another source of distillation positions in the Elo tournament games. The tournament was expensive to generate, but we needed it to assess our models accurately anyway, so reusing its games had effectively no additional game-generation cost. For each new student, the current pretrained model relabelled the recorded positions with its policy and value from a single forward pass: v19beta labelled the data for v19, and v20beta for v20. The value target also retained the recorded game outcome. We did not replay the games or run a new search for each position.

The benefit came with a limitation: relabelling changes the targets, but the games and positions stay the same as new models are trained. In particular, the tournament included older, weaker models with opening preferences different from those of stronger models. The pretrained models tended to have different favorite openings despite evaluating several top choices as close. Older models often lost after playing their preferred openings, and outcome-based value labels could make an opening look worse because of who played it, rather than because the opening itself was bad. We observed distilled models concentrating on fewer openings, favoring those played by newer models even when the older openings appeared sound to people and to the newer networks.

The v20 analysis gives an example. On the first move Bombay → Goa, 38% of tournament games had v8 as British; British won only 34% of those games. The pretrained v19beta and v20beta evaluated the move positively, while the distilled v19 and v20 evaluated it negatively. Comparisons within the same British bot did not show the same large disadvantage. The analysis found that outcome targets carried player-strength bias mainly in turns one and two, consistent with the opening effect; it did not establish that correcting the bias improved playing strength.

This potential failure mode is not specific to board games. In reinforcement learning or supervised training on trajectories generated by policies of different strengths, outcomes can reflect differences between the policies as well as differences between the states they visit. A model may then learn that a state is bad because a weaker policy often reached it and went on to fail, even when the state would be promising under a stronger policy. Our experiment demonstrates this concern in the tournament dataset; its importance in other settings would need to be tested there.

### Attempts to reduce tournament outcome bias

We tested whether replacing the fixed outcome blend with TD(0.97) targets would reduce player-strength bias while retaining a useful outcome signal. The tested target mixed later positions' v20beta evaluations with the game outcome, reducing the outcome's weight for earlier positions. It reduced the measured early-game bias by roughly 10–15%. Against v20 it scored 1,547–1,453 over 3,000 games, an estimated +11 Elo with a 95% interval of −2 to +23; against v20beta, the two students scored almost identically on paired games. This was encouraging but did not establish a strength gain.

We also tried Tree-Backup, cutting the trace according to the teacher's probability for each played move. This largely removed the measured bias, but the resulting targets stayed close to the teacher's evaluations. The student scored 452–548 against v20 and roughly even against v20beta. Other trace-cut rules were calculated on the data without training students: the rules that removed the bias also removed much of the outcome signal, while retaining more outcome weight retained more bias. We also tested dropping weaker bots' games against random-drop controls. It did not establish an advantage from filtering, and using fewer games risked losing useful data.

The distillation code was changed to a TD(0.97) variant, but the exact implementation includes the current position's teacher value and was not itself trained or rated in the recorded experiment. Self-play retained its separate outcome/root-value blend. Choosing between TD(λ) and a turn-based schedule remains open, as does the broader choice of whether to continue distilling.

For future work, we may generate fresh distillation games to avoid repeatedly reusing this dataset. Alternatively, recent improvements in pretraining efficiency may make it economical to train models directly rather than distill them at all. We have not yet settled on the approach.

### Making pretraining fast enough to compare research approaches

The recent push to make pretraining more efficient has a research purpose: we want to run larger-scale comparisons with frontier models. For example, we could give a model a limited period in a virtual machine to develop and train an agent, then compare it with the best agent we can train in a similar amount of time, or give the model several times more time as a handicap. The question is whether it can discover an approach as effective as the one we developed.

Training efficiency has improved substantially since v8, which we recall taking about three days to train to roughly 1,000 Elo. The gains came from several sources over time, including faster engine and inference code, as well as changes in network architecture and training design. We recall recent mini-model runs reaching roughly 1,600–1,700 Elo in under an hour, and reaching v8-level strength in roughly 10–15 minutes with an appropriate setup. These are recollections; hardware, training recipes and evaluation methods matter when comparing times and ratings.

The preserved [v20mini model card](ai/models/v20mini%20Model%20Card.md) documents a 22.4-minute T4 run estimated around 1,020 Elo, while the [simulation sweep](SIMS_SWEEP_RESULTS.md) reports a roughly 1,626 Elo run in 128 minutes on four CPU cores. These documented runs demonstrate the gains but do not by themselves verify the remembered sub-hour 1,600–1,700 Elo result; their hardware and recipes differ.

The simulation sweep asks how to allocate a fixed search budget between more games and deeper searches. In its initial grid, 50 simulations per full search was the strongest tested setting at each of three budgets, but the replay buffer and number of training updates also changed in their relationship to the amount of data. Follow-up tests found that matching the buffer changed results substantially, while the split between simulations and games still mattered. A growing replay window under the tested schedule underperformed a fixed buffer, and a later grid found that the best buffer size depended on the training ratio. These results are promising guidance for cheap experiments, not a settled scaling law: many comparisons used one training seed, and further seed and full-scale tests were planned.

We want to investigate these tradeoffs more systematically, both to choose settings for future runs and because stronger evidence could be useful to other researchers. [Andy Jones's 2021 Hex study](https://arxiv.org/abs/2104.03113) explored scaling and the tradeoff between training compute and search at evaluation; it used a fixed search size during training, so it is related to, but does not directly answer, our games-versus-training-search question. The [2026 study by Ota and colleagues](https://arxiv.org/abs/2602.10894) includes a more directly relevant search-budget experiment in its appendix. We think there is room to study this question more deeply and across more conditions.

Replay-window size is another choice we want to put on firmer empirical ground. In our initial 50-simulation buffer sweeps at two budgets, the best fitted fixed sizes were about 12–13% of the positions generated in a run; this is an observation for those settings, not a general rule. KataGo's [original paper](https://arxiv.org/abs/1902.10565) specifies a growing, sublinear training window, rather than a fixed percentage. Its main run ended with about 22 million samples in the window out of 241 million generated, and the paper reports an informal window-size adjustment to address overfitting at a high search budget. We want to test buffer sizes and their interactions with search budget and training ratio directly, with enough repeats and scales to assess when the choices transfer.

### Further checks on endgames and width

In [endgame-only self-play experiments](ENDGAME_EXPERIMENTS.md), a 50/50 blend of outcome and search value scored 503–497 against outcome-only training, while a pure search-value target scored 471–529. This did not show a benefit from blending in that short-horizon setup; it differs from the later full-game distillation comparison where blending helped. Increasing outcome-only training from 1,000 to 5,000 games produced a large gain, and 10,000 beat 5,000 more modestly. An endgame-only model trained on 10,000 games scored 462–538 against v15 when evaluated from those endgame positions. These were specific endgame tests and do not establish the best target or budget for the whole curriculum.

The [width-transfer experiments](MUP_RESULTS.md) tested whether maximal update parametrization (muP) could transfer a learning rate across network widths. On fixed distillation data, the best tested peak rate under muP was the same at widths 64, 128, and 256. A 512-wide muP student fit the targets better than v20 but scored 495–505 against it, an estimated −3 Elo with a 95% interval of −22 to +15. We kept width 256: the experiment did not establish a playing-strength benefit from doubling width on this dataset. Depth scaling and the wider network's performance with more data remain open.
