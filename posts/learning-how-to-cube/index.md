*Companion post to [Learning How to Cube](https://arxiv.org/abs/2605.16632), accepted as a poster at NeurIPS 2026 on September 24, 2026. Updated September 24, 2026.*

Our paper *Learning How to Cube* has been accepted as a poster at NeurIPS 2026. We trained a 4B-parameter transformer to choose splits in SAT formulas, using feedback grounded in the work a symbolic solver actually performs. Across five runs, it solves **53 of 100 held-out competition benchmarks**, tying the best symbolic cubing heuristic under that budget. The interesting part is how the training changes the decisions the model makes.

## The setup, briefly

Cube-and-Conquer splits a propositional satisfiability (SAT) formula into subproblems called cubes, then uses a symbolic solver to solve them. A cubing heuristic chooses the variable to split on. In *Learning How to Cube*, a transformer learns that choice; the symbolic solver remains responsible for solving the resulting subproblems.

We train the 4B transformer in two stages: **supervised fine-tuning (SFT)** on teacher-generated reasoning traces, followed by **direct preference optimisation (DPO)**. Monte Carlo Tree Search (MCTS) produces preference pairs grounded in solver outcomes. These preferences connect the model's splitting choices to downstream solver performance.

<figure class="post-figure">
<img src="figures/neural_heuristic.png" alt="Cube-and-Conquer with a neural cubing heuristic: the transformer chooses a splitting variable, and a symbolic solver works on the resulting subproblems." />
<figcaption>The learned component chooses how to split the formula; the symbolic solver handles the resulting subproblems.</figcaption>
</figure>

## What the result actually says

The evaluation uses 100 held-out SAT competition benchmarks, with five runs per heuristic and a 30-minute timeout per run. Here, **pass@5** counts benchmarks solved in at least one of the five runs. It measures coverage across those attempts, whereas the per-run mean measures average coverage in one attempt.

| Heuristic | pass@5 (out of 100) | Per-run mean (out of 100) |
|---|---|---|
| Qwen3-4B-SFT-DPO | 53 | 47.4 |
| `unit` (best symbolic heuristic by pass@5 in this evaluation) | 53 | 51.6 |

The two methods tie on pass@5, while `unit` has the higher per-run mean. Our model reaches the same overall coverage by solving different instances on different attempts. This comparison uses up to 30 minutes per run, or 2.5 hours per benchmark across five runs, and does not establish equal single-run performance or equal decision cost.

## What changes with training

The training ablation separates the contributions of the two stages. SFT takes the base model from **46 to 51** benchmarks solved at pass@5. Adding DPO reaches **53**, a further gain of two benchmarks.

| Training stage | pass@5 (out of 100) |
|---|---|
| Base model | 46 |
| SFT | 51 |
| SFT + DPO | 53 |

<figure class="post-figure">
<img src="figures/training_stage_ablation.png" alt="Training ablations comparing pass@5 and first-split diversity measurements for the base model, DPO-only, SFT-only, SFT plus DPO, and the teacher model." />
<figcaption>The training ablations report both coverage and first-split diversity. Along the base → SFT → SFT+DPO sequence, pass@5 is 46 → 51 → 53. The diversity measurements describe observed behaviour; they do not establish why coverage improves.</figcaption>
</figure>

## Interpreting diversity measurements

Across five runs, our model covers more benchmarks than its per-run mean suggests: 53 versus 47.4. First-split measurements also show more variation after SFT than in the base model. These observations motivate studying how different runs complement one another.

They do not establish that diversity caused the coverage gain, that exploration is calibrated, or that increasing diversity would improve performance. The ablations change training as well as behaviour. Isolating the role of diversity would require an experiment designed for that question.

## Why I find this useful

The contribution is a concrete way to train a small transformer for a decision inside a symbolic reasoning system. Symbolic search helps construct the training data, the transformer learns to propose splits, and solver rollouts let us measure whether those choices help. The five-run results show that this can produce coverage competitive with established cubing heuristics.

I want to understand which formulas benefit most from those learned choices and how to reduce the cost of making them. The difference between the SFT and DPO stages also raises a useful training question: how can we teach a model to make several productive attempts at a difficult search problem? We now have a working system and a benchmark on which to investigate it.
