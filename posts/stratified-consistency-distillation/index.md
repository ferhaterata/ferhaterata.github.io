The policies might describe authorization rules, filesystem access, or network permissions. Once those rules have a formal representation, a system can check an agent's actions or a formalized interpretation of an LLM's answer against them. **But the check is only meaningful if the translation preserved what the policy meant.**

A frontier model with a carefully tuned prompt can do these translations well. Using it for every translation is expensive, though, and closed weights limit our options for adapting and deploying it. We wanted to distill that ability into smaller open models we could fine-tune and run ourselves.

That is the motivation behind [*Stratified Consistency Distillation for Natural Language Formalization*](https://arxiv.org/abs/2608.30258).

## Choosing what to teach the student

Our experiments focus on [SMT-LIB](https://smt-lib.org/), the language used to express formulas for SMT solvers. For each input, we sample multiple translations from a frontier model and use [Z3](https://github.com/Z3Prover/z3) to group logically equivalent ones. Two translations can use different syntax and still express the same rule.

We then compute semantic entropy across those clusters. A large, dominant cluster means the model mostly agrees with itself. Several similarly sized clusters mean it has produced competing interpretations.

That disagreement determines how we select the *pseudo-label*: the translation we will use as the student's training target.

- **Low entropy:** take a translation from the largest cluster, the majority answer.
- **Medium entropy:** ask an LLM judge to choose between the two leading clusters.
- **High entropy:** try to unify the leading translations, or drop the example.

The intuition is that agreement gives us a useful training signal, some disagreement warrants another look, and too much makes the label unreliable. Equivalence checking tells us when translations agree; checking whether they preserve the original meaning remains the central problem.

<figure class="post-figure post-figure--scientific">
<a href="/news/stratified-consistency-distillation/scd-overview.png" target="_blank" rel="noopener" aria-label="Open the Stratified Consistency Distillation figure at full resolution"><img src="/news/stratified-consistency-distillation/scd-overview.png" width="2600" height="1628" loading="lazy" decoding="async" alt="Stratified Consistency Distillation: generate SMT-LIB translations, cluster them by logical equivalence, select training targets according to semantic entropy, and fine-tune a smaller model. Three examples below show the correct translation in different clusters as disagreement increases." /></a>
<figcaption>We use the distribution of meanings to decide what to distill. The stars in the examples below mark the correct translations; a majority vote becomes less reliable as disagreement grows. <a href="/news/stratified-consistency-distillation/scd-overview.png" target="_blank" rel="noopener">Open the full-resolution figure ↗</a></figcaption>
</figure>

During his internship with the AWS Automated Reasoning group, Zhichao Hou built this into a distillation pipeline. We use the selected translations to fine-tune Qwen2.5-7B-Instruct.

## What changes after fine-tuning

On FOLIO, the student improves from **21.9% to 55.2% Pass@10**. Here, a translation succeeds when Z3 finds it logically equivalent to the reference formula. Pass@10 measures whether at least one of ten attempts succeeds; Pass@1 measures performance with one attempt.

| Qwen2.5-7B-Instruct | Pass@1 | Pass@10 |
|---|---:|---:|
| Before fine-tuning, with few-shot prompting | 15.6% | 21.9% |
| Ordinary distillation | 39.9% | 50.3% |
| Stratified Consistency Distillation | **44.1%** | **55.2%** |

The comparison with ordinary distillation matters. Learning from the teacher already accounts for a large improvement. Choosing labels according to disagreement adds another 4.9 percentage points at Pass@10. These results are reported in [Table 1 of the paper](https://arxiv.org/pdf/2608.30258#page=5).

The student is also faster in the paper's inference comparison. Its median translation time is **4.040 seconds**, compared with **16.680 seconds** for Claude Sonnet 3.7, about **4.1× faster** in that setup ([Table 2](https://arxiv.org/pdf/2608.30258#page=6)).

## From translations to policy checks

SMT-LIB is one way to represent a policy. [Cedar](https://cedarpolicy.com/) expresses authorization rules, while [Common Expression Language (CEL)](https://cel.dev/overview/cel-overview) lets applications express and evaluate conditions. [Lean](https://lean-lang.org/) provides a language and proof assistant for stating and proving properties.

For agents, [Dogwood](https://aws.amazon.com/blogs/opensource/introducing-dogwood-runtime-verification-for-ai-agents/) combines Cedar policies with temporal conditions over actions. [OpenShell's YAML policies](https://docs.nvidia.com/openshell/how-it-works/policies/schema) describe filesystem and network access. NVIDIA's [write-up on formal methods and Z3](https://nvidia.github.io/OpenShell-Research/dev-notes/posts/2026-09-10-learning-formal-methods-agent-policy-prover/) shows a concrete use of formal reasoning here: checking whether a proposed policy stays within an approved access boundary, and producing a counterexample when it does not.

The distillation recipe could extend to other target languages if we can supply a suitable semantic equivalence checker. The implementation and evaluation in this paper use SMT-LIB and Z3. Applying it elsewhere would mean defining what equivalence means for that language, implementing the check, and evaluating the resulting translations.

An early version of this work will appear at the [**AI for Verifiable Coding** workshop at NeurIPS 2026](https://vericodegen.github.io/).

Joint work with Zhichao Hou, Joseph Lilien, Rémi Delmas, and Ali Torkamani.

[Read the paper: *Stratified Consistency Distillation for Natural Language Formalization*](https://arxiv.org/abs/2608.30258) · [Workshop announcement](/news/stratified-consistency-distillation/)
