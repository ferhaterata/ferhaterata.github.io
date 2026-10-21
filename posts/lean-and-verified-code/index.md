Our interns [Ben Mikek](https://mikekben.github.io/), [George Ma](https://georgemlp.github.io/), and [Haoyu Li](https://haoyuli02.github.io/) have done incredible work on AI for formal reasoning. Three papers from this collaboration are now on arXiv.

- **[SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190)** pairs real software issues with formal specifications and verified implementations, with [Lean](https://lean-lang.org/) among its backends. It also examines a crucial question: does the specification actually capture the intended fix?

- **[Verifier-Grounded Agentic RL for Provably Correct Code](https://arxiv.org/abs/2610.19298)** trains coding agents using detailed verifier feedback. The reward recognizes progress on proof obligations, while integrity checks reject attempts to change the specification or assume away the proof.

- **[Maestro: Agentic Formal Proof Development by Coordination](https://arxiv.org/abs/2610.25920)** coordinates proof-writing agents in Lean. When an attempt fails, a coordinator can guide a prover, correct a false lemma, or revise the plan. Checked counterexamples can also send generated code back for repair.

Together, they connect three parts of the problem: what to specify, how to learn from verification, and how to organize the search for a proof. A successful check is still only as useful as the specification it checks against.

Thanks also to Zeren Shui, Yuhao Zhang, Murali Krishna Ramanathan, and Hao Zhou for the great collaboration, and to all our coauthors. It's been a pleasure working on this together.
