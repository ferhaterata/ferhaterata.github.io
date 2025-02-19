---
title: Scaling Combinatorial Reasoning with Conflict-Driven Clause Learning In-Context
authors:
- Yoshiki Takashima
- admin 
date: '2025-02-16'
publishDate: '2025-02-16'
publication_types:
- paper-conference
# publication: '**'
# doi: 10.1109/EuroSP57164.2023.00047
abstract: "Recent advances in reasoning models have led to impressive performance in several areas of reason: first-order logic, mathematics, and computer programming to name a few. Yet these models do not scale to combinatorial reasoning tasks. These tasks require the model to find a solution out of a combinatorially large search space while reasoning correctly at each step of the search. Existing techniques for such problems are neuro-symbolic: they translate the problem into a formal representation and use symbolic solvers to conquer the combinatorial search space.  These approaches are often limited to rigid reasoning tasks that have exact formal translations and even then, translation incurs an error overhead, leading to lower performance. We propose CDCL-IC, a Chain-of-Thought (CoT) reasoning technique that conquers combinatorial search problems through Conflict-Driven Clause Learning~(CDCL). Our technique uses CDCL to learn bad patterns during backtracking CoT reasoning and blocks it using in-context learning to prune the search space. We implement CDCL-IC for Sudoku, and show that our approach greatly outperforms both traditional CoT and o3-mini on 9x9 Sudoku problems."
tags:
- CDCL 
- Test-time compute
- Neuro-symbolic Reasoning
- Reward Modeling 
- Inference Time
# links:
# - name: Preprint
#   url: https://arxiv.org/abs/2412.18134
# - name: Tool Website
#   url: https://bitween.fun

# projects:
# - randomized-reductions

# Display this page in the Featured widget?
featured: true

# url_slides: 'https://eurosp2023.ieee-security.org/slides/EuroSP-Pascal-Slides.pdf'
# url_video: 'https://youtu.be/1w_jSuvThD4'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  placement: 1 
  caption: ''
  focal_point: 'Smart'
  preview_only: false

show_breadcrumb: false

# Summary. An optional shortened abstract.
summary: "Recent advances in reasoning models have improved performance in logic, mathematics, and programming but struggle with combinatorial reasoning tasks due to the vast search space. Existing neuro-symbolic techniques rely on formal translations and symbolic solvers but are limited and error-prone. We introduce CDCL-IC, a Chain-of-Thought (CoT) reasoning approach leveraging Conflict-Driven Clause Learning (CDCL) to prune search spaces via in-context learning. Applied to Sudoku, CDCL-IC significantly outperforms traditional CoT and o3-mini on 9x9 puzzles."

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
#   - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set 
categories:
- selected
---
