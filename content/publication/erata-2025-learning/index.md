---
title: Learning Randomized Reductions and Program Properties
authors:
- admin 
- Orr Paradise
- Timos Antonopoulos
- ThanhVu Nguyen
- Shafi Goldwasser
- Ruzica Piskac
date: '2025-01-16'
publishDate: '2024-11-16'
publication_types:
- paper-conference
# publication: '**'
# doi: 10.1109/EuroSP57164.2023.00047
abstract: "The correctness of computations remains a significant challenge in computer science, with traditional approaches relying on automated testing or formal verification. Self-testing/correcting programs introduce an alternative paradigm, allowing a program to verify and correct its own outputs via randomized reductions, a concept that previously required manual derivation. In this paper, we present Bitween, a method and tool for automated learning of randomized (self)-reductions and program properties in numerical programs. Bitween combines symbolic analysis and machine learning, with a surprising finding: polynomial-time linear regression, a basic optimization method, is not only sufficient but also highly effective for deriving complex randomized self-reductions and program invariants, often outperforming sophisticated mixed-integer linear programming solvers. We establish a theoretical framework for learning these reductions and introduce RSR-Bench, a benchmark suite for evaluating Bitween's capabilities on scientific and machine learning functions. Our empirical results show that Bitween surpasses state-of-the-art tools in scalability, stability, and sample efficiency when evaluated on nonlinear invariant benchmarks like NLA-DigBench. Bitween is open-source as a Python package and accessible via a web interface that supports C language programs."
tags:
- Randomized Self-Reductions
- Self-Correctness
- Machine Learning
- Formal Methods
- Dynamic Analysis
- Program Properties
- Linear Regression
- Nonlinear equations
- Supervised learning by regression
- Loop Invariants
links:
- name: Preprint
  url: https://arxiv.org/abs/2412.18134
- name: Tool Website
  url: https://bitween.fun

projects:
- randomized-reductions

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
summary: "We introduce Bitween, a linear regression-based automated learning algorithm that effectively discovers complex nonlinear randomized (self)-reductions and other arbitrary program invariants (e.g., loop invariants and post conditions) in mixed integer and floating-point programs."

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
