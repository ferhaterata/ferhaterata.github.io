---
title: Learning Randomized Reductions and Program Properties 
authors:
- admin 
- Orr Paradise
- Timos Antonopoulos
- ThanhVu Nguyen
- Shafi Goldwasser
- Ruzica Piskac
date: '2024-01-16'
publishDate: '2024-11-16'
abstract: "We introduce Bitween, a linear regression-based automated learning algorithm that effectively discovers complex nonlinear randomized (self)-reductions and other arbitrary program invariants (e.g., loop invariants and post conditions) in mixed integer and floating-point programs."

tags:
- Randomized Self-Reductions
- Loop Invariants
- Nonlinear equations
- Self-Correctness
- Machine Learning
- Formal Methods
- Linear Regression
- Program Properties
- Dynamic Analysis
- Supervised learning by regression
links:
- name: Tool Website
  url: https://bitween.fun
- name: Preprint
  url: https://arxiv.org/abs/xxxx.xxxxx

# Display this page in the Featured widget?
featured: true

url_slides: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  placement: 1
  caption: ''
  focal_point: 'Bottom'
  preview_only: false

show_breadcrumb: false

# Summary. An optional shortened abstract.
summary: "We introduce Bitween, a linear regression-based automated learning algorithm that effectively discovers complex nonlinear randomized (self)-reductions and other arbitrary program invariants (e.g., loop invariants and post conditions) in mixed integer and floating-point programs."

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set 
---
1. We introduce Bitween, a linear regression-based automated learning algorithm that effectively discovers complex nonlinear randomized (self)-reductions and other arbitrary program invariants (e.g., loop invariants and post conditions) in mixed integer and floating-point programs. 
2. We present a rigorous theoretical framework for learning randomized (self)-reductions, advancing formal foundations in this area. 
3. We create a benchmark suite, RSR-Bench, to evaluate Bitween's effectiveness in learning randomized reductions. This suite includes a diverse set of mathematical functions commonly used in scientific and machine learning applications. Our evaluation compares Bitween's linear regression backend, which we call Bitween, against a Mixed-Integer Linear Programming (MILP) backend, which we call Bitween-Milp, showing that linear regression-based learning outperforms MILP in both scalability and stability within this domain. 
4. We implement Bitween in Python and demonstrate its performance on an extended version of NLA-DigBench, which includes nonlinear loop invariants and post-conditions. Our empirical evaluation shows that Bitween surpasses leading tools DIG and SymInfer in capability, runtime efficiency, and sampling complexity.