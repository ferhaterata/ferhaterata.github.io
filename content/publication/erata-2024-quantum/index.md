---
title: Quantum Circuit Reconstruction from Power Side-Channel Attacks on Quantum Computer Controllers
authors:
- admin 
- Chuanqi Xu
- Ruzica Piskac
- Jakub Szefer
date: '2024-01-15'
publishDate: '2023-12-16'
publication_types:
- article-journal
publication: '*IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)*'
# doi: 10.1145/3564785
abstract: 'The interest in quantum computing has grown rapidly in recent years, and with it grows the importance of securing quantum circuits. A novel type of threat to quantum circuits that dedicated attackers could launch are power trace attacks. To address this threat, this paper presents first formalization and demonstration of using power traces to unlock and steal quantum circuit secrets. With access to power traces, attackers can recover information about the control pulses sent to quantum computers. From the control pulses, the gate level description of the circuits, and eventually the secret algorithms can be reverse engineered. This work demonstrates how and what information could be recovered. This work uses algebraic reconstruction from power traces to realize two new types of single trace attacks: per-channel and total power attacks. The former attack relies on per-channel measurements to perform a brute-force attack to reconstruct the quantum circuits. The latter attack performs a single-trace attack using Mixed-Integer Linear Programming optimization. Through the use of algebraic reconstruction, this work demonstrates that quantum circuit secrets can be stolen with high accuracy. Evaluation on 32 real benchmark quantum circuits shows that our technique is highly effective at reconstructing quantum circuits. The findings not only show the veracity of the potential attacks, but also the need to develop new means to protect quantum circuits from power trace attacks. Throughout this work real control pulse information from real quantum computers is used to demonstrate potential attacks based on simulation of collection of power traces.' 
tags:
- Quantum Circuits
- Quantum Computers
- Side Channel Attacks
- Power Trace Attack
- Automated Reasoning
- Mixed Integer Linear Programming
- MILP

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
  - side-channel-analysis

---
