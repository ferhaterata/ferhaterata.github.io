---
title: Exploration of Power Side-Channel Vulnerabilities in Quantum Computer Controllers
authors:
- Chuanqi Xu
- admin 
- Jakub Szefer
date: '2023-11-01'
publishDate: '2023-11-25T07:08:35.394792Z'
publication_types:
- paper-conference
publication: '*Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications
  Security (CCS)*'
doi: 10.1145/3576915.3623118
abstract: The rapidly growing interest in quantum computing also increases the importance
  of securing these computers from various physical attacks. Constantly increasing
  qubit counts and improvements to the fidelity of the quantum computers hold great
  promise for the ability of these computers to run novel algorithms with highly sensitive
  intellectual property. However, in today's cloud-based quantum computer setting,
  users lack physical control over the computers. Physical attacks, such as those
  perpetrated by malicious insiders in data centers, could be used to extract sensitive
  information about the circuits being executed on these computers. This work shows
  the first exploration and study of power-based side-channel attacks in quantum computers.
  The explored attacks could be used to recover information about the control pulses
  sent to these computers. By analyzing these control pulses, attackers can reverse-engineer
  the equivalent gate-level description of the circuits, and the algorithms being
  run, or data hard-coded into the circuits. This work introduces five new types of
  attacks, and evaluates them using control pulse information available from cloud-based
  quantum computers. This work demonstrates how and what circuits could be recovered,
  and then in turn how to defend from the newly demonstrated side-channel attacks
  on quantum computing systems.
tags:
- power side-channel vulnerabilities
- quantum computer controllers
- quantum computers
links:
- name: URL
  url: https://doi.org/10.1145/3576915.3623118

# Display this page in the Featured widget?
featured: true

url_slides: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  placement: 3 
  caption: ''
  focal_point: 'Bottom'
  preview_only: false

show_breadcrumb: false

# Summary. An optional shortened abstract.
summary: This work shows
  the first exploration and study of power-based side-channel attacks in quantum computers.
  The explored attacks could be used to recover information about the control pulses
  sent to these computers. By analyzing these control pulses, attackers can reverse-engineer
  the equivalent gate-level description of the circuits, and the algorithms being
  run, or data hard-coded into the circuits.

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
---
