---
title: Towards Automated Detection of Single-Trace Side-Channel Vulnerabilities in
  Constant-Time Cryptographic Code
authors:
- admin 
- Ruzica Piskac
- Victor Mateu
- Jakub Szefer
date: '2023-07-01'
publishDate: '2023-11-25T07:08:35.405532Z'
publication_types:
- paper-conference
publication: '*European Symposium on Security and Privacy (EuroS&P)*'
doi: 10.1109/EuroSP57164.2023.00047
abstract: 'Although cryptographic algorithms may be mathematically secure, it is often
  possible to leak secret information from the implementation of the algorithms. Timing
  and power side-channel vulnerabilities are some of the most widely considered threats
  to cryptographic algorithm implementations. Timing vulnerabilities may be easier
  to detect and exploit, and all high-quality cryptographic code today should be written
  in constant-time style. However, this does not prevent power side-channels from
  existing. With constant time code, potential attackers can resort to power side-channel
  attacks to try leaking secrets. Detecting potential power side-channel vulnerabilities
  is a tedious task, as it requires analyzing code at the assembly level and needs
  reasoning about which instructions could be leaking information based on their operands
  and their values. To help make the process of detecting potential power side-channel
  vulnerabilities easier for cryptographers, this work presents Pascal: Power Analysis
  Side Channel Attack Locator, a tool that introduces novel symbolic register analysis
  techniques for binary analysis of constant-time cryptographic algorithms, and verifies
  locations of potential power side-channel vulnerabilities with high precision. Pascal
  is evaluated on a number of implementations of post-quantum cryptographic algorithms,
  and it is able to find dozens of previously reported single-trace power side-channel
  vulnerabilities in these algorithms, all in an automated manner.'
tags:
- power side-channels
- hamming weight
- differential program analysis
- post-quantum cryptography
- symbolic execution
- relational program analysis
- binary analysis
links:
- name: URL
  url: https://doi.ieeecomputersociety.org/10.1109/EuroSP57164.2023.00047

# Display this page in the Featured widget?
featured: true

url_slides: 'https://eurosp2023.ieee-security.org/slides/EuroSP-Pascal-Slides.pdf'
url_video: 'https://youtu.be/1w_jSuvThD4'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  placement: 1 
  caption: ''
  focal_point: 'Smart'
  preview_only: false

show_breadcrumb: false

# Summary. An optional shortened abstract.
summary: Pascal is a tool that introduces novel symbolic register analysis techniques for constant-time low-level cryptographic code, and verifies locations of potential single-trace power side-channel vulnerabilities with high precision. Pascal is evaluated on a number of implementations of post-quantum cryptographic algorithms, and it is able to find dozens of previously reported single-trace power side-channel vulnerabilities in these algorithms, all in an automated manner.

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
