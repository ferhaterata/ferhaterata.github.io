---
title: Systematic Use of Random Self-Reducibility against Physical Attacks
authors:
- admin 
- TingHung Chiu
- Anthony Etim
- Srilalith Nampally
- Tejas Raju
- Rajashree Ramu
- Ruzica Piskac
- Timos Antonopoulos
- Wenjie Xiong
- Jakub Szefer
date: '2024-01-15'
publishDate: '2024-01-15'
publication_types:
- article-journal
publication: '*IEEE/ACM International Conference on Computer-Aided Design (ICCAD)*'
doi: 10.1145/3676536.3689920
abstract: 'This work presents a novel, black-box software-based countermeasure against physical attacks including power side-channel and fault-injection attacks. The approach uses the concept of random self-reducibility and self-correctness to add randomness and redundancy in the execution for protection. Our approach is at the operation level, is not algorithm-specific, and thus, can be applied for protecting a wide range of algorithms. The countermeasure is empirically evaluated against attacks over operations like modular exponentiation, modular multiplication, polynomial multiplication, and number theoretic transforms. An end-to-end implementation of this countermeasure is demonstrated for RSA-CRT signature algorithm and Kyber Key Generation public key cryptosystems. The countermeasure reduced the power side-channel leakage by two orders of magnitude, to an acceptably secure level in TVLA analysis. For fault injection, the countermeasure reduces the number of faults to 95.4% in average.' 
tags:
- Random Self-Reducibility 
- Fault Injection Attacks
- Power Side-Channel Attacks
- Countermeasure
- NTT
- PQC
- RSA-CRT
- Randomly Testable Functions
links:
# - name: URL
#   url: https://tches.iacr.org/index.php/TCHES/article/view/11445

url_slides: '/slides/erata-2024-systematic.pdf'
url_video: 'https://youtu.be/0Srjasg9eR8'

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
  - side-channel-analysis

categories:
- selected
---
