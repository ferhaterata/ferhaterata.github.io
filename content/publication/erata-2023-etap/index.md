---
title: 'ETAP: Energy-aware timing analysis of intermittent programs'
authors:
- Ferhat Erata
- Eren Yildiz
- Arda Goknil
- Kasim Sinan Yildirim
- Jakub Szefer
- Ruzica Piskac
- Gokcin Sezgin
date: '2023-01-01'
publishDate: '2023-11-25T04:44:50.528717Z'
publication_types:
- article-journal
publication: '*ACM Transactions on Embedded Computing Systems (TECS)*'
doi: 10.1145/3563216
abstract: Energy harvesting battery-free embedded devices rely only on ambient energy
  harvesting that enables stand-alone and sustainable IoT applications. These devices
  execute programs when the harvested ambient energy in their energy reservoir is
  sufficient to operate and stop execution abruptly (and start charging) otherwise.
  These intermittent programs have varying timing behavior under different energy
  conditions, hardware configurations, and program structures. This article presents
  Energy-aware Timing Analysis of intermittent Programs (ETAP), a probabilistic symbolic
  execution approach that analyzes the timing and energy behavior of intermittent
  programs at compile time. ETAP symbolically executes the given program while taking
  time and energy cost models for ambient energy and dynamic energy consumption into
  account. We evaluate ETAP by comparing the compile-time analysis results of our
  benchmark codes and real-world application with the results of their executions
  on real hardware. Our evaluation shows that ETAP’s prediction error rate is between
  0.0076% and 10.8%, and it speeds up the timing analysis by at least two orders of
  magnitude compared to manual testing.
tags:
- symbolic execution
- timing analysis
- Intermittent computing
- energy harvesting
links:
- name: URL
  url: https://doi.org/10.1145/3563216
---
