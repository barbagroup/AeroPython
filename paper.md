---
title: 'Aero Python: classical aerodynamics of potential flow using Python'
tags:
- Python
- Aerodynamics
- differential equations
- numerical methods
- finite differences
authors:
- name: Lorena A. Barba
  orcid: 0000-0001-5812-2711
  affiliation: 1
- name: Olivier Mesnard
  orcid: 0000-0001-5335-7853
  affiliation: "1"
affiliations:
- name: The George Washington University
  index: 1
date: 11 November 2018
bibliography: paper.bib
---

# Summary

The **AeroPython** learning module is a collection of Jupyter notebooks: one "lesson zero" introduction to Python and NumPy arrays; 11 lessons in potential flow using Python; three student assignments involving coding; and one extra notebook with an exercise deriving the panel-method equations.

The list of lessons is:

* _Python crash course_: quick introduction to Python, NumPy arrays and plotting with Matplotlib.
* _Source \& sink_: introduction to potential-flow theory; computing and plotting a source-sink pair.
* _Source \& sink in a freestream_: adds a freestream to a source to get a Rankine half-body; then adds a freestream to a source-sink pair to get a Rankine oval; introduces Python functions.
* _Doublet_: develops a doublet singularity from a source-sink pair, at the limit of zero distance; adds a freestream to get flow around a cylinder.
* _Assignment 1: source distribution on an airfoil_.
* _Vortex_: a potential vortex, a vortex and sink; idea of irrotational flow.
* _Infinite row of vortices_: superposition of many vortices to represent a vortex sheet.
* _Lift on a cylinder_: superposition of a doublet, a freestream, and a vortex; computing lift and drag.
* _Assignment 2: the Joukowski transformation_.
* _Method of images_: source near a plane wall; vortex near a wall; vortex pair near a wall; doublet near a wall parallel to a uniform flow. Introduces Python classes.
* _Source sheet_: a finite row of sources, then an infinite row of sources along one line. Introduces SciPy for integration.
* _Flow over a cylinder with source panels_: calculates the source-strength distribution that can produce potential flow around a circular cylinder.
* _Source panel method_: solves for the source-sheet strengths to get flow around a NACA0012 airfoil.
* _Vortex-source panel method_: start with the source panel method of the previous lesson, and add circulation to get a lift force. Introduces the idea of the Kutta condition.
* _Exercise: Derivation of the vortex-source panel method_.
* _Assignment: 2D multi-component airfoil_.