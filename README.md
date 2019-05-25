# Classical Aerodynamics with Python

[![Version](https://img.shields.io/badge/version-1.0-blue.svg)](None)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/barbagroup/AeroPython/raw/master/LICENSE)
[![License](https://img.shields.io/badge/license-CC--BY%204.0-lightgrey.svg)](https://github.com/barbagroup/AeroPython/raw/master/LICENSE)
[![status](https://jose.theoj.org/papers/b679b34c976beec0bc64807bf087a468/status.svg)](http://jose.theoj.org/papers/b679b34c976beec0bc64807bf087a468)

The _AeroPython_ series of lessons is the core of a university course (Aerodynamics-Hydrodynamics, MAE-6226) by Prof. Lorena A. Barba at the George Washington University.
The first version ran in Spring 2014 and these Jupyter Notebooks were prepared for that class, with assistance from Barba-group PhD student Olivier Mesnard.
In Spring 2015, we revised and extended the collection, adding student assignments to strengthen the learning experience.
The course is also supported by an open learning space in the [GW SEAS Open edX](https://openedx.seas.gwu.edu/) platform.

The materials are distributed publicly and openly under a Creative Commons Attribution license, [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Cite as:

Barba, Lorena A., Mesnard, Olivier (2019). Aero Python: classical aerodynamics of potential flow using Python. Journal of Open Source Education, 2(15), 45, https://doi.org/10.21105/jose.00045

## Archive

— Barba, Lorena A.; Mesnard, Olivier (2014): AeroPython. figshare. Code.
DOI: [10.6084/m9.figshare.1004727.v5](https://doi.org/10.6084/m9.figshare.1004727.v5)

## List of notebooks

### 0. Getting Started

* [Quick Python Intro](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/00_Lesson00_QuickPythonIntro.ipynb)

### Module 1. Building blocks of potential flow

1. [Source & Sink](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/01_Lesson01_sourceSink.ipynb)
2. [Source & Sink in a Freestream](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/02_Lesson02_sourceSinkFreestream.ipynb)
3. [Doublet](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/03_Lesson03_doublet.ipynb)
4. [Assignment: Source distribution on an airfoil](http://nbviewer.ipython.org/github/barbagroup/AeroPython/blob/master/lessons/03_Lesson03_Assignment.ipynb)

### Module 2. Potential vortices and lift

1. [Vortex](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/04_Lesson04_vortex.ipynb)
2. [Infinite row of vortices](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/05_Lesson05_InfiniteRowOfVortices.ipynb)
3. [Vortex Lift on a cylinder](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/06_Lesson06_vortexLift.ipynb)
4. [Assignment: Joukowski transformation](http://nbviewer.ipython.org/github/barbagroup/AeroPython/blob/master/lessons/06_Lesson06_Assignment.ipynb)

### Module 3. Source-panel method for non-lifting bodies

1. [Method of Images](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/07_Lesson07_methodOfImages.ipynb)
2. [Source Sheet](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/08_Lesson08_sourceSheet.ipynb)
3. [Flow over a cylinder with source panels](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/09_Lesson09_flowOverCylinder.ipynb)
4. [Source panel method](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/10_Lesson10_sourcePanelMethod.ipynb)

### Module 4. Vortex-source panel method for lifting bodies

1. [Vortex-source panel method](http://nbviewer.ipython.org/urls/github.com/barbagroup/AeroPython/blob/master/lessons/11_Lesson11_vortexSourcePanelMethod.ipynb)
2. [Exercise: Derivation of the vortex-source panel method](http://nbviewer.ipython.org/github/barbagroup/AeroPython/blob/master/lessons/11_Lesson11_Exercise.ipynb)
3. [Assignment: 2D multi-component airfoil](http://nbviewer.ipython.org/github/barbagroup/AeroPython/blob/master/lessons/11_Lesson11_Assignment.ipynb)

## Statement of need

Classical aerodynamics based on potential theory can be an arid subject when presented in the traditional "pen-and-paper" approach. It is a fact that the mathematical framework of potential flow was the only tractable way to apply theoretical calculations in aeronautics through all the early years of aviation, including the development of commercial aircraft into the 1980s and later. Yet, the only way to exercise the power of potential-flow aerodynamics is through numerical computation. Without computing, the student can explore only the simplest fundamental solutions of the potential equation: point sinks and sources, point vortex, doublet, uniform flow.

The essential tool for applying this theoretical framework to aerodynamics is the panel method, which obtains the strength of a distribution of singularities on a body that makes the body a closed streamline. The addition of vortex singularities to satisfy a Kutta condition allows treating lifting bodies (like airfoils). The AeroPython series begins with simple point-singularity solutions of the potential equation, and applies the principle of superposition to show how to obtain streamline patterns corresponding to flow around objects. Around the half-way point, the module presents the learner with the fundamental relationship between circulation (via a point vortex) and the production of a lift force. Using a distribution of many point singularities on an airfoil, finally, the module shows how we can obtain pressure distributions, and the lift around an airfoil. With this foundation, the student is ready to apply the panel method in authentic engineering situations.

## Dependencies

To use these lessons, you need Python 3, and the standard stack of scientific Python: NumPy, Matplotlib, SciPy.
And of course, you need [Jupyter](http://jupyter.org)—an interactive computational environment that runs on a web browser.

This mini-course is built as a set of [Jupyter notebooks](https://jupyter-notebook.readthedocs.org/en/latest/notebook.html) containing the written materials and worked-out solutions on Python code.
To work with the material, we recommend that you start each lesson with a fresh new notebook, and follow along, typing each line of code (don't copy-and-paste!), and exploring by changing parameters and seeing what happens.

### Installing via Anaconda

We *highly* recommend that you install the [Anaconda Python Distribution](https://docs.anaconda.com/anaconda/install/).
It will make your life so much easier.
You can download and install Anaconda on Windows, OSX, and Linux.

After installing, to ensure that your packages are up to date, run the following commands in a terminal:

```shell
conda update conda
conda update jupyter numpy scipy matplotlib
```

If you prefer Miniconda (a mini version of Anaconda that saves you disk space), install all the necessary libraries to follow this course by running the following commands in a terminal:

```shell
conda update conda
conda install jupyter numpy scipy matplotlib
```

### Without Anaconda

If you already have Python installed on your machine, you can install Jupyter using pip:

```shell
pip install jupyter
```

Please also make sure that you have the necessary libraries installed by running

```shell
pip install numpy scipy matplotlib
```

## Running the notebook server

Once Jupyter is installed, open up a terminal and then run

```shell
jupyter notebook
```

This will start up a Jupyter session in your browser!

## How to contribute to AeroPython

We accept contributions via pull request.
You can also open an issue if you find a bug, or have a suggestion.

## Copyright and License

(c) 2017 Lorena A. Barba, Olivier Mesnard. All content is under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt), and all [code is under BSD-3 clause](https://github.com/barbagroup/AeroPython/blob/master/LICENSE) (previously under MIT, and changed on November 12, 2018).

We are happy if you re-use the content in any way!

## Note

Another Python course exists under the [AeroPython](https://github.com/AeroPython/Curso_AeroPython/) title with different content, and in the Spanish language. (See [tweet](https://twitter.com/LorenaABarba/status/464041427169583104) from 2014.)
