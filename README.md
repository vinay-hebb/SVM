---
title: SVM
emoji: 😻
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
short_description: SVM Demo
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Simple demo of few XAI algorithms

As an initial step, implemented initial versions

# Points to Note

1. Better model can be considered. I just choose simplest model possible(simple in terms of coding rather than number of paramters!)
-------------------

# Simple demo to get insights into SVM

| Notation            | Description                                          |
|---------------------|------------------------------------------------------|
| $D$                 | Dimension of the input samples in the dataset        |
| $N$                 | Total number of samples in the dataset               |
| $\mathbf{x_n}$      | $n_{th}$ input sample or feature vector. $\mathbf{x_n} \in \mathbf{R}^D$                                |
| $y_n$               | $n_{th}$ output sample. $y_n \in \{-1, +1\}$                               |
| $\mathbf{w}$        | weight vector. $\mathbf{w} \in \mathbf{R}^D$                                               |
| $b$                 | bias                                                 |
| $C$                 | Hyperparameter                                       |
| $\xi_n$             | $n_{th}$ Slack variable                              |

# Introduction
A Support Vector Machine (SVM) is a supervised learning model used for classification and regression tasks. It finds the optimal hyperplane that separates data points of different classes in a high-dimensional feature space. The objective of an SVM is to maximize the margin between the closest data points (support vectors) of each class.

# Prerequistes:
For brevity, I assume that reader understands about Hyperplane, Support vectors, Maximal Margin classifier.

## Hard SVM

The goal is to find a hyperplane $<\mathbf{w}, x> + b = 0$ which maximizes the worstcase distance of a sample from the hyperplane considering all samples in the dataset. When we enforce above conditions, we will get below optimization problem[1]

$minimize_{\mathbf{w},b}$ $\frac{1}{2} \lVert w \rVert^2$

`subject to` $y_n (<w, x_n> + b) \geq 1$ for all $n$

If the dataset is such that positive labels overlap with negative labels then the optimization problem is infeasible (i.e., we cannot linearly separate them strictly or there is no solution to the problem).

## Soft SVM
Hard SVM poses hard constraints sometimes on the optimization problem. It restricts us in solving practical problems in a meaninful way. So, few slack variables are introduced to relax it (through which some samples are allowed to violate the maximal margin assumption). It is called Soft-SVM and is as follows (where $\xi_n$ are slack variables)

$minimize_{\mathbf{w},b,\xi_n}$ $\frac{1}{2} \lVert w \rVert^2 + C \sum_{i=n}^{N} \xi_n$

`subject to` $y_n (<w, x_n> + b) \geq 1 - \xi_n$ for all $n$

C is hyperparameter to tradeoff the margin and total amount of the slack we have

# Demo
I have created a simple interactive demonstration for a setting where there are 2 classes(positive and negative). The concepts can be extended to multiple classes. User can choose number of samples, hyperparameter C can be modified to understand their effects

# References

1. Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Chapter 12: Classification with Support Vector Machines in Mathematics for machine learning (pp. 370-405). Cambridge University Press. https://mml-book.github.io/book/mml-book.pdf