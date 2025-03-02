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

# Simple webapp to gain insights into SVM

## Goal
To provide a simple interactive demonstration of linear SVM with 2 classes(positive and negative) and gain insights. The concepts can be extended to multiple classes. User can choose number of samples, hyperparameter C can be modified to understand their effects. 

If reader is quite faimiliar with SVM then they can directly jump to [Insights](#insights) or the webapp

## Introduction
A Support Vector Machine (SVM) is a supervised learning model used for classification and regression tasks. It finds the optimal hyperplane that separates data points of different classes in a high-dimensional feature space. The objective of an SVM is to maximize the margin between the data points (support vectors) of each class.

## Prerequistes:
For brevity, I assume that reader understands about Hyperplane, Support vectors, Maximal Margin classifier.

## Notations

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

## Formulations
### Hard SVM

The goal is to find a hyperplane $<\mathbf{w}, x> + b = 0$ which maximizes the worstcase distance of a sample from the hyperplane considering all samples in the dataset. When we enforce above conditions, we will get below optimization problem(see [1] for more details)

$minimize_{\mathbf{w},b}$ $\frac{1}{2} \lVert w \rVert^2$

`subject to` $y_n (<w, x_n> + b) \geq 1$ for all $n$

If the dataset is such that positive labels overlap (i.e., we cannot linearly separate them strictly) with negative labels then the optimization problem is infeasible (i.e., there is no solution to the problem).

### Soft SVM
Hard SVM can pose impractical constraints on the optimization problem. It restricts us in solving practical problems in a meaninful way. So, few slack variables are introduced to relax it (through which some samples are allowed to violate the maximal margin assumption). This formulation is called Soft-SVM and is as follows (where $\xi_n$ are slack variables)

$minimize_{\mathbf{w},b,\xi_n}$ $\frac{1}{2} \lVert w \rVert^2 + C \sum_{i=n}^{N} \xi_n$

`subject to` $y_n (<w, x_n> + b) \geq 1 - \xi_n \forall n$

C is hyperparameter to tradeoff the margin and total amount of the slack. Separating hyperplane is denoted by $<w, x_n> + b = 0$ and supporting hyperplanes are denoted by $<w, x_n> + b = 1$ and $<w, x_n> + b = -1$

## Workflow

1. Open the link <???>
2. Input #samples and value of C
3. Click *Generate* to produce random set of data
4. If data is satisfactory, click *Classify* to classify and observe separating and supporting hyperplanes
5. Modify C and click *Classify* to see the effect of C
6. If required, you can generate different set of data by clicking *Generate*

## Insights

1. Typically, optimization problem optimizes such that there are no misclassificaitons to the extent possible
2. It is possible that support vectors does not lie on supporting hyperplanes
3. Outliers may not have influence on the separating hyperplane as they dont contribute to the objective function.
4. Varying C
    1. $C$ can change the hyperplane significantly
    2. Increasing $C$, generally, leads to Hard-SVM formulation (i.e., it strives to not missclassify examples) and can potentially overfit (or classify noisy samples)
    3. Decreasing $C$, can make optimization problem lazy and potentially underfit (without finding any meaningful hyperplane)
5. Slack variables are (in a way) proportional to distance between their supporting hyperplanes
6. Farther a miss-classification from the hyperplane, it contributes more loss to the objective function though contribution varies linearly with distance

## Few open questions

1. SVC package seems to produce unexpected outputs and support vectors for extreme values of C(not converged?)
2. SVC always finds, in theory, a solution irrespective of linear separability or value of C
    1. Ignoring convergence?

## References

1. Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Chapter 12: Classification with Support Vector Machines in Mathematics for machine learning (pp. 370-405). Cambridge University Press. https://mml-book.github.io/book/mml-book.pdf