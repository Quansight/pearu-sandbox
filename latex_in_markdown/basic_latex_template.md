# Basic LaTeX Template

The content of this document is copied from [template.tex](http://persweb.wabash.edu/facstaff/turnerw/Writing/LaTeX/).

## Abstract
This paper computes the distance between two points and fits both linear and
exponential functions through the two points.
\end{abstract}

## Introduction

Consider the two points <!--$(-1,16)$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28-1%2C16%29"> and <!--$(3,1)$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%283%2C1%29">.  Section~\ref{sec: distance}
computes the distance between these two points.  Section~\ref{sec: linear fit}
computes a linear equation <!--$y = m x + b$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%20%3D%20m%20x%20%2B%20b"> through the two points, and
Section~\ref{sec: exponential fit} fits a exponential equation <!--$y = A e^{k x}$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%20%3D%20A%20e%5E%7Bk%20x%7D">
through the two points.

## Distance

\label{sec: distance}
We can use the distance formula
<!--$$\begin{equation}
\label{eqn: distance}
        d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\end{equation}$$--><img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bequation%7D%0A%5Clabel%7Beqn%3A%20distance%7D%0A%20%20%20%20%20%20%20%20d%20%3D%20%5Csqrt%7B%28x_2%20-%20x_1%29%5E2%20%2B%20%28y_2%20-%20y_1%29%5E2%7D%0A%5Cend%7Bequation%7D">
to determine the distance between any two points <!--$(x_1, y_1)$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_1%2C%20y_1%29"> and <!--$(x_2, y_2)$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_2%2C%20y_2%29">
in <!--$\mathbb{R}^2$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbb%7BR%7D%5E2">.  For our example, <!--$(x_1, y_1) = (-1, 16)$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_1%2C%20y_1%29%20%3D%20%28-1%2C%2016%29"> and <!--$(x_2, y_2) =
(3, 1)$--><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_2%2C%20y_2%29%20%3D%0A%283%2C%201%29">, so plugging these values into the distance formula~\eqref{eqn:
distance} tell us the distance between the two points is
<!--$$d 
        = \sqrt{(3 - (-1))^2 + (1 - 16)^2}
        = \sqrt{4^2 + (-15)^2}
        = \sqrt{241}
        .$$--><img src="https://latex.codecogs.com/svg.latex?d%20%0A%20%20%20%20%20%20%20%20%3D%20%5Csqrt%7B%283%20-%20%28-1%29%29%5E2%20%2B%20%281%20-%2016%29%5E2%7D%0A%20%20%20%20%20%20%20%20%3D%20%5Csqrt%7B4%5E2%20%2B%20%28-15%29%5E2%7D%0A%20%20%20%20%20%20%20%20%3D%20%5Csqrt%7B241%7D%0A%20%20%20%20%20%20%20%20.">
