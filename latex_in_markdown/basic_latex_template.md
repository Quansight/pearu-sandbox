<!--watch-latex-md

This document is processed by watch_latex_md.py program, see

  https://github.com/Quansight/pearu-sandbox/latex_in_markdown/

You can edit this document as you wish. You can also edit the LaTeX
data in img elements but note:

  1. to automatically update the LaTeX rendering in img element, edit
     the file under the supervision of watch_latex_md.py

  2. don't change the beginning (`<img latex-data="...`) and the end
     (`...alt="latex">`) parts of the img element as these are used by
     the watch_latex_md.py script.
-->

The content for this Markdown document is copied from
[template.tex](http://persweb.wabash.edu/facstaff/turnerw/Writing/LaTeX/)
and processed with
[watch_latex_md.py](https://github.com/Quansight/pearu-sandbox/latex_in_markdown/)
script.

# Basic <img data-latex="\huge\LaTeX" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Ctext%7B%5Chuge%5CLaTeX%7D%5C%21"  style="display:inline;" alt="latex"> Template


<center>
<bold>Abstract</bold>
<br/>
<small>
This paper computes the distance between two points and fits both linear and
exponential functions through the two points.
</small>
</center>

## Introduction

Consider the two points <img data-latex="$(-1,16)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28-1%2C16%29%5C%21"  style="display:inline;" alt="latex"> and <img data-latex="$(3,1)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%283%2C1%29%5C%21"  style="display:inline;" alt="latex">.  Section [distance](#sec:distance)
computes the distance between these two points.  Section [linear fit](#sec:linear-fit) computes a linear equation <img data-latex="$y = m x + b$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%20%3D%20m%20x%20%2B%20b%5C%21"  style="display:inline;" alt="latex"> through the two points, and
Section [exponential fit](#sec:exponential-fit) fits a exponential equation <img data-latex="$y = A e^{k x}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%20%3D%20A%20e%5E%7Bk%20x%7D%5C%21"  style="display:inline;" alt="latex">
through the two points.

## <a name="sec:distance"></a>Distance

We can use the distance formula
<a name="eqn:distance"></a>
<img data-latex="
\begin{equation*}
        d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\end{equation*}
" src="https://latex.codecogs.com/svg.latex?d%20%3D%20%5Csqrt%7B%28x_2%20-%20x_1%29%5E2%20%2B%20%28y_2%20-%20y_1%29%5E2%7D%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
to determine the distance between any two points <img data-latex="$(x_1, y_1)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_1%2C%20y_1%29%5C%21"  style="display:inline;" alt="latex"> and <img data-latex="$(x_2, y_2)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_2%2C%20y_2%29%5C%21"  style="display:inline;" alt="latex">
in <img data-latex="$\mathbb{R}^2$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbb%7BR%7D%5E2%5C%21"  style="display:inline;" alt="latex">.  For our example, <img data-latex="$(x_1, y_1) = (-1, 16)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_1%2C%20y_1%29%20%3D%20%28-1%2C%2016%29%5C%21"  style="display:inline;" alt="latex"> and <img data-latex="$(x_2, y_2) =
(3, 1)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_2%2C%20y_2%29%20%3D%0A%283%2C%201%29%5C%21"  style="display:inline;" alt="latex">, so plugging these values into the distance formula [distance](#eqn:distance) tell us the distance between the two points is
<img data-latex="
$$
        d 
        = \sqrt{(3 - (-1))^2 + (1 - 16)^2}
        = \sqrt{4^2 + (-15)^2}
        = \sqrt{241}
        .
$$
" src="https://latex.codecogs.com/svg.latex?d%20%0A%20%20%20%20%20%20%20%20%3D%20%5Csqrt%7B%283%20-%20%28-1%29%29%5E2%20%2B%20%281%20-%2016%29%5E2%7D%0A%20%20%20%20%20%20%20%20%3D%20%5Csqrt%7B4%5E2%20%2B%20%28-15%29%5E2%7D%0A%20%20%20%20%20%20%20%20%3D%20%5Csqrt%7B241%7D%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">

## <a name="sec:linear-fit"></a> Linear Fit

Consider a linear equation <img data-latex="$y = m x + b$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%20%3D%20m%20x%20%2B%20b%5C%21"  style="display:inline;" alt="latex"> through the two points.  We will
first determine the slope <img data-latex="$m$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20m%5C%21"  style="display:inline;" alt="latex"> of the line in Section [slope](#sec:slope), and we
will then determine the <img data-latex="$y$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%5C%21"  style="display:inline;" alt="latex">-intercept <img data-latex="$b$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20b%5C%21"  style="display:inline;" alt="latex"> of the line in Section [intercept](#sec:intercept).

### <a name="sec:slope"></a> Slope

The slope of the line passing through the two points is given by the forumula
<img data-latex="
$$
        m 
        = \frac{\Delta y}{\Delta x} 
        = \frac{y_2 - y_1}{x_2 - x_1}
        .
$$
" src="https://latex.codecogs.com/svg.latex?m%20%0A%20%20%20%20%20%20%20%20%3D%20%5Cfrac%7B%5CDelta%20y%7D%7B%5CDelta%20x%7D%20%0A%20%20%20%20%20%20%20%20%3D%20%5Cfrac%7By_2%20-%20y_1%7D%7Bx_2%20-%20x_1%7D%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
Plugging in our two points, we find the slope of the line between them is
<a name="eqn:slope"></a>
<img data-latex="
\begin{equation}
        m 
        = \frac{1 - 16}{3 - (-1)}
        = - \frac{15}{4}
        .
\end{equation}
" src="https://latex.codecogs.com/svg.latex?m%20%0A%20%20%20%20%20%20%20%20%3D%20%5Cfrac%7B1%20-%2016%7D%7B3%20-%20%28-1%29%7D%0A%20%20%20%20%20%20%20%20%3D%20-%20%5Cfrac%7B15%7D%7B4%7D%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">

### <a name="sec:intercept"></a> Intercept

To find the <img data-latex="$y$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%5C%21"  style="display:inline;" alt="latex">-intercept of the line, we start with the point-slope form of
the line of slope <img data-latex="$m$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20m%5C%21"  style="display:inline;" alt="latex"> through the point <img data-latex="$(x_0, y_0)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_0%2C%20y_0%29%5C%21"  style="display:inline;" alt="latex">:
<img data-latex="
$$
        y - y_0 = m (x - x_0)
        .
$$
" src="https://latex.codecogs.com/svg.latex?y%20-%20y_0%20%3D%20m%20%28x%20-%20x_0%29%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
We plug in the point <img data-latex="$(x_0, y_0) = (-1, 16)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28x_0%2C%20y_0%29%20%3D%20%28-1%2C%2016%29%5C%21"  style="display:inline;" alt="latex"> and the slope we found
previously [slope](#eqn:slope) to obtain the equation
<img data-latex="
$$
        y - 16 = - \frac{15}{4} (x + 1)
        .
$$
" src="https://latex.codecogs.com/svg.latex?y%20-%2016%20%3D%20-%20%5Cfrac%7B15%7D%7B4%7D%20%28x%20%2B%201%29%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
Solving for <img data-latex="$y$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%5C%21"  style="display:inline;" alt="latex">, we find the slope-intercept form of the line:
<img data-latex="
\begin{align*}
        y
        &= - \frac{15}{4} x - \frac{15}{4} + 16 \\
        &= - \frac{15}{4} x + \frac{49}{4}
        .
\end{align*}
" src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign%2A%7D%0A%20%20%20%20%20%20%20%20y%0A%20%20%20%20%20%20%20%20%26%3D%20-%20%5Cfrac%7B15%7D%7B4%7D%20x%20-%20%5Cfrac%7B15%7D%7B4%7D%20%2B%2016%20%5C%5C%0A%20%20%20%20%20%20%20%20%26%3D%20-%20%5Cfrac%7B15%7D%7B4%7D%20x%20%2B%20%5Cfrac%7B49%7D%7B4%7D%0A%20%20%20%20%20%20%20%20.%0A%5Cend%7Balign%2A%7D%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
Therefore, the <img data-latex="$y$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%5C%21"  style="display:inline;" alt="latex">-intercept is <img data-latex="$b = 49/4$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20b%20%3D%2049/4%5C%21"  style="display:inline;" alt="latex">, and the equation
<img data-latex="$y = - \frac{15}{4} x + \frac{49}{4}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%20%3D%20-%20%5Cfrac%7B15%7D%7B4%7D%20x%20%2B%20%5Cfrac%7B49%7D%7B4%7D%5C%21"  style="display:inline;" alt="latex"> describes the line through the two
points.

## <a name="sec:exponential-fit"></a> Exponential Fit

Let us consider the exponential function <img data-latex="$y = A e^{k x}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20y%20%3D%20A%20e%5E%7Bk%20x%7D%5C%21"  style="display:inline;" alt="latex">.  For this function
to pass through both points, we must find constants <img data-latex="$A$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20A%5C%21"  style="display:inline;" alt="latex"> and <img data-latex="$k$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20k%5C%21"  style="display:inline;" alt="latex"> that satisfy
both equations <img data-latex="$16 = A e^{-k}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%2016%20%3D%20A%20e%5E%7B-k%7D%5C%21"  style="display:inline;" alt="latex"> and <img data-latex="$1 = A e^{3 k}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%201%20%3D%20A%20e%5E%7B3%20k%7D%5C%21"  style="display:inline;" alt="latex">.  To solve these two
simultaneous equations, we first take the ratio of the two equations, which
gives us a single equation involving only <img data-latex="$k$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20k%5C%21"  style="display:inline;" alt="latex">:
<img data-latex="
$$
        16
        = \frac{A e^{-k}}{A e^{3 k}}
        = e^{-4 k}
        .
$$
" src="https://latex.codecogs.com/svg.latex?16%0A%20%20%20%20%20%20%20%20%3D%20%5Cfrac%7BA%20e%5E%7B-k%7D%7D%7BA%20e%5E%7B3%20k%7D%7D%0A%20%20%20%20%20%20%20%20%3D%20e%5E%7B-4%20k%7D%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
We can take the natural logarithm of this equation to solve for <img data-latex="$k$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20k%5C%21"  style="display:inline;" alt="latex">:
<img data-latex="
$$
        -4k = \ln(16) = 4 \ln (2)
        ,
$$
" src="https://latex.codecogs.com/svg.latex?-4k%20%3D%20%5Cln%2816%29%20%3D%204%20%5Cln%20%282%29%0A%20%20%20%20%20%20%20%20%2C%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
which means <img data-latex="$k = - \ln(2)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20k%20%3D%20-%20%5Cln%282%29%5C%21"  style="display:inline;" alt="latex">.

We can then use this value of <img data-latex="$k$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20k%5C%21"  style="display:inline;" alt="latex">, along with either of the two points to
solve for <img data-latex="$A$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20A%5C%21"  style="display:inline;" alt="latex">.  Let us consider the point <img data-latex="$(-1, 16)$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%28-1%2C%2016%29%5C%21"  style="display:inline;" alt="latex">:
<img data-latex="
$$
        16 = A e^{(-\ln(2))(-1)} = A e^{\ln{2}} = 2 A
        .
$$
" src="https://latex.codecogs.com/svg.latex?16%20%3D%20A%20e%5E%7B%28-%5Cln%282%29%29%28-1%29%7D%20%3D%20A%20e%5E%7B%5Cln%7B2%7D%7D%20%3D%202%20A%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
Solving for <img data-latex="$A$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20A%5C%21"  style="display:inline;" alt="latex">, we find <img data-latex="$A = 8$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20A%20%3D%208%5C%21"  style="display:inline;" alt="latex">, and the exponential equation through both
points is
<img data-latex="
$$
        y
        = 8 e^{-\ln(2) x}
        = 8 2^{-x}
        = 8 \left( \frac{1}{2} \right)^x
        .
$$
" src="https://latex.codecogs.com/svg.latex?y%0A%20%20%20%20%20%20%20%20%3D%208%20e%5E%7B-%5Cln%282%29%20x%7D%0A%20%20%20%20%20%20%20%20%3D%208%202%5E%7B-x%7D%0A%20%20%20%20%20%20%20%20%3D%208%20%5Cleft%28%20%5Cfrac%7B1%7D%7B2%7D%20%5Cright%29%5Ex%0A%20%20%20%20%20%20%20%20.%5C%21"  style="display:block;50px:auto;margin-right:auto;padding:25px" alt="latex">
