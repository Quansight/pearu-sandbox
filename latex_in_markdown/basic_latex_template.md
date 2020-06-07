<!--watch-latex-md

This document is processed by watch_latex_md.py program, see

  https://github.com/Quansight/pearu-sandbox/latex_in_markdown/

You can edit this document as you wish. You can also edit the LaTeX
data in img elements, but only the content of `latex-data`:

  1. To automatically update the LaTeX rendering in img element, edit
     the file while watch_latex_md.py is running.

  2. Never change the beginning (`<img latex-data="...`) and the end
     (`...alt="latex">`) parts of the LaTeX img elements as these are
     used by the watch_latex_md.py script.

  3. Changes to other parts of the LaTeX img elements will be
     overwritten.

Enjoy LaTeXing!
-->



The content for this Markdown document is copied from
[template.tex](http://persweb.wabash.edu/facstaff/turnerw/Writing/LaTeX/)
and processed with
[watch_latex_md.py](https://github.com/Quansight/pearu-sandbox/latex_in_markdown/)
script.

# Basic <img data-latex="\huge\LaTeX" src=".images/9bf3e2cf4b3d81ac596da8e8a90da5d6.svg"  valign="-7.684px" width="86.797px" height="32.071px" style="display:inline;" alt="latex"> Template


<center>
<bold>Abstract</bold>
<br/>
<small>
This paper computes the distance between two points and fits both linear and
exponential functions through the two points.
</small>
</center>

## Introduction

Consider the two points <img data-latex="$(-1,16)$" src=".images/400fa1f9a379f32d4d40cab6d3d4cbd1.svg"  valign="-4.289px" width="61.118px" height="17.186px" style="display:inline;" alt="latex"> and <img data-latex="$(3,1)$" src=".images/7da59ed0c2f06bf5f9c544c81ffa009d.svg"  valign="-4.289px" width="39.833px" height="17.186px" style="display:inline;" alt="latex">.  Section [distance](#sec:distance)
computes the distance between these two points.  Section [linear fit](#sec:linear-fit) computes a linear equation <img data-latex="$y = m x + b$" src=".images/ded3e28f19b026949f6eca36ba64ca93.svg"  valign="-3.347px" width="86.625px" height="15.303px" style="display:inline;" alt="latex"> through the two points, and
Section [exponential fit](#sec:exponential-fit) fits a exponential equation <img data-latex="$y = A e^{k x}$" src=".images/d525374cbb7128e4d1518f8da7baccf0.svg"  valign="-3.347px" width="68.682px" height="17.897px" style="display:inline;" alt="latex">
through the two points.

## <a name="sec:distance"></a>Distance

We can use the distance formula

<a name="eqn:distance"></a>
<img data-latex="
\begin{equation*}
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\end{equation*}
" src=".images/17b37165e230bf3e645afc814450b9dc.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

to determine the distance between any two points <img data-latex="$(x_1, y_1)$" src=".images/9e8098a889a2d093afcd20fbb07856bc.svg"  valign="-4.289px" width="54.543px" height="17.186px" style="display:inline;" alt="latex"> and <img data-latex="$(x_2, y_2)$" src=".images/49c752a12ae08cb035584c6853828f3a.svg"  valign="-4.289px" width="54.543px" height="17.186px" style="display:inline;" alt="latex">
in <img data-latex="$\mathbb{R}^2$" src=".images/90d777bda1d64f482bbd7ee431963e17.svg"  width="22.584px" height="13.952px" style="display:inline;" alt="latex">.  For our example, <img data-latex="$(x_1, y_1) = (-1, 16)$" src=".images/a30d42f2f3ce8a453e77ddb3e48da254.svg"  valign="-4.289px" width="133.319px" height="17.186px" style="display:inline;" alt="latex"> and <img data-latex="$(x_2, y_2) = (3, 1)$" src=".images/99e54d55cdfe5d0112e6c73dcf652c5a.svg"  valign="-4.289px" width="112.034px" height="17.186px" style="display:inline;" alt="latex">, so plugging these values into the distance formula [distance](#eqn:distance) tell us the distance between the two points is

<img data-latex="
\begin{equation}
        d 
        = \sqrt{(3 - (-1))^2 + (1 - 16)^2}
        = \sqrt{4^2 + (-15)^2}
        = \sqrt{241}
        .
\end{equation}
" src=".images/5cb7cfb2a2bedefcac9c291821c596f2.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

## <a name="sec:linear-fit"></a> Linear Fit

Consider a linear equation <img data-latex="$y = m x + b$" src=".images/ded3e28f19b026949f6eca36ba64ca93.svg"  valign="-3.347px" width="86.625px" height="15.303px" style="display:inline;" alt="latex"> through the two points.  We will
first determine the slope <img data-latex="$m$" src=".images/3289f1f3038516158022b6f14b8fe0c9.svg"  width="19.042px" height="7.412px" style="display:inline;" alt="latex"> of the line in Section [slope](#sec:slope), and we
will then determine the <img data-latex="$y$" src=".images/76cc814eb790ce3c94002e2c22b65534.svg"  valign="-3.347px" width="13.134px" height="10.76px" style="display:inline;" alt="latex">-intercept <img data-latex="$b$" src=".images/a826d4507bb86e911d0f44a68d0773c4.svg"  width="11.465px" height="11.955px" style="display:inline;" alt="latex"> of the line in Section [intercept](#sec:intercept).

### <a name="sec:slope"></a> Slope

The slope of the line passing through the two points is given by the forumula

<img data-latex="
$$
        m 
        = \frac{\Delta y}{\Delta x} 
        = \frac{y_2 - y_1}{x_2 - x_1}
        .
$$
" src=".images/dd5363b68f9ba74f436c891bb2edac5d.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Plugging in our two points, we find the slope of the line between them is

<a name="eqn:slope"></a>
<img data-latex="
\begin{equation}
        m 
        = \frac{1 - 16}{3 - (-1)}
        = - \frac{15}{4}
        .
\end{equation}
" src=".images/96c65477199e7ce5ce08884ec8ad8f3b.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

### <a name="sec:intercept"></a> Intercept

To find the <img data-latex="$y$" src=".images/76cc814eb790ce3c94002e2c22b65534.svg"  valign="-3.347px" width="13.134px" height="10.76px" style="display:inline;" alt="latex">-intercept of the line, we start with the point-slope form of
the line of slope <img data-latex="$m$" src=".images/3289f1f3038516158022b6f14b8fe0c9.svg"  width="19.042px" height="7.412px" style="display:inline;" alt="latex"> through the point <img data-latex="$(x_0, y_0)$" src=".images/5e0f411b1034db3caacaffdd3260fc00.svg"  valign="-4.289px" width="54.543px" height="17.186px" style="display:inline;" alt="latex">:

<img data-latex="
$$
        y - y_0 = m (x - x_0)
        .
$$
" src=".images/79d40592c0bc60f8c21b3df7c8e18cef.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

We plug in the point <img data-latex="$(x_0, y_0) = (-1, 16)$" src=".images/6d89bf59f45f816e03abd31d58e58022.svg"  valign="-4.289px" width="133.319px" height="17.186px" style="display:inline;" alt="latex"> and the slope we found
previously [slope](#eqn:slope) to obtain the equation

<img data-latex="
$$
        y - 16 = - \frac{15}{4} (x + 1)
        .
$$
" src=".images/c957bdd13cfde4c800b0cec6cb0968d7.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Solving for <img data-latex="$y$" src=".images/76cc814eb790ce3c94002e2c22b65534.svg"  valign="-3.347px" width="13.134px" height="10.76px" style="display:inline;" alt="latex">, we find the slope-intercept form of the line:

<img data-latex="
\begin{align*}
        y
        &= - \frac{15}{4} x - \frac{15}{4} + 16 \\
        &= - \frac{15}{4} x + \frac{49}{4}
        .
\end{align*}
" src=".images/595b683d9df64077313c6506297cdf8d.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Therefore, the <img data-latex="$y$" src=".images/76cc814eb790ce3c94002e2c22b65534.svg"  valign="-3.347px" width="13.134px" height="10.76px" style="display:inline;" alt="latex">-intercept is <img data-latex="$b = 49/4$" src=".images/fa26a6baa727f8bcfb6506e0e2ff527c.svg"  valign="-4.304px" width="65.535px" height="17.215px" style="display:inline;" alt="latex">, and the equation
<img data-latex="$y = - \frac{15}{4} x + \frac{49}{4}$" src=".images/93c6f2fea9492317a03dbd3b97d5a812.svg"  valign="-5.937px" width="105.102px" height="20.419px" style="display:inline;" alt="latex"> describes the line through the two
points.

## <a name="sec:exponential-fit"></a> Exponential Fit

Let us consider the exponential function <img data-latex="$y = A e^{k x}$" src=".images/d525374cbb7128e4d1518f8da7baccf0.svg"  valign="-3.347px" width="68.682px" height="17.897px" style="display:inline;" alt="latex">.  For this function
to pass through both points, we must find constants <img data-latex="$A$" src=".images/bf178f97bb21c0e45a177271d3c0554a.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> and <img data-latex="$k$" src=".images/bee7b96e0233a5c9db75ddf7bde63a40.svg"  width="13.643px" height="11.955px" style="display:inline;" alt="latex"> that satisfy
both equations <img data-latex="$16 = A e^{-k}$" src=".images/76d2467ba930d606493c415323b35139.svg"  width="78.282px" height="14.55px" style="display:inline;" alt="latex"> and <img data-latex="$1 = A e^{3 k}$" src=".images/7878796e34ebc288137d264f72010cce.svg"  width="66.941px" height="14.55px" style="display:inline;" alt="latex">.  To solve these two
simultaneous equations, we first take the ratio of the two equations, which
gives us a single equation involving only <img data-latex="$k$" src=".images/bee7b96e0233a5c9db75ddf7bde63a40.svg"  width="13.643px" height="11.955px" style="display:inline;" alt="latex">:

<img data-latex="
$$
        16
        = \frac{A e^{-k}}{A e^{3 k}}
        = e^{-4 k}
        .
$$
" src=".images/611f640c7df9068b014dbf394ad86a09.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

We can take the natural logarithm of this equation to solve for <img data-latex="$k$" src=".images/bee7b96e0233a5c9db75ddf7bde63a40.svg"  width="13.643px" height="11.955px" style="display:inline;" alt="latex">:

<img data-latex="
$$
        -4k = \ln(16) = 4 \ln (2)
        ,
$$
" src=".images/936dbfc8614e987200b04d0a26c5fcef.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

which means <img data-latex="$k = - \ln(2)$" src=".images/b9cb7ff24483892a44f61aa759debee3.svg"  valign="-4.289px" width="85.039px" height="17.186px" style="display:inline;" alt="latex">.

We can then use this value of <img data-latex="$k$" src=".images/bee7b96e0233a5c9db75ddf7bde63a40.svg"  width="13.643px" height="11.955px" style="display:inline;" alt="latex">, along with either of the two points to
solve for <img data-latex="$A$" src=".images/bf178f97bb21c0e45a177271d3c0554a.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">.  Let us consider the point <img data-latex="$(-1, 16)$" src=".images/0d0f874dd4bbfe3d9a8765a7632d466a.svg"  valign="-4.289px" width="61.118px" height="17.186px" style="display:inline;" alt="latex">:

<img data-latex="
$$
        16 = A e^{(-\ln(2))(-1)} = A e^{\ln{2}} = 2 A
        .
$$
" src=".images/0999d1a3098296a6cc84d4119da287c6.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Solving for <img data-latex="$A$" src=".images/bf178f97bb21c0e45a177271d3c0554a.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">, we find <img data-latex="$A = 8$" src=".images/4f52e58fb1bd3486250940a04338a1e7.svg"  width="46.786px" height="11.764px" style="display:inline;" alt="latex">, and the exponential equation through both
points is

<img data-latex="
$$
        y
        = 8 e^{-\ln(2) x}
        = 8 2^{-x}
        = 8 \left( \frac{1}{2} \right)^x
        .
$$
" src=".images/9f26e27cce5771191aef62d58697d52a.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">
