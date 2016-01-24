---
layout: post
title: Optimizing Functions
category: regressions
labels:
  - introduction
  - theoretic
---
{% assign im = "<span class='inline-math'>" %}
{% assign dm = "<span class='display-math'>" %}
{% assign em = "</span>" %}

Given a function with a nonnegative numeric return type, how do you
find the input parameters that minimize the function --- given that
such a parameter actually exists? We will consider the following two
cases as an initial foray into solving this problem:

- there are finite number of possible inputs
- the function is piece-wise differentiable with a "small" 
  collection of nondifferentiable inputs

If we can enumerate through all the inputs of a function, then finding the 
minimum is relatively straightforward. At worst, we iterate through the 
input set one-by-one. If we can leverage any structure on the input, we may 
be able to cut down the runtime. In any case, the problem is a search 
problem and the runtime is {{im}}O(n){{em}} where {{im}}n{{em}} is the size 
of the input. If anything, linear search is highly parallelizable and 
adaptable to architectures with multitude of topologies and dataflows.

The problem becomes less straightforward when computing minimum of a 
function where the input space is not finitely enumerable. For example, 
finding the minimum of a function like

{{dm}}f(x, y) = 1 - \dfrac{1}{x^2|x - y| + x^4 + 1}{{em}}

would not be possible via enumeration (though rather transparent
analytically.) One motivating example --- which we will expand on in the
next section --- is that of error minimization in one step of
training a linear model. 

The goal is to come up with a relatively general method for solving 
minimization problems for a large class of functions, and to do so with
a view towards parallelization and architectural flexibility.

## Gradients

For the remainder, assume {{im}}E{{em}} is some multivariate numeric 
function that has a global minimum and is continuously differentiable in 
its domain, e.g. we can calculate the
partial derivatives of {{im}}E{{em}} with respect to each of its factors.
Without appealing to Lagrangian methods for constraint optimization, the
first-pass to solving this problem is to calculate the gradient field of 
{{im}}E{{em}}, which we write as {{im}}\nabla E{{em}}, given by the partial 
derivative functions:

{{dm}}
\nabla (E)\_i = \dfrac{\partial E}{\partial w\_i}.
{{em}}

Using Euler's method on {{im}}-\nabla E{{em}}, we can define an approximate
trajectory from any given point to a local minimum of {{im}}E{{em}}.
In the case where {{im}}E{{em}} is convex --- say, the Hessian 
{{im}}H(E){{em}} is positive definite --- this effectively solves the
minimization problem for {{im}}E{{em}}.

Let us now examine the solution in a more concrete fashion. Given an 
abstract class `C1Function` representing a continuously differentiable 
real-valued function, defined as follows:

```scala
/**
 * Trait representing an abstract continuous diff.
 * function
 */
trait C1Function {
  /**
   * returns the gradient field of the function
   */
  def gradField(): VectorK[C0Function]

  /**
   * Evaluates the function at a point
   */
  def evaluate(point: VectorK[Real]): Real
}
```

The naive minimization algorithm may be implemented as follows:

```scala
object GDMinFinder {
  
  /**
   * Gradient Descend - while the gradient is still "big"
   * go in the direction to minimize value.
   */
  def gradDescend(E: C1Function,
                  startingPoint: VectorK[Real], 
                  step: Real,
                  threshold: Real): VectorK[Real] = { 
    // obtain the gradientField
    val gradientField = E.gradField
    var currentPoint = startPoint

    // walk in the direction of the gradient until the
    // gradient is "small" enough, which indicates that
    // we have reached a local minimum
    do {
      // use the current point to find the gradient vector
      // at the point and normalize it
      var gradient = E.gradField.map {
        _.evaluate(currentPoint)
      }
      .normalize
      * step
      currentPoint -= gradient
    } while (gradient.length > threshold)

    currentPoint
  }
}
```

### Example: Least-Square Linear Regression

Consider the very simple case of a 2-dimensional linear regression.
Let {{im}}E{{em}} be the square error of the line
{{im}}y = w\_0 + w\_1x{{em}} fitting {{im}}n{{em}} data points 
{{im}}\\\{(x\_1, y\_1) | i = 1,...,n\\\}{{em}}, e.g.

{{dm}}
\displaystyle E(w\_0, w\_1) = 
(\mathbf{y - \mathbf{x}\mathbf{w}})^T(\mathbf{y - \mathbf{x}\mathbf{w}})
{{em}}

where {{im}}\mathbf{y}{{em}} is the vector of values {{im}}y\_i{{em}},
{{im}}\mathbf{x}{{em}} is the matrix whose {{im}}i{{em}}th row is the
row vector {{im}}[1, x\_i]{{em}} and 
{{im}}\mathbf{w} = [w\_0, w\_1]^T{{em}}. Then,

{{dm}}
\displaystyle 
\nabla E = 2(\mathbf{y} - \mathbf{w}\mathbf{x})^T\mathbf{x}.
{{em}}
