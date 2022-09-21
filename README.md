 # Oracle-Structured Bundle Distributed Optimization (OSBDO)
 
This repository accompanies the manuscript [An Oracle-Structured Bundle Method for Distributed Optimization](https://web.stanford.edu/~boyd/papers/os_bundle_distr_opt.html).

We consider the optimization problem
```math
\begin{array}{ll}
\mbox{minimize} & h(x) = f(x) + g(x),
\end{array}
```
where $x \in {\mbox{\bf R}}^n$ is the variable, and
$f, g:{\mbox{\bf R}}^n \to {\mbox{\bf R}}\cup \{\infty\}$
are the oracle and structured convex objective functions, respectively. 
   
The oracle objective function $f$ 
is block separable, i.e., is of the form
```math
          f(x) = \sum_{i=1}^M f_i(x_i),
```
where $x_i \in {\mbox{\bf R}}^{n_i}$ and $x=(x_1, \ldots, x_M)$.
We refer to $x_i$ as the public variable and $f_i$ as the objective function
of the agent $i$. Our access to $f_i$ is only via an oracle that evaluates 
the function value and a subgradient at a given point $x_i$, i.e.,
$f_i(x_i)$ and $q_i \in \partial f_i(x_i)$.

The function $g$ is structured in the sense that we are given its complete 
description in disciplined convex programming (DCP) format. 
Presumably the function $g$ couples the block variables 
$x_1, \ldots, x_M$.

In this repository we provide an implementation of the bundle-type method
proposed in our [manuscript](https://web.stanford.edu/~boyd/papers/os_bundle_distr_opt.html).

## Installation
OSBDO is available on the Python Package Index, use
```
pip install osbdo
```
Requirements
* python >= 3.8
* CVXPY >= 1.2.0
* numpy >= 1.22.2
* matplotlib >= 1.16.0
* scipy >= 1.8.0

## Getting started

To start using `osbdo` solver, follow the procedure below.

1. Describe each objective function $f_i$ in a class that inherits from `osbdo.Agent`
    * `Agent_i(osbdo.Agent)`
    * create dictionary with parameters `params` relevant for function $f_i$ 
       * dictionary `params` contains items with the dimension of public variable $x_i$ and its lower and upper bound 
           * `params = {"dimension"= ..., "lower_bound"=..., "upper_bound":..., }` 
    * implement methods 
       * `Agent_i.query(v)`: returns output of the subgradient oracle at point $v$ as a `Point(v, q, f_i(v))` 
       * `Agent_i._construct_params()`: construct necessary parameters for $f_i$ from `params`
       * `Agent_i.get_init_minorant()`: returns initial minorant of agent's objective $\hat f^0_i$ as a `cvxpy.Expression`

2. Define a structured function $g$ as an `osbdo.Coupling(agents, function, domain)` by specifying
    * `agents`: list of $M$ agents of type `Agent_i`
    * `function`: function $g$ on its domain, given as a `cvxpy.Expression` 
    * `domain`: domain of $g$ given as a list of `cvxpy` constraints

3. Define a distributed optimization problem as an `osbdo.Problem(agents, g)`  by specifying
    * `agents`: list of $M$ agents of type `Agent_i`
    * `g`: a structured function $g$ of type `Coupling`
       
4. Solve a distributed optimization problem 
    * `osbdo.Problem.solve()`
        * `memory` is an optional parameter that limits the memory (set
           to infinity by default); see the [manuscript](https://web.stanford.edu/~boyd/papers/os_bundle_distr_opt.html) 
    * `osbdo.Problem.upper_bnd`, `osbdo.Problem.lower_bnd`: upper and lower bounds on optimal 
        problem value in each iteration, populated after calling the `Problem.solve()`


### Hello world

We provide a guideline on how to use our method using the [hello world example](https://github.com/cvxgrp/OSBDO/blob/main/examples/hello_world/hello_world.ipynb) Jupyter notebook. 


## Example notebooks
We have [example notebooks](https://github.com/cvxgrp/OSBDO/tree/main/examples) 
that show how to use our method on a number of different problems.

* [supply chain problem](https://github.com/cvxgrp/OSBDO/tree/main/examples/supply_chain)                             
* [resource allocation problem](https://github.com/cvxgrp/OSBDO/tree/main/examples/resource_allocation)
* [multi-commodity flow problem](https://github.com/cvxgrp/OSBDO/tree/main/examples/multicommodity_flow)
* [federated learning problem](https://github.com/cvxgrp/OSBDO/tree/main/examples/federated_learning)

Please consult our [manuscript](https://web.stanford.edu/~boyd/papers/os_bundle_distr_opt.html) 
for the details of mentioned problems and their oracle-structured form. 

### Extra example
We also use our method for finding the intersection of the convex sets. 
* [intersection of convex sets](https://github.com/cvxgrp/OSBDO/tree/main/examples/intersection_cvx_sets)
