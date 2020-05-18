# CMF.jl

Convolutive Matrix Factorizations (CMF) in Julia.

**NOTE: this package is currently under development. You can expect a stable package in June 2020, and major rehauls soon thereafter. Feel free to raise an issue if you have questions in the meantime.**


### Simple Example

Fitting CMF models is as simple as calling `fit_cnmf`:

```julia
>> using CMF
>> data = CMF.gen_synthetic(N=500, T=2000)
>> 
>> results = fit_cnmf(data; L=10, K=5, alg=:hals)
>> println(results.loss_hist[end])
```

Produces:

```julia
0.012
```

### Basic Usage

The command `results = fit_cnmf(data)` fits a convolutive NMF model to the dataset. Several optional parameters are available, including:

- `alg`: the algorithm used to fit the model. The package currently supports:
  - Multiplicative Updates (`:mult`)
  - Hierarchical Alternating Least Squares (`:hals`)
  - Alternating Nonnegative Least Squares (`:anls`)
- `K`: the number of components in the model (default `K=5`).
- `L`: the width, or lag, of each component (default `L=10`).
- `max_time`: the maximum runtime in seconds (default `max_time=Inf`).
- `max_itr`: the maximum number of iterations (default `max_itr=100`).

### Regularization  
  
Multiplicative Updates (MU) and Hierarchical Alternating Least Squares (HALS) support L1 and L2 regularization. Using regularization with the Alternating Nonnegative Least Squares solver will raise an error. 

Regularization is added with additional keyword arguments to the `fit_cnmf` function. The following keyword arguments may be used:
- `l1_H`
- `l2_H`
- `l1_W`
- `l2_W`

These add L1 and L2 regularization to W or H as you would expect. Below is an example:

```julia
results = fit_cnmf(data; L=10, K=5, alg=:hals, l1_H=0.1, l2_H=0.2, l1_W=0.1, l2_W=0.5)
```

### Installation

First, install the latest version of Julia. Then use the `]` key to enter package mode and type:

```julia
(v1.0) pkg> add https://github.com/degleris1/CMF.jl
```
