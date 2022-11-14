Low-level functions for evaluating and manipulating polynomials.

# Examples

The vector of coefficients for the polynomial `f(x, y) = 3 x y + x^2` is
`[0, 3, 0, 1, 0, 0]`.

With `eval()` we can evaluate this polynomial:

```rust
use nutils_poly;

let coeffs = [0, 3, 0, 1, 0, 0];
assert_eq!(nutils_poly::eval(&coeffs, &[1, 0], 2), Ok( 1)); // f(1, 0) =  1
assert_eq!(nutils_poly::eval(&coeffs, &[1, 1], 2), Ok( 4)); // f(1, 1) =  4
assert_eq!(nutils_poly::eval(&coeffs, &[2, 3], 2), Ok(22)); // f(2, 3) = 22
```

`PartialDerivPlan::apply()` computes the coefficients for the partial
derivative of a polynomial to one of the variables. The partial derivative
of `f` to `x`, the first variable, is `∂_x f(x, y) = 3 y + 2 x`
(coefficients: `[3, 2, 0]`):

```rust
use nutils_poly::PartialDerivPlan;

let coeffs = [0, 3, 0, 1, 0, 0];
let pd = PartialDerivPlan::new(
    2, // number of variables
    2, // degree
    0, // variable to compute the partial derivative to
).unwrap();
assert_eq!(Vec::from_iter(pd.apply(coeffs)?), vec![3, 2, 0]);
# Ok::<_, nutils_poly::Error>(())
```

# Further reading

See the [crate documentation] for a detailed description.

This crate is part of the [Nutils project].

[crate documentation]: https://docs.rs/sqnc
[Nutils project]: https://nutils.org
