//! Low-level functions for evaluating and manipulating polynomials.
//!
//! The polynomials considered in this crate are of the form
//!
//! ```text
//! Σ_{k ∈ ℤ^n | Σ_i k_i ≤ p} c_k ∏_i x_i^(k_i)
//! ```
//!
//! where `c` is a vector of coefficients of type [`f64`], `x` a vector of `n`
//! variables of type [`f64`] and `p` a nonnegative integer degree.
//!
//! # Representation
//!
//! This crate does not feature a polynomial type. Instead, all functions take
//! either a [`slice`] or [`Iterator`] of coefficients. The coefficients are
//! assumed to be in reverse [lexicographic order]: the coefficient for powers `j
//! ∈ ℤ^n` comes before the coefficient for powers `k ∈ ℤ^n / {j}` iff `j_i >
//! k_i`, where `i = max_l(j_l ≠ k_l)`, the index of the *last* non-matching
//! power.
//!
//! The functions [`powers_to_index`] and [`index_to_powers`] perform the
//! conversion from a slice of powers to an index and vice versa. The number of
//! coefficients for a polynomial of given degree and number of variables
//! can be obtained via [`ncoeffs`].
//!
//! # Examples
//!
//! Consider the polynomial `1 - x`. The corresponding vector of coefficients
//! is `[-1, 1]`. You can evaluate this polynomial for some `x` using [`eval`]:
//!
//! ```
//! use nutils_poly as poly;
//!
//! let coeffs = [-1, 1];
//! // x = 0
//! assert_eq!(poly::eval(&coeffs, &[0], 1), 1);
//! // x = 1
//! assert_eq!(poly::eval(&coeffs, &[1], 1), 0);
//! ```
//!
//! [lexicographic order]: https://en.wikipedia.org/wiki/Lexicographic_order

#![cfg_attr(feature = "bench", feature(test))]

use num_traits::cast::FromPrimitive;
use num_traits::Zero;
use std::iter;
use std::ops::{Add, AddAssign, Mul};

fn uniform_vec<T: Clone>(item: T, len: usize) -> Vec<T> {
    iter::repeat(item).take(len).collect()
}

fn uniform_vec_with<T, F>(gen_item: F, len: usize) -> Vec<T>
where
    F: FnMut() -> T,
{
    iter::repeat_with(gen_item).take(len).collect()
}

/// Returns the number of coefficients for a polynomial of given degree and number of variables.
#[inline]
pub const fn ncoeffs(degree: usize, nvars: usize) -> usize {
    // To improve the performance the implementation is specialized for
    // polynomials in zero to three variables.
    match nvars {
        0 => ncoeffs_impl(degree, 0),
        1 => ncoeffs_impl(degree, 1),
        2 => ncoeffs_impl(degree, 2),
        3 => ncoeffs_impl(degree, 3),
        _ => ncoeffs_impl(degree, nvars),
    }
}

#[inline]
const fn ncoeffs_impl(degree: usize, nvars: usize) -> usize {
    let mut n = 1;
    let mut i = 1;
    while i <= nvars {
        n = n * (degree + i) / i;
        i += 1;
    }
    n
}

/// Returns the sum of the number of coefficients up to (excluding) the given degree.
#[inline]
const fn ncoeffs_sum(degree: usize, nvars: usize) -> usize {
    // To improve the performance the implementation is specialized for
    // polynomials in zero to three variables.
    match nvars {
        0 => ncoeffs_sum_impl(degree, 0),
        1 => ncoeffs_sum_impl(degree, 1),
        2 => ncoeffs_sum_impl(degree, 2),
        3 => ncoeffs_sum_impl(degree, 3),
        _ => ncoeffs_sum_impl(degree, nvars),
    }
}

#[inline]
const fn ncoeffs_sum_impl(degree: usize, nvars: usize) -> usize {
    let mut n = 1;
    let mut i = 0;
    while i <= nvars {
        n = n * (degree + i) / (i + 1);
        i += 1;
    }
    n
}

/// Returns the index of the coefficient for the given powers.
///
/// Returns `None` if the sum of powers exceeds the given degree.
#[inline]
pub fn powers_to_index(powers: &[usize], degree: usize) -> Option<usize> {
    powers
        .iter()
        .copied()
        .enumerate()
        .rev()
        .try_fold((0, degree), |(index, degree), (nvars, power)| {
            let degree = degree.checked_sub(power)?;
            let index = index + ncoeffs_sum(degree, nvars);
            Some((index, degree))
        })
        .map(|(index, _)| index)
}

#[inline]
fn powers_rev_iter_to_index(
    mut rev_powers: impl Iterator<Item = usize>,
    mut degree: usize,
    nvars: usize,
) -> Option<usize> {
    let mut index = 0;
    for nvars in (1..nvars).rev() {
        degree = degree.checked_sub(rev_powers.next().unwrap())?;
        index += ncoeffs_sum(degree, nvars);
    }
    degree
        .checked_sub(rev_powers.next().unwrap())
        .map(|degree| index + degree)
}

/// Returns the powers for the given index.
///
/// Returns `None` if the index is larger or equal to the number of coeffients
/// for the given degree and number of variables.
#[inline]
pub fn index_to_powers(index: usize, degree: usize, nvars: usize) -> Option<Vec<usize>> {
    // FIXME: return None if index is out of bounds
    let mut powers = uniform_vec(0, nvars);
    index_to_powers_increment(index, degree, &mut powers).map(|()| powers)
}

#[inline]
fn index_to_powers_increment(
    mut index: usize,
    mut degree: usize,
    powers: &mut [usize],
) -> Option<()> {
    if powers.is_empty() {
        return Some(());
    }
    'outer: for ivar in (1..powers.len()).rev() {
        for i in 0..=degree {
            let n = ncoeffs(i, ivar);
            if index < n {
                powers[ivar] += degree - i;
                degree = i;
                continue 'outer;
            }
            index -= n;
        }
        return None;
    }
    if let Some(power) = degree.checked_sub(index) {
        powers[0] += power;
        Some(())
    } else {
        None
    }
}

struct PowersIter {
    powers: Vec<usize>,
    rem: usize,
}

impl PowersIter {
    #[inline]
    fn next(&mut self) -> Option<&[usize]> {
        if let Some(power) = self.powers[0].checked_sub(1) {
            self.powers[0] = power;
            self.rem += 1;
            return Some(&self.powers);
        }
        for ivar in 0..self.powers.len() - 1 {
            if let Some(power) = self.powers[ivar + 1].checked_sub(1) {
                self.powers[ivar + 1] = power;
                self.powers[ivar] = self.rem;
                self.rem = 1;
                return Some(&self.powers);
            }
        }
        None
    }
    #[inline]
    fn get(self, index: usize) -> Option<NthPowerIter> {
        (index < self.powers.len()).then(|| NthPowerIter {
            powers: self,
            index,
        })
    }
    #[inline]
    fn reset(&mut self, degree: usize) {
        if !self.powers.is_empty() {
            self.powers.fill(0);
            *self.powers.last_mut().unwrap() = degree;
            *self.powers.first_mut().unwrap() += 1;
        }
        self.rem = 0;
    }
}

fn powers_iter(degree: usize, nvars: usize) -> PowersIter {
    let mut powers = uniform_vec(0, nvars);
    if nvars > 0 {
        powers[nvars - 1] = degree;
        powers[0] += 1;
    }
    PowersIter {
        powers: powers,
        rem: 0,
    }
}

struct NthPowerIter {
    powers: PowersIter,
    index: usize,
}

impl Iterator for NthPowerIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.powers.next().map(|powers| powers[self.index])
    }
}

/// Evaluate a polynomial.
///
/// # Examples
///
/// Consider the polynomial `1 - x`. The corresponding vector of coefficients
/// is `[-1, 1]`. You can evaluate this polynomial for some `x` using [`eval`]:
///
/// ```
/// use nutils_poly as poly;
///
/// let coeffs = [-1, 1];
/// // x = 0
/// assert_eq!(poly::eval(&coeffs, &[0], 1), 1);
/// // x = 1
/// assert_eq!(poly::eval(&coeffs, &[1], 1), 0);
/// ```
#[inline]
pub fn eval<'a, C>(coeffs: &[C], vars: &'a [C], degree: usize) -> C
where
    C: Clone + Add<C, Output = C> + Mul<C, Output = C> + Mul<&'a C, Output = C>,
{
    // TODO: convert the asserts into errors
    let mut coeffs_iter = coeffs.into_iter().cloned();
    match vars.len() {
        0 => {
            assert_eq!(coeffs.len(), ncoeffs(degree, 0));
            eval_0d(&mut coeffs_iter, vars, degree)
        }
        1 => {
            assert_eq!(coeffs.len(), ncoeffs(degree, 1));
            eval_1d(&mut coeffs_iter, vars, degree)
        }
        2 => {
            assert_eq!(coeffs.len(), ncoeffs(degree, 2));
            eval_2d(&mut coeffs_iter, vars, degree)
        }
        3 => {
            assert_eq!(coeffs.len(), ncoeffs(degree, 3));
            eval_3d(&mut coeffs_iter, vars, degree)
        }
        nvars => {
            assert_eq!(coeffs.len(), ncoeffs(degree, nvars));
            eval_nd(&mut coeffs_iter, vars, degree)
        }
    }
}

#[inline]
fn eval_0d<C>(coeffs: &mut impl Iterator<Item = C>, _vars: &[C], _degree: usize) -> C {
    coeffs.next().unwrap()
}

#[inline]
fn eval_1d<'a, C>(coeffs: &mut impl Iterator<Item = C>, vars: &'a [C], degree: usize) -> C
where
    C: Add<C, Output = C> + Mul<C, Output = C> + Mul<&'a C, Output = C>,
{
    (1..=degree).fold(eval_0d(coeffs, vars, 0), |v, p| {
        v * &vars[0] + eval_0d(coeffs, vars, p)
    })
}

#[inline]
fn eval_2d<'a, C>(coeffs: &mut impl Iterator<Item = C>, vars: &'a [C], degree: usize) -> C
where
    C: Add<C, Output = C> + Mul<C, Output = C> + Mul<&'a C, Output = C>,
{
    (1..=degree).fold(eval_1d(coeffs, vars, 0), |v, p| {
        v * &vars[1] + eval_1d(coeffs, vars, p)
    })
}

#[inline]
fn eval_3d<'a, C>(coeffs: &mut impl Iterator<Item = C>, vars: &'a [C], degree: usize) -> C
where
    C: Add<C, Output = C> + Mul<C, Output = C> + Mul<&'a C, Output = C>,
{
    (1..=degree).fold(eval_2d(coeffs, vars, 0), |v, p| {
        v * &vars[2] + eval_2d(coeffs, vars, p)
    })
}

#[cfg(not(feature = "poly_eval_loop"))]
#[inline]
fn eval_nd<'a, C>(coeffs: &mut impl Iterator<Item = C>, vars: &'a [C], degree: usize) -> C
where
    C: Add<C, Output = C> + Mul<C, Output = C> + Mul<&'a C, Output = C>,
{
    match vars.len() {
        0 => coeffs.next().unwrap(),
        1 => {
            let v = coeffs.next().unwrap();
            coeffs.take(degree).fold(v, |v, c| c + v * &vars[0])
        }
        nvars => {
            let last_var = &vars[nvars - 1];
            let vars = &vars[..nvars - 1];
            (1..=degree).fold(eval_nd(coeffs, vars, 0), |v, p| {
                v * last_var + eval_nd(coeffs, vars, p)
            })
        }
    }
}

#[cfg(feature = "poly_eval_loop")]
#[inline]
fn eval_nd<'a, C>(coeffs: &mut impl Iterator<Item = C>, vars: &'a [C], degree: usize) -> C
where
    C: Add<C, Output = C> + Mul<C, Output = C> + Mul<&'a C, Output = C> + Zero,
{
    match vars.len() {
        0 => coeffs.next().unwrap(),
        nvars => {
            let mut v = uniform_vec_with(|| C::zero(), nvars);
            let mut n = uniform_vec(0, nvars);
            n[nvars - 1] = degree;
            'outer: loop {
                v[0] = coeffs
                    .take(n[0] + 1)
                    .fold(C::zero(), |v, c| v * vars[0] + c);
                for i in 0..nvars - 1 {
                    v[i + 1] = v[i + 1] * vars[i + 1] + v[i];
                    v[i] = C::zero();
                    if n[i] < n[i + 1] {
                        n[i] += 1;
                        continue 'outer;
                    }
                    n[i] = 0;
                }
                return v[nvars - 1];
            }
        }
    }
}

/// Returns coefficients for the product of two polynomials of the same variables.
///
/// The degree of the returned polynomial is the sum of the degrees of the two
/// operands.
///
/// # Examples
///
/// The product of the polynomials `1 + x` (coefficients `[1, 1]`) and `1 - x`
/// (coefficients `[-1, 1]`) is `(1 - x^2)` (coefficients `[-1, 0, 1]`):
///
/// ```
/// use nutils_poly as poly;
///
/// let coeffs = poly::mul(&[1, 1], &[-1, 1], 1, 1, 1);
/// assert_eq!(coeffs, vec![-1, 0, 1]);
/// assert_eq!(coeffs.len(), poly::ncoeffs(2, 1));
/// ```
#[inline]
pub fn mul<'a, 'b, C1, C2, O>(
    coeffs1: &'a [C1],
    coeffs2: &'b [C2],
    degree1: usize,
    degree2: usize,
    nvars: usize,
) -> Vec<O>
where
    &'a C1: Mul<&'b C2, Output = O>,
    C1: Zero,
    C2: Zero,
    O: AddAssign + Zero,
{
    // TODO: convert into error
    assert_eq!(coeffs1.len(), ncoeffs(degree1, nvars));
    assert_eq!(coeffs2.len(), ncoeffs(degree2, nvars));
    let mut result = uniform_vec_with(|| O::zero(), ncoeffs(degree1 + degree2, nvars));
    let mut coeffs1 = coeffs1.iter();
    let mut powers1 = powers_iter(degree1, nvars);
    let mut powers2 = powers_iter(degree2, nvars);
    while let (Some(c1), Some(p1)) = (coeffs1.next(), powers1.next()) {
        if c1.is_zero() {
            continue;
        }
        let mut coeffs2 = coeffs2.iter();
        powers2.reset(degree2);
        while let (Some(c2), Some(p2)) = (coeffs2.next(), powers2.next()) {
            if c2.is_zero() {
                continue;
            }
            let p = iter::zip(p1.iter().rev(), p2.iter().rev()).map(|(j1, j2)| j1 + j2);
            let i = powers_rev_iter_to_index(p, degree1 + degree2, nvars).unwrap();
            result[i] += c1 * c2;
        }
    }
    result
}

/// Returns coefficients for the product of two polynomials of different variables.
///
/// The degree of the returned polynomial is the sum of the degrees of the two
/// operands and the number of variables is the sum of the number of variables
/// of the two operands. The first `nvars1` variables correspond to the left
/// operand, the last `nvars2` variables to the right operand.
///
/// # Examples
///
/// The product of the polynomials `1 + x` (coefficients `[1, 1]`) and `1 - y`
/// (coefficients `[-1, 1]`) is `(1 + x - y - x * y)` (coefficients `[0, -1,
/// -1, 0, 1, 1]`):
///
/// ```
/// use nutils_poly as poly;
///
/// let coeffs = poly::outer_mul(&[1, 1], &[-1, 1], 1, 1, 1, 1);
/// assert_eq!(coeffs, vec![0, -1, -1, 0, 1, 1]);
/// assert_eq!(coeffs.len(), poly::ncoeffs(2, 2));
/// ```
#[inline]
pub fn outer_mul<'a, 'b, C1, C2, O>(
    coeffs1: &'a [C1],
    coeffs2: &'b [C2],
    degree1: usize,
    degree2: usize,
    nvars1: usize,
    nvars2: usize,
) -> Vec<O>
where
    &'a C1: Mul<&'b C2, Output = O>,
    C1: Zero,
    C2: Zero,
    O: AddAssign + Zero,
{
    // TODO: convert into error
    assert_eq!(coeffs1.len(), ncoeffs(degree1, nvars1));
    assert_eq!(coeffs2.len(), ncoeffs(degree2, nvars2));
    let mut result = uniform_vec_with(|| O::zero(), ncoeffs(degree1 + degree2, nvars1 + nvars2));
    let mut coeffs1 = coeffs1.iter();
    let mut powers1 = powers_iter(degree1, nvars1);
    let mut powers2 = powers_iter(degree2, nvars2);
    while let (Some(c1), Some(p1)) = (coeffs1.next(), powers1.next()) {
        if c1.is_zero() {
            continue;
        }
        let mut coeffs2 = coeffs2.iter();
        powers2.reset(degree2);
        while let (Some(c2), Some(p2)) = (coeffs2.next(), powers2.next()) {
            if c2.is_zero() {
                continue;
            }
            let p = p2.iter().rev().chain(p1.iter().rev()).copied();
            let i = powers_rev_iter_to_index(p, degree1 + degree2, nvars1 + nvars2).unwrap();
            result[i] += c1 * c2;
        }
    }
    result
}

/// Returns the power of a polynomial to a non-negative integer.
pub fn pow(coeffs: &[f64], degree: usize, nvars: usize, exp: usize) -> Vec<f64> {
    if exp == 0 {
        vec![1.0]
    } else if exp == 1 {
        coeffs.to_vec()
    } else {
        let sqr = mul(coeffs, coeffs, degree, degree, nvars);
        if exp == 2 {
            sqr
        } else {
            let even = pow(&sqr, degree * 2, nvars, exp / 2);
            if exp % 2 == 0 {
                even
            } else {
                mul(&even, coeffs, degree * 2 * (exp / 2), degree, nvars)
            }
        }
    }
}

/// Returns the coefficients for the partial derivative of a polynomial to a variable.
#[inline]
pub fn partial_derivative<Coeffs, Coeff, Output>(
    coeffs: Coeffs,
    degree: usize,
    nvars: usize,
    ivar: usize,
) -> impl Iterator<Item = Output>
where
    Coeffs: IntoIterator<Item = Coeff>,
    Output: Mul<Coeff, Output = Output> + FromPrimitive,
{
    iter::zip(coeffs, powers_iter(degree, nvars).get(ivar).unwrap()).filter_map(
        move |(coeff, power)| (power > 0).then(|| Output::from_usize(power).unwrap() * coeff),
    )
}

/// Returns a transformation matrix.
///
/// The matrix is such that the following two expressions are equivalent:
///
/// ```text
/// eval(coeffs, eval(transform_coeffs, vars, transform_degree, from_nvars), degree, to_nvars)
/// eval(matvec(matrix, coeffs), vars, transform_degree * degree, from_nvars)
/// ```
///
/// where `matrix` is the result of
///
/// ```text
/// transform_matrix(transform_coeffs, transform_degree, from_nvars, degree, to_nvars)
/// ```
pub fn transform_matrix(
    transform_coeffs: &[f64],
    transform_degree: usize,
    from_nvars: usize,
    degree: usize,
    to_nvars: usize,
) -> Vec<f64> {
    let transform_ncoeffs = ncoeffs(transform_degree, from_nvars);
    assert_eq!(transform_coeffs.len(), to_nvars * transform_ncoeffs);
    let row_degree = transform_degree * degree;
    let nrows = ncoeffs(transform_degree * degree, from_nvars);
    let ncols = ncoeffs(degree, to_nvars);
    let mut matrix = uniform_vec(0.0, nrows * ncols);
    let mut col_iter = (0..).into_iter();
    let mut col_powers = powers_iter(degree, to_nvars);
    let mut row_powers = powers_iter(degree, from_nvars);
    while let (Some(col), Some(col_powers)) = (col_iter.next(), col_powers.next()) {
        let (col_coeffs, col_degree) = iter::zip(
            transform_coeffs.chunks_exact(transform_ncoeffs),
            col_powers.iter().copied(),
        )
        .fold(
            (vec![1.0], 0),
            |(col_coeffs, col_degree), (t_coeffs, power)| {
                (
                    mul(
                        &col_coeffs,
                        &pow(t_coeffs, transform_degree, from_nvars, power),
                        col_degree,
                        transform_degree * power,
                        from_nvars,
                    ),
                    col_degree + transform_degree * power,
                )
            },
        );
        let mut col_coeffs = col_coeffs.into_iter();
        row_powers.reset(col_degree);
        while let (Some(coeff), Some(powers)) = (col_coeffs.next(), row_powers.next()) {
            let row =
                powers_rev_iter_to_index(powers.iter().rev().copied(), row_degree, from_nvars)
                    .unwrap();
            matrix[row * ncols + col] = coeff;
        }
    }
    matrix
}

pub fn change_degree<Coeff, Coeffs>(
    coeffs: Coeffs,
    degree: usize,
    nvars: usize,
    new_degree: usize,
) -> Vec<f64>
where
    Coeff: std::borrow::Borrow<f64>,
    Coeffs: IntoIterator<Item = Coeff>,
{
    let mut new_coeffs = uniform_vec(0.0, ncoeffs(new_degree, nvars));
    change_degree_into(coeffs, degree, nvars, &mut new_coeffs, new_degree);
    new_coeffs
}

pub fn change_degree_into<Coeff, Coeffs>(
    coeffs: Coeffs,
    degree: usize,
    nvars: usize,
    new_coeffs: &mut [f64],
    new_degree: usize,
) where
    Coeff: std::borrow::Borrow<f64>,
    Coeffs: IntoIterator<Item = Coeff>,
{
    assert!(new_degree >= degree);
    assert_eq!(new_coeffs.len(), ncoeffs(new_degree, nvars));
    let mut coeffs = coeffs.into_iter().map(|c| *c.borrow());
    let mut powers = powers_iter(degree, nvars);
    while let (Some(c), Some(p)) = (coeffs.next(), powers.next()) {
        if c == 0.0 {
            continue;
        }
        let i = powers_rev_iter_to_index(p.iter().rev().copied(), new_degree, nvars).unwrap();
        new_coeffs[i] = c;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    #[test]
    fn ncoeffs() {
        assert_eq!(super::ncoeffs(0, 0), 1);
        assert_eq!(super::ncoeffs(1, 0), 1);
        assert_eq!(super::ncoeffs(0, 1), 1);
        assert_eq!(super::ncoeffs(1, 1), 2);
        assert_eq!(super::ncoeffs(2, 1), 3);
        assert_eq!(super::ncoeffs(0, 2), 1);
        assert_eq!(super::ncoeffs(1, 2), 3);
        assert_eq!(super::ncoeffs(2, 2), 6);
        assert_eq!(super::ncoeffs(3, 2), 10);
        assert_eq!(super::ncoeffs(0, 3), 1);
        assert_eq!(super::ncoeffs(1, 3), 4);
        assert_eq!(super::ncoeffs(2, 3), 10);
        assert_eq!(super::ncoeffs(3, 3), 20);
    }

    #[test]
    fn powers_to_index_to_powers() {
        macro_rules! assert_index_powers {
            ($degree:literal, $powers:tt) => {
                let mut powers_iter = super::powers_iter($degree, $powers[0].len());
                for (index, powers) in $powers.iter().enumerate() {
                    // index_to_powers
                    assert_eq!(
                        super::index_to_powers(index, $degree, powers.len()),
                        Some(powers.to_vec()),
                    );
                    // powers_to_index
                    assert_eq!(super::powers_to_index(&powers[..], $degree), Some(index));
                    // powers_rev_iter_to_index
                    assert_eq!(
                        super::powers_rev_iter_to_index(
                            powers.iter().rev().copied(),
                            $degree,
                            powers.len(),
                        ),
                        Some(index),
                    );
                    // powers_iter
                    assert_eq!(powers_iter.next(), Some(&powers[..]));
                }
                assert_eq!(powers_iter.next(), None);
            };
        }

        assert_index_powers!(2, [[2], [1], [0]]);
        assert_index_powers!(2, [[0, 2], [1, 1], [0, 1], [2, 0], [1, 0], [0, 0]]);
        assert_index_powers!(
            2,
            [
                [0, 0, 2],
                [0, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
                [0, 2, 0],
                [1, 1, 0],
                [0, 1, 0],
                [2, 0, 0],
                [1, 0, 0],
                [0, 0, 0]
            ]
        );
    }

    #[test]
    fn eval_0d() {
        assert_eq!(super::eval(&[1], &[] as &[usize], 0), 1);
        assert_eq!(super::eval(&[1], &[] as &[usize], 1), 1);
        assert_eq!(super::eval(&[1], &[] as &[usize], 2), 1);
    }

    #[test]
    fn eval_1d() {
        assert_eq!(super::eval(&[1], &[5], 0), 1);
        assert_eq!(super::eval(&[2, 1], &[5], 1), 11);
        assert_eq!(super::eval(&[3, 2, 1], &[5], 2), 86);
    }

    #[test]
    fn eval_2d() {
        assert_eq!(super::eval(&[1], &[5, 3], 0), 1);
        assert_eq!(super::eval(&[0, 0, 1], &[5, 3], 1), 1);
        assert_eq!(super::eval(&[0, 1, 0], &[5, 3], 1), 5);
        assert_eq!(super::eval(&[1, 0, 0], &[5, 3], 1), 3);
        assert_eq!(super::eval(&[3, 2, 1], &[5, 3], 1), 20);
        assert_eq!(super::eval(&[6, 5, 4, 3, 2, 1], &[5, 3], 2), 227);
    }

    #[test]
    fn eval_3d() {
        assert_eq!(super::eval(&[1], &[5, 3, 2], 0), 1);
        assert_eq!(super::eval(&[0, 0, 0, 1], &[5, 3, 2], 1), 1);
        assert_eq!(super::eval(&[0, 0, 1, 0], &[5, 3, 2], 1), 5);
        assert_eq!(super::eval(&[0, 1, 0, 0], &[5, 3, 2], 1), 3);
        assert_eq!(super::eval(&[1, 0, 0, 0], &[5, 3, 2], 1), 2);
        assert_eq!(super::eval(&[4, 3, 2, 1], &[5, 3, 2], 1), 28);
        assert_eq!(
            super::eval(&[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], &[5, 3, 2], 2),
            415,
        );
    }

    #[test]
    fn mul() {
        assert_eq!(super::mul(&[2, 1], &[4, 3], 1, 1, 1), vec![8, 10, 3]);
        assert_eq!(
            super::mul(&[3, 2, 1], &[6, 5, 4], 1, 1, 2),
            vec![18, 27, 18, 10, 13, 4],
        );
    }

    #[test]
    fn pow() {
        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 0)[..], [1.]);
        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 1)[..], [0., 2.]);
        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 2)[..], [0., 0., 4.]);
        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 3)[..], [0., 0., 0., 8.]);
        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 4)[..], [0., 0., 0., 0., 16.]);
    }

    #[test]
    fn partial_derivative() {
        assert_eq!(
            super::partial_derivative(&[4, 3, 2, 1], 3, 1, 0).collect::<Vec<usize>>(),
            vec![12, 6, 2],
        );
        assert_eq!(
            super::partial_derivative(&[6, 5, 4, 3, 2, 1], 2, 2, 0).collect::<Vec<usize>>(),
            vec![5, 6, 2],
        );
        assert_eq!(
            super::partial_derivative(&[6, 5, 4, 3, 2, 1], 2, 2, 1).collect::<Vec<usize>>(),
            vec![12, 5, 4],
        );
    }

    #[test]
    fn transform_matrix_1d() {
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.5, 0.0], 1, 1, 2, 1)[..],
            [
                0.25, 0.0, 0.0, //
                0.0, 0.5, 0.0, //
                0.0, 0.0, 1.0, //
            ]
        );
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.5, 0.5], 1, 1, 2, 1)[..],
            [
                0.25, 0.0, 0.0, //
                0.5, 0.5, 0.0, //
                0.25, 0.5, 1.0, //
            ]
        );
    }

    #[test]
    fn transform_matrix_2d() {
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.0, 0.5, 0.0, 0.5, 0.0, 0.0], 1, 2, 2, 2)[..],
            [
                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
                0.00, 0.00, 0.50, 0.00, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.00, 0.50, 0.00, //
                0.00, 0.00, 0.00, 0.00, 0.00, 1.00, //
            ]
        );
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.0, 0.5, 0.5, 0.5, 0.0, 0.0], 1, 2, 2, 2)[..],
            [
                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
                0.00, 0.25, 0.50, 0.00, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.50, 0.50, 0.00, //
                0.00, 0.00, 0.00, 0.25, 0.50, 1.00, //
            ]
        );
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.0, 0.5, 0.0, 0.5, 0.0, 0.5], 1, 2, 2, 2)[..],
            [
                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
                0.50, 0.00, 0.50, 0.00, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
                0.00, 0.25, 0.00, 0.00, 0.50, 0.00, //
                0.25, 0.00, 0.50, 0.00, 0.00, 1.00, //
            ]
        );
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.0, 0.5, 0.5, 0.5, 0.0, 0.5], 1, 2, 2, 2)[..],
            [
                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
                0.50, 0.25, 0.50, 0.00, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
                0.00, 0.25, 0.00, 0.50, 0.50, 0.00, //
                0.25, 0.25, 0.50, 0.25, 0.50, 1.00, //
            ]
        );
    }
}

#[cfg(all(feature = "bench", test))]
mod benches {
    extern crate test;
    use self::test::Bencher;

    macro_rules! mk_bench_eval {
        ($name:ident, $degree:literal, $nvars:literal) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let coeffs: Vec<_> = (1..=super::ncoeffs($degree, $nvars))
                    .map(|i| i as f64)
                    .collect();
                let vars: Vec<_> = (1..=$nvars).map(|x| x as f64).collect();
                b.iter(|| super::eval(test::black_box(&coeffs), test::black_box(&vars), $degree));
            }
        };
    }

    mk_bench_eval! {eval_1d_degree1, 1, 1}
    mk_bench_eval! {eval_1d_degree2, 2, 1}
    mk_bench_eval! {eval_1d_degree3, 3, 1}
    mk_bench_eval! {eval_1d_degree4, 4, 1}
    mk_bench_eval! {eval_2d_degree1, 1, 2}
    mk_bench_eval! {eval_2d_degree2, 2, 2}
    mk_bench_eval! {eval_2d_degree3, 3, 2}
    mk_bench_eval! {eval_2d_degree4, 4, 2}
    mk_bench_eval! {eval_3d_degree1, 1, 3}
    mk_bench_eval! {eval_3d_degree2, 2, 3}
    mk_bench_eval! {eval_3d_degree3, 3, 3}
    mk_bench_eval! {eval_3d_degree4, 4, 3}
    mk_bench_eval! {eval_4d_degree1, 1, 4}
    mk_bench_eval! {eval_4d_degree2, 2, 4}
    mk_bench_eval! {eval_4d_degree3, 3, 4}
    mk_bench_eval! {eval_4d_degree4, 4, 4}

    #[bench]
    fn ncoeffs_3d_degree4(b: &mut Bencher) {
        b.iter(|| super::ncoeffs(test::black_box(4), test::black_box(3)));
    }

    #[bench]
    fn mul_3d_degree4_degree2(b: &mut Bencher) {
        b.iter(|| {
            super::mul(
                test::black_box(
                    &(0..super::ncoeffs(4, 3))
                        .into_iter()
                        .map(|i| i as f64)
                        .collect::<Vec<_>>()[..],
                ),
                test::black_box(
                    &(0..super::ncoeffs(2, 3))
                        .into_iter()
                        .map(|i| i as f64)
                        .collect::<Vec<_>>()[..],
                ),
                test::black_box(4),
                test::black_box(2),
                test::black_box(3),
            )
        });
    }

    #[bench]
    fn outer_mul_1d_degree4_2d_degree2(b: &mut Bencher) {
        b.iter(|| {
            super::outer_mul(
                test::black_box(
                    &(0..super::ncoeffs(4, 1))
                        .into_iter()
                        .map(|i| i as f64)
                        .collect::<Vec<_>>()[..],
                ),
                test::black_box(
                    &(0..super::ncoeffs(2, 2))
                        .into_iter()
                        .map(|i| i as f64)
                        .collect::<Vec<_>>()[..],
                ),
                test::black_box(4),
                test::black_box(2),
                test::black_box(1),
                test::black_box(2),
            )
        });
    }

    #[bench]
    fn pow_3d_degree4_exp3(b: &mut Bencher) {
        b.iter(|| {
            super::pow(
                test::black_box(
                    &(0..super::ncoeffs(4, 3))
                        .into_iter()
                        .map(|i| i as f64)
                        .collect::<Vec<_>>()[..],
                ),
                test::black_box(4),
                test::black_box(3),
                test::black_box(3),
            )
        });
    }

    #[bench]
    fn transform_matrix_2d_degree2(b: &mut Bencher) {
        b.iter(|| {
            super::transform_matrix(
                test::black_box(&[0.5, 0.5, 0.0, 0.5, 0.0, 0.5]),
                test::black_box(1),
                test::black_box(2),
                test::black_box(2),
                test::black_box(2),
            )
        });
    }

    #[bench]
    fn change_degree_2d_degree3_to_5(b: &mut Bencher) {
        b.iter(|| {
            super::change_degree(
                test::black_box(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                test::black_box(3),
                test::black_box(2),
                test::black_box(5),
            )
        });
    }
}
