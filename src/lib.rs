//! Low-level functions for evaluating and manipulating polynomials.
//!
//! The polynomials considered in this crate are [power series] in zero or
//! more variables centered at zero and truncated to order $p$,
//!
//! $$ \\sum_{\\substack{k ∈ ℤ^n \\\\ \\sum_i k_i ≤ p}} c_k \\prod_i x_i^{k_i} $$
//!
//! where $c$ is a vector of coefficients, $x$ a vector of $n$ variables and
//! $p$ a nonnegative integer degree.
//!
//! This crate requires the coefficients to be stored in a linear array in
//! reverse [lexicographic order]: the coefficient for powers $j ∈ ℤ^n$ comes
//! before the coefficient for powers $k ∈ ℤ^n \\setminus \\{j\\}$ iff $j_i >
//! k_i$, where $i = \\max_l(j_l ≠ k_l)$, the index of the *last* non-matching
//! power.
//!
//! This crate provides functions for [evaluating polynomials][`eval()`],
//! computing coefficients for the [partial derivativative][`PartialDerivPlan`]
//! and [products][`MulPlan`] of polynomials.
//!
//! # Examples
//!
//! The vector of coefficients for the polynomial $f(x, y) = 3 x y + x^2$ is
//! `[0, 3, 0, 1, 0, 0]`.
//!
//! With [`eval()`] we can evaluate this polynomial:
//!
//! ```
//! use nutils_poly;
//!
//! let coeffs = [0, 3, 0, 1, 0, 0];
//! assert_eq!(nutils_poly::eval(&coeffs, &[1, 0], 2), Ok( 1)); // f(1, 0) =  1
//! assert_eq!(nutils_poly::eval(&coeffs, &[1, 1], 2), Ok( 4)); // f(1, 1) =  4
//! assert_eq!(nutils_poly::eval(&coeffs, &[2, 3], 2), Ok(22)); // f(2, 3) = 22
//! ```
//!
//! [`PartialDerivPlan::apply()`] computes the coefficients for the partial
//! derivative of a polynomial to one of the variables. The partial derivative
//! of $f$ to $x$, the first variable, is $∂_x f(x, y) = 3 y + 2 x$
//! (coefficients: `[3, 2, 0]`):
//!
//! ```
//! use nutils_poly::PartialDerivPlan;
//!
//! let coeffs = [0, 3, 0, 1, 0, 0];
//! let pd = PartialDerivPlan::new(
//!     2, // number of variables
//!     2, // degree
//!     0, // variable to compute the partial derivative to
//! ).unwrap();
//! assert_eq!(Vec::from_iter(pd.apply(coeffs)?), vec![3, 2, 0]);
//! # Ok::<_, nutils_poly::Error>(())
//! ```
//!
//! # Nutils project
//!
//! This crate is part of the [Nutils project].
//!
//! [power series]: https://en.wikipedia.org/wiki/Power_series
//! [lexicographic order]: https://en.wikipedia.org/wiki/Lexicographic_order
//! [Nutils project]: https://nutils.org

// Enable unstable feature `test` if we're running benchmarks.
#![cfg_attr(feature = "bench", feature(test))]

use ndarray::Array2;
use num_traits::{One, Zero};
use sqnc::traits::*;
use std::cmp::Ordering;
use std::fmt;
use std::iter::{self, FusedIterator};
use std::marker::PhantomData;
use std::ops;

pub type Power = u8;

/// Enum used by [`Error::IncorrectNumberOfCoefficients`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueMoreOrLess {
    Value(usize),
    More,
    Less,
}

use ValueMoreOrLess::{Less, More};

impl From<usize> for ValueMoreOrLess {
    #[inline]
    fn from(value: usize) -> Self {
        Self::Value(value)
    }
}

impl fmt::Display for ValueMoreOrLess {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Value(value) => value.fmt(f),
            Self::More => write!(f, "more"),
            Self::Less => write!(f, "less"),
        }
    }
}

/// The error type for fallible operations in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The length of the coefficients sequence or iterator is incorrect.
    IncorrectNumberOfCoefficients {
        expected: usize,
        got: ValueMoreOrLess,
        detail: Option<&'static str>,
    },
    TooManyVariables,
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IncorrectNumberOfCoefficients {
                expected,
                got,
                detail,
            } => {
                if let Some(detail) = detail {
                    write!(f, "{detail}: ")?;
                }
                if *expected == 1 {
                    write!(f, "Expected 1 coefficient but got {got}.")
                } else {
                    write!(f, "Expected {expected} coefficients but got {got}.")
                }
            }
            Self::TooManyVariables => {
                write!(
                    f,
                    "The number of variables exceeds the maximum (`usize::MAX`)."
                )
            }
        }
    }
}

#[inline]
fn check_ncoeffs_sqnc<S: Sequence>(
    coeffs: &S,
    expected: usize,
    detail: Option<&'static str>,
) -> Result<(), Error> {
    let got = coeffs.len();
    if got == expected {
        Ok(())
    } else {
        Err(Error::IncorrectNumberOfCoefficients {
            expected,
            got: got.into(),
            detail,
        })
    }
}

// Function `ncoeffs_impl` computes the number of coefficients for a polynomial
// of certain degree and number of variables using a loop over all variables.
// If the number of variables is not know at copmile time, this loop cannot be
// unrolled. Polynomials of interest typically have only a few variables. To
// improve the perfomance of the typical case, function `ncoeffs` specializes
// the calls to `ncoeffs_impl` for zero to three variables. The same holds for
// function `ncoeffs_sum_impl`.

/// Returns the number of coefficients for a polynomial of given degree and number of variables.
///
/// # Related
///
/// See [`degree()`] for the inverse operation, [`ncoeffs_iter()`] for an
/// iterator that yields the number of coefficients for degree zero and up and
/// [`degree_ncoeffs_iter()`] for an iterator that yields pairs of degrees and
/// number of coefficients.
#[inline]
pub const fn ncoeffs(nvars: usize, degree: Power) -> usize {
    match nvars {
        0 => ncoeffs_impl(0, degree),
        1 => ncoeffs_impl(1, degree),
        2 => ncoeffs_impl(2, degree),
        3 => ncoeffs_impl(3, degree),
        _ => ncoeffs_impl(nvars, degree),
    }
}

#[inline]
const fn ncoeffs_impl(nvars: usize, degree: Power) -> usize {
    let mut n = 1;
    let mut i = 0;
    while i < nvars {
        i += 1;
        n = n * (degree as usize + i) / i;
    }
    n
}

/// Returns an iterator of the number of coefficients for a polynomial of degree zero and up.
///
/// # Related
///
/// See [`degree_ncoeffs_iter()`] for an iterator that yields the degree
/// together with the number of coefficients and [`ncoeffs()`] for a function
/// that returns the number of coefficients for a single degree.
#[inline]
pub fn ncoeffs_iter(nvars: usize) -> impl Iterator<Item = usize> {
    degree_ncoeffs_iter(nvars).map(|(_, ncoeffs)| ncoeffs)
}

/// Returns an iterator of degrees and number of coefficients.
///
/// # Related
///
/// See [`ncoeffs_iter()`] for an iterator that yields the number of
/// coefficients only.
#[inline]
pub fn degree_ncoeffs_iter(nvars: usize) -> impl Iterator<Item = (Power, usize)> {
    (0u8..).scan(1, move |ncoeffs, degree| {
        let current = (degree, *ncoeffs);
        *ncoeffs = *ncoeffs * (degree as usize + 1 + nvars) / (degree as usize + 1);
        Some(current)
    })
}

/// Returns the sum of the number of coefficients up to (excluding) the given degree.
#[inline]
const fn ncoeffs_sum(nvars: usize, degree: Power) -> usize {
    match nvars {
        0 => ncoeffs_sum_impl(0, degree),
        1 => ncoeffs_sum_impl(1, degree),
        2 => ncoeffs_sum_impl(2, degree),
        3 => ncoeffs_sum_impl(3, degree),
        _ => ncoeffs_sum_impl(nvars, degree),
    }
}

#[inline]
const fn ncoeffs_sum_impl(nvars: usize, degree: Power) -> usize {
    let mut n = 1;
    let mut i = 0;
    while i <= nvars {
        n = n * (degree as usize + i) / (i + 1);
        i += 1;
    }
    n
}

/// Returns the degree of a polynomial given the number of variables and coefficients.
///
/// This is the inverse of [`ncoeffs()`]. If there is no degree for which
/// `ncoeffs(nvars, degree) == ncoeffs`, this function returns `None`. For
/// example, a polynomial in two variables has one coefficient for degree zero
/// and three for degree one, but there is nothing in between.
///
/// # Related
///
/// See [`ncoeffs()`] for the inverse operation and [`degree_ncoeffs_iter()`]
/// for an iterator that yields pairs of degrees and number of coefficients for
/// degree zero and up.
pub fn degree(nvars: usize, ncoeffs: usize) -> Option<Power> {
    match nvars {
        0 => (ncoeffs == 1).then_some(0),
        1 => ncoeffs
            .checked_sub(1)
            .and_then(|degree| degree.try_into().ok()),
        _ => {
            // There's nothing else we can do than to iterate over all degrees
            // starting at zero and to stop as soon as we have found or stepped
            // over `ncoeffs`.
            for (tdegree, tncoeffs) in degree_ncoeffs_iter(nvars) {
                match tncoeffs.cmp(&ncoeffs) {
                    Ordering::Equal => {
                        return Some(tdegree);
                    }
                    Ordering::Greater => {
                        return None;
                    }
                    _ => {}
                }
            }
            unreachable! {}
        }
    }
}

/// Returns an iterator that yields powers (in reverse order) for the given index.
///
/// Returns `None` if the index is larger than the number of coefficients for a
/// polynomial in the given number of variables and of the given degree.
///
/// # Related
///
/// The inverse operation is [`powers_rev_iter_to_index()`].
fn index_to_powers_rev_iter(
    mut index: usize,
    nvars: usize,
    mut degree: Power,
) -> Option<impl Iterator<Item = Power> + ExactSizeIterator> {
    // For the last variable the sets of possible indices for each power `k`
    // are adjacent ranges with lengths `n_k`, ordered from power `k = degree`
    // to zero. The length of the range for power `k` is the number of
    // coefficients for a polynomial in one variable less and of degree `degree
    // - k`. We can find the power `k` of the last variable by finding the
    // range that contains the index.
    //
    // Having found the this power, we can subtract power `k` from `degree`
    // (now treated as remaining degree), subtract the lower bound of the range
    // from `index` (now treated as remaining index), and repeat the process.
    //
    // For example a polynomial in 2 variables and of degree 3 has the
    // following ranges of indices for the last variable:
    //
    //     {0}          for power 3
    //     {1,2}        for power 2
    //     {3,4,5}      for power 1
    //     {6,7,8,9}    for power 0
    //
    // Index 4 corresponds to power 1 for the last variable. The remaining
    // degree is 2 and the remaining index is 1. Repeating the process gives
    // power 1 for the first variable.

    (index < ncoeffs(nvars, degree)).then(|| {
        Iterator::rev(0..nvars).map(move |var| {
            if var == 0 {
                // The else branch works fine, but since `ncoeffs(0, ?)` is one
                // regardless `?` we can simplify the loop over the degree to:
                degree - index as Power
            } else {
                // `degree - i` is the tentative power for variable `var` and
                // `i` is the tentative remaining degree. We loop over power
                // from high to low, continuously subtract the length of the
                // range (`n`) from `index`, and stop just before `index`
                // would become negative.
                for (i, n) in degree_ncoeffs_iter(var) {
                    if index < n {
                        // We have checked that initial `index` does not exceed
                        // the number of coefficients above, hence `i` is never
                        // larger than `degree`.
                        let power = degree - i;
                        degree = i;
                        return power;
                    }
                    index -= n;
                }
                // The `degree_ncoeffs_iter` never ends.
                unreachable! {}
            }
        })
    })
}

/// Returns the index given an iterator of powers in reverse order.
///
/// Returns `None` if the sum of the powers exceeds the given `degree`. If the
/// length of `rev_powers` does not match `nvars`, this function returns an
/// undefined value.
///
/// # Related
///
/// The inverse operation is [`index_to_powers_rev_iter()`].
fn powers_rev_iter_to_index<PowersRevIter>(
    rev_powers: PowersRevIter,
    nvars: usize,
    mut degree: Power,
) -> Option<usize>
where
    PowersRevIter: Iterator<Item = Power>,
{
    // For the last variable the range of possible indices for power `k` is
    // `{i_k,...,i_{k+1}-1}` where `i_k` is the partial sum of `{n_.}`,
    //
    //     i_k = Σ_{j | 0 ≤ j < k} n_j
    //
    // and `n_k` is the number of coefficients for a polynomial in one variable
    // less and of degree `degree - k`. We subtract `k` from `degree` and
    // repeat this process for the remaining variables. The index is the sum of
    // all `i_k`.
    let mut index = 0;
    for (var, power) in iter::zip(Iterator::rev(0..nvars), rev_powers) {
        degree = degree.checked_sub(power)?;
        index += ncoeffs_sum(var, degree);
    }
    Some(index)
}

/// Index map relating coefficients from one degree to another.
///
/// For each coefficient for a polynomial in `nvars` variables and of degree
/// `from_degree`, this sequence gives the index of the same coefficient (the
/// same powers) for a polynomial with degree `to_degree`, where `to_degree`
/// must be larger or equal to `from_degree`.
///
/// The indices are strict monotonic increasing.
///
/// # Example
///
/// The following example shows how to map a vector of coefficients from one
/// degree to another.
///
/// ```
/// use nutils_poly::{self, MapDegree};
/// use sqnc::traits::*;
///
/// // Coefficients for a polynomial in two variables and of degree one.
/// let coeffs1 = [4, 5, 6];
/// // Index map from degree two to degree three for a polynomial in one variable.
/// let map = MapDegree::new(2, 1, 2).unwrap();
/// // Apply the map to obtain a coefficients vector for a polynomial of degree two.
/// let mut coeffs2 = [0; nutils_poly::ncoeffs(2, 2)];
/// coeffs2.as_mut_sqnc().select(map).unwrap().assign(coeffs1);
/// assert_eq!(coeffs2, [0, 0, 4, 0, 5, 6]);
/// // Both polynomials evaluate to the same value.
/// assert_eq!(
///     nutils_poly::eval(coeffs1, &[2, 3], 1)?,
///     nutils_poly::eval(coeffs2, &[2, 3], 2)?,
/// );
/// # Ok::<_, nutils_poly::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapDegree {
    nvars: usize,
    from_degree: Power,
    to_degree: Power,
}

impl MapDegree {
    /// Returns an index map or `None` if `to_degree` is smaller than `from_degree`.
    pub fn new(nvars: usize, from_degree: Power, to_degree: Power) -> Option<Self> {
        (to_degree >= from_degree).then_some(Self {
            nvars,
            from_degree,
            to_degree,
        })
    }
}

impl<'this> SequenceTypes<'this> for MapDegree {
    type Item = usize;
    type Iter = sqnc::derive::Iter<'this, Self>;
}

impl Sequence for MapDegree {
    #[inline]
    fn len(&self) -> usize {
        ncoeffs(self.nvars, self.from_degree)
    }

    #[inline]
    fn get(&self, index: usize) -> Option<usize> {
        let powers = index_to_powers_rev_iter(index, self.nvars, self.from_degree)?;
        powers_rev_iter_to_index(powers, self.nvars, self.to_degree)
    }

    #[inline]
    fn iter(&self) -> sqnc::derive::Iter<'_, Self> {
        self.into()
    }
}

impl IntoIterator for MapDegree {
    type Item = usize;
    type IntoIter = sqnc::derive::IntoIter<Self>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }
}

// SAFETY: The coefficients of the polynomials are ordered such that the
// corresponding powers are in reverse lexicographic order (see crate-level
// documentation). The powers for a polynomial in `n` variables of some degree
// `q` is a superset of those for a polynomial in `n` variables of a degree `p
// ≤ q` and `MapDegree` lists the indices of the powers in polynomial `q` that
// also exist in polynomial `p`. Due to the lexicographic ordering of both
// polynomials, this sequence is strict monotonic increasing, hence unique.
// `MapDegree` cannot be instantiated using public methods if `p > q`.
unsafe impl UniqueSequence for MapDegree {}

/// Operations for evaluating polynomials.
///
/// The polynomials considered in this crate are of the form
///
/// $$
///     s_{0,\[\\ \]} = \\left(
///         \\sum_{k_{n-1}} x_{n-1}^{k_{n-1}} \\left(
///             \\cdots
///             \\sum_{k_1} x_1^{k_1} \\left(
///                 \\sum_{k_0} x_0^{k_0} c_k
///             \\right)
///         \\right)
///     \\right).
/// $$
///
/// For brevity the bounds of the summations are omited. The scopes, delimited
/// by parentheses, indicate the order in which the polynomial is being
/// evaluated numerically. The innermost scope eliminates the first variable,
/// the next scope the second, etc.
///
/// Let $s_{n,k} = c_k$ and let $s_{i,k}$ be the value of the $i$-th scope,
///
/// $$ s_{i,k} = \\sum_{j=0}^{m_{i,k}} x_i^j s_{i+1,k+\[j\]}, $$
///
/// where $m_{i,k}$ is the appropriate bound and $k+\[j\]$ denotes the
/// concatenation of vector $k ∈ ℤ^i$ with vector $\[j\]$.
/// The summation is rewritten such that the powers of $x_i$ are eliminated:
///
/// $$ s_{i,k} = ((((s_{i+1,k+\[m_{i,k}\]}) \\cdots) x_i + s_{i+1,k+\[2\]}) x_i + s_{i+1,k+\[1\]}) x_i + s_{i,k+\[0\]} $$
///
/// or equivalently,
///
/// $$ s_{i,k} = a_{i,k,0} $$
///
/// with $a_{i,k,j}$ defined recursively as
///
/// $$
///     a_{i,k,j} = \\begin{cases}
///         s_{i+1,k+\[j\]} & \\text{if } j = m_{i,k} \\\\
///         a_{i,k,j+1} x_i + s_{i+1,k+\[j\]} & \\text{if } j < m_{i,k} \\\\
///     \\end{cases}
/// $$
///
/// Method [`EvalOps::eval()`] uses this recursion to evaluate the
/// polynomial. The four required methods in this trait perform operations on
/// accumulator $a_{i,k}$ (the aggregation of $a_{i,k,0},a_{i,k,1},\\ldots$).
pub trait EvalOps<Coeff, Value: ?Sized> {
    /// The type of the result of the evaluation.
    type Output;

    /// Returns an accumulator initialized with zero.
    ///
    /// This function performs the following operation:
    ///
    /// ```no_compile
    /// acc = 0
    /// ```
    fn init_acc_zero() -> Self::Output;

    /// Returns an accumulator initialized with a polynomial coefficient.
    ///
    /// This function performs the following operation:
    ///
    /// ```no_compile
    /// acc = coeff
    /// ```
    fn init_acc_coeff(coeff: Coeff) -> Self::Output;

    /// Updates an accumulator with a coefficient.
    ///
    /// This function performs the following operation:
    ///
    /// ```no_compile
    /// acc = acc * value + coeff
    /// ```
    fn update_acc_coeff(acc: &mut Self::Output, coeff: Coeff, value: &Value);

    /// Updates an accumulator with the result of an inner loop.
    ///
    /// This function performs the following operation:
    ///
    /// ```no_compile
    /// acc = acc * value + inner
    /// ```
    ///
    /// where `inner` has the same type as the accumulator.
    fn update_acc_inner(acc: &mut Self::Output, inner: Self::Output, value: &Value);

    /// Evaluate a polynomial for the given values.
    ///
    /// # Errors
    ///
    /// This function returns an error if the `coeffs` iterator is too short.
    #[inline]
    fn eval<Coeffs, Values>(
        coeffs: &mut Coeffs,
        values: &Values,
        degree: Power,
    ) -> Result<Self::Output, Error>
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: SequenceRef<OwnedItem = Value> + ?Sized,
    {
        <EvalImpl<Self, Coeffs, Values>>::eval_nv(coeffs, values, degree, values.len()).ok_or_else(
            || Error::IncorrectNumberOfCoefficients {
                expected: ncoeffs(values.len(), degree),
                got: Less,
                detail: None,
            },
        )
    }
}

// While it would be easier to merge the private methods defined below with
// [`EvalOps`], we don't want the functions to be public. There are
// [tricks][Private Methods on a Public Trait] to do this, but this struct does
// the job just fine.
//
// [Private Methods on a Public Trait]: https://web.archive.org/web/20220220002300/https://jack.wrenn.fyi/blog/private-trait-methods/

/// Private methods for [`EvalOps`].
struct EvalImpl<Ops: ?Sized, Coeffs, Values: ?Sized>(
    PhantomData<Ops>,
    PhantomData<Coeffs>,
    PhantomData<Values>,
);

impl<Ops, Coeffs, Values> EvalImpl<Ops, Coeffs, Values>
where
    Ops: EvalOps<Coeffs::Item, Values::OwnedItem> + ?Sized,
    Coeffs: Iterator,
    Values: SequenceRef + ?Sized,
{
    /// Evaluate a polynomial in zero variables.
    #[inline]
    fn eval_0v(coeffs: &mut Coeffs) -> Option<Ops::Output> {
        coeffs.next().map(Ops::init_acc_coeff)
    }

    /// Evaluate a polynomial in one variable.
    #[inline]
    fn eval_1v(coeffs: &mut Coeffs, values: &Values, degree: Power) -> Option<Ops::Output> {
        let mut acc = Ops::init_acc_zero();
        if let Some(value) = values.get(0) {
            if coeffs
                .take(degree as usize + 1)
                .map(|coeff| Ops::update_acc_coeff(&mut acc, coeff, value))
                .count()
                != degree as usize + 1
            {
                return None;
            }
        }
        Some(acc)
    }

    /// Evaluate a polynomial in two variables.
    #[inline]
    fn eval_2v(coeffs: &mut Coeffs, values: &Values, degree: Power) -> Option<Ops::Output> {
        let mut acc = Ops::init_acc_zero();
        if let Some(value) = values.get(1) {
            for p in 0..=degree {
                let inner = Self::eval_1v(coeffs, values, p)?;
                Ops::update_acc_inner(&mut acc, inner, value);
            }
        }
        Some(acc)
    }

    /// Evaluate a polynomial in three variables.
    #[inline]
    fn eval_3v(coeffs: &mut Coeffs, values: &Values, degree: Power) -> Option<Ops::Output> {
        let mut acc = Ops::init_acc_zero();
        if let Some(value) = values.get(2) {
            for p in 0..=degree {
                let inner = Self::eval_2v(coeffs, values, p)?;
                Ops::update_acc_inner(&mut acc, inner, value);
            }
        }
        Some(acc)
    }

    /// Evaluate a polynomial in any number of variables.
    #[inline]
    fn eval_nv(
        coeffs: &mut Coeffs,
        values: &Values,
        degree: Power,
        nvars: usize,
    ) -> Option<Ops::Output> {
        match nvars {
            0 => Self::eval_0v(coeffs),
            1 => Self::eval_1v(coeffs, values, degree),
            2 => Self::eval_2v(coeffs, values, degree),
            3 => Self::eval_3v(coeffs, values, degree),
            _ => {
                let mut acc = Ops::init_acc_zero();
                if let Some(value) = values.get(nvars - 1) {
                    for p in 0..=degree {
                        let inner = Self::eval_nv(coeffs, values, p, nvars - 1)?;
                        Ops::update_acc_inner(&mut acc, inner, value);
                    }
                }
                Some(acc)
            }
        }
    }
}

/// Implementation for [`EvalOps`] using [`std::ops::AddAssign`] and [`std::ops::MulAssign`].
///
/// This implementation is used by [`eval()`].
enum DefaultEvalOps {}

impl<Coeff, Value> EvalOps<Coeff, Value> for DefaultEvalOps
where
    Value: Zero + ops::AddAssign + ops::AddAssign<Coeff> + for<'v> ops::MulAssign<&'v Value>,
{
    type Output = Value;

    #[inline]
    fn init_acc_coeff(coeff: Coeff) -> Value {
        let mut acc = Value::zero();
        acc += coeff;
        acc
    }

    #[inline]
    fn init_acc_zero() -> Value {
        Value::zero()
    }

    #[inline]
    fn update_acc_coeff(acc: &mut Value, coeff: Coeff, value: &Value) {
        *acc *= value;
        *acc += coeff;
    }

    #[inline]
    fn update_acc_inner(acc: &mut Value, inner: Value, value: &Value) {
        *acc *= value;
        *acc += inner;
    }
}

/// Evaluates a polynomial for the given values.
///
/// The number of coefficients of the polynomial is determined by the length of
/// the `values` (the number of variables) and the `degree`. The coefficients
/// have to be iterable. Only the expected number of coefficients are taken
/// from the iterator.
///
/// # Errors
///
/// If the iterator of coefficients has fewer elements than expected, an error
/// is returned.
///
/// # Examples
///
/// The vector of coefficients for the polynomial $f(x) = x^2 - x + 2$ is
/// `[1, -1, 2]`. Evaluating $f(0)$, $f(1)$ and $f(2)$:
///
/// ```
/// use nutils_poly;
///
/// let coeffs = [1, -1, 2];
/// assert_eq!(nutils_poly::eval(&coeffs, &[0], 2), Ok(2)); // f(0) = 2
/// assert_eq!(nutils_poly::eval(&coeffs, &[1], 2), Ok(2)); // f(1) = 2
/// assert_eq!(nutils_poly::eval(&coeffs, &[2], 2), Ok(4)); // f(2) = 4
/// ```
///
/// Let $g(x) = x^2 + x + 1$ be another polynomial. Since [`eval()`] consumes
/// only the expected number of coefficients, we can chain the coefficients for
/// $f$ and $g$ in a single iterator and call [`eval()`] twice to obtain the
/// values for $f$ and $g$:
///
/// ```
/// use nutils_poly;
///
/// let mut coeffs = [1, -1, 2, 1, 1, 1].into_iter();
/// // Evaluate `f`, consumes the first three coefficients.
/// assert_eq!(nutils_poly::eval(&mut coeffs, &[2], 2), Ok(4)); // f(2) = 4
/// // Evaluate `g`, consumes the last three coefficients.
/// assert_eq!(nutils_poly::eval(&mut coeffs, &[2], 2), Ok(7)); // g(2) = 7
/// ```
#[inline]
pub fn eval<Coeff, Value, Coeffs, Values>(
    coeffs: Coeffs,
    values: &Values,
    degree: Power,
) -> Result<Value, Error>
where
    Value: Zero + ops::AddAssign + ops::AddAssign<Coeff> + for<'v> ops::MulAssign<&'v Value>,
    Coeffs: IntoIterator<Item = Coeff>,
    Values: SequenceRef<OwnedItem = Value> + ?Sized,
{
    DefaultEvalOps::eval(&mut coeffs.into_iter(), values, degree)
}

/// Interface for computing a [multiple].
///
/// [multiple]: https://en.wikipedia.org/wiki/Multiple_(mathematics)
pub trait Multiple {
    /// The return type of [`Multiple::multiple()`].
    type Output;

    /// Returns the product of `self` and `n`.
    fn multiple(self, n: usize) -> Self::Output;
}

macro_rules! impl_int_mul {
    ($T:ty $(,)?) => {
        impl Multiple for $T {
            type Output = $T;

            #[inline]
            fn multiple(self, n: usize) -> Self::Output {
                self * n as $T
            }
        }

        impl Multiple for &$T {
            type Output = $T;

            #[inline]
            fn multiple(self, n: usize) -> Self::Output {
                self * n as $T
            }
        }
    };
    ($T:ty, $($tail:tt)*) => {
        impl_int_mul!{$T}
        impl_int_mul!{$($tail)*}
    };
}

impl_int_mul! {u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64}

/// Plan for computing coefficients for partial derivatives.
///
/// This struct holds a plan for efficiently evaluating the partial derivative
/// of polynomials.
///
/// # Examples
///
/// The partial derivative of $f(x, y) = x y - x^2 + 2$ (coefficients: `[0, 1,
/// 0, -1, 0, 2]`) to the first variable, $x$:
///
/// ```
/// use nutils_poly::PartialDerivPlan;
/// use sqnc::traits::*;
///
/// let f = [0, 1, 0, -1, 0, 2];
/// let plan = PartialDerivPlan::new(
///     2, // number of variables
///     2, // degree
///     0, // variable to compute the partial derivative to
/// ).unwrap();
/// assert!(Iterator::eq(
///     plan.apply(f)?.iter(),
///     [1, -2, 0],
/// ));
/// # Ok::<_, nutils_poly::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct PartialDerivPlan {
    indices_powers: Box<[(usize, usize)]>,
    ncoeffs_input: usize,
    degree_output: Power,
    nvars: usize,
}

impl PartialDerivPlan {
    /// Plan the partial derivative of a polynomial.
    pub fn new(nvars: usize, degree: Power, var: usize) -> Option<Self> {
        let n = nvars.checked_sub(var + 1)?;
        let ncoeffs_input = ncoeffs(nvars, degree);
        let indices_powers = if degree == 0 {
            [(0, 0)].into()
        } else {
            (0..ncoeffs_input)
                .filter_map(move |index| {
                    let power = index_to_powers_rev_iter(index, nvars, degree)?.nth(n)?;
                    (power > 0).then_some((index, power as usize))
                })
                .collect()
        };
        Some(Self {
            indices_powers,
            ncoeffs_input,
            degree_output: degree.saturating_sub(1),
            nvars,
        })
    }

    /// Returns the partial derivative of a polynomial.
    ///
    /// # Errors
    ///
    /// If the number of coefficients of the input doesn't match
    /// [`PartialDerivPlan::ncoeffs_input()`] then this function returns an
    /// error.
    #[inline]
    pub fn apply<Coeffs>(&self, coeffs: Coeffs) -> Result<PartialDeriv<'_, Coeffs>, Error>
    where
        Coeffs: Sequence,
    {
        check_ncoeffs_sqnc(&coeffs, self.ncoeffs_input, None)?;
        Ok(PartialDeriv { plan: self, coeffs })
    }

    /// Returns the number of coefficients for the input polynomial.
    #[inline]
    pub fn ncoeffs_input(&self) -> usize {
        self.ncoeffs_input
    }

    /// Returns the number of coefficients for the partial derivative.
    #[inline]
    pub fn ncoeffs_output(&self) -> usize {
        self.indices_powers.len()
    }

    /// Returns the degree of the partial derivative.
    #[inline]
    pub fn degree_output(&self) -> Power {
        self.degree_output
    }

    /// Returns the number of variables of both the input and the output.
    #[inline]
    pub fn nvars(&self) -> usize {
        self.nvars
    }
}

/// The partial derivative of a polynomial.
///
/// This struct is created by [`PartialDerivPlan`]. See its documentation for
/// more information.
#[derive(Debug, Clone)]
pub struct PartialDeriv<'plan, Coeffs> {
    plan: &'plan PartialDerivPlan,
    coeffs: Coeffs,
}

impl<'this, 'plan, Coeffs, OCoeff> SequenceTypes<'this> for PartialDeriv<'plan, Coeffs>
where
    for<'a> <Coeffs as SequenceTypes<'a>>::Item: Multiple<Output = OCoeff>,
    Coeffs: Sequence,
{
    type Item = OCoeff;
    type Iter = PartialDerivIter<'plan, sqnc::Wrapper<&'this Coeffs, ((),)>>;
}

impl<'plan, Coeffs, OCoeff> Sequence for PartialDeriv<'plan, Coeffs>
where
    for<'a> <Coeffs as SequenceTypes<'a>>::Item: Multiple<Output = OCoeff>,
    Coeffs: Sequence,
{
    #[inline]
    fn len(&self) -> usize {
        self.plan.indices_powers.len()
    }

    #[inline]
    fn get(&self, index: usize) -> Option<OCoeff> {
        let (index, power) = self.plan.indices_powers.get(index)?;
        Some(self.coeffs.get(*index)?.multiple(*power))
    }

    #[inline]
    fn iter(&self) -> PartialDerivIter<'plan, sqnc::Wrapper<&'_ Coeffs, ((),)>> {
        PartialDerivIter {
            indices_powers: self.plan.indices_powers.iter(),
            coeffs: self.coeffs.as_sqnc(),
        }
    }
}

impl<'plan, Coeffs, OCoeff> IntoIterator for PartialDeriv<'plan, Coeffs>
where
    Coeffs: Sequence,
    for<'a> <Coeffs as SequenceTypes<'a>>::Item: Multiple<Output = OCoeff>,
{
    type Item = OCoeff;
    type IntoIter = PartialDerivIter<'plan, Coeffs>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        PartialDerivIter {
            indices_powers: self.plan.indices_powers.iter(),
            coeffs: self.coeffs,
        }
    }
}

/// An iterator of the coefficients of a partial derivative.
///
/// This struct is created by [`PartialDeriv::iter()`].
#[derive(Debug, Clone)]
pub struct PartialDerivIter<'plan, Coeffs> {
    indices_powers: std::slice::Iter<'plan, (usize, usize)>,
    coeffs: Coeffs,
}

impl<'plan, Coeffs, OCoeff> Iterator for PartialDerivIter<'plan, Coeffs>
where
    Coeffs: Sequence,
    for<'a> <Coeffs as SequenceTypes<'a>>::Item: Multiple<Output = OCoeff>,
{
    type Item = OCoeff;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (index, power) = self.indices_powers.next()?;
        Some(self.coeffs.get(*index)?.multiple(*power))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices_powers.size_hint()
    }
}

impl<'plan, Coeffs, OCoeff> DoubleEndedIterator for PartialDerivIter<'plan, Coeffs>
where
    Coeffs: Sequence,
    for<'a> <Coeffs as SequenceTypes<'a>>::Item: Multiple<Output = OCoeff>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (index, power) = self.indices_powers.next_back()?;
        Some(self.coeffs.get(*index)?.multiple(*power))
    }
}

impl<'plan, Coeffs, OCoeff> ExactSizeIterator for PartialDerivIter<'plan, Coeffs>
where
    Coeffs: Sequence,
    for<'a> <Coeffs as SequenceTypes<'a>>::Item: Multiple<Output = OCoeff>,
{
}

impl<'plan, Coeffs, OCoeff> FusedIterator for PartialDerivIter<'plan, Coeffs>
where
    Coeffs: Sequence,
    for<'a> <Coeffs as SequenceTypes<'a>>::Item: Multiple<Output = OCoeff>,
{
}

/// Existence of a variable in the operands of a product polynomial.
///
/// The documentation for [`MulPlan::new()`] explains how to use this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MulVar {
    Left,
    Right,
    Both,
}

/// Plan for computing products of polynomials.
///
/// This struct holds a plan for efficiently evaluating products of polynomials.
///
/// # Examples
///
/// Consider the following polynomials: $f(x) = x^2$ (coefficients: `[1, 0,
/// 0]`), $g(x) = 2 x + 1$ (coefficients: `[2, 1]`) and $h(x, y) = x y$
/// (coefficients: `[0, 1, 0, 0, 0, 0]`).
///
/// ```
/// use nutils_poly::{MulPlan, MulVar};
/// use sqnc::traits::*;
///
/// let f = [1, 0, 0];          // degree: 2, nvars: 1
/// let g = [2, 1];             // degree: 1, nvars: 1
/// let h = [0, 1, 0, 0, 0, 0]; // degree: 2, nvars: 2
/// ```
///
/// Computing the coefficients for $x ↦ f(x) g(x)$:
///
/// ```
/// # use nutils_poly::{MulPlan, MulVar};
/// use sqnc::traits::*;
/// #
/// # let f = [1, 0, 0];
/// # let g = [2, 1];
/// # let h = [0, 1, 0, 0, 0, 0];
/// #
/// let plan = MulPlan::same_vars(
///     1, // number of variables
///     2, // degree of left operand
///     1, // degree of right operand
/// );
/// let fx_gx = plan.apply(f.as_sqnc(), g.as_sqnc()).unwrap();
/// assert!(fx_gx.iter().eq([2, 1, 0, 0]));
/// ```
///
/// Similarly, but with different variables, $(x, y) ↦ f(x) g(y)$:
///
/// ```
/// # use nutils_poly::{MulPlan, MulVar};
/// use sqnc::traits::*;
/// #
/// # let f = [1, 0, 0];
/// # let g = [2, 1];
/// # let h = [0, 1, 0, 0, 0, 0];
/// #
/// let plan = MulPlan::different_vars(
///     1, // number of variables of the left operand
///     1, // number of variables of the right operand
///     2, // degree of left operand
///     1, // degree of right operand
/// )?;
/// let fx_gy = plan.apply(f.as_sqnc(), g.as_sqnc()).unwrap();
/// assert!(fx_gy.iter().eq([0, 0, 0, 2, 0, 0, 0, 1, 0, 0]));
/// # Ok::<_, nutils_poly::Error>(())
/// ```
///
/// Computing the coefficients for $(x, y) ↦ f(y) h(x,y)$:
///
/// ```
/// # use nutils_poly::{MulPlan, MulVar};
/// use sqnc::traits::*;
/// #
/// # let f = [1, 0, 0];
/// # let g = [2, 1];
/// # let h = [0, 1, 0, 0, 0, 0];
/// #
/// let plan = MulPlan::new(
///     &[
///         MulVar::Right, // variable x, exists only in the right operand
///         MulVar::Both,  // variable y, exists in both operands
///     ].copied(),
///     2, // degree of left operand
///     2, // degree of right operand
/// );
/// let fy_hxy = plan.apply(f.as_sqnc(), h.as_sqnc()).unwrap();
/// assert!(fy_hxy.iter().eq([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
/// ```
#[derive(Debug, Clone)]
pub struct MulPlan {
    count: Box<[usize]>,
    offsets: Box<[usize]>,
    indices: Box<[[usize; 2]]>,
    ncoeffs_left: usize,
    ncoeffs_right: usize,
    nvars_output: usize,
    degree_output: Power,
}

impl MulPlan {
    /// Plan the product of two polynomials.
    ///
    /// For each output variable, `vars` lists the relation between output and input variables:
    ///
    /// *   [`MulVar::Left`] if the variable exists only in the left polynomial,
    /// *   [`MulVar::Right`] if the variable exists only in the right polynomial or
    /// *   [`MulVar::Both`] if the variable exists in both polynomials.
    ///
    /// It is not possible to reorder the output variables with respect to the input variables.
    #[inline]
    pub fn new<'vars, Vars>(vars: &'vars Vars, degree_left: Power, degree_right: Power) -> Self
    where
        Vars: SequenceOwned<OwnedItem = MulVar>,
        <Vars as SequenceTypes<'vars>>::Iter: DoubleEndedIterator,
    {
        Self::for_output_degree(vars, degree_left, degree_right, degree_left + degree_right)
    }

    /// Plan the product of two polynomials for the given output degree.
    ///
    /// If the output degree is smaller than the sum of the degrees of the input operands,
    /// then the product is truncated.
    pub fn for_output_degree<'vars, Vars>(
        vars: &'vars Vars,
        degree_left: Power,
        degree_right: Power,
        degree_output: Power,
    ) -> Self
    where
        Vars: SequenceOwned<OwnedItem = MulVar>,
        <Vars as SequenceTypes<'vars>>::Iter: DoubleEndedIterator,
    {
        let nvars_left = vars.iter().filter(|var| *var != MulVar::Right).count();
        let nvars_right = vars.iter().filter(|var| *var != MulVar::Left).count();
        let nvars_output = vars.len();
        let ncoeffs_left = ncoeffs(nvars_left, degree_left);
        let ncoeffs_right = ncoeffs(nvars_right, degree_right);
        let ncoeffs_out = ncoeffs(nvars_output, degree_output);

        // For every combination of left and right indices, we compute the
        // corresponding index of the product polynomial (by computing the
        // powers left and right, summing, and computing the index from powers)
        // and store the index triple (output first) in `indices`.
        let mut indices: Vec<[usize; 3]> = Vec::with_capacity(ncoeffs_left * ncoeffs_right);
        for ileft in 0..ncoeffs_left {
            for iright in 0..ncoeffs_right {
                let mut lpowers = index_to_powers_rev_iter(ileft, nvars_left, degree_left).unwrap();
                let mut rpowers =
                    index_to_powers_rev_iter(iright, nvars_right, degree_right).unwrap();
                let opowers = vars.iter().rev().map(|var| match var {
                    MulVar::Left => lpowers.next().unwrap(),
                    MulVar::Right => rpowers.next().unwrap(),
                    MulVar::Both => lpowers.next().unwrap() + rpowers.next().unwrap(),
                });
                if let Some(iout) = powers_rev_iter_to_index(opowers, vars.len(), degree_output) {
                    indices.push([iout, ileft, iright]);
                }
            }
        }

        // Given `indices` we can allocate a vector of zeros as output
        // coefficients, walk over `indices` and update the output:
        //
        //     let mut coeffs_out = vec_of_enough_zeros();
        //     for (iout, ileft, iright) in indices {
        //         coeffs_out[iout] += coeffs_left[ileft] * coeffs_right[iright];
        //     }
        //
        // However, we also want to be able to get a single coefficient or
        // iterate over the coefficients efficiently without allocating the
        // output vector first. By sorting `indices` on the output index, we
        // can evaluate a single output coefficient by iterating over a slice
        // of `indices` and summing the products of coefficients:
        //
        //     coeffs_out[iout] = indices[offsets[iout]..offsets[iout + 1]].map(|(_, ileft, iright) {
        //         coeffs_left[*ileft] * coeffs_right[*iright]
        //     }).sum()
        //
        // with vec `offsets` defined below.
        //
        // We can use unstable sort because the `ileft` and `iright` pairs are
        // unique.
        indices.sort_unstable();

        let mut oiter = indices.iter().map(|[iout, _, _]| *iout).peekable();
        let count: Box<[usize]> = Iterator::map(0..ncoeffs_out, |i| {
            let mut n = 0;
            while oiter.next_if(|j| i == *j).is_some() {
                n += 1;
            }
            n
        })
        .collect();

        let offsets = iter::once(0)
            .chain(count.iter().copied())
            .scan(0, |acc, n| {
                *acc += n;
                Some(*acc)
            })
            .collect();

        // Strip the output index from `indices` because we use `offsets` and
        // `count` in `Sequence` instead.
        let indices = indices
            .into_iter()
            .map(|[_, ileft, iright]| [ileft, iright])
            .collect();

        Self {
            count,
            offsets,
            indices,
            ncoeffs_left,
            ncoeffs_right,
            nvars_output,
            degree_output,
        }
    }

    /// Plan the product of two polynomials in the same variables.
    #[inline]
    pub fn same_vars(nvars: usize, degree_left: Power, degree_right: Power) -> Self {
        Self::same_vars_for_output_degree(
            nvars,
            degree_left,
            degree_right,
            degree_left + degree_right,
        )
    }

    /// Plan the product of two polynomials in the same variables for the given output degree.
    ///
    /// If the output degree is smaller than the sum of the degrees of the input operands,
    /// then the product is truncated.
    #[inline]
    pub fn same_vars_for_output_degree(
        nvars: usize,
        degree_left: Power,
        degree_right: Power,
        degree_output: Power,
    ) -> Self {
        Self::for_output_degree(
            &[MulVar::Both].copied().repeat(nvars),
            degree_left,
            degree_right,
            degree_output,
        )
    }

    /// Plan the product of two polynomials in different variables.
    ///
    /// The coefficients returned by [`MulPlan::apply()`] are ordered such that
    /// the first `nvars_left` variables are the variables of the left operand and
    /// the last `nvars_right` are the variables of the right operand.
    #[inline]
    pub fn different_vars(
        nvars_left: usize,
        nvars_right: usize,
        degree_left: Power,
        degree_right: Power,
    ) -> Result<Self, Error> {
        Self::different_vars_for_output_degree(
            nvars_left,
            nvars_right,
            degree_left,
            degree_right,
            degree_left + degree_right,
        )
    }

    /// Plan the product of two polynomials in different variables for the given output degree.
    ///
    /// The coefficients returned by [`MulPlan::apply()`] are ordered such that
    /// the first `nvars_left` variables are the variables of the left operand and
    /// the last `nvars_right` are the variables of the right operand.
    ///
    /// If the output degree is smaller than the sum of the degrees of the input operands,
    /// then the product is truncated.
    #[inline]
    pub fn different_vars_for_output_degree(
        nvars_left: usize,
        nvars_right: usize,
        degree_left: Power,
        degree_right: Power,
        degree_output: Power,
    ) -> Result<Self, Error> {
        let vars = [MulVar::Left]
            .copied()
            .repeat(nvars_left)
            .concat([MulVar::Right].copied().repeat(nvars_right))
            .ok_or(Error::TooManyVariables)?;
        Ok(Self::for_output_degree(
            &vars,
            degree_left,
            degree_right,
            degree_output,
        ))
    }

    /// Returns the coefficients for the product of two polynomials.
    ///
    /// # Errors
    ///
    /// This function returns an error if the number of coefficients of the
    /// left or right polynomial doesn't match [`MulPlan::ncoeffs_left()`] or
    /// [`MulPlan::ncoeffs_right()`], respectively.
    #[inline]
    pub fn apply<LCoeffs, RCoeffs>(
        &self,
        coeffs_left: LCoeffs,
        coeffs_right: RCoeffs,
    ) -> Result<Mul<'_, LCoeffs, RCoeffs>, Error>
    where
        LCoeffs: Sequence,
        RCoeffs: Sequence,
    {
        check_ncoeffs_sqnc(&coeffs_left, self.ncoeffs_left(), Some("left polynomial"))?;
        check_ncoeffs_sqnc(
            &coeffs_right,
            self.ncoeffs_right(),
            Some("right polynomial"),
        )?;
        Ok(Mul {
            plan: self,
            coeffs_left,
            coeffs_right,
        })
    }

    /// Returns the number of coefficients for the left operand.
    #[inline]
    pub fn ncoeffs_left(&self) -> usize {
        self.ncoeffs_left
    }

    /// Returns the number of coefficients for the left operand.
    #[inline]
    pub fn ncoeffs_right(&self) -> usize {
        self.ncoeffs_right
    }

    /// Returns the number of coefficients for the product.
    #[inline]
    pub fn ncoeffs_output(&self) -> usize {
        self.count.len()
    }

    /// Returns the number of variables for the product.
    #[inline]
    pub fn nvars_output(&self) -> usize {
        self.nvars_output
    }

    /// Returns the degree of the product.
    #[inline]
    pub fn degree_output(&self) -> Power {
        self.degree_output
    }
}

/// The product of two polynomials.
///
/// This struct is created by [`MulPlan`]. See its documentation for more
/// information.
#[derive(Debug, Clone)]
pub struct Mul<'plan, LCoeffs, RCoeffs> {
    plan: &'plan MulPlan,
    coeffs_left: LCoeffs,
    coeffs_right: RCoeffs,
}

impl<'this, 'plan, LCoeffs, RCoeffs, OCoeff> SequenceTypes<'this> for Mul<'plan, LCoeffs, RCoeffs>
where
    for<'a> <LCoeffs as SequenceTypes<'a>>::Item:
        ops::Mul<<RCoeffs as SequenceTypes<'a>>::Item, Output = OCoeff>,
    OCoeff: ops::AddAssign + Zero,
    LCoeffs: Sequence,
    RCoeffs: Sequence,
{
    type Item = OCoeff;
    type Iter = sqnc::derive::Iter<'this, Self>;
}

impl<'plan, LCoeffs, RCoeffs, OCoeff> Sequence for Mul<'plan, LCoeffs, RCoeffs>
where
    for<'a> <LCoeffs as SequenceTypes<'a>>::Item:
        ops::Mul<<RCoeffs as SequenceTypes<'a>>::Item, Output = OCoeff>,
    OCoeff: ops::AddAssign + Zero,
    LCoeffs: Sequence,
    RCoeffs: Sequence,
{
    #[inline]
    fn len(&self) -> usize {
        self.plan.ncoeffs_output()
    }

    #[inline]
    fn get(&self, index: usize) -> Option<OCoeff> {
        let start = *self.plan.offsets.get(index)?;
        let stop = *self.plan.offsets.get(index + 1)?;
        let n = stop.saturating_sub(start);
        let mut value = OCoeff::zero();
        for [l, r] in self.plan.indices.iter().skip(start).take(n) {
            value += self.coeffs_left.get(*l)? * self.coeffs_right.get(*r)?;
        }
        Some(value)
    }

    #[inline]
    fn iter(&self) -> sqnc::derive::Iter<'_, Self> {
        self.into()
    }
}

impl<'plan, LCoeffs, RCoeffs, OCoeff> IntoIterator for Mul<'plan, LCoeffs, RCoeffs>
where
    for<'a> <LCoeffs as SequenceTypes<'a>>::Item:
        ops::Mul<<RCoeffs as SequenceTypes<'a>>::Item, Output = OCoeff>,
    OCoeff: ops::AddAssign + Zero,
    LCoeffs: Sequence,
    RCoeffs: Sequence,
{
    type Item = OCoeff;
    type IntoIter = sqnc::derive::IntoIter<Self>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }
}

/// Returns a matrix that flattens coefficients for a composition of polynomials.
///
/// Let $f$ be a polynomial in $m$ variables of degree $p$ and $g$ a vector of
/// $m$ polynomials in $n$ variables of degree $q$. The composition $f ∘ g$ is
/// a polynomial in $n$ variables of degree $p q$.
///
/// This function returns a matrix $M$ that, when multiplied with a vector of
/// coefficients belonging to the outer polynomial $p$, produces the
/// coefficients for the composition, for a fixed vector of inner polynomials
/// $g$.
///
/// Argument `inner_coeffs` is a flattened sequence of the coefficients for the
/// vector of inner polynomials $g$, with `inner_nvars` the number of inner
/// variables $n$ and `inner_degree` the degree $q$.
///
/// # Errors
///
/// If the length of `inner_coeffs` sequence differs from the expected length,
/// this function returns an error.
///
/// # Examples
///
/// Let $f(x) = x_1^2 + 2 x_0 x_1$ (coefficients: `[1, 2, 0, 0, 0, 0]`) and
/// $g(y) = [y_1 - y_2, 3 y_0 + 2]$ (flattened coefficients: `[-1, 1, 0, 0, 0,
/// 0, 3, 2]`]). The composition is
///
/// $$\\begin{align}
///     f(g(y)) &= (3 y_0 + 2)^2 + 2 (y_1 - y_2) (3 y_0 + 2) \\\\
///             &= -6 y_2 y_0 - 4 y_2 + 6 y_1 y_0 + 4 y_1 + 9 y_0^2 + 12 y_0 + 4,
/// \\end{align}$$
///
/// (coefficients: `[0, 0, -6, -4, 0, 6, 4, 9, 12, 4]`).
///
/// ```
/// use ndarray::array;
/// use nutils_poly;
/// use sqnc::traits::*;
///
/// let f = array![1, 2, 0, 0, 0, 0];
/// let g = array![-1, 1, 0, 0, 0, 0, 3, 2];
/// let m = nutils_poly::composition_with_inner_matrix(g.iter().copied(), 3, 1, 2, 2)?;
/// assert_eq!(m.dot(&f), array![0, 0, -6, -4, 0, 6, 4, 9, 12, 4]);
/// # Ok::<_, nutils_poly::Error>(())
/// ```
pub fn composition_with_inner_matrix<Coeff, InnerCoeffs>(
    mut inner_coeffs: InnerCoeffs,
    inner_nvars: usize,
    inner_degree: Power,
    outer_nvars: usize,
    outer_degree: Power,
) -> Result<Array2<Coeff>, Error>
where
    Coeff: One + Zero + Clone + ops::AddAssign,
    for<'a> &'a Coeff: ops::Mul<Output = Coeff>,
    InnerCoeffs: Iterator<Item = Coeff>,
{
    let outer_ncoeffs = ncoeffs(outer_nvars, outer_degree);

    let result_degree = inner_degree * outer_degree;
    let result_ncoeffs = ncoeffs(inner_nvars, result_degree);

    let mut matrix: Array2<Coeff> = Array2::zeros((result_ncoeffs, outer_ncoeffs));

    // Initialize the column with total degree zero.
    matrix[(result_ncoeffs - 1, outer_ncoeffs - 1)] = Coeff::one();
    if outer_ncoeffs == 1 {
        return Ok(matrix);
    }

    // Initialize the columns with total degree one.
    let mapped_indices = MapDegree::new(inner_nvars, inner_degree, result_degree).unwrap();
    for ivar in 0..outer_nvars {
        let icolumn = outer_ncoeffs - 1 - ncoeffs(ivar, outer_degree);
        for irow in mapped_indices.iter() {
            matrix[(irow, icolumn)] =
                inner_coeffs
                    .next()
                    .ok_or_else(|| Error::IncorrectNumberOfCoefficients {
                        expected: ncoeffs(inner_nvars, inner_degree) * outer_nvars,
                        got: Less,
                        detail: None,
                    })?;
        }
    }
    if inner_coeffs.next().is_some() {
        return Err(Error::IncorrectNumberOfCoefficients {
            expected: ncoeffs(inner_nvars, inner_degree) * outer_nvars,
            got: More,
            detail: None,
        });
    }

    let mul = MulPlan::for_output_degree(
        &[MulVar::Both].copied().repeat(inner_nvars),
        result_degree,
        result_degree,
        result_degree,
    );
    let mut rev_powers: Vec<Power> = iter::repeat(0).take(outer_nvars).collect();
    for icol in Iterator::rev(0..outer_ncoeffs) {
        iter::zip(
            rev_powers.iter_mut(),
            index_to_powers_rev_iter(icol, outer_nvars, outer_degree).unwrap(),
        )
        .for_each(|(d, s)| *d = s);
        // Skip the columns that correspond to total degree 0 and 1: we have
        // populated these above.
        if rev_powers.iter().sum::<Power>() <= 1 {
            continue;
        }
        let rev_ivar = rev_powers.iter().position(|i| *i > 0).unwrap();
        let icol1 = outer_ncoeffs - 1 - ncoeffs(outer_nvars - rev_ivar - 1, outer_degree);
        rev_powers[rev_ivar] -= 1;
        let icol2 = powers_rev_iter_to_index(rev_powers.iter().copied(), outer_nvars, outer_degree)
            .unwrap();
        let (icol1, icol2) = if icol1 <= icol2 {
            (icol1, icol2)
        } else {
            (icol2, icol1)
        };

        let mut cols = matrix.columns_mut().into_iter();
        let cols = cols.by_ref();
        let mut col = cols.nth(icol).unwrap();
        let col1 = cols.nth(icol1 - icol - 1).unwrap();
        let col1 = col1.as_sqnc();
        let col2;
        let col2 = if icol2 == icol1 {
            col1
        } else {
            col2 = cols.nth(icol2 - icol1 - 1).unwrap();
            col2.as_sqnc()
        };
        for (dst, src) in iter::zip(col.iter_mut(), mul.apply(col1, col2).unwrap().iter()) {
            *dst = src;
        }
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::{Error, Less, MapDegree, More, MulPlan, MulVar, Multiple, PartialDerivPlan, Power};
    use core::iter;
    use ndarray::array;
    use sqnc::traits::*;

    #[test]
    fn error() {
        assert_eq!(
            Error::IncorrectNumberOfCoefficients {
                expected: 2,
                got: 3.into(),
                detail: None,
            }
            .to_string(),
            "Expected 2 coefficients but got 3.",
        );
        assert_eq!(
            Error::IncorrectNumberOfCoefficients {
                expected: 1,
                got: More,
                detail: None,
            }
            .to_string(),
            "Expected 1 coefficient but got more.",
        );
        assert_eq!(
            Error::IncorrectNumberOfCoefficients {
                expected: 2,
                got: Less,
                detail: Some("left polynomial"),
            }
            .to_string(),
            "left polynomial: Expected 2 coefficients but got less.",
        );
        assert_eq!(
            Error::TooManyVariables.to_string(),
            "The number of variables exceeds the maximum (`usize::MAX`).",
        );
    }

    #[test]
    fn ncoeffs_degree() {
        macro_rules! t {
            ($nvars:literal, $ncoeffs_array:expr) => {
                let mut ncoeffs_sum = 0;
                for (degree, ncoeffs) in $ncoeffs_array.iter().copied().enumerate() {
                    let degree = degree as Power;
                    assert_eq!(
                        super::ncoeffs($nvars, degree),
                        ncoeffs,
                        "ncoeffs({}, {degree}) != {ncoeffs}",
                        $nvars
                    );
                    assert_eq!(
                        super::ncoeffs_sum($nvars, degree),
                        ncoeffs_sum,
                        "ncoeffs_sum({}, {degree}) != {ncoeffs_sum}",
                        $nvars
                    );
                    if $nvars > 0 || degree == 0 {
                        assert_eq!(
                            super::degree($nvars, ncoeffs),
                            Some(degree),
                            "degree({}, {ncoeffs}) != Some({degree})",
                            $nvars
                        );
                    }
                    ncoeffs_sum += ncoeffs;
                }
                assert!(
                    super::ncoeffs_iter($nvars)
                        .take($ncoeffs_array.len())
                        .eq($ncoeffs_array),
                    "ncoeffs_iter({}).take({}) != {:?}",
                    $nvars,
                    $ncoeffs_array.len(),
                    $ncoeffs_array
                );
                assert!(
                    super::degree_ncoeffs_iter($nvars)
                        .take($ncoeffs_array.len())
                        .eq(iter::zip(0.., $ncoeffs_array)),
                    "degree_ncoeffs_iter({}).take({}) != {:?}",
                    $nvars,
                    $ncoeffs_array.len(),
                    $ncoeffs_array
                );
            };
        }

        t! {0, [1, 1, 1, 1, 1]}
        t! {1, [1, 2, 3, 4, 5]}
        t! {2, [1, 3, 6, 10, 15]}
        t! {3, [1, 4, 10, 20, 35]}
        t! {4, [1, 5, 15]}

        assert_eq!(super::degree(0, 0), None);
        assert_eq!(super::degree(0, 2), None);
        assert_eq!(super::degree(1, 0), None);
        assert_eq!(super::degree(2, 0), None);
        assert_eq!(super::degree(2, 2), None);
        assert_eq!(super::degree(2, 4), None);
        assert_eq!(super::degree(2, 9), None);
        assert_eq!(super::degree(2, 11), None);
        assert_eq!(super::degree(3, 0), None);
        assert_eq!(super::degree(3, 2), None);
        assert_eq!(super::degree(3, 3), None);
        assert_eq!(super::degree(4, 3), None);
    }

    #[test]
    fn powers_index_maps() {
        macro_rules! assert_index_powers {
            ($degree:literal, $desired_powers:tt) => {
                let nvars = $desired_powers[0].len();
                for (desired_index, desired_powers) in $desired_powers.into_iter().enumerate() {
                    assert_eq!(
                        super::powers_rev_iter_to_index(
                            desired_powers.iter().rev().copied(),
                            nvars,
                            $degree
                        ),
                        Some(desired_index)
                    );
                    assert!(
                        super::index_to_powers_rev_iter(desired_index, nvars, $degree)
                            .unwrap()
                            .eq(desired_powers.iter().rev().copied())
                    )
                }
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
    fn map_degree() {
        assert_eq!(MapDegree::new(1, 3, 2), None);
        assert!(MapDegree::new(0, 0, 0).unwrap().iter().eq([0]));
        assert!(MapDegree::new(1, 2, 2).unwrap().iter().eq([0, 1, 2]));
        assert!(MapDegree::new(1, 2, 3).unwrap().iter().eq([1, 2, 3]));
        assert!(MapDegree::new(1, 2, 4).unwrap().iter().eq([2, 3, 4]));
        assert!(MapDegree::new(2, 1, 2).unwrap().iter().eq([2, 4, 5]));

        assert!(MapDegree::new(0, 0, 0).unwrap().into_iter().eq([0]));
        assert!(MapDegree::new(1, 2, 2).unwrap().into_iter().eq([0, 1, 2]));
    }

    #[test]
    fn eval_0d() {
        let values: [usize; 0] = [];
        assert_eq!(super::eval([1], &values, 0), Ok(1));
        assert!(super::eval(0..0, &values, 0).is_err());
    }

    #[test]
    fn eval_1d() {
        let values = [5];
        assert_eq!(super::eval([1], &values, 0), Ok(1));
        assert_eq!(super::eval([2, 1], &values, 1), Ok(11));
        assert_eq!(super::eval([3, 2, 1], &values, 2), Ok(86));
        assert!(super::eval(0..1, &values, 1).is_err());
    }

    #[test]
    fn eval_2d() {
        let values = [5, 3];
        assert_eq!(super::eval([1], &values, 0), Ok(1));
        assert_eq!(super::eval([0, 0, 1], &values, 1), Ok(1));
        assert_eq!(super::eval([0, 1, 0], &values, 1), Ok(5));
        assert_eq!(super::eval([1, 0, 0], &values, 1), Ok(3));
        assert_eq!(super::eval([3, 2, 1], &values, 1), Ok(20));
        assert_eq!(super::eval(Iterator::rev(1..7), &values, 2), Ok(227));
        assert!(super::eval(0..2, &values, 1).is_err());
    }

    #[test]
    fn eval_3d() {
        let values = [5, 3, 2];
        assert_eq!(super::eval([1], &values, 0), Ok(1));
        assert_eq!(super::eval([0, 0, 0, 1], &values, 1), Ok(1));
        assert_eq!(super::eval([0, 0, 1, 0], &values, 1), Ok(5));
        assert_eq!(super::eval([0, 1, 0, 0], &values, 1), Ok(3));
        assert_eq!(super::eval([1, 0, 0, 0], &values, 1), Ok(2));
        assert_eq!(super::eval([4, 3, 2, 1], &values, 1), Ok(28));
        assert_eq!(super::eval(Iterator::rev(1..11), &values, 2), Ok(415));
        assert!(super::eval(0..3, &values, 1).is_err());
    }

    #[test]
    fn eval_4d() {
        let values = [5, 3, 2, 1];
        assert_eq!(super::eval([1], &values, 0), Ok(1));
        assert_eq!(super::eval([0, 0, 0, 0, 1], &values, 1), Ok(1));
        assert_eq!(super::eval([0, 0, 0, 1, 0], &values, 1), Ok(5));
        assert_eq!(super::eval([0, 0, 1, 0, 0], &values, 1), Ok(3));
        assert_eq!(super::eval([0, 1, 0, 0, 0], &values, 1), Ok(2));
        assert_eq!(super::eval([1, 0, 0, 0, 0], &values, 1), Ok(1));
        assert!(super::eval(0..4, &values, 1).is_err());
    }

    #[test]
    fn multiple() {
        assert_eq!(2.multiple(3), 6);
        assert_eq!((&2).multiple(3), 6);
    }

    #[test]
    fn partial_deriv_x0_x() {
        let plan = PartialDerivPlan::new(1, 0, 0).unwrap();
        assert_eq!(plan.ncoeffs_input(), 1);
        assert_eq!(plan.ncoeffs_output(), 1);
        assert_eq!(plan.degree_output(), 0);
        assert_eq!(plan.nvars(), 1);
        let pd = plan.apply([1]).unwrap();
        assert!(pd.iter().eq([0]));
        assert_eq!(pd.iter().size_hint(), (1, Some(1)));
        assert_eq!(pd.get(0), Some(0));
        assert_eq!(pd.len(), 1);
        let iter = pd.into_iter();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert!(iter.eq([0]));
    }

    #[test]
    fn partial_deriv_x1_x() {
        let plan = PartialDerivPlan::new(1, 1, 0).unwrap();
        assert_eq!(plan.ncoeffs_input(), 2);
        assert_eq!(plan.ncoeffs_output(), 1);
        assert_eq!(plan.degree_output(), 0);
        assert_eq!(plan.nvars(), 1);
        let pd = plan.apply([2, 1]).unwrap();
        assert!(pd.iter().eq([2]));
        assert_eq!(pd.get(1), None);
        assert_eq!(pd.len(), 1);
    }

    #[test]
    fn partial_deriv_x3_x() {
        let plan = PartialDerivPlan::new(1, 3, 0).unwrap();
        assert_eq!(plan.ncoeffs_input(), 4);
        assert_eq!(plan.ncoeffs_output(), 3);
        assert_eq!(plan.degree_output(), 2);
        assert_eq!(plan.nvars(), 1);
        let pd = plan.apply([4, 3, 2, 1]).unwrap();
        assert!(pd.iter().eq([12, 6, 2]));
        assert_eq!(pd.get(2), Some(2));
        assert_eq!(pd.len(), 3);
    }

    #[test]
    fn partial_deriv_xy2_x() {
        let plan = PartialDerivPlan::new(2, 2, 0).unwrap();
        assert_eq!(plan.ncoeffs_input(), 6);
        assert_eq!(plan.ncoeffs_output(), 3);
        assert_eq!(plan.degree_output(), 1);
        assert_eq!(plan.nvars(), 2);
        let pd = plan.apply([6, 5, 4, 3, 2, 1]).unwrap();
        assert!(pd.iter().eq([5, 6, 2]));
        assert_eq!(pd.get(1), Some(6));
        assert_eq!(pd.len(), 3);
    }

    #[test]
    fn partial_deriv_xy2_y() {
        let plan = PartialDerivPlan::new(2, 2, 1).unwrap();
        assert_eq!(plan.ncoeffs_input(), 6);
        assert_eq!(plan.ncoeffs_output(), 3);
        assert_eq!(plan.degree_output(), 1);
        assert_eq!(plan.nvars(), 2);
        let pd = plan.apply([6, 5, 4, 3, 2, 1]).unwrap();
        assert!(pd.iter().eq([12, 5, 4]));
        assert!(pd.iter().rev().eq([4, 5, 12]));
        assert_eq!(pd.get(2), Some(4));
        assert_eq!(pd.len(), 3);
    }

    #[test]
    fn partial_deriv_out_of_range() {
        assert!(PartialDerivPlan::new(1, 2, 2).is_none());
    }

    #[test]
    fn mul_x1_x1() {
        let plan = MulPlan::same_vars(1, 1, 1);
        assert_eq!(plan.ncoeffs_left(), 2);
        assert_eq!(plan.ncoeffs_right(), 2);
        assert_eq!(plan.ncoeffs_output(), 3);
        assert_eq!(plan.nvars_output(), 1);
        assert_eq!(plan.degree_output(), 2);
        // (2 x + 1) (4 x + 3) = 8 x^2 + 10 x + 3
        let mul = plan.apply([2, 1], [4, 3]).unwrap();
        assert!(mul.iter().eq([8, 10, 3]));
        assert!(mul.into_iter().eq([8, 10, 3]));
    }

    #[test]
    fn mul_x1_y1() {
        let plan = MulPlan::different_vars(1, 1, 1, 1).unwrap();
        assert_eq!(plan.ncoeffs_left(), 2);
        assert_eq!(plan.ncoeffs_right(), 2);
        assert_eq!(plan.ncoeffs_output(), 6);
        assert_eq!(plan.nvars_output(), 2);
        assert_eq!(plan.degree_output(), 2);
        // (2 x + 1) (4 y + 3) = 8 x y + 4 y + 6 x + 3
        let mul = plan.apply([2, 1], [4, 3]).unwrap();
        assert!(mul.iter().eq([0, 8, 4, 0, 6, 3]));
        assert!(mul.into_iter().eq([0, 8, 4, 0, 6, 3]));
    }

    #[test]
    fn mul_xy1_y2() {
        let plan = MulPlan::new(&[MulVar::Left, MulVar::Both].copied(), 1, 2);
        assert_eq!(plan.ncoeffs_left(), 3);
        assert_eq!(plan.ncoeffs_right(), 3);
        assert_eq!(plan.ncoeffs_output(), 10);
        assert_eq!(plan.nvars_output(), 2);
        assert_eq!(plan.degree_output(), 3);
        // (3 y + 2 x + 1) (6 y^2 + 5 y + 4)
        // = 18 y^3 + 12 x y^2 + 21 y^2 + 10 x y + 17 y + 8 x + 4
        let mul = plan.apply([3, 2, 1], [6, 5, 4]).unwrap();
        assert!(mul.iter().eq([18, 12, 21, 0, 10, 17, 0, 0, 8, 4]));
        assert!(mul.into_iter().eq([18, 12, 21, 0, 10, 17, 0, 0, 8, 4]));
    }

    #[test]
    fn mul_errors() {
        let plan = MulPlan::same_vars(1, 1, 1);
        assert_eq!(
            plan.apply([2, 1, 3], [4, 3]).unwrap_err(),
            Error::IncorrectNumberOfCoefficients {
                expected: 2,
                got: 3.into(),
                detail: Some("left polynomial"),
            }
        );
        assert_eq!(
            plan.apply([2, 1], [4, 3, 2]).unwrap_err(),
            Error::IncorrectNumberOfCoefficients {
                expected: 2,
                got: 3.into(),
                detail: Some("right polynomial"),
            }
        );
    }

    #[test]
    fn composition() {
        assert_eq!(
            super::composition_with_inner_matrix(0..0, 0, 0, 0, 0),
            Ok(array![[1]])
        );

        let f = array![1, 2, 0, 0, 0, 0];
        let g = array![-1, 1, 0, 0, 0, 0, 3, 2];
        let m = super::composition_with_inner_matrix(g.iter().copied(), 3, 1, 2, 2).unwrap();
        assert_eq!(m.dot(&f), array![0, 0, -6, -4, 0, 6, 4, 9, 12, 4]);
    }

    #[test]
    fn composition_errors() {
        assert_eq!(
            super::composition_with_inner_matrix(0..1, 1, 3, 1, 2),
            Err(Error::IncorrectNumberOfCoefficients {
                expected: 4,
                got: Less,
                detail: None,
            })
        );
        assert_eq!(
            super::composition_with_inner_matrix(0..9, 1, 3, 1, 2),
            Err(Error::IncorrectNumberOfCoefficients {
                expected: 4,
                got: More,
                detail: None,
            })
        );
    }
}

#[cfg(all(feature = "bench", test))]
mod benches {
    extern crate test;
    use self::test::Bencher;
    use super::MulPlan;
    use sqnc::traits::*;
    use std::iter;

    macro_rules! mk_bench_eval {
        ($name:ident, $nvars:literal, $degree:literal) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let coeffs =
                    Vec::from_iter(Iterator::map(1..=super::ncoeffs($nvars, $degree), |i| {
                        i as f64
                    }));
                let values: Vec<_> = (1..=$nvars).map(|x| x as f64).collect();
                b.iter(|| {
                    super::eval(
                        test::black_box(coeffs.as_slice()),
                        test::black_box(&values[..]),
                        $degree,
                    )
                    .unwrap()
                })
            }
        };
    }

    mk_bench_eval! {eval_1d_degree1, 1, 1}
    mk_bench_eval! {eval_1d_degree2, 1, 2}
    mk_bench_eval! {eval_1d_degree3, 1, 3}
    mk_bench_eval! {eval_1d_degree4, 1, 4}
    mk_bench_eval! {eval_2d_degree1, 2, 1}
    mk_bench_eval! {eval_2d_degree2, 2, 2}
    mk_bench_eval! {eval_2d_degree3, 2, 3}
    mk_bench_eval! {eval_2d_degree4, 2, 4}
    mk_bench_eval! {eval_3d_degree1, 3, 1}
    mk_bench_eval! {eval_3d_degree2, 3, 2}
    mk_bench_eval! {eval_3d_degree3, 3, 3}
    mk_bench_eval! {eval_3d_degree4, 3, 4}
    mk_bench_eval! {eval_4d_degree1, 4, 1}
    mk_bench_eval! {eval_4d_degree2, 4, 2}
    mk_bench_eval! {eval_4d_degree3, 4, 3}
    mk_bench_eval! {eval_4d_degree4, 4, 4}

    #[bench]
    fn ncoeffs_3d_degree4(b: &mut Bencher) {
        b.iter(|| super::ncoeffs(test::black_box(3), test::black_box(4)));
    }

    #[bench]
    fn mul_xyz4_xyz2(b: &mut Bencher) {
        let plan = MulPlan::same_vars(3, 4, 2);
        let lcoeffs = Vec::from_iter(Iterator::map(0..super::ncoeffs(3, 4), |i| i as f64));
        let rcoeffs = Vec::from_iter(Iterator::map(0..super::ncoeffs(3, 2), |i| i as f64));
        let mut ocoeffs = Vec::from_iter(iter::repeat(0f64).take(plan.ncoeffs_output()));
        b.iter(|| {
            ocoeffs
                .as_mut_sqnc()
                .assign(
                    plan.apply(
                        test::black_box(lcoeffs.as_sqnc()),
                        test::black_box(rcoeffs.as_sqnc()),
                    )
                    .unwrap(),
                )
                .unwrap()
        })
    }

    #[bench]
    fn mul_x4_yz2(b: &mut Bencher) {
        let plan = MulPlan::different_vars(1, 2, 4, 2).unwrap();
        let lcoeffs: Vec<f64> = Iterator::map(0..super::ncoeffs(1, 4), |i| i as f64).collect();
        let rcoeffs: Vec<f64> = Iterator::map(0..super::ncoeffs(2, 2), |i| i as f64).collect();
        let mut ocoeffs: Vec<f64> = iter::repeat(0f64).take(plan.ncoeffs_output()).collect();
        b.iter(|| {
            ocoeffs
                .as_mut_sqnc()
                .assign(
                    MulPlan::apply(
                        test::black_box(&plan),
                        test::black_box(lcoeffs.as_sqnc()),
                        test::black_box(rcoeffs.as_sqnc()),
                    )
                    .unwrap(),
                )
                .unwrap()
        })
    }
}
