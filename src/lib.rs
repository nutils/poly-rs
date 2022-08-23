//! Functions for evaluating and manipulating polynomials.
//!
//! The polynomials considered in this crate are of the form
//!
//! ```text
//! Σ_{k ∈ ℤ^n | Σ_i k_i ≤ p} c_k ∏_i x_i^(k_i)
//! ```
//!
//! where `c` is a vector of coefficients, `x` a vector of `n` variables and
//! `p` a nonnegative integer degree.
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
//! The vector of coefficients for the polynomial `p(x) = x^2 - x + 2` is
//! `[1, -2, 2]`.
//!
//! ```
//! use nutils_poly::{Eval as _, PartialDeriv as _, PolySequence};
//!
//! let p = PolySequence::new([1, -1, 2], 2, 1).unwrap();
//! ```
//!
//! You can evaluate this polynomial for some `x` using [`Eval::eval()`]:
//!
//! ```
//! # use nutils_poly::{Eval as _, PartialDeriv as _, PolySequence};
//! #
//! # let p = PolySequence::new([1, -1, 2], 2, 1).unwrap();
//! assert_eq!(p.eval(&[0]), 2); // x = 0
//! assert_eq!(p.eval(&[1]), 2); // x = 1
//! assert_eq!(p.eval(&[2]), 4); // x = 2
//! ```
//!
//! Or compute the partial derivative `∂p/∂x` using [`PartialDeriv::partial_deriv()`]:
//!
//! ```
//! # use nutils_poly::{Eval as _, PartialDeriv as _, PolySequence};
//! #
//! # let p = PolySequence::new([1, -1, 2], 2, 1).unwrap();
//! assert_eq!(p.partial_deriv(0), Some(PolySequence::new(vec![2, -1], 1, 1).unwrap()));
//! ```
//!
//! [lexicographic order]: https://en.wikipedia.org/wiki/Lexicographic_order

#![cfg_attr(feature = "bench", feature(test))]

use num_traits::cast::FromPrimitive;
use num_traits::Zero;
use std::iter;
use std::marker::PhantomData;
use std::ops;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    AssignDifferentNumberOfVariables,
    AssignLowerDegree,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssignDifferentNumberOfVariables => write!(f, "The number of variables of the destination polynomial differs from that of the source."),
            Self::AssignLowerDegree => write!(f, "The degree of the destination polynomial is lower than the degree of the source."),
        }
    }
}

#[inline]
fn uniform_vec<T: Clone>(item: T, len: usize) -> Vec<T> {
    iter::repeat(item).take(len).collect()
}

#[inline]
fn uniform_vec_with<T, F>(gen_item: F, len: usize) -> Vec<T>
where
    F: FnMut() -> T,
{
    iter::repeat_with(gen_item).take(len).collect()
}

#[inline]
fn uniform_box<T: Clone>(item: T, len: usize) -> Box<[T]> {
    iter::repeat(item).take(len).collect()
}

#[inline]
fn uniform_box_with<T, F>(gen_item: F, len: usize) -> Box<[T]>
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

// TODO: Compact representation of `Powers`:
//
// #[derive(Debug, Clone)]
// enum Powers {
//     /// Limited to six variables and degree 255.
//     Inline {
//         len: u8,
//         powers: [u8; 6],
//     },
//     /// Unlimited.
//     Boxed(Box<Vec<usize>>),
// }
//
// impl Sequence<usize> for Powers { ... }
// impl MutSequence<usize> for Powers { ... }

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

/// Lending iterator of powers in reverse lexicographic order.
///
/// This struct is created by [`powers_iter()`].
#[derive(Debug, Clone, PartialEq)]
struct PowersIter {
    powers: Vec<usize>,
    rem: usize,
}

impl PowersIter {
    /// Returns a reference to the next vector of powers in reverse lexicographic order.
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

    /// Returns an iterator of powers corresponding to one variable.
    #[inline]
    fn get(self, index: usize) -> Option<NthPowerIter> {
        (index < self.powers.len()).then(|| NthPowerIter {
            powers: self,
            index,
        })
    }

    /// Returns an iterator of the sum of the powers.
    #[inline]
    fn sum_of_powers(self) -> SumOfPowersIter {
        SumOfPowersIter(self)
    }

    /// Resets the iterator to the initial state for the given degree.
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

/// Returns a lending iterator of powers in reverse lexicographic order.
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
struct SumOfPowersIter(PowersIter);

impl Iterator for SumOfPowersIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.0.next().map(|powers| powers.iter().sum())
    }
}

// Workaround for associated type `Iter<'a>: Iterator<Item = &'a T>` of
// `Sequence` from
// https://web.archive.org/web/20220530082425/https://sabrinajewson.org/blog/the-better-alternative-to-lifetime-gats#the-better-gats

/// An interface for finite sequences.
pub trait Sequence<T>
where
    Self: for<'this> SequenceIterType<'this, &'this T>,
{
    /// Returns the number of elements in the sequence.
    fn len(&self) -> usize;

    /// Returns a reference to an element or `None` if the index is out of bounds.
    fn get(&self, index: usize) -> Option<&T>;

    /// Returns a reference to an element, without doing bounds checking.
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.get(index).unwrap()
    }

    /// Returns an iterator over the sequence.
    fn iter(&self) -> <Self as SequenceIterType<'_, &T>>::Iter;
}

/// An interface for finite sequences with mutable elements.
pub trait MutSequence<T>: Sequence<T>
where
    Self: for<'this> SequenceIterMutType<'this, &'this mut T>,
{
    /// Returns a mutable reference to an element or `None` if the index is out of bounds.
    fn get_mut(&mut self, index: usize) -> Option<&mut T>;

    /// Returns a mutable reference to an element, without doing bounds checking.
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).unwrap()
    }

    /// Returns an iterator over the sequence that allows modifying each element.
    fn iter_mut(&mut self) -> <Self as SequenceIterMutType<'_, &mut T>>::IterMut;
}

/// Return type of [`Sequence::iter()`].
pub trait SequenceIterType<'this, T> {
    /// Return type of [`Sequence::iter()`].
    type Iter: Iterator<Item = T>;
}

/// Return type of [`MutSequence::iter_mut()`].
pub trait SequenceIterMutType<'this, T> {
    /// Return type of [`MutSequence::iter_mut()`].
    type IterMut: Iterator<Item = T>;
}

macro_rules! impl_sequence_for_as_ref_slice {
    ($T:ident, $ty:ty, <$($params:tt)*) => {
        impl<'this, $($params)* SequenceIterType<'this, &'this $T> for $ty {
            type Iter = std::slice::Iter<'this, T>;
        }

        impl<'this, $($params)* SequenceIterMutType<'this, &'this mut $T> for $ty {
            type IterMut = std::slice::IterMut<'this, $T>;
        }

        impl<$($params)* Sequence<$T> for $ty {
            #[inline]
            fn len(&self) -> usize {
                <Self as AsRef::<[$T]>>::as_ref(self).len()
            }
            #[inline]
            fn get(&self, index: usize) -> Option<&$T> {
                <Self as AsRef::<[$T]>>::as_ref(self).get(index)
            }
            #[inline]
            unsafe fn get_unchecked(&self, index: usize) -> &$T {
                <Self as AsRef::<[$T]>>::as_ref(self).get_unchecked(index)
            }
            #[inline]
            fn iter(&self) -> <Self as SequenceIterType<'_, &$T>>::Iter {
                <Self as AsRef::<[$T]>>::as_ref(self).iter()
            }
        }

        impl<$($params)* MutSequence<$T> for $ty {
            #[inline]
            fn get_mut(&mut self, index: usize) -> Option<&mut $T> {
                <Self as AsMut::<[$T]>>::as_mut(self).get_mut(index)
            }
            #[inline]
            unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut $T {
                <Self as AsMut::<[$T]>>::as_mut(self).get_unchecked_mut(index)
            }
            #[inline]
            fn iter_mut(&mut self) -> <Self as SequenceIterMutType<'_, &mut $T>>::IterMut {
                <Self as AsMut::<[$T]>>::as_mut(self).iter_mut()
            }
        }
    };
}

impl_sequence_for_as_ref_slice! {T, [T], <T>}
impl_sequence_for_as_ref_slice! {T, [T; N], <T, const N: usize>}
impl_sequence_for_as_ref_slice! {T, Vec<T>, <T>}
impl_sequence_for_as_ref_slice! {T, Box<[T]>, <T>}

impl<'this, T, S: SequenceIterType<'this, &'this T> + ?Sized> SequenceIterType<'this, &'this T>
    for &S
{
    type Iter = <S as SequenceIterType<'this, &'this T>>::Iter;
}

impl<'this, T, S: SequenceIterType<'this, &'this T> + ?Sized> SequenceIterType<'this, &'this T>
    for &mut S
{
    type Iter = <S as SequenceIterType<'this, &'this T>>::Iter;
}

impl<'this, T, S: SequenceIterMutType<'this, &'this mut T> + ?Sized>
    SequenceIterMutType<'this, &'this mut T> for &mut S
{
    type IterMut = <S as SequenceIterMutType<'this, &'this mut T>>::IterMut;
}

impl<T, S: Sequence<T> + ?Sized> Sequence<T> for &S {
    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
    #[inline]
    fn get(&self, index: usize) -> Option<&T> {
        (**self).get(index)
    }
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        (**self).get_unchecked(index)
    }
    #[inline]
    fn iter(&self) -> <Self as SequenceIterType<'_, &T>>::Iter {
        (**self).iter()
    }
}

impl<T, S: Sequence<T> + ?Sized> Sequence<T> for &mut S {
    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
    #[inline]
    fn get(&self, index: usize) -> Option<&T> {
        (**self).get(index)
    }
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        (**self).get_unchecked(index)
    }
    #[inline]
    fn iter(&self) -> <Self as SequenceIterType<'_, &T>>::Iter {
        (**self).iter()
    }
}

impl<T, S: MutSequence<T> + ?Sized> MutSequence<T> for &mut S {
    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        (**self).get_mut(index)
    }
    #[inline]
    fn iter_mut(&mut self) -> <Self as SequenceIterMutType<'_, &mut T>>::IterMut {
        (**self).iter_mut()
    }
}

pub trait Dimensions {
    fn degree(&self) -> usize;
    fn nvars(&self) -> usize;
}

impl<'this, T: Dimensions> Dimensions for &'this T {
    #[inline]
    fn degree(&self) -> usize {
        (**self).degree()
    }
    #[inline]
    fn nvars(&self) -> usize {
        (**self).nvars()
    }
}

/// Polynomial with the coefficients stored in a [`Sequence`].
#[derive(Debug, Clone, PartialEq)]
pub struct PolySequence<Coeff, Coeffs>
where
    Coeffs: Sequence<Coeff>,
{
    coeffs: Coeffs,
    degree: usize,
    nvars: usize,
    phantom: PhantomData<Coeff>,
}

impl<Coeff, Coeffs> Dimensions for PolySequence<Coeff, Coeffs>
where
    Coeffs: Sequence<Coeff>,
{
    #[inline]
    fn degree(&self) -> usize {
        self.degree
    }

    #[inline]
    fn nvars(&self) -> usize {
        self.nvars
    }
}

// Return type of `PolySequence::iter()`.
type PolySequenceIter<'a, Coeff, Coeffs> =
    PolyIter<&'a Coeff, <Coeffs as SequenceIterType<'a, &'a Coeff>>::Iter>;

impl<Coeff, Coeffs> PolySequence<Coeff, Coeffs>
where
    Coeffs: Sequence<Coeff>,
{
    #[inline]
    pub fn new(coeffs: Coeffs, degree: usize, nvars: usize) -> Option<Self> {
        (coeffs.len() == ncoeffs(degree, nvars)).then(|| Self {
            coeffs,
            degree,
            nvars,
            phantom: PhantomData,
        })
    }

    /// Returns a [`PolyIter`] with references to the coefficients of this polynomial.
    #[inline]
    pub fn iter(&self) -> PolySequenceIter<Coeff, Coeffs> {
        PolyIter {
            coeffs_iter: self.coeffs.iter(),
            degree: self.degree,
            nvars: self.nvars,
        }
    }

    /// Converts this polynomial into a [`PolyIter`].
    #[inline]
    pub fn into_iter(self) -> PolyIter<Coeff, <Coeffs as IntoIterator>::IntoIter>
    where
        Coeffs: IntoIterator<Item = Coeff>,
    {
        PolyIter {
            coeffs_iter: self.coeffs.into_iter(),
            degree: self.degree,
            nvars: self.nvars,
        }
    }

    /// Assign a polynomial to this polynomial.
    ///
    /// Returns an error if the number of variables of the source is not equal
    /// to the number of this polynomial, or if the degree of the source is
    /// higher than the degree of this polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use nutils_poly::{PartialDeriv as _, PolySequence};
    ///
    /// let p = PolySequence::new([1, -1, 2], 2, 1).unwrap();
    /// let mut q = PolySequence::new([0, 0, 0, 0], 3, 1).unwrap();
    /// q.assign(p.into_iter()).unwrap();
    /// assert_eq!(q, PolySequence::new([0, 1, -1, 2], 3, 1).unwrap());
    /// ```
    #[must_use]
    pub fn assign<SrcCoeffs>(&mut self, iter: PolyIter<Coeff, SrcCoeffs>) -> Result<(), Error>
    where
        Coeff: Zero,
        Coeffs: MutSequence<Coeff>,
        SrcCoeffs: Iterator<Item = Coeff>,
    {
        if iter.nvars != self.nvars {
            Err(Error::AssignDifferentNumberOfVariables)
        } else if iter.degree > self.degree {
            Err(Error::AssignLowerDegree)
        } else if iter.degree == self.degree {
            for (dst, src) in iter::zip(self.coeffs.iter_mut(), iter.coeffs_iter) {
                *dst = src;
            }
            Ok(())
        } else {
            let pick = powers_iter(self.degree, self.nvars)
                .sum_of_powers()
                .map(|p| p < self.degree);
            let mut src = iter.coeffs_iter;
            for (dst, pick) in iter::zip(self.coeffs.iter_mut(), pick) {
                *dst = if pick {
                    src.next().unwrap()
                } else {
                    Coeff::zero()
                };
            }
            Ok(())
        }
    }
}

impl<Coeff, Coeffs> PolySlice<Coeff, Coeffs> where Coeffs: AsRef<[Coeff]> {}

#[derive(Debug, Clone, PartialEq)]
pub struct PolySlice<Coeff, Coeffs>
where
    Coeffs: AsRef<[Coeff]>,
{
    coeffs: Coeffs,
    degree: usize,
    nvars: usize,
    phantom: PhantomData<Coeff>,
}

impl<Coeff, Coeffs> PolySlice<Coeff, Coeffs>
where
    Coeffs: AsRef<[Coeff]>,
{
    pub fn new(coeffs: Coeffs, degree: usize, nvars: usize) -> Option<Self> {
        (coeffs.as_ref().len() == ncoeffs(degree, nvars)).then(|| Self {
            coeffs,
            degree,
            nvars,
            phantom: PhantomData,
        })
    }
}

// Return type of `PolySlice::iter()`.
type PolySliceIter<'a, Coeff> = PolyIter<&'a Coeff, std::slice::Iter<'a, Coeff>>;

impl<Coeff, Coeffs> PolySlice<Coeff, Coeffs>
where
    Coeffs: AsRef<[Coeff]>,
{
    #[inline]
    pub fn iter(&self) -> PolySliceIter<Coeff> {
        PolyIter {
            coeffs_iter: self.coeffs.as_ref().iter(),
            degree: self.degree,
            nvars: self.nvars,
        }
    }
}

/// Polynomial with the coefficients stored in an [`Iterator`].
///
/// This struct is typically created by [`PolySequence::iter()`] or functions
/// that derive a polynomial from one or more polynomials, e.g.
/// [`PartialDeriv::partial_deriv_iter()`].
#[derive(Debug, Clone, PartialEq)]
pub struct PolyIter<Coeff, CoeffsIter>
where
    CoeffsIter: Iterator<Item = Coeff>,
{
    coeffs_iter: CoeffsIter,
    degree: usize,
    nvars: usize,
}

impl<Coeff, CoeffsIter> Dimensions for PolyIter<Coeff, CoeffsIter>
where
    CoeffsIter: Iterator<Item = Coeff>,
{
    #[inline]
    fn degree(&self) -> usize {
        self.degree
    }

    #[inline]
    fn nvars(&self) -> usize {
        self.nvars
    }
}

impl<Coeff, CoeffsIter> PolyIter<Coeff, CoeffsIter>
where
    CoeffsIter: Iterator<Item = Coeff>,
{
    /// Creates a new [`PolyIter`].
    #[inline]
    pub fn new(coeffs_iter: CoeffsIter, degree: usize, nvars: usize) -> Option<Self>
    where
        CoeffsIter: ExactSizeIterator,
    {
        (coeffs_iter.len() == ncoeffs(degree, nvars)).then(|| Self {
            coeffs_iter,
            degree,
            nvars,
        })
    }
    /// Creates a [`PolySequence`] from this polynomial.
    #[inline]
    pub fn collect<Coeffs>(self) -> PolySequence<Coeff, Coeffs>
    where
        Coeffs: Sequence<Coeff> + FromIterator<Coeff>,
    {
        PolySequence {
            coeffs: self.coeffs_iter.collect(),
            degree: self.degree,
            nvars: self.nvars,
            phantom: PhantomData,
        }
    }
}

impl<'a, Coeff, CoeffsIter> PolyIter<&'a Coeff, CoeffsIter>
where
    CoeffsIter: Iterator<Item = &'a Coeff>,
    Coeff: 'a + Copy,
{
    /// Creates a [`PolyIter`] which copies all coefficients.
    pub fn copied(self) -> PolyIter<Coeff, iter::Copied<CoeffsIter>> {
        PolyIter {
            coeffs_iter: self.coeffs_iter.copied(),
            degree: self.degree,
            nvars: self.nvars,
        }
    }
}

impl<'a, Coeff, CoeffsIter> PolyIter<&'a Coeff, CoeffsIter>
where
    CoeffsIter: Iterator<Item = &'a Coeff>,
    Coeff: 'a + Clone,
{
    /// Creates a [`PolyIter`] which clones all coefficients.
    pub fn cloned(self) -> PolyIter<Coeff, iter::Cloned<CoeffsIter>> {
        PolyIter {
            coeffs_iter: self.coeffs_iter.cloned(),
            degree: self.degree,
            nvars: self.nvars,
        }
    }
}

// TODO: impl PolyIter::change_degree(self, degree: usize) -> PolyIter

/// Evaluate a polynomial.
pub trait Eval<Var> {
    /// Evaluates the polynomial for the given variables.
    ///
    /// # Examples
    ///
    /// Consider the polynomial `2 - x + x^2` (coefficients: `[1, -1, 2]`).
    ///
    /// ```
    /// use nutils_poly::{PolySequence, Eval as _};
    ///
    /// let poly = PolySequence::new([1, -1, 2], 2, 1).unwrap();
    /// assert_eq!(poly.eval(&[0]), 2); // x = 0
    /// assert_eq!(poly.eval(&[1]), 2); // x = 1
    /// assert_eq!(poly.eval(&[2]), 4); // x = 2
    /// ```
    fn eval<Vars>(self, vars: &Vars) -> Var
    where
        Vars: Sequence<Var> + ?Sized;
}

impl<Coeff, Coeffs, Var> Eval<Var> for &PolySequence<Coeff, Coeffs>
where
    Coeffs: Sequence<Coeff>,
    Var: Zero + ops::AddAssign,
    for<'a> Var: ops::MulAssign<&'a Var> + ops::AddAssign<&'a Coeff>,
{
    #[inline]
    fn eval<Vars>(self, vars: &Vars) -> Var
    where
        Vars: Sequence<Var> + ?Sized,
    {
        self.iter().eval(vars)
    }
}

impl<Coeff, Coeffs, Var> Eval<Var> for &PolySlice<Coeff, Coeffs>
where
    Coeffs: AsRef<[Coeff]>,
    Var: Zero + ops::AddAssign,
    for<'a> Var: ops::MulAssign<&'a Var> + ops::AddAssign<&'a Coeff>,
{
    #[inline]
    fn eval<Vars>(self, vars: &Vars) -> Var
    where
        Vars: Sequence<Var> + ?Sized,
    {
        self.iter().eval(vars)
    }
}

impl<Coeff, Coeffs, Var> Eval<Var> for PolyIter<Coeff, Coeffs>
where
    Coeffs: Iterator<Item = Coeff>,
    Var: Zero + ops::AddAssign + ops::AddAssign<Coeff>,
    for<'a> Var: ops::MulAssign<&'a Var>,
{
    #[inline]
    fn eval<Vars>(mut self, vars: &Vars) -> Var
    where
        Vars: Sequence<Var> + ?Sized,
    {
        assert_eq!(vars.len(), self.nvars);
        DefaultEvalCoeffsIter.eval(&mut self.coeffs_iter, self.degree, &vars)
    }
}

trait EvalCoeffsIter<Var, Coeff, Output> {
    fn from_coeff(&self, coeff: Coeff) -> Output;
    fn init_acc(&self) -> Output;
    fn update_acc_coeff(&self, acc: &mut Output, coeff: Coeff, var: &Var);
    fn update_acc_inner(&self, acc: &mut Output, inner: Output, var: &Var);

    #[inline]
    fn eval<Coeffs, Vars>(&self, coeffs: &mut Coeffs, degree: usize, vars: &Vars) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Vars: Sequence<Var> + ?Sized,
    {
        match vars.len() {
            0 => self.eval_0d(coeffs),
            1 => self.eval_1d(coeffs, degree, vars),
            2 => self.eval_2d(coeffs, degree, vars),
            3 => self.eval_3d(coeffs, degree, vars),
            nvars => self.eval_nd(coeffs, degree, vars, nvars),
        }
    }

    #[inline]
    fn eval_0d<Coeffs>(&self, coeffs: &mut Coeffs) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
    {
        self.from_coeff(coeffs.next().unwrap())
    }

    #[inline]
    fn eval_1d<Coeffs, Vars>(&self, coeffs: &mut Coeffs, degree: usize, vars: &Vars) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Vars: Sequence<Var> + ?Sized,
    {
        let mut acc = self.init_acc();
        let var = vars.get(0).unwrap();
        for coeff in coeffs.take(degree + 1) {
            self.update_acc_coeff(&mut acc, coeff, var);
        }
        acc
    }

    #[inline]
    fn eval_2d<Coeffs, Vars>(&self, coeffs: &mut Coeffs, degree: usize, vars: &Vars) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Vars: Sequence<Var> + ?Sized,
    {
        let mut acc = self.init_acc();
        let var = vars.get(1).unwrap();
        for p in 0..=degree {
            let inner = self.eval_1d(coeffs, p, vars);
            self.update_acc_inner(&mut acc, inner, var);
        }
        acc
    }

    #[inline]
    fn eval_3d<Coeffs, Vars>(&self, coeffs: &mut Coeffs, degree: usize, vars: &Vars) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Vars: Sequence<Var> + ?Sized,
    {
        let mut acc = self.init_acc();
        let var = vars.get(2).unwrap();
        for p in 0..=degree {
            let inner = self.eval_2d(coeffs, p, vars);
            self.update_acc_inner(&mut acc, inner, var);
        }
        acc
    }

    #[inline]
    fn eval_nd<Coeffs, Vars>(
        &self,
        coeffs: &mut Coeffs,
        degree: usize,
        vars: &Vars,
        nvars: usize,
    ) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Vars: Sequence<Var> + ?Sized,
    {
        if nvars == 3 {
            self.eval_3d(coeffs, degree, vars)
        } else {
            let mut acc = self.init_acc();
            let var = vars.get(nvars - 1).unwrap();
            for p in 0..=degree {
                let inner = self.eval_nd(coeffs, p, vars, nvars - 1);
                self.update_acc_inner(&mut acc, inner, var);
            }
            acc
        }
    }
}

struct DefaultEvalCoeffsIter;

impl<Var, Coeff> EvalCoeffsIter<Var, Coeff, Var> for DefaultEvalCoeffsIter
where
    Var: Zero + ops::AddAssign + ops::AddAssign<Coeff>,
    for<'v> Var: ops::MulAssign<&'v Var>,
{
    #[inline]
    fn from_coeff(&self, coeff: Coeff) -> Var {
        let mut acc = Var::zero();
        acc += coeff;
        acc
    }
    #[inline]
    fn init_acc(&self) -> Var {
        Var::zero()
    }
    #[inline]
    fn update_acc_coeff(&self, acc: &mut Var, coeff: Coeff, var: &Var) {
        *acc *= var;
        *acc += coeff;
    }
    #[inline]
    fn update_acc_inner(&self, acc: &mut Var, inner: Var, var: &Var) {
        *acc *= var;
        *acc += inner;
    }
}

/// The partial derivative of a polynomial.
pub trait PartialDeriv<Coeff>: Sized {
    type CoeffsIter: Iterator<Item = Coeff>;

    /// Returns the partial derivative of a polynomial to the given variable as [`PolyIter`].
    ///
    /// If the index of the variable is out of range, returns `None`.
    fn partial_deriv_iter(self, ivar: usize) -> Option<PolyIter<Coeff, Self::CoeffsIter>>;

    /// Returns the partial derivative of a polynomial to the given variable as [`PolySequence`].
    ///
    /// If the index of the variable is out of range, returns `None`.
    ///
    /// # Examples
    ///
    /// The derivative of polynomial `2 - x + x^2` (coefficients: `[1, -1, 2]`) is
    /// `-1 + 2 * x` (coefficients: `[2, -1]`):
    ///
    /// ```
    /// use nutils_poly::{PartialDeriv as _, PolySequence};
    /// assert_eq!(
    ///     PolySequence::new([1, -1, 2], 2, 1).unwrap().partial_deriv(0),
    ///     Some(PolySequence::new(vec![2, -1], 1, 1).unwrap()));
    /// ```
    #[inline]
    fn partial_deriv<Coeffs>(self, ivar: usize) -> Option<PolySequence<Coeff, Coeffs>>
    where
        Coeffs: Sequence<Coeff> + iter::FromIterator<Coeff>,
    {
        self.partial_deriv_iter(ivar).map(|pditer| pditer.collect())
    }
}

impl<'this, Coeff, Coeffs, OutCoeff> PartialDeriv<OutCoeff> for &'this PolySequence<Coeff, Coeffs>
where
    Coeffs: Sequence<Coeff>,
    OutCoeff: FromPrimitive,
    for<'a> OutCoeff: ops::Mul<&'a Coeff, Output = OutCoeff>,
{
    type CoeffsIter =
        <PolySequenceIter<'this, Coeff, Coeffs> as PartialDeriv<OutCoeff>>::CoeffsIter;

    #[inline]
    fn partial_deriv_iter(self, ivar: usize) -> Option<PolyIter<OutCoeff, Self::CoeffsIter>> {
        self.iter().partial_deriv_iter(ivar)
    }
}

impl<'this, Coeff, Coeffs, OutCoeff> PartialDeriv<OutCoeff> for &'this PolySlice<Coeff, Coeffs>
where
    Coeffs: AsRef<[Coeff]>,
    OutCoeff: FromPrimitive,
    for<'a> OutCoeff: ops::Mul<&'a Coeff, Output = OutCoeff>,
{
    type CoeffsIter = <PolySliceIter<'this, Coeff> as PartialDeriv<OutCoeff>>::CoeffsIter;

    #[inline]
    fn partial_deriv_iter(self, ivar: usize) -> Option<PolyIter<OutCoeff, Self::CoeffsIter>> {
        self.iter().partial_deriv_iter(ivar)
    }
}

#[derive(Debug, Clone)]
pub struct PartialDerivCoeffsIter<Coeff, CoeffsIter, OutCoeff>
where
    CoeffsIter: Iterator<Item = Coeff>,
    OutCoeff: ops::Mul<Coeff, Output = OutCoeff> + FromPrimitive,
{
    coeffs_iter: CoeffsIter,
    powers_iter: NthPowerIter,
    is_constant: bool,
    phantom: PhantomData<OutCoeff>,
}

impl<Coeff, CoeffsIter, OutCoeff> Iterator for PartialDerivCoeffsIter<Coeff, CoeffsIter, OutCoeff>
where
    CoeffsIter: Iterator<Item = Coeff>,
    OutCoeff: ops::Mul<Coeff, Output = OutCoeff> + FromPrimitive,
{
    type Item = OutCoeff;

    #[inline]
    fn next(&mut self) -> Option<OutCoeff> {
        while let Some(coeff) = self.coeffs_iter.next() {
            let power = self.powers_iter.next().unwrap();
            if power > 0 || self.is_constant {
                return Some(OutCoeff::from_usize(power).unwrap() * coeff);
            }
        }
        None
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.coeffs_iter.size_hint()
    }
}

impl<Coeff, CoeffsIter, OutCoeff> ExactSizeIterator
    for PartialDerivCoeffsIter<Coeff, CoeffsIter, OutCoeff>
where
    CoeffsIter: Iterator<Item = Coeff> + ExactSizeIterator,
    OutCoeff: ops::Mul<Coeff, Output = OutCoeff> + FromPrimitive,
{
}

impl<Coeff, CoeffsIter, OutCoeff> PartialDeriv<OutCoeff> for PolyIter<Coeff, CoeffsIter>
where
    CoeffsIter: Iterator<Item = Coeff>,
    OutCoeff: ops::Mul<Coeff, Output = OutCoeff> + FromPrimitive,
{
    type CoeffsIter = PartialDerivCoeffsIter<Coeff, CoeffsIter, OutCoeff>;

    #[inline]
    fn partial_deriv_iter(self, ivar: usize) -> Option<PolyIter<OutCoeff, Self::CoeffsIter>> {
        Some(PolyIter {
            coeffs_iter: PartialDerivCoeffsIter {
                coeffs_iter: self.coeffs_iter,
                powers_iter: powers_iter(self.degree, self.nvars).get(ivar)?,
                is_constant: self.degree == 0 || self.nvars == 0,
                phantom: PhantomData,
            },
            degree: self.degree.checked_sub(1).unwrap_or(0),
            nvars: self.nvars,
        })
    }
}

/// Multiply two polynomials.
pub trait Mul<LCoeff, RCoeff, RCoeffsIter, OCoeff>
where
    Self: Sized + Dimensions,
    for<'a> &'a LCoeff: ops::Mul<RCoeff, Output = OCoeff>,
    RCoeff: Clone,
    RCoeffsIter: Iterator<Item = RCoeff> + Clone,
    OCoeff: Zero + ops::AddAssign,
{
    #[must_use]
    fn mul_same_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>;

    #[inline]
    fn mul_same_vars<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
    ) -> Result<PolySequence<OCoeff, OCoeffs>, Error>
    where
        OCoeffs: Sequence<OCoeff> + MutSequence<OCoeff> + iter::FromIterator<OCoeff>,
    {
        let odegree = self.degree() + rpoly.degree();
        let mut opoly = PolySequence::new(
            iter::repeat_with(|| OCoeff::zero())
                .take(ncoeffs(odegree, self.nvars()))
                .collect(),
            odegree,
            self.nvars(),
        )
        .unwrap();
        self.mul_same_vars_into(rpoly, &mut opoly)?;
        Ok(opoly)
    }

    #[must_use]
    fn mul_different_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>;

    #[inline]
    fn mul_different_vars<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
    ) -> Result<PolySequence<OCoeff, OCoeffs>, Error>
    where
        OCoeffs: Sequence<OCoeff> + MutSequence<OCoeff> + iter::FromIterator<OCoeff>,
    {
        let odegree = self.degree() + rpoly.degree();
        let onvars = self.nvars() + rpoly.nvars();
        let mut opoly = PolySequence::new(
            iter::repeat_with(|| OCoeff::zero())
                .take(ncoeffs(odegree, onvars))
                .collect(),
            odegree,
            onvars,
        )
        .unwrap();
        self.mul_different_vars_into(rpoly, &mut opoly)?;
        Ok(opoly)
    }
}

impl<LCoeff, LCoeffsIter, RCoeff, RCoeffsIter, OCoeff> Mul<LCoeff, RCoeff, RCoeffsIter, OCoeff>
    for PolyIter<LCoeff, LCoeffsIter>
where
    for<'a> &'a LCoeff: ops::Mul<RCoeff, Output = OCoeff>,
    LCoeffsIter: Iterator<Item = LCoeff>,
    RCoeff: Clone,
    RCoeffsIter: Iterator<Item = RCoeff> + Clone,
    OCoeff: Zero + ops::AddAssign,
{
    fn mul_same_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>,
    {
        let onvars = opoly.nvars();
        if onvars != self.nvars() || onvars != rpoly.nvars() {
            return Err(Error::AssignDifferentNumberOfVariables);
        }
        let odegree = opoly.degree();
        if odegree < self.degree() + rpoly.degree() {
            return Err(Error::AssignLowerDegree);
        }
        for ocoeff in opoly.coeffs.iter_mut() {
            *ocoeff = OCoeff::zero();
        }
        let mut lpowers = powers_iter(self.degree(), onvars);
        let mut rpowers = powers_iter(rpoly.degree(), onvars);
        let mut lcoeffs = self.coeffs_iter;
        while let (Some(lc), Some(lp)) = (lcoeffs.next(), lpowers.next()) {
            //if lc.is_zero() {
            //    continue;
            //}
            let mut rcoeffs = rpoly.clone().coeffs_iter;
            rpowers.reset(rpoly.degree());
            while let (Some(rc), Some(rp)) = (rcoeffs.next(), rpowers.next()) {
                //if rc.is_zero() {
                //    continue;
                //}
                let op = iter::zip(lp.iter().rev(), rp.iter().rev()).map(|(lj, rj)| lj + rj);
                let oi = powers_rev_iter_to_index(op, odegree, onvars).unwrap();
                *opoly.coeffs.get_mut(oi).unwrap() += &lc * rc;
            }
        }
        Ok(())
    }

    fn mul_different_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>,
    {
        let onvars = opoly.nvars();
        if onvars != self.nvars() + rpoly.nvars() {
            return Err(Error::AssignDifferentNumberOfVariables);
        }
        let odegree = opoly.degree();
        if odegree < self.degree() + rpoly.degree() {
            return Err(Error::AssignLowerDegree);
        }
        for ocoeff in opoly.coeffs.iter_mut() {
            *ocoeff = OCoeff::zero();
        }
        let mut lpowers = powers_iter(self.degree(), self.nvars());
        let mut rpowers = powers_iter(rpoly.degree(), rpoly.nvars());
        let mut lcoeffs = self.coeffs_iter;
        while let (Some(lc), Some(lp)) = (lcoeffs.next(), lpowers.next()) {
            //if lc.is_zero() {
            //    continue;
            //}
            let mut rcoeffs = rpoly.clone().coeffs_iter;
            rpowers.reset(rpoly.degree());
            while let (Some(rc), Some(rp)) = (rcoeffs.next(), rpowers.next()) {
                //if rc.is_zero() {
                //    continue;
                //}
                let op = rp.iter().rev().chain(lp.iter().rev()).copied();
                let oi = powers_rev_iter_to_index(op, odegree, onvars).unwrap();
                *opoly.coeffs.get_mut(oi).unwrap() += &lc * rc;
            }
        }
        Ok(())
    }
}

impl<'this, LCoeff, LCoeffsIter, RCoeff, RCoeffsIter, OCoeff>
    Mul<LCoeff, RCoeff, RCoeffsIter, OCoeff> for PolyIter<&'this LCoeff, LCoeffsIter>
where
    for<'a> &'a LCoeff: ops::Mul<RCoeff, Output = OCoeff>,
    LCoeffsIter: Iterator<Item = &'this LCoeff>,
    RCoeff: Clone,
    RCoeffsIter: Iterator<Item = RCoeff> + Clone,
    OCoeff: Zero + ops::AddAssign,
{
    fn mul_same_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>,
    {
        let onvars = opoly.nvars();
        if onvars != self.nvars() || onvars != rpoly.nvars() {
            return Err(Error::AssignDifferentNumberOfVariables);
        }
        let odegree = opoly.degree();
        if odegree < self.degree() + rpoly.degree() {
            return Err(Error::AssignLowerDegree);
        }
        for ocoeff in opoly.coeffs.iter_mut() {
            *ocoeff = OCoeff::zero();
        }
        let mut lpowers = powers_iter(self.degree(), onvars);
        let mut rpowers = powers_iter(rpoly.degree(), onvars);
        let mut lcoeffs = self.coeffs_iter;
        while let (Some(lc), Some(lp)) = (lcoeffs.next(), lpowers.next()) {
            //if lc.is_zero() {
            //    continue;
            //}
            let mut rcoeffs = rpoly.clone().coeffs_iter;
            rpowers.reset(rpoly.degree());
            while let (Some(rc), Some(rp)) = (rcoeffs.next(), rpowers.next()) {
                //if rc.is_zero() {
                //    continue;
                //}
                let op = iter::zip(lp.iter().rev(), rp.iter().rev()).map(|(lj, rj)| lj + rj);
                let oi = powers_rev_iter_to_index(op, odegree, onvars).unwrap();
                *opoly.coeffs.get_mut(oi).unwrap() += lc * rc;
            }
        }
        Ok(())
    }

    fn mul_different_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>,
    {
        let onvars = opoly.nvars();
        if onvars != self.nvars() + rpoly.nvars() {
            return Err(Error::AssignDifferentNumberOfVariables);
        }
        let odegree = opoly.degree();
        if odegree < self.degree() + rpoly.degree() {
            return Err(Error::AssignLowerDegree);
        }
        for ocoeff in opoly.coeffs.iter_mut() {
            *ocoeff = OCoeff::zero();
        }
        let mut lpowers = powers_iter(self.degree(), self.nvars());
        let mut rpowers = powers_iter(rpoly.degree(), rpoly.nvars());
        let mut lcoeffs = self.coeffs_iter;
        while let (Some(lc), Some(lp)) = (lcoeffs.next(), lpowers.next()) {
            //if lc.is_zero() {
            //    continue;
            //}
            let mut rcoeffs = rpoly.clone().coeffs_iter;
            rpowers.reset(rpoly.degree());
            while let (Some(rc), Some(rp)) = (rcoeffs.next(), rpowers.next()) {
                //if rc.is_zero() {
                //    continue;
                //}
                let op = rp.iter().rev().chain(lp.iter().rev()).copied();
                let oi = powers_rev_iter_to_index(op, odegree, onvars).unwrap();
                *opoly.coeffs.get_mut(oi).unwrap() += lc * rc;
            }
        }
        Ok(())
    }
}

impl<'this, LCoeff, LCoeffs, RCoeff, RCoeffsIter, OCoeff> Mul<LCoeff, RCoeff, RCoeffsIter, OCoeff>
    for &'this PolySequence<LCoeff, LCoeffs>
where
    for<'a> &'a LCoeff: ops::Mul<RCoeff, Output = OCoeff>,
    LCoeffs: Sequence<LCoeff>,
    RCoeff: Clone,
    RCoeffsIter: Iterator<Item = RCoeff> + Clone,
    OCoeff: Zero + ops::AddAssign,
{
    #[inline]
    fn mul_same_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>,
    {
        self.iter().mul_same_vars_into(rpoly, opoly)
    }

    #[inline]
    fn mul_different_vars_into<OCoeffs>(
        self,
        rpoly: PolyIter<RCoeff, RCoeffsIter>,
        opoly: &mut PolySequence<OCoeff, OCoeffs>,
    ) -> Result<(), Error>
    where
        OCoeffs: MutSequence<OCoeff>,
    {
        self.iter().mul_different_vars_into(rpoly, opoly)
    }
}
//
//pub fn composition(outer: PolyIter, inner: &Inner) -> PolySequence
//where
//    Inner: Sequence<PolySequence<InnerCoeff12
//{
//    EvalCompositionCoeffsIter.eval(&mut outer.coeffs_iter, outer.degree, &inner)
//}
//
//struct EvalCompositionCoeffsIter {}
//
//impl EvalCoeffsIter<InnerPoly, OuterCoeff, OutPoly> for EvalCompositionCoeffsIter {
//    fn from_coeff(&self, coeff: OuterCoeff) -> OutPoly {
//        OutPoly::new(iter::once(coeff).collect(), 0, 0)
//    }
//    fn init_acc(&self) -> OutPoly {
//        OutPoly::new(iter::once(OutCoeff::zero()), 0, 0)
//    }
//    fn update_acc_coeff(&self, acc: &mut OutPoly, coeff: OuterCoeff, var: &InnerPoly) {
//        *acc = acc.mul_different_vars(var);
//        acc += coeff;
//    }
//    fn update_acc_inner(&self, acc: &mut OutPoly, inner: OutPoly, var: &InnerPoly) {
//        *acc = acc.mul_different_vars(var);
//        acc += inner;
//    }
//}

///// Returns the power of a polynomial to a non-negative integer.
//pub fn pow(coeffs: &[f64], degree: usize, nvars: usize, exp: usize) -> Vec<f64> {
//    if exp == 0 {
//        vec![1.0]
//    } else if exp == 1 {
//        coeffs.to_vec()
//    } else {
//        let sqr = mul(coeffs, coeffs, degree, degree, nvars);
//        if exp == 2 {
//            sqr
//        } else {
//            let even = pow(&sqr, degree * 2, nvars, exp / 2);
//            if exp % 2 == 0 {
//                even
//            } else {
//                mul(&even, coeffs, degree * 2 * (exp / 2), degree, nvars)
//            }
//        }
//    }
//}

///// Returns a transformation matrix.
/////
///// The matrix is such that the following two expressions are equivalent:
/////
///// ```text
///// eval(coeffs, eval(transform_coeffs, vars, transform_degree, from_nvars), degree, to_nvars)
///// eval(matvec(matrix, coeffs), vars, transform_degree * degree, from_nvars)
///// ```
/////
///// where `matrix` is the result of
/////
///// ```text
///// transform_matrix(transform_coeffs, transform_degree, from_nvars, degree, to_nvars)
///// ```
//pub fn transform_matrix(
//    transform_coeffs: &[f64],
//    transform_degree: usize,
//    from_nvars: usize,
//    degree: usize,
//    to_nvars: usize,
//) -> Vec<f64> {
//    let transform_ncoeffs = ncoeffs(transform_degree, from_nvars);
//    assert_eq!(transform_coeffs.len(), to_nvars * transform_ncoeffs);
//    let row_degree = transform_degree * degree;
//    let nrows = ncoeffs(transform_degree * degree, from_nvars);
//    let ncols = ncoeffs(degree, to_nvars);
//    let mut matrix = uniform_vec(0.0, nrows * ncols);
//    let mut col_iter = (0..).into_iter();
//    let mut col_powers = powers_iter(degree, to_nvars);
//    let mut row_powers = powers_iter(degree, from_nvars);
//    while let (Some(col), Some(col_powers)) = (col_iter.next(), col_powers.next()) {
//        let (col_coeffs, col_degree) = iter::zip(
//            transform_coeffs.chunks_exact(transform_ncoeffs),
//            col_powers.iter().copied(),
//        )
//        .fold(
//            (vec![1.0], 0),
//            |(col_coeffs, col_degree), (t_coeffs, power)| {
//                (
//                    mul(
//                        &col_coeffs,
//                        &pow(t_coeffs, transform_degree, from_nvars, power),
//                        col_degree,
//                        transform_degree * power,
//                        from_nvars,
//                    ),
//                    col_degree + transform_degree * power,
//                )
//            },
//        );
//        let mut col_coeffs = col_coeffs.into_iter();
//        row_powers.reset(col_degree);
//        while let (Some(coeff), Some(powers)) = (col_coeffs.next(), row_powers.next()) {
//            let row =
//                powers_rev_iter_to_index(powers.iter().rev().copied(), row_degree, from_nvars)
//                    .unwrap();
//            matrix[row * ncols + col] = coeff;
//        }
//    }
//    matrix
//}

#[cfg(test)]
mod tests {
    use super::{Eval, Mul, PartialDeriv, PolySequence};

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
        assert_eq!(
            PolySequence::new([1], 0, 0).unwrap().eval(&[] as &[usize]),
            1
        );
    }

    #[test]
    fn eval_1d() {
        assert_eq!(PolySequence::new([1], 0, 1).unwrap().eval(&[5]), 1);
        assert_eq!(PolySequence::new([2, 1], 1, 1).unwrap().eval(&[5]), 11);
        assert_eq!(PolySequence::new([3, 2, 1], 2, 1).unwrap().eval(&[5]), 86);
    }

    #[test]
    fn eval_2d() {
        assert_eq!(PolySequence::new([1], 0, 2).unwrap().eval(&[5, 3]), 1);
        assert_eq!(PolySequence::new([0, 0, 1], 1, 2).unwrap().eval(&[5, 3]), 1);
        assert_eq!(PolySequence::new([0, 1, 0], 1, 2).unwrap().eval(&[5, 3]), 5);
        assert_eq!(PolySequence::new([1, 0, 0], 1, 2).unwrap().eval(&[5, 3]), 3);
        assert_eq!(
            PolySequence::new([3, 2, 1], 1, 2).unwrap().eval(&[5, 3]),
            20
        );
        assert_eq!(
            PolySequence::new([6, 5, 4, 3, 2, 1], 2, 2)
                .unwrap()
                .eval(&[5, 3]),
            227
        );
    }

    #[test]
    fn eval_3d() {
        assert_eq!(PolySequence::new([1], 0, 3).unwrap().eval(&[5, 3, 2]), 1);
        assert_eq!(
            PolySequence::new([0, 0, 0, 1], 1, 3)
                .unwrap()
                .eval(&[5, 3, 2]),
            1
        );
        assert_eq!(
            PolySequence::new([0, 0, 1, 0], 1, 3)
                .unwrap()
                .eval(&[5, 3, 2]),
            5
        );
        assert_eq!(
            PolySequence::new([0, 1, 0, 0], 1, 3)
                .unwrap()
                .eval(&[5, 3, 2]),
            3
        );
        assert_eq!(
            PolySequence::new([1, 0, 0, 0], 1, 3)
                .unwrap()
                .eval(&[5, 3, 2]),
            2
        );
        assert_eq!(
            PolySequence::new([4, 3, 2, 1], 1, 3)
                .unwrap()
                .eval(&[5, 3, 2]),
            28
        );
        assert_eq!(
            PolySequence::new([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 2, 3)
                .unwrap()
                .eval(&[5, 3, 2]),
            415,
        );
    }

    #[test]
    fn partial_deriv() {
        assert_eq!(
            PolySequence::new([1], 0, 1).unwrap().partial_deriv(0),
            Some(PolySequence::new(vec![0], 0, 1).unwrap()),
        );
        assert_eq!(
            PolySequence::new([4, 3, 2, 1], 3, 1)
                .unwrap()
                .partial_deriv(0),
            Some(PolySequence::new(vec![12, 6, 2], 2, 1).unwrap()),
        );
        assert_eq!(
            PolySequence::new([6, 5, 4, 3, 2, 1], 2, 2)
                .unwrap()
                .partial_deriv(0),
            Some(PolySequence::new(vec![5, 6, 2], 1, 2).unwrap()),
        );
        assert_eq!(
            PolySequence::new([6, 5, 4, 3, 2, 1], 2, 2)
                .unwrap()
                .partial_deriv(1),
            Some(PolySequence::new(vec![12, 5, 4], 1, 2).unwrap()),
        );
    }

    #[test]
    fn mul() {
        let l = PolySequence::new([2, 1], 1, 1).unwrap();
        let r = PolySequence::new([4, 3], 1, 1).unwrap();
        assert_eq!(
            l.mul_same_vars(r.iter()),
            Ok(PolySequence::new(vec![8, 10, 3], 2, 1).unwrap()),
        );

        let l = PolySequence::new([3, 2, 1], 1, 2).unwrap();
        let r = PolySequence::new([6, 5, 4], 1, 2).unwrap();
        assert_eq!(
            l.mul_same_vars(r.iter()),
            Ok(PolySequence::new(vec![18, 27, 18, 10, 13, 4], 2, 2).unwrap()),
        );
    }

//    #[test]
//    fn pow() {
//        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 0)[..], [1.]);
//        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 1)[..], [0., 2.]);
//        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 2)[..], [0., 0., 4.]);
//        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 3)[..], [0., 0., 0., 8.]);
//        assert_abs_diff_eq!(super::pow(&[0., 2.], 1, 1, 4)[..], [0., 0., 0., 0., 16.]);
//    }
//
//    #[test]
//    fn transform_matrix_1d() {
//        assert_abs_diff_eq!(
//            super::transform_matrix(&[0.5, 0.0], 1, 1, 2, 1)[..],
//            [
//                0.25, 0.0, 0.0, //
//                0.0, 0.5, 0.0, //
//                0.0, 0.0, 1.0, //
//            ]
//        );
//        assert_abs_diff_eq!(
//            super::transform_matrix(&[0.5, 0.5], 1, 1, 2, 1)[..],
//            [
//                0.25, 0.0, 0.0, //
//                0.5, 0.5, 0.0, //
//                0.25, 0.5, 1.0, //
//            ]
//        );
//    }
//
//    #[test]
//    fn transform_matrix_2d() {
//        assert_abs_diff_eq!(
//            super::transform_matrix(&[0.0, 0.5, 0.0, 0.5, 0.0, 0.0], 1, 2, 2, 2)[..],
//            [
//                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
//                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
//                0.00, 0.00, 0.50, 0.00, 0.00, 0.00, //
//                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
//                0.00, 0.00, 0.00, 0.00, 0.50, 0.00, //
//                0.00, 0.00, 0.00, 0.00, 0.00, 1.00, //
//            ]
//        );
//        assert_abs_diff_eq!(
//            super::transform_matrix(&[0.0, 0.5, 0.5, 0.5, 0.0, 0.0], 1, 2, 2, 2)[..],
//            [
//                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
//                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
//                0.00, 0.25, 0.50, 0.00, 0.00, 0.00, //
//                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
//                0.00, 0.00, 0.00, 0.50, 0.50, 0.00, //
//                0.00, 0.00, 0.00, 0.25, 0.50, 1.00, //
//            ]
//        );
//        assert_abs_diff_eq!(
//            super::transform_matrix(&[0.0, 0.5, 0.0, 0.5, 0.0, 0.5], 1, 2, 2, 2)[..],
//            [
//                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
//                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
//                0.50, 0.00, 0.50, 0.00, 0.00, 0.00, //
//                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
//                0.00, 0.25, 0.00, 0.00, 0.50, 0.00, //
//                0.25, 0.00, 0.50, 0.00, 0.00, 1.00, //
//            ]
//        );
//        assert_abs_diff_eq!(
//            super::transform_matrix(&[0.0, 0.5, 0.5, 0.5, 0.0, 0.5], 1, 2, 2, 2)[..],
//            [
//                0.25, 0.00, 0.00, 0.00, 0.00, 0.00, //
//                0.00, 0.25, 0.00, 0.00, 0.00, 0.00, //
//                0.50, 0.25, 0.50, 0.00, 0.00, 0.00, //
//                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
//                0.00, 0.25, 0.00, 0.50, 0.50, 0.00, //
//                0.25, 0.25, 0.50, 0.25, 0.50, 1.00, //
//            ]
//        );
//    }
}

#[cfg(all(feature = "bench", test))]
mod benches {
    extern crate test;
    use self::test::Bencher;
    use super::{Eval, Mul, PartialDeriv, PolySequence};

    macro_rules! mk_bench_eval {
        ($name:ident, $degree:literal, $nvars:literal) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let coeffs: Vec<_> = (1..=super::ncoeffs($degree, $nvars))
                    .map(|i| i as f64)
                    .collect();
                let coeffs = &coeffs[..];
                let vars: Vec<_> = (1..=$nvars).map(|x| x as f64).collect();
                let poly = PolySequence::new(coeffs, $degree, $nvars).unwrap();
                b.iter(|| test::black_box(&poly).eval(&vars))
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
    fn mul_same_vars_3d_degree4_degree2(b: &mut Bencher) {
        let l = PolySequence::new(
            (0..super::ncoeffs(4, 3))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            4,
            3,
        )
        .unwrap();
        let r = PolySequence::new(
            (0..super::ncoeffs(2, 3))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            2,
            3,
        )
        .unwrap();
        b.iter(|| {
            test::black_box(&l)
                .mul_same_vars::<Vec<_>>(test::black_box(r.iter()))
                .unwrap()
        });
    }

    #[bench]
    fn mul_different_vars_1d_degree4_2d_degree2(b: &mut Bencher) {
        let l = PolySequence::new(
            (0..super::ncoeffs(4, 1))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            4,
            1,
        )
        .unwrap();
        let r = PolySequence::new(
            (0..super::ncoeffs(2, 2))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            2,
            2,
        )
        .unwrap();
        b.iter(|| {
            test::black_box(&l)
                .mul_different_vars::<Vec<_>>(test::black_box(r.iter()))
                .unwrap()
        });
    }

    //#[bench]
    //fn pow_3d_degree4_exp3(b: &mut Bencher) {
    //    b.iter(|| {
    //        super::pow(
    //            test::black_box(
    //                &(0..super::ncoeffs(4, 3))
    //                    .into_iter()
    //                    .map(|i| i as f64)
    //                    .collect::<Vec<_>>()[..],
    //            ),
    //            test::black_box(4),
    //            test::black_box(3),
    //            test::black_box(3),
    //        )
    //    });
    //}

    //#[bench]
    //fn transform_matrix_2d_degree2(b: &mut Bencher) {
    //    b.iter(|| {
    //        super::transform_matrix(
    //            test::black_box(&[0.5, 0.5, 0.0, 0.5, 0.0, 0.5]),
    //            test::black_box(1),
    //            test::black_box(2),
    //            test::black_box(2),
    //            test::black_box(2),
    //        )
    //    });
    //}

    //#[bench]
    //fn change_degree_2d_degree3_to_5(b: &mut Bencher) {
    //    b.iter(|| {
    //        super::change_degree(
    //            test::black_box(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    //            test::black_box(3),
    //            test::black_box(2),
    //            test::black_box(5),
    //        )
    //    });
    //}
}
