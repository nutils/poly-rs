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
//! `[1, -1, 2]`.
//!
//! ```
//! use nutils_poly::{Poly, PolySequence, Variable};
//!
//! let p = PolySequence::new([1, -1, 2], 0..1, 2).unwrap();
//! ```
//!
//! You can evaluate this polynomial for some `x` using [`PolySequence::eval()`]:
//!
//! ```
//! # use nutils_poly::{Poly, PolySequence, Variable};
//! #
//! # let p = PolySequence::new([1, -1, 2], 0..1, 2).unwrap();
//! assert_eq!(p.by_ref().eval(&[0]), 2); // x = 0
//! assert_eq!(p.by_ref().eval(&[1]), 2); // x = 1
//! assert_eq!(p.by_ref().eval(&[2]), 4); // x = 2
//! ```
//!
//! Or compute the partial derivative `∂p/∂x` using [`PartialDeriv::partial_deriv()`]:
//!
//! ```
//! # use nutils_poly::{Poly, PolySequence, Variable};
//! #
//! # let p = PolySequence::new([1, -1, 2], 0..1, 2).unwrap();
//! assert_eq!(
//!     p.by_ref().partial_deriv(Variable::try_from(0).unwrap()).collect(),
//!     PolySequence::new(vec![2, -1], 0..1, 1).unwrap(),
//! );
//! ```
//!
//! [lexicographic order]: https://en.wikipedia.org/wiki/Lexicographic_order

#![cfg_attr(feature = "bench", feature(test))]

use num_traits::Zero;
use std::iter;
use std::ops;

// Workaround for associated type `Iter<'a>: Iterator<Item = &'a T>` of
// `Sequence` from
// https://web.archive.org/web/20220530082425/https://sabrinajewson.org/blog/the-better-alternative-to-lifetime-gats#the-better-gats

/// An interface for finite sequences.
pub trait Sequence
where
    Self: for<'me> SequenceIterType<'me, &'me Self::Item>,
{
    type Item;

    /// Returns the number of elements in the sequence.
    fn len(&self) -> usize;

    /// Returns `true` if the slice is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to an element or `None` if the index is out of bounds.
    fn get(&self, index: usize) -> Option<&Self::Item>;

    /// Returns a reference to the first element or `None`.
    #[inline]
    fn first(&self) -> Option<&Self::Item> {
        self.get(0)
    }

    /// Returns a reference to the last element or `None`.
    #[inline]
    fn last(&self) -> Option<&Self::Item> {
        self.len().checked_sub(1).and_then(|index| self.get(index))
    }

    /// Returns an iterator over the sequence.
    fn iter(&self) -> <Self as SequenceIterType<'_, &Self::Item>>::Iter;
}

/// An interface for finite sequences with mutable elements.
pub trait SequenceMut: Sequence
where
    Self: for<'me> SequenceIterMutType<'me, &'me mut Self::Item>,
{
    /// Returns a mutable reference to an element or `None` if the index is out of bounds.
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item>;

    /// Returns a reference to the first element or `None`.
    #[inline]
    fn first_mut(&mut self) -> Option<&mut Self::Item> {
        self.get_mut(0)
    }

    /// Returns a reference to the last element or `None`.
    #[inline]
    fn last_mut(&mut self) -> Option<&mut Self::Item> {
        self.len()
            .checked_sub(1)
            .and_then(|index| self.get_mut(index))
    }

    #[inline]
    fn fill(&mut self, value: Self::Item)
    where
        Self::Item: Clone,
    {
        self.fill_with(|| value.clone());
    }

    #[inline]
    fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> Self::Item,
    {
        for v in self.iter_mut() {
            *v = f();
        }
    }

    /// Returns an iterator over the sequence that allows modifying each element.
    fn iter_mut(&mut self) -> <Self as SequenceIterMutType<'_, &mut Self::Item>>::IterMut;
}

/// Return type of [`Sequence::iter()`].
pub trait SequenceIterType<'me, Item> {
    /// Return type of [`Sequence::iter()`].
    type Iter: Iterator<Item = Item>;
}

/// Return type of [`SequenceMut::iter_mut()`].
pub trait SequenceIterMutType<'me, Item> {
    /// Return type of [`SequenceMut::iter_mut()`].
    type IterMut: Iterator<Item = Item>;
}

macro_rules! impl_sequence_for_as_ref_slice {
    ($T:ident, $ty:ty, <$($params:tt)*) => {
        impl<'me, $($params)* SequenceIterType<'me, &'me $T> for $ty {
            type Iter = std::slice::Iter<'me, $T>;
        }

        impl<'me, $($params)* SequenceIterMutType<'me, &'me mut $T> for $ty {
            type IterMut = std::slice::IterMut<'me, $T>;
        }

        impl<$($params)* Sequence for $ty {
            type Item = $T;

            #[inline]
            fn len(&self) -> usize {
                <Self as AsRef::<[$T]>>::as_ref(self).len()
            }
            #[inline]
            fn get(&self, index: usize) -> Option<&$T> {
                <Self as AsRef::<[$T]>>::as_ref(self).get(index)
            }
            #[inline]
            fn iter(&self) -> <Self as SequenceIterType<'_, &$T>>::Iter {
                <Self as AsRef::<[$T]>>::as_ref(self).iter()
            }
        }

        impl<$($params)* SequenceMut for $ty {
            #[inline]
            fn get_mut(&mut self, index: usize) -> Option<&mut $T> {
                <Self as AsMut::<[$T]>>::as_mut(self).get_mut(index)
            }
            #[inline]
            fn fill(&mut self, value: Self::Item)
            where
                Self::Item: Clone,
            {
                <Self as AsMut::<[$T]>>::as_mut(self).fill(value);
            }
            #[inline]
            fn fill_with<F>(&mut self, f: F)
            where
                F: FnMut() -> Self::Item,
            {
                <Self as AsMut::<[$T]>>::as_mut(self).fill_with(f);
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

impl<'me, T, S: SequenceIterType<'me, &'me T> + ?Sized> SequenceIterType<'me, &'me T> for &S {
    type Iter = <S as SequenceIterType<'me, &'me T>>::Iter;
}

impl<'me, T, S: SequenceIterType<'me, &'me T> + ?Sized> SequenceIterType<'me, &'me T> for &mut S {
    type Iter = <S as SequenceIterType<'me, &'me T>>::Iter;
}

impl<'me, T, S: SequenceIterMutType<'me, &'me mut T> + ?Sized> SequenceIterMutType<'me, &'me mut T>
    for &mut S
{
    type IterMut = <S as SequenceIterMutType<'me, &'me mut T>>::IterMut;
}

impl<S: Sequence + ?Sized> Sequence for &S {
    type Item = S::Item;

    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Item> {
        (**self).get(index)
    }
    #[inline]
    fn iter(&self) -> <Self as SequenceIterType<'_, &Self::Item>>::Iter {
        (**self).iter()
    }
}

impl<S: Sequence + ?Sized> Sequence for &mut S {
    type Item = S::Item;

    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Item> {
        (**self).get(index)
    }
    #[inline]
    fn iter(&self) -> <Self as SequenceIterType<'_, &Self::Item>>::Iter {
        (**self).iter()
    }
}

impl<S: SequenceMut + ?Sized> SequenceMut for &mut S {
    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item> {
        (**self).get_mut(index)
    }
    #[inline]
    fn fill(&mut self, value: Self::Item)
    where
        Self::Item: Clone,
    {
        (**self).fill(value);
    }
    #[inline]
    fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> Self::Item,
    {
        (**self).fill_with(f);
    }
    #[inline]
    fn iter_mut(&mut self) -> <Self as SequenceIterMutType<'_, &mut Self::Item>>::IterMut {
        (**self).iter_mut()
    }
}

#[cfg(feature = "ndarray")]
mod impl_ndarray {
    use super::{Sequence, SequenceIterMutType, SequenceIterType, SequenceMut};
    use ndarray::iter::{Iter, IterMut};
    use ndarray::{ArrayBase, Data, DataMut, Ix1};

    impl<'me, S: Data> SequenceIterType<'me, &'me S::Elem> for ArrayBase<S, Ix1> {
        type Iter = Iter<'me, S::Elem, Ix1>;
    }

    impl<'me, S: Data> SequenceIterMutType<'me, &'me mut S::Elem> for ArrayBase<S, Ix1> {
        type IterMut = IterMut<'me, S::Elem, Ix1>;
    }

    impl<S: Data> Sequence for ArrayBase<S, Ix1> {
        type Item = S::Elem;

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }
        #[inline]
        fn get(&self, index: usize) -> Option<&Self::Item> {
            self.get(index)
        }
        #[inline]
        fn iter(&self) -> <Self as SequenceIterType<'_, &Self::Item>>::Iter {
            self.iter()
        }
    }

    impl<S: Data + DataMut> SequenceMut for ArrayBase<S, Ix1> {
        #[inline]
        fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item> {
            self.get_mut(index)
        }
        #[inline]
        fn fill(&mut self, value: Self::Item)
        where
            Self::Item: Clone,
        {
            self.fill(value);
        }
        #[inline]
        fn iter_mut(&mut self) -> <Self as SequenceIterMutType<'_, &mut Self::Item>>::IterMut {
            self.iter_mut()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::{Sequence, SequenceMut};
        use ndarray::array;

        #[test]
        fn len() {
            assert_eq!(Sequence::len(&array![0, 1, 2]), 3);
        }

        #[test]
        fn get() {
            assert_eq!(Sequence::get(&array![0, 1, 2], 1), Some(&1));
            assert_eq!(Sequence::get(&array![0, 1, 2], 3), None);
        }

        #[test]
        fn iter() {
            assert_eq!(
                Sequence::iter(&array![0, 1, 2])
                    .copied()
                    .collect::<Vec<_>>(),
                vec![0, 1, 2]
            );
        }

        #[test]
        fn get_mut() {
            let mut a = array![1, 3, 5];
            *SequenceMut::get_mut(&mut a, 1).unwrap() = 7;
            assert_eq!(Sequence::get(&a, 1), Some(&7));
        }

        #[test]
        fn fill() {
            let mut a = array![1, 3, 5];
            SequenceMut::fill(&mut a, 7);
            assert_eq!(
                Sequence::iter(&a).copied().collect::<Vec<_>>(),
                vec![7, 7, 7]
            );
        }

        #[test]
        fn iter_mut() {
            let mut a = array![1, 3, 5];
            let mut iter = SequenceMut::iter_mut(&mut a);
            *iter.next().unwrap() = 7;
            *iter.next().unwrap() = 9;
            assert_eq!(
                Sequence::iter(&a).copied().collect::<Vec<_>>(),
                vec![7, 9, 5]
            );
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    AssignMissingVariables,
    AssignLowerDegree,
    NCoeffsNVarsDegreeMismatch,
    VariableOutOfRange,
    DuplicateVariable,
    VariablesNotSorted,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssignMissingVariables => write!(
                f,
                "Some or all variables of the source are missing in the destination."
            ),
            Self::AssignLowerDegree => write!(
                f,
                "The degree of the destination polynomial is lower than the degree of the source."
            ),
            Self::NCoeffsNVarsDegreeMismatch => write!(
                f,
                "The number of coefficients is not compatible with the number of variables and degree."
            ),
            Self::VariableOutOfRange => write!(
                f,
                "The variable is out of range."
            ),
            Self::DuplicateVariable => write!(
                f,
                "Duplicate variable.",
            ),
            Self::VariablesNotSorted => write!(
                f,
                "The variables are not sorted.",
            ),
        }
    }
}

impl From<std::convert::Infallible> for Error {
    fn from(_infallible: std::convert::Infallible) -> Self {
        unreachable! {}
    }
}

/// Returns the number of coefficients for a polynomial of given degree and number of variables.
#[inline]
pub const fn ncoeffs(nvars: usize, degree: Power) -> usize {
    // To improve the performance the implementation is specialized for
    // polynomials in zero to three variables.
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
    let mut i = 1;
    while i <= nvars {
        n = n * (degree as usize + i) / i;
        i += 1;
    }
    n
}

/// Returns the sum of the number of coefficients up to (excluding) the given degree.
#[inline]
const fn ncoeffs_sum(nvars: usize, degree: Power) -> usize {
    // To improve the performance the implementation is specialized for
    // polynomials in zero to three variables.
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

type VariablesBits = u8;

type VarData = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variable(VarData);

impl TryFrom<usize> for Variable {
    type Error = Error;

    #[inline]
    fn try_from(var: usize) -> Result<Self, Error> {
        if var >= VariablesBits::BITS as usize {
            Err(Error::VariableOutOfRange)
        } else {
            Ok(Self(var as VarData))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variables(VariablesBits);

impl Variables {
    pub fn iter_all() -> impl Iterator<Item = Variable> {
        (0..NVARIABLES as VarData).into_iter().map(|v| Variable(v))
    }
    /// Returns the number of variables in the set.
    #[inline]
    pub const fn len(self) -> usize {
        let mut rem = self.0;
        let mut len = 0;
        while rem != 0 {
            len += (rem & 1) as usize;
            rem >>= 1;
        }
        len
    }

    /// Returns `true` if the set of variables is empty.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Creates an empty set of variables.
    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Returns `true` if the variable is in the set.
    #[inline]
    pub const fn get(self, var: Variable) -> bool {
        (self.0 >> var.0) & 1 == 1
    }

    /// Returns the index of the variable in the set or `None` if the variable is not in the set.
    #[inline]
    pub const fn index(self, var: Variable) -> Option<usize> {
        if self.get(var) {
            Some(Variables(self.0 & !(VariablesBits::MAX << var.0)).len())
        } else {
            None
        }
    }

    pub const fn first(self) -> Option<Variable> {
        if self.0 == 0 {
            None
        } else {
            let mut val = self.0;
            let mut var = 0;
            while (val & 1) == 0 {
                val >>= 1;
                var += 1;
            }
            Some(Variable(var))
        }
    }

    pub const fn last(self) -> Option<Variable> {
        if self.0 == 0 {
            None
        } else {
            let mut val = self.0 >> 1;
            let mut var = 0;
            while val != 0 {
                val >>= 1;
                var += 1;
            }
            Some(Variable(var))
        }
    }

    pub fn iter(self) -> impl Iterator<Item = Variable> + DoubleEndedIterator {
        (0..NVARIABLES as VarData)
            .map(|i| Variable(i))
            .filter(move |v| self.get(*v))
    }

    /// Returns `true` if all variables in this set are sorted before those in the other set.
    #[inline]
    pub const fn all_less_than(self, other: Variables) -> bool {
        let first = if let Some(first) = other.first() {
            first.0
        } else {
            0
        };
        self.0 >> first == 0
    }

    /// Returns `true` if all variables in this set are contained in the other set.
    #[inline]
    pub const fn is_contained_in(self, other: Variables) -> bool {
        self.0 & !other.0 == 0
    }
}

impl std::fmt::Display for Variables {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Variables[")?;
        let mut iter = self.iter();
        if let Some(var) = iter.next() {
            write!(f, "{}", var.0)?;
            for var in iter {
                write!(f, ", {}", var.0)?;
            }
        }
        write!(f, "]")
    }
}

impl ops::BitAnd for Variables {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl ops::BitOr for Variables {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl ops::BitXor for Variables {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl ops::Sub for Variables {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 & !rhs.0)
    }
}

impl TryFrom<ops::Range<usize>> for Variables {
    type Error = Error;

    #[inline]
    fn try_from(value: ops::Range<usize>) -> Result<Self, Error> {
        if value.end > VariablesBits::BITS as usize {
            Err(Error::VariableOutOfRange)
        } else {
            let mut vars = 0;
            for i in value {
                vars |= 1 << i;
            }
            Ok(Variables(vars))
        }
    }
}

impl TryFrom<&[usize]> for Variables {
    type Error = Error;

    #[inline]
    fn try_from(value: &[usize]) -> Result<Self, Error> {
        let mut vars = 0;
        for i in value {
            let i = *i;
            if i >= VariablesBits::BITS as usize {
                return Err(Error::VariableOutOfRange);
            } else if (vars >> i) & 1 == 1 {
                return Err(Error::DuplicateVariable);
            } else if (vars >> i) != 0 {
                return Err(Error::VariablesNotSorted);
            } else {
                vars |= 1 << i;
            }
        }
        Ok(Variables(vars))
    }
}

impl<const N: usize> TryFrom<[usize; N]> for Variables {
    type Error = Error;

    #[inline]
    fn try_from(value: [usize; N]) -> Result<Self, Error> {
        let mut vars = 0;
        for i in value {
            if i >= VariablesBits::BITS as usize {
                return Err(Error::VariableOutOfRange);
            } else if (vars >> i) & 1 == 1 {
                return Err(Error::DuplicateVariable);
            } else if (vars >> i) != 0 {
                return Err(Error::VariablesNotSorted);
            } else {
                vars |= 1 << i;
            }
        }
        Ok(Variables(vars))
    }
}

pub struct VariablesIter(VariablesBits, VarData);

impl Iterator for VariablesIter {
    type Item = Variable;

    #[inline]
    fn next(&mut self) -> Option<Variable> {
        if self.0 == 0 {
            None
        } else {
            while (self.0 & 1) == 0 {
                self.1 += 1;
                self.0 >>= 1;
            }
            let v = Variable(self.1);
            self.1 += 1;
            self.0 >>= 1;
            Some(v)
        }
    }
}

pub type Power = u8;

const NVARIABLES: usize = VariablesBits::BITS as usize;

/// Powers of variables.
///
/// The sum of the powers fits in a [`Power`].
#[derive(Clone, Copy)]
pub union Powers {
    vec: [Power; NVARIABLES],
    int: u64,
}

impl Powers {
    #[inline]
    pub fn iter_all(vars: Variables, degree: Power) -> PowersIter {
        let mut powers = Powers::zeros();
        if let Some(var) = vars.last() {
            powers[var] = degree;
        }
        PowersIter {
            vars,
            rem: 1,
            len: ncoeffs(vars.len(), degree),
            powers,
        }
    }
    #[inline]
    pub fn zeros() -> Self {
        Self { int: 0 }
    }
    pub fn from_index(mut index: usize, vars: Variables, mut degree: Power) -> Option<Self> {
        let mut result = Self::zeros();
        let mut nvars = vars.len();
        'outer: for var in vars.iter().rev() {
            nvars -= 1;
            if nvars == 0 {
                if let Ok(index) = Power::try_from(index) {
                    if let Some(power) = degree.checked_sub(index as Power) {
                        result[var] += power;
                        return Some(result);
                    }
                }
                return None;
            } else {
                for i in 0..=degree {
                    let n = ncoeffs(nvars, i);
                    if index < n {
                        result[var] += degree - i;
                        #[allow(clippy::mut_range_bound)]
                        {
                            degree = i;
                        }
                        continue 'outer;
                    }
                    index -= n;
                }
                return None;
            }
        }
        None
    }
    pub fn to_index(self, vars: Variables, mut degree: Power) -> Option<usize> {
        let mut index = 0;
        let mut nvars = vars.len();
        for (i, power) in self.iter().enumerate().rev() {
            if vars.get(Variable(i as VarData)) {
                nvars -= 1;
                degree = degree.checked_sub(power)?;
                index += ncoeffs_sum(nvars, degree);
            } else if power > 0 {
                return None;
            }
        }
        Some(index)
    }
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = Power> + DoubleEndedIterator + ExactSizeIterator {
        unsafe {
            self.vec.into_iter()
        }
    }
    #[inline]
    pub fn iter_vars(self, vars: Variables) -> impl Iterator<Item = Power> {
        vars.iter().map(move |v| self[v])
    }
    #[inline]
    pub fn sum(self) -> Power {
        self.iter().sum()
    }
    #[inline]
    pub fn is_zeros(self) -> bool {
        unsafe { self.int == 0 }
    }
    #[inline]
    pub fn unchecked_add(self, rhs: Powers) -> Self {
        unsafe {
            Self { int: self.int.overflowing_add(rhs.int).0 }
        }
    }
}

impl std::fmt::Debug for Powers {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            self.vec.fmt(f)
        }
    }
}

impl std::cmp::PartialEq for Powers {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            self.int == other.int
        }
    }
}

impl ops::Index<Variable> for Powers {
    type Output = Power;

    #[inline]
    fn index(&self, var: Variable) -> &Power {
        unsafe {
            &self.vec[var.0 as usize]
        }
    }
}

impl ops::IndexMut<Variable> for Powers {
    #[inline]
    fn index_mut(&mut self, var: Variable) -> &mut Power {
        unsafe {
            &mut self.vec[var.0 as usize]
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PowersIter {
    vars: Variables,
    rem: Power,
    len: usize,
    powers: Powers,
}

impl Iterator for PowersIter {
    type Item = Powers;

    fn next(&mut self) -> Option<Powers> {
        if let Some(len) = self.len.checked_sub(1) {
            self.len = len;
            let next = self.powers;
            if len == 0 {
                return Some(next);
            }
            let mut vars = self.vars.iter();
            if let Some(mut v0) = vars.next() {
                if let Some(power) = self.powers[v0].checked_sub(1) {
                    self.powers[v0] = power;
                    self.rem += 1;
                    return Some(next);
                }
                for v1 in vars {
                    if let Some(power) = self.powers[v1].checked_sub(1) {
                        self.powers[v1] = power;
                        self.powers[v0] = self.rem;
                        self.rem = 1;
                        return Some(next);
                    }
                    v0 = v1;
                }
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl ExactSizeIterator for PowersIter {}

pub trait IntegerMultiple {
    type Output;
    fn integer_multiple(self, n: Power) -> Self::Output;
}

macro_rules! impl_int_mul {
    ($T:ty $(,)?) => {
        impl IntegerMultiple for $T {
            type Output = $T;
            #[inline]
            fn integer_multiple(self, n: Power) -> $T {
                self * n as $T
            }
        }

        impl IntegerMultiple for &$T {
            type Output = $T;
            #[inline]
            fn integer_multiple(self, n: Power) -> $T {
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

pub trait Poly: Sized {
    type Coeff;
    type CoeffsIter: Iterator<Item = Self::Coeff>;

    fn vars(&self) -> Variables;

    #[inline]
    fn nvars(&self) -> usize {
        self.vars().len()
    }

    fn degree(&self) -> Power;

    #[inline]
    fn is_constant(&self) -> bool {
        self.degree() == 0 || self.vars().is_empty()
    }

    fn coeffs_iter(self) -> Self::CoeffsIter;

    #[inline]
    fn coeffs_iter_with_powers(self) -> iter::Zip<PowersIter, Self::CoeffsIter> {
        iter::zip(
            Powers::iter_all(self.vars(), self.degree()),
            self.coeffs_iter(),
        )
    }

    #[inline]
    fn eval<Value, Values>(self, values: &Values) -> Value
    where
        Value: Zero + ops::AddAssign + ops::AddAssign<Self::Coeff>,
        for<'v> Value: ops::MulAssign<&'v Value>,
        Values: Sequence<Item = Value> + ?Sized,
    {
        assert!(values.len() >= self.nvars());
        let degree = self.degree();
        DefaultEvalCoeffsIter.eval_iter(&mut self.coeffs_iter(), degree, &values)
    }

    #[inline]
    fn partial_deriv(self, var: Variable) -> PartialDeriv<Self>
    where
        Self::Coeff: IntegerMultiple,
        <Self::Coeff as IntegerMultiple>::Output: Zero,
    {
        PartialDeriv { poly: self, var }
    }

    #[inline]
    fn mul<Rhs, OCoeff>(self, rhs: Rhs) -> Mul<Self, Rhs>
    where
        Rhs: Poly,
        Rhs::CoeffsIter: Clone,
        Self::Coeff: ops::Mul<Rhs::Coeff, Output = OCoeff> + Clone,
        OCoeff: ops::AddAssign + Zero,
    {
        // TODO: return error of self.degree() + rhs.degree() overflows.
        Mul { lhs: self, rhs }
    }

    fn assign_to<TargetCoeffs>(self, target: &mut PolySequence<TargetCoeffs>) -> Result<(), Error>
    where
        Self::Coeff: Zero,
        TargetCoeffs: Sequence<Item = Self::Coeff> + SequenceMut,
    {
        let svars = self.vars();
        let sdegree = self.degree();
        let tvars = target.vars();
        let tdegree = target.degree();
        if tdegree < sdegree {
            Err(Error::AssignLowerDegree)
        } else if !svars.is_contained_in(tvars) {
            Err(Error::AssignMissingVariables)
        } else if tdegree == sdegree && tvars == svars {
            for (t, s) in iter::zip(target.coeffs.iter_mut(), self.coeffs_iter()) {
                *t = s;
            }
            Ok(())
        } else {
            let mut scoeffs = self.coeffs_iter();
            let tpowers = Powers::iter_all(tvars, tdegree);
            for (tpowers, tcoeff) in iter::zip(tpowers, target.coeffs.iter_mut()) {
                let mut tdegree_svars = 0;
                let mut tdegree_other = 0;
                for var in Variables::iter_all() {
                    if svars.get(var) {
                        tdegree_svars += tpowers[var];
                    } else {
                        tdegree_other += tpowers[var];
                    }
                }
                if tdegree_other == 0 && tdegree_svars <= sdegree {
                    *tcoeff = scoeffs.next().unwrap_or_else(|| Self::Coeff::zero());
                } else {
                    *tcoeff = Self::Coeff::zero();
                }
            }
            Ok(())
        }
    }

    fn add_to<TargetCoeff, TargetCoeffs>(
        self,
        target: &mut PolySequence<TargetCoeffs>,
    ) -> Result<(), Error>
    where
        TargetCoeff: ops::AddAssign<Self::Coeff>,
        TargetCoeffs: Sequence<Item = TargetCoeff> + SequenceMut,
    {
        let svars = self.vars();
        let sdegree = self.degree();
        let tvars = target.vars();
        let tdegree = target.degree();
        if tdegree < sdegree {
            Err(Error::AssignLowerDegree)
        } else if !self.vars().is_contained_in(tvars) {
            Err(Error::AssignMissingVariables)
        } else if tdegree == sdegree && tvars == svars {
            for (t, s) in iter::zip(target.coeffs.iter_mut(), self.coeffs_iter()) {
                *t += s;
            }
            Ok(())
        } else {
            let mut scoeffs = self.coeffs_iter();
            let tpowers = Powers::iter_all(tvars, tdegree);
            for (tpowers, tcoeff) in iter::zip(tpowers, target.coeffs.iter_mut()) {
                let mut tdegree_svars = 0;
                let mut tdegree_other = 0;
                for var in Variables::iter_all() {
                    if svars.get(var) {
                        tdegree_svars += tpowers[var];
                    } else {
                        tdegree_other += tpowers[var];
                    }
                }
                if tdegree_other == 0 && tdegree_svars <= sdegree {
                    if let Some(scoeff) = scoeffs.next() {
                        *tcoeff += scoeff;
                    }
                }
            }
            Ok(())
        }
    }

    fn collect<TargetCoeffs>(self) -> PolySequence<TargetCoeffs>
    where
        TargetCoeffs: Sequence<Item = Self::Coeff> + FromIterator<Self::Coeff>,
    {
        let vars = self.vars();
        let degree = self.degree();
        let coeffs = self.coeffs_iter().collect();
        PolySequence::new_unchecked(coeffs, vars, degree)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolySequence<Coeffs> {
    coeffs: Coeffs,
    vars: Variables,
    degree: Power,
}

impl<Coeff, Coeffs> PolySequence<Coeffs>
where
    Coeffs: Sequence<Item = Coeff>,
{
    #[inline]
    pub fn new<Vars>(coeffs: Coeffs, vars: Vars, mut degree: Power) -> Result<Self, Error>
    where
        Vars: TryInto<Variables>,
        Error: From<<Vars as TryInto<Variables>>::Error>,
    {
        let mut vars = vars.try_into()?;
        // Normalize degree and vars.
        if degree == 0 || vars.is_empty() {
            degree = 0;
            vars = Variables::empty();
        }
        if coeffs.len() != ncoeffs(vars.len(), degree) {
            Err(Error::NCoeffsNVarsDegreeMismatch)
        } else {
            Ok(Self::new_unchecked(coeffs, vars, degree))
        }
    }

    #[inline]
    fn new_unchecked(coeffs: Coeffs, vars: Variables, degree: Power) -> Self {
        #[cfg(debug)]
        if degree == 0 || vars.len() == 0 {
            assert_eq!(degree, 0);
            assert_eq!(vars.len(), 0);
        }
        Self {
            coeffs,
            vars,
            degree,
        }
    }

    #[inline]
    pub fn zeros(mut vars: Variables, mut degree: Power) -> Self
    where
        Coeff: Zero,
        Coeffs: FromIterator<Coeff>,
    {
        // Normalize degree and vars.
        if degree == 0 || vars.is_empty() {
            degree = 0;
            vars = Variables::empty();
        }
        let ncoeffs = ncoeffs(vars.len(), degree);
        let coeffs: Coeffs = iter::repeat_with(|| Coeff::zero()).take(ncoeffs).collect();
        Self::new_unchecked(coeffs, vars, degree)
    }

    #[inline]
    pub fn by_ref(&self) -> &Self {
        self
    }
}

impl<Coeff, Coeffs> Poly for PolySequence<Coeffs>
where
    Coeffs: Sequence<Item = Coeff> + IntoIterator<Item = Coeff>,
{
    type Coeff = Coeff;
    type CoeffsIter = Coeffs::IntoIter;

    #[inline]
    fn vars(&self) -> Variables {
        self.vars
    }

    #[inline]
    fn degree(&self) -> Power {
        self.degree
    }

    fn coeffs_iter(self) -> Self::CoeffsIter {
        self.coeffs.into_iter()
    }
}

impl<'me, Coeff, Coeffs> Poly for &'me PolySequence<Coeffs>
where
    Coeff: 'me,
    Coeffs: Sequence<Item = Coeff>,
{
    type Coeff = &'me Coeff;
    type CoeffsIter = <Coeffs as SequenceIterType<'me, &'me Coeff>>::Iter;

    #[inline]
    fn vars(&self) -> Variables {
        self.vars
    }

    #[inline]
    fn degree(&self) -> Power {
        self.degree
    }

    fn coeffs_iter(self) -> Self::CoeffsIter {
        self.coeffs.iter()
    }
}

impl<'me, Coeff, Coeffs> Poly for &'me mut PolySequence<Coeffs>
where
    Coeff: 'me,
    Coeffs: Sequence<Item = Coeff>,
{
    type Coeff = &'me Coeff;
    type CoeffsIter = <Coeffs as SequenceIterType<'me, &'me Coeff>>::Iter;

    #[inline]
    fn vars(&self) -> Variables {
        self.vars
    }

    #[inline]
    fn degree(&self) -> Power {
        self.degree
    }

    fn coeffs_iter(self) -> Self::CoeffsIter {
        self.coeffs.iter()
    }
}

pub struct Constant<Coeff>(Coeff);

impl<Coeff> Poly for Constant<Coeff> {
    type Coeff = Coeff;
    type CoeffsIter = iter::Once<Coeff>;

    #[inline]
    fn vars(&self) -> Variables {
        Variables::empty()
    }

    #[inline]
    fn degree(&self) -> Power {
        0
    }

    #[inline]
    fn coeffs_iter(self) -> Self::CoeffsIter {
        iter::once(self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mul<Lhs, Rhs> {
    lhs: Lhs,
    rhs: Rhs,
}

impl<Lhs, Rhs, Coeff> Poly for Mul<Lhs, Rhs>
where
    Lhs: Poly,
    Rhs: Poly,
    Rhs::CoeffsIter: Clone,
    Lhs::Coeff: ops::Mul<Rhs::Coeff, Output = Coeff> + Clone,
    Coeff: ops::AddAssign + Zero,
{
    type Coeff = Coeff;
    type CoeffsIter = std::vec::IntoIter<Coeff>;

    #[inline]
    fn vars(&self) -> Variables {
        self.lhs.vars() | self.rhs.vars()
    }

    #[inline]
    fn degree(&self) -> Power {
        self.lhs.degree() + self.rhs.degree()
    }

    #[inline]
    fn coeffs_iter(self) -> Self::CoeffsIter {
        let mut output: PolySequence<Vec<_>> = PolySequence::zeros(self.vars(), self.degree());
        self.add_to(&mut output).unwrap();
        output.coeffs.into_iter()
    }

    #[inline]
    fn assign_to<TargetCoeffs>(self, target: &mut PolySequence<TargetCoeffs>) -> Result<(), Error>
    where
        TargetCoeffs: Sequence<Item = Self::Coeff> + SequenceMut,
    {
        target.coeffs.fill_with(|| Self::Coeff::zero());
        self.add_to(target)
    }

    fn add_to<TargetCoeff, TargetCoeffs>(
        self,
        target: &mut PolySequence<TargetCoeffs>,
    ) -> Result<(), Error>
    where
        TargetCoeff: ops::AddAssign<Self::Coeff>,
        TargetCoeffs: Sequence<Item = TargetCoeff> + SequenceMut,
    {
        let tvars = target.vars();
        let tdegree = target.degree();
        if tdegree < self.degree() {
            return Err(Error::AssignLowerDegree);
        } else if !self.vars().is_contained_in(tvars) {
            return Err(Error::AssignMissingVariables);
        } else if self.lhs.degree() == 0 && self.rhs.degree() == 0 {
            if let (Some(lc), Some(rc)) =
                (self.lhs.coeffs_iter().next(), self.rhs.coeffs_iter().next())
            {
                if let Some(tc) = target.coeffs.last_mut() {
                    *tc += lc * rc;
                }
            }
            return Ok(());
        } else {
            let rhs_coeffs_powers = self.rhs.coeffs_iter_with_powers();
            for (lp, lc) in self.lhs.coeffs_iter_with_powers() {
                //if lc.is_zero() {
                //    continue;
                //}
                for (rp, rc) in rhs_coeffs_powers.clone() {
                    //if rc.is_zero() {
                    //    continue;
                    //}
                    if let Some(ti) = lp.unchecked_add(rp).to_index(tvars, tdegree) {
                        if let Some(tc) = target.coeffs.get_mut(ti) {
                            *tc += lc.clone() * rc;
                        }
                    }
                }
            }
            return Ok(());
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Add<Lhs, Rhs> {
    lhs: Lhs,
    rhs: Rhs,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialDeriv<P: Poly> {
    poly: P,
    var: Variable,
}

impl<P: Poly> Poly for PartialDeriv<P>
where
    P::Coeff: IntegerMultiple,
    <P::Coeff as IntegerMultiple>::Output: Zero,
{
    type Coeff = <P::Coeff as IntegerMultiple>::Output;
    type CoeffsIter = PartialDerivCoeffsIter<P::CoeffsIter>;

    #[inline]
    fn vars(&self) -> Variables {
        if self.poly.degree() <= 1 || !self.poly.vars().get(self.var) {
            Variables::empty()
        } else {
            self.poly.vars()
        }
    }

    #[inline]
    fn degree(&self) -> Power {
        if self.poly.degree() <= 1 || !self.poly.vars().get(self.var) {
            0
        } else {
            self.poly.degree() - 1
        }
    }

    #[inline]
    fn coeffs_iter(self) -> Self::CoeffsIter {
        if self.poly.vars().get(self.var) {
            // If the degree of `self.poly` is zero, the iterator we return
            // here will be empty, which violates the requirement of `Poly`:
            // the number of coefficients for a polynomial of degree zero is
            // one. However, since the `Poly::vars()` should be empty for a
            // polynomial of degree zero, this situation cannot occur.
            PartialDerivCoeffsIter::NonZero(self.poly.coeffs_iter_with_powers(), self.var)
        } else {
            PartialDerivCoeffsIter::Zero(iter::once(Self::Coeff::zero()))
        }
    }
}

pub enum PartialDerivCoeffsIter<CoeffsIter>
where
    CoeffsIter: Iterator,
    CoeffsIter::Item: IntegerMultiple,
    <CoeffsIter::Item as IntegerMultiple>::Output: Zero,
{
    Zero(iter::Once<<CoeffsIter::Item as IntegerMultiple>::Output>),
    NonZero(iter::Zip<PowersIter, CoeffsIter>, Variable),
}

impl<CoeffsIter> Iterator for PartialDerivCoeffsIter<CoeffsIter>
where
    CoeffsIter: Iterator,
    CoeffsIter::Item: IntegerMultiple,
    <CoeffsIter::Item as IntegerMultiple>::Output: Zero,
{
    type Item = <CoeffsIter::Item as IntegerMultiple>::Output;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Zero(iter) => iter.next(),
            Self::NonZero(iter, var) => {
                while let Some((powers, coeff)) = iter.next() {
                    if powers[*var] > 0 {
                        return Some(coeff.integer_multiple(powers[*var]));
                    }
                }
                None
            }
        }
    }
}

trait EvalCoeffsIter<Value, Coeff, Output> {
    fn from_coeff(&self, coeff: Coeff) -> Output;
    fn init_acc(&self) -> Output;
    fn update_acc_coeff(&self, acc: &mut Output, coeff: Coeff, value: &Value);
    fn update_acc_inner(&self, acc: &mut Output, inner: Output, value: &Value);

    #[inline]
    fn eval<P, Values>(&self, poly: P, values: &Values) -> Output
    where
        P: Poly<Coeff = Coeff>,
        Values: Sequence<Item = Value> + ?Sized,
    {
        assert!(values.len() >= poly.nvars());
        let degree = poly.degree();
        self.eval_iter(&mut poly.coeffs_iter(), degree, &values)
    }

    #[inline]
    fn eval_iter<Coeffs, Values>(
        &self,
        coeffs: &mut Coeffs,
        degree: Power,
        values: &Values,
    ) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: Sequence<Item = Value> + ?Sized,
    {
        match values.len() {
            0 => self.eval_0d(coeffs),
            1 => self.eval_1d(coeffs, degree, values),
            2 => self.eval_2d(coeffs, degree, values),
            3 => self.eval_3d(coeffs, degree, values),
            nvars => self.eval_nd(coeffs, degree, values, nvars),
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
    fn eval_1d<Coeffs, Values>(&self, coeffs: &mut Coeffs, degree: Power, values: &Values) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: Sequence<Item = Value> + ?Sized,
    {
        let mut acc = self.init_acc();
        let value = values.get(0).unwrap();
        for coeff in coeffs.take(degree as usize + 1) {
            self.update_acc_coeff(&mut acc, coeff, value);
        }
        acc
    }

    #[inline]
    fn eval_2d<Coeffs, Values>(&self, coeffs: &mut Coeffs, degree: Power, values: &Values) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: Sequence<Item = Value> + ?Sized,
    {
        let mut acc = self.init_acc();
        let value = values.get(1).unwrap();
        for p in 0..=degree {
            let inner = self.eval_1d(coeffs, p, values);
            self.update_acc_inner(&mut acc, inner, value);
        }
        acc
    }

    #[inline]
    fn eval_3d<Coeffs, Values>(&self, coeffs: &mut Coeffs, degree: Power, values: &Values) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: Sequence<Item = Value> + ?Sized,
    {
        let mut acc = self.init_acc();
        let value = values.get(2).unwrap();
        for p in 0..=degree {
            let inner = self.eval_2d(coeffs, p, values);
            self.update_acc_inner(&mut acc, inner, value);
        }
        acc
    }

    #[inline]
    fn eval_nd<Coeffs, Values>(
        &self,
        coeffs: &mut Coeffs,
        degree: Power,
        values: &Values,
        nvars: usize,
    ) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: Sequence<Item = Value> + ?Sized,
    {
        if nvars == 3 {
            self.eval_3d(coeffs, degree, values)
        } else {
            let mut acc = self.init_acc();
            let value = values.get(nvars - 1).unwrap();
            for p in 0..=degree {
                let inner = self.eval_nd(coeffs, p, values, nvars - 1);
                self.update_acc_inner(&mut acc, inner, value);
            }
            acc
        }
    }
}

struct DefaultEvalCoeffsIter;

impl<Value, Coeff> EvalCoeffsIter<Value, Coeff, Value> for DefaultEvalCoeffsIter
where
    Value: Zero + ops::AddAssign + ops::AddAssign<Coeff>,
    for<'v> Value: ops::MulAssign<&'v Value>,
{
    #[inline]
    fn from_coeff(&self, coeff: Coeff) -> Value {
        let mut acc = Value::zero();
        acc += coeff;
        acc
    }
    #[inline]
    fn init_acc(&self) -> Value {
        Value::zero()
    }
    #[inline]
    fn update_acc_coeff(&self, acc: &mut Value, coeff: Coeff, value: &Value) {
        *acc *= value;
        *acc += coeff;
    }
    #[inline]
    fn update_acc_inner(&self, acc: &mut Value, inner: Value, value: &Value) {
        *acc *= value;
        *acc += inner;
    }
}

struct EvalCompositionCoeffsIter;

impl<Value, Coeff, OCoeff> EvalCoeffsIter<Value, Coeff, PolySequence<Vec<OCoeff>>>
    for EvalCompositionCoeffsIter
where
    for<'a> &'a Value: Poly,
    for<'a> <&'a Value as Poly>::CoeffsIter: Clone,
    OCoeff: Zero
        + ops::AddAssign
        + for<'a> ops::AddAssign<&'a OCoeff>
        + ops::AddAssign<Coeff>
        + for<'a> ops::AddAssign<<&'a Value as Poly>::Coeff>
        + for<'a> ops::Mul<<&'a Value as Poly>::Coeff, Output = OCoeff>
        + Clone,
{
    #[inline]
    fn from_coeff(&self, coeff: Coeff) -> PolySequence<Vec<OCoeff>> {
        let mut acc: PolySequence<Vec<OCoeff>> = PolySequence::zeros(Variables::empty(), 0);
        *acc.coeffs.last_mut().unwrap() += coeff;
        acc
    }
    #[inline]
    fn init_acc(&self) -> PolySequence<Vec<OCoeff>> {
        PolySequence::zeros(Variables::empty(), 0)
    }
    #[inline]
    fn update_acc_coeff(&self, acc: &mut PolySequence<Vec<OCoeff>>, coeff: Coeff, value: &Value) {
        if acc.degree() == 0 && acc.coeffs.get(0).unwrap().is_zero() {
            *acc.coeffs.last_mut().unwrap() += coeff;
        } else {
            let mut old_acc =
                PolySequence::zeros(acc.vars() | value.vars(), acc.degree() + value.degree());
            std::mem::swap(acc, &mut old_acc);
            old_acc.mul(value).add_to(acc).unwrap();
            *acc.coeffs.last_mut().unwrap() += coeff;
        }
    }
    #[inline]
    fn update_acc_inner(
        &self,
        acc: &mut PolySequence<Vec<OCoeff>>,
        mut inner: PolySequence<Vec<OCoeff>>,
        value: &Value,
    ) {
        if acc.degree() == 0 && acc.coeffs.get(0).unwrap().is_zero() {
            std::mem::swap(acc, &mut inner)
        } else {
            let mut old_acc = PolySequence::zeros(
                acc.vars() | inner.vars() | value.vars(),
                std::cmp::max(acc.degree() + value.degree(), inner.degree()),
            );
            std::mem::swap(acc, &mut old_acc);
            old_acc.mul(value).add_to(acc).unwrap();
            inner.add_to(acc).unwrap();
        }
    }
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
    transform_degree: Power,
    from_nvars: usize,
    degree: Power,
    to_nvars: usize,
) -> Vec<f64> {
    let transform_ncoeffs = ncoeffs(from_nvars, transform_degree);
    assert_eq!(transform_coeffs.len(), to_nvars * transform_ncoeffs);
    let row_degree = transform_degree * degree;

    let inner_vars = (0..from_nvars).try_into().unwrap();
    let outer_vars = (0..to_nvars).try_into().unwrap();
    let transform_polys: Vec<_> = transform_coeffs
        .chunks_exact(transform_ncoeffs)
        .map(|c| PolySequence::new_unchecked(c, inner_vars, transform_degree))
        .collect();

    let nrows = ncoeffs(from_nvars, row_degree);
    let ncols = ncoeffs(to_nvars, degree);
    let mut matrix: Vec<f64> = Vec::new();
    matrix.resize(nrows * ncols, 0.0);

    for (i, col) in matrix.chunks_exact_mut(nrows).enumerate() {
        let mut col = PolySequence::new_unchecked(col, inner_vars, row_degree);
        let mut outer: PolySequence<Vec<f64>> = PolySequence::zeros(outer_vars, degree);
        *outer.coeffs.get_mut(i).unwrap() = 1.0;
        EvalCompositionCoeffsIter
            .eval(outer, &transform_polys)
            .assign_to(&mut col)
            .unwrap();
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::{
        EvalCoeffsIter, EvalCompositionCoeffsIter, Poly, PolySequence, Powers, Variable, Variables,
    };
    use approx::assert_abs_diff_eq;
    use std::iter;

    #[test]
    fn ncoeffs() {
        assert_eq!(super::ncoeffs(0, 0), 1);
        assert_eq!(super::ncoeffs(0, 1), 1);
        assert_eq!(super::ncoeffs(1, 0), 1);
        assert_eq!(super::ncoeffs(1, 1), 2);
        assert_eq!(super::ncoeffs(1, 2), 3);
        assert_eq!(super::ncoeffs(2, 0), 1);
        assert_eq!(super::ncoeffs(2, 1), 3);
        assert_eq!(super::ncoeffs(2, 2), 6);
        assert_eq!(super::ncoeffs(2, 3), 10);
        assert_eq!(super::ncoeffs(3, 0), 1);
        assert_eq!(super::ncoeffs(3, 1), 4);
        assert_eq!(super::ncoeffs(3, 2), 10);
        assert_eq!(super::ncoeffs(3, 3), 20);
    }

    #[test]
    fn variables_index() {
        let vars = Variables::try_from([2, 4, 5]).unwrap();
        assert_eq!(vars.index(Variable::try_from(0).unwrap()), None);
        assert_eq!(vars.index(Variable::try_from(1).unwrap()), None);
        assert_eq!(vars.index(Variable::try_from(2).unwrap()), Some(0));
        assert_eq!(vars.index(Variable::try_from(3).unwrap()), None);
        assert_eq!(vars.index(Variable::try_from(4).unwrap()), Some(1));
        assert_eq!(vars.index(Variable::try_from(5).unwrap()), Some(2));
        assert_eq!(vars.index(Variable::try_from(6).unwrap()), None);
    }

    #[test]
    fn powers_to_index_to_powers() {
        macro_rules! assert_index_powers {
            ($degree:literal, $powers:tt) => {
                let vars = Variables::try_from(0..$powers[0].len()).unwrap();
                let mut iter = Powers::iter_all(vars, $degree);
                for (index, raw_powers) in $powers.iter().enumerate() {
                    let mut powers = Powers::zeros();
                    for (v, p) in iter::zip(vars.iter(), raw_powers) {
                        powers[v] = *p;
                    }
                    println!("index: {index}, powers: {powers:?}");
                    assert_eq!(
                        Powers::from_index(index, vars, $degree),
                        Some(powers),
                        "Powers::from_index"
                    );
                    assert_eq!(
                        powers.to_index(vars, $degree),
                        Some(index),
                        "Powers::to_index"
                    );
                    assert_eq!(iter.next(), Some(powers), "Powers::iter_all");
                }
                assert_eq!(iter.next(), None);
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
            PolySequence::new([1], 0..0, 0)
                .unwrap()
                .eval(&[] as &[usize]),
            1
        );
    }

    #[test]
    fn eval_1d() {
        assert_eq!(PolySequence::new([1], 0..1, 0).unwrap().eval(&[5]), 1);
        assert_eq!(PolySequence::new([2, 1], 0..1, 1).unwrap().eval(&[5]), 11);
        assert_eq!(
            PolySequence::new([3, 2, 1], 0..1, 2).unwrap().eval(&[5]),
            86
        );
    }

    #[test]
    fn eval_2d() {
        assert_eq!(PolySequence::new([1], 0..2, 0).unwrap().eval(&[5, 3]), 1);
        assert_eq!(
            PolySequence::new([0, 0, 1], 0..2, 1).unwrap().eval(&[5, 3]),
            1
        );
        assert_eq!(
            PolySequence::new([0, 1, 0], 0..2, 1).unwrap().eval(&[5, 3]),
            5
        );
        assert_eq!(
            PolySequence::new([1, 0, 0], 0..2, 1).unwrap().eval(&[5, 3]),
            3
        );
        assert_eq!(
            PolySequence::new([3, 2, 1], 0..2, 1).unwrap().eval(&[5, 3]),
            20
        );
        assert_eq!(
            PolySequence::new([6, 5, 4, 3, 2, 1], 0..2, 2)
                .unwrap()
                .eval(&[5, 3]),
            227
        );
    }

    #[test]
    fn eval_3d() {
        assert_eq!(PolySequence::new([1], 0..3, 0).unwrap().eval(&[5, 3, 2]), 1);
        assert_eq!(
            PolySequence::new([0, 0, 0, 1], 0..3, 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            1
        );
        assert_eq!(
            PolySequence::new([0, 0, 1, 0], 0..3, 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            5
        );
        assert_eq!(
            PolySequence::new([0, 1, 0, 0], 0..3, 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            3
        );
        assert_eq!(
            PolySequence::new([1, 0, 0, 0], 0..3, 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            2
        );
        assert_eq!(
            PolySequence::new([4, 3, 2, 1], 0..3, 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            28
        );
        assert_eq!(
            PolySequence::new([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 0..3, 2)
                .unwrap()
                .eval(&[5, 3, 2]),
            415,
        );
    }

    #[test]
    fn eval_ref() {
        let coeffs: Vec<usize> = vec![3, 2, 1];
        let poly = PolySequence::new(&coeffs, 0..1, 2).unwrap();
        assert_eq!(poly.eval(&[5]), 86);
    }

    #[test]
    fn partial_deriv() {
        let x0 = Variable::try_from(0).unwrap();
        let x1 = Variable::try_from(1).unwrap();
        assert_eq!(
            PolySequence::new([1], 0..1, 0)
                .unwrap()
                .partial_deriv(x0)
                .collect(),
            PolySequence::new(vec![0], 0..1, 0).unwrap(),
        );
        assert_eq!(
            PolySequence::new([2, 1], 0..1, 1)
                .unwrap()
                .partial_deriv(x0)
                .collect(),
            PolySequence::new(vec![2], 0..1, 0).unwrap(),
        );
        assert_eq!(
            PolySequence::new([4, 3, 2, 1], 0..1, 3)
                .unwrap()
                .partial_deriv(x0)
                .collect(),
            PolySequence::new(vec![12, 6, 2], 0..1, 2).unwrap(),
        );
        assert_eq!(
            PolySequence::new([6, 5, 4, 3, 2, 1], 0..2, 2)
                .unwrap()
                .partial_deriv(x0)
                .collect(),
            PolySequence::new(vec![5, 6, 2], 0..2, 1).unwrap(),
        );
        assert_eq!(
            PolySequence::new([6, 5, 4, 3, 2, 1], 0..2, 2)
                .unwrap()
                .partial_deriv(x1)
                .collect(),
            PolySequence::new(vec![12, 5, 4], 0..2, 1).unwrap(),
        );
    }

    #[test]
    fn mul() {
        let l = PolySequence::new([2], 0..1, 0).unwrap();
        let r = PolySequence::new([3], 0..1, 0).unwrap();
        assert_eq!(
            l.mul(&r).collect(),
            PolySequence::new(vec![6], 0..1, 0).unwrap(),
        );

        let l = PolySequence::new([2, 1], 0..1, 1).unwrap();
        let r = PolySequence::new([4, 3], 0..1, 1).unwrap();
        assert_eq!(
            l.mul(&r).collect(),
            PolySequence::new(vec![8, 10, 3], 0..1, 2).unwrap(),
        );

        let l = PolySequence::new([3, 2, 1], 0..2, 1).unwrap();
        let r = PolySequence::new([6, 5, 4], 0..2, 1).unwrap();
        assert_eq!(
            l.mul(&r).collect(),
            PolySequence::new(vec![18, 27, 18, 10, 13, 4], 0..2, 2).unwrap(),
        );
    }

    #[test]
    fn composition() {
        let p = PolySequence::new([2, 1], 0..1, 1).unwrap();
        let q = PolySequence::new([4, 3], 1..2, 1).unwrap();
        let desired = PolySequence::new(vec![8, 7], 1..2, 1).unwrap();
        assert_eq!(EvalCompositionCoeffsIter.eval(p, &[q]), desired,);

        let p = PolySequence::new([3, 2, 1], 0..1, 2).unwrap();
        let q = PolySequence::new([4, 3], 1..2, 1).unwrap();
        let desired = PolySequence::new(vec![48, 80, 34], 1..2, 2).unwrap();
        assert_eq!(EvalCompositionCoeffsIter.eval(p, &[q]), desired,);

        let p = PolySequence::new([3, 2, 1], 0..2, 1).unwrap();
        let q = PolySequence::new([4, 3], 2..3, 1).unwrap();
        let r = PolySequence::new([2, 1], 3..4, 1).unwrap();
        let desired = PolySequence::new(vec![6, 8, 10], 2..4, 1).unwrap();
        assert_eq!(EvalCompositionCoeffsIter.eval(p, &[q, r]), desired,);
    }

    #[test]
    fn transform_matrix_1d() {
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.5, 0.0], 1, 1, 2, 1)[..],
            [
                0.25, 0.00, 0.00, //
                0.00, 0.50, 0.00, //
                0.00, 0.00, 1.00, //
            ]
        );
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.5, 0.5], 1, 1, 2, 1)[..],
            [
                0.25, 0.50, 0.25, //
                0.00, 0.50, 0.50, //
                0.00, 0.00, 1.00, //
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
                0.00, 0.25, 0.25, 0.00, 0.00, 0.00, //
                0.00, 0.00, 0.50, 0.00, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.25, 0.50, 0.25, //
                0.00, 0.00, 0.00, 0.00, 0.50, 0.50, //
                0.00, 0.00, 0.00, 0.00, 0.00, 1.00, //
            ]
        );
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.0, 0.5, 0.0, 0.5, 0.0, 0.5], 1, 2, 2, 2)[..],
            [
                0.25, 0.00, 0.50, 0.00, 0.00, 0.25, //
                0.00, 0.25, 0.00, 0.00, 0.25, 0.00, //
                0.00, 0.00, 0.50, 0.00, 0.00, 0.50, //
                0.00, 0.00, 0.00, 0.25, 0.00, 0.00, //
                0.00, 0.00, 0.00, 0.00, 0.50, 0.00, //
                0.00, 0.00, 0.00, 0.00, 0.00, 1.00, //
            ]
        );
        assert_abs_diff_eq!(
            super::transform_matrix(&[0.0, 0.5, 0.5, 0.5, 0.0, 0.5], 1, 2, 2, 2)[..],
            [
                0.25, 0.00, 0.50, 0.00, 0.00, 0.25, //
                0.00, 0.25, 0.25, 0.00, 0.25, 0.25, //
                0.00, 0.00, 0.50, 0.00, 0.00, 0.50, //
                0.00, 0.00, 0.00, 0.25, 0.50, 0.25, //
                0.00, 0.00, 0.00, 0.00, 0.50, 0.50, //
                0.00, 0.00, 0.00, 0.00, 0.00, 1.00, //
            ]
        );
    }
}

#[cfg(all(feature = "bench", test))]
mod benches {
    extern crate test;
    use self::test::Bencher;
    use super::{Poly, PolySequence};

    macro_rules! mk_bench_eval {
        ($name:ident, $degree:literal, $nvars:literal) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let coeffs: Vec<_> = (1..=super::ncoeffs($nvars, $degree))
                    .map(|i| i as f64)
                    .collect();
                let values: Vec<_> = (1..=$nvars).map(|x| x as f64).collect();
                let poly = PolySequence::new(coeffs, 0..$nvars, $degree).unwrap();
                b.iter(|| test::black_box(&poly).eval(&values[..]))
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
        b.iter(|| super::ncoeffs(test::black_box(3), test::black_box(4)));
    }

    #[bench]
    fn mul_same_vars_3d_degree4_degree2(b: &mut Bencher) {
        let l = PolySequence::new(
            (0..super::ncoeffs(3, 4))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            0..3,
            4,
        )
        .unwrap();
        let r = PolySequence::new(
            (0..super::ncoeffs(3, 2))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            0..3,
            2,
        )
        .unwrap();
        b.iter(|| {
            test::black_box(&l)
                .mul(test::black_box(&r))
                .collect::<Vec<_>>()
        });
    }

    #[bench]
    fn mul_different_vars_1d_degree4_2d_degree2(b: &mut Bencher) {
        let l = PolySequence::new(
            (0..super::ncoeffs(1, 4))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            0..1,
            4,
        )
        .unwrap();
        let r = PolySequence::new(
            (0..super::ncoeffs(2, 2))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            1..3,
            2,
        )
        .unwrap();
        b.iter(|| {
            test::black_box(&l)
                .mul(test::black_box(&r))
                .collect::<Vec<_>>()
        });
    }

    //#[bench]
    //fn pow_3d_degree4_exp3(b: &mut Bencher) {
    //    b.iter(|| {
    //        super::pow(
    //            test::black_box(
    //                &(0..super::ncoeffs(3, 4))
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
}
