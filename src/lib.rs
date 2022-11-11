//! Functions for evaluating and manipulating polynomials.
//!
//! The polynomials considered in this crate are of the form
//!
//! ```text
//! Σ_{k ∈ ℤ^n | Σ_i k_i ≤ p} c_k ∏_i x_i^(k_i)
//! ```
//!
//! where `c` is a vector of coefficients, `x` a vector of `n` variables and
//! `p` a nonnegative integer degree. For performance reasons `n` is bounded
//! above by eight.
//!
//! This crate provides a [`Poly`] trait defining functions like [`eval`],
//! evaluate a polynomial, and [`mul`], multiply a polynomial, and several
//! implementations of which [`PolySequence`] is the most important.
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
//! The vector of coefficients for the polynomial `p(x) = x0^2 - x0 + 2` is
//! `[1, -1, 2]`.
//!
//! ```
//! use nutils_poly::{ExplicitPoly, Variable, Variables, traits::*};
//!
//! let x = Variable::new(0).unwrap();
//! let p = ExplicitPoly::new([1, -1, 2], x, 2).unwrap();
//! ```
//!
//! You can evaluate this polynomial for some `x` using [`PolySequence::eval()`]:
//!
//! ```
//! # use nutils_poly::{ExplicitPoly, Variable, Variables, traits::*};
//! #
//! # let x = Variable::new(0).unwrap();
//! # let p = ExplicitPoly::new([1, -1, 2], x, 2).unwrap();
//! assert_eq!(p.eval(&[0]), 2); // x = [0]
//! assert_eq!(p.eval(&[1]), 2); // x = [1]
//! assert_eq!(p.eval(&[2]), 4); // x = [2]
//! ```
//!
//! Or compute the partial derivative `∂p/∂x` using [`PartialDeriv::partial_deriv()`]:
//!
//! ```
//! # use nutils_poly::{ExplicitPoly, Variable, Variables, traits::*};
//! #
//! # let x = Variable::new(0).unwrap();
//! # let p = ExplicitPoly::new([1, -1, 2], x, 2).unwrap();
//! assert_eq!(
//!     ExplicitPoly::from_iter(p.partial_deriv(Variable::new(0).unwrap())),
//!     ExplicitPoly::new(vec![2, -1], x, 1).unwrap(),
//! );
//! ```
//!
//! [lexicographic order]: https://en.wikipedia.org/wiki/Lexicographic_order

#![cfg_attr(feature = "bench", feature(test))]

mod explicit;
mod power;
pub mod traits;
mod variable;

pub use explicit::ExplicitPoly;
pub use power::{Power, Powers, PowersIter};
pub use variable::{Variable, Variables};

use num_traits::Zero;
use sqnc::{SequenceRef, IndexableSequence, IndexableMutSequence};
use std::borrow::Borrow;
use std::iter;
use std::marker::PhantomData;
use std::ops;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    AssignMissingVariables,
    AssignLowerDegree,
    NCoeffsNVarsDegreeMismatch,
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

use crate::traits::*;

pub struct Mul<L, R>(L, R);

impl<'l, 'r, L, R, OCoeff> Poly for Mul<&'l L, &'r R>
where
    L: Poly,
    R: Poly,
    &'l L::Coeff: ops::Mul<&'r R::Coeff, Output = OCoeff>,
{
    type Coeff = OCoeff;

    fn vars(&self) -> Variables {
        self.0.vars() | self.1.vars()
    }

    fn degree(&self) -> Power {
        self.0.degree() + self.1.degree()
    }
}

impl<'l, 'r, L, R, OCoeff> PolyAssign for Mul<&'l L, &'r R>
where
    L: PolyCoeffsIter,
    R: PolyCoeffsIter,
    L::Coeff: Zero,
    R::Coeff: Zero,
    &'l L::Coeff: ops::Mul<&'r R::Coeff, Output = OCoeff>,
    OCoeff: ops::AddAssign,
{
    #[inline]
    fn assign_to<Target>(self, target: &mut Target) -> Result<(), Error>
    where
        OCoeff: Zero,
        Target: Poly<Coeff = OCoeff> + PolyCoeffsMut + PolyCoeffsIterMut,
    {
        target
            .coeffs_iter_mut()
            .for_each(|coeff| *coeff = OCoeff::zero());
        self.add_to(target)
    }

    fn add_to<Target>(self, target: &mut Target) -> Result<(), Error>
    where
        Target: PolyCoeffsMut + PolyCoeffsIterMut,
        Target::Coeff: ops::AddAssign<OCoeff>,
    {
        let tvars = target.vars();
        let tdegree = target.degree();
        if tdegree < self.degree() {
            Err(Error::AssignLowerDegree)
        } else if !self.vars().is_contained_in(tvars) {
            Err(Error::AssignMissingVariables)
        } else if self.0.degree() == 0 && self.1.degree() == 0 {
            if let (Some(lc), Some(rc)) = (self.0.coeffs_iter().next(), self.1.coeffs_iter().next())
            {
                if let Some(tc) = target.coeff_mut(target.ncoeffs().saturating_sub(1)) {
                    *tc += lc * rc;
                }
            }
            Ok(())
        } else {
            for (lp, lc) in self.0.coeffs_iter_with_powers() {
                if lc.is_zero() {
                    continue;
                }
                for (rp, rc) in self.1.coeffs_iter_with_powers() {
                    if rc.is_zero() {
                        continue;
                    }
                    if let Some(ti) = lp.unchecked_add(rp).to_index(tvars, tdegree) {
                        if let Some(tc) = target.coeff_mut(ti) {
                            *tc += lc * rc;
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

//impl<'r, L, R, OCoeff> PolyAssign for Mul<L, &'r R>
//where
//    L: PolyIntoCoeffsIter,
//    R: PolyCoeffsIter,
//    L::Coeff: Zero,
//    R::Coeff: Zero,
//    L::Coeff: ops::Mul<&'r R::Coeff, Output = OCoeff>,
//    for<'l> &'l L::Coeff: ops::Mul<&'r R::Coeff, Output = OCoeff>,
//    OCoeff: ops::AddAssign,
//{
//    fn assign_to<Target>(self, target: &mut Target) -> Result<(), Error>
//    where
//        OCoeff: Zero,
//        Target: Poly<Coeff = OCoeff> + PolyCoeffsMut + PolyCoeffsIterMut,
//    {
//        target
//            .coeffs_iter_mut()
//            .for_each(|coeff| *coeff = OCoeff::zero());
//        self.add_to(target)
//    }
//
//    fn add_to<Target>(self, target: &mut Target) -> Result<(), Error>
//    where
//        Target: PolyCoeffsMut + PolyCoeffsIterMut,
//        Target::Coeff: ops::AddAssign<OCoeff>,
//    {
//        let tvars = target.vars();
//        let tdegree = target.degree();
//        if tdegree < self.degree() {
//            Err(Error::AssignLowerDegree)
//        } else if !self.vars().is_contained_in(tvars) {
//            Err(Error::AssignMissingVariables)
//        } else if self.0.degree() == 0 && self.1.degree() == 0 {
//            if let (Some(lc), Some(rc)) = (self.0.into_coeffs_iter().next(), self.1.coeffs_iter().next())
//            {
//                if let Some(tc) = target.coeff_mut(target.ncoeffs().saturating_sub(1)) {
//                    *tc += lc * rc;
//                }
//            }
//            Ok(())
//        } else {
//            for (lp, lc) in self.0.into_coeffs_iter_with_powers() {
//                if lc.is_zero() {
//                    continue;
//                }
//                for (rp, rc) in self.1.coeffs_iter_with_powers() {
//                    if rc.is_zero() {
//                        continue;
//                    }
//                    if let Some(ti) = lp.unchecked_add(rp).to_index(tvars, tdegree) {
//                        if let Some(tc) = target.coeff_mut(ti) {
//                            *tc += &lc * rc;
//                        }
//                    }
//                }
//            }
//            Ok(())
//        }
//    }
//}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialDeriv<Parent, OwnedParent> {
    parent: Parent,
    var: Variable,
    phantom: PhantomData<OwnedParent>,
}

impl<Parent, OwnedParent> PartialDeriv<Parent, OwnedParent> {
    #[inline]
    pub(crate) fn new(parent: Parent, var: Variable) -> Self {
        Self {
            parent,
            var,
            phantom: PhantomData,
        }
    }
}

impl<Parent, OwnedParent, Coeff> Poly for PartialDeriv<Parent, OwnedParent>
where
    Parent: Borrow<OwnedParent>,
    OwnedParent: Poly,
    OwnedParent::Coeff: IntegerMultiple<Output = Coeff>,
{
    type Coeff = Coeff;

    #[inline]
    fn vars(&self) -> Variables {
        if self.degree() == 0 {
            Variables::none()
        } else {
            self.parent.borrow().vars()
        }
    }

    #[inline]
    fn degree(&self) -> Power {
        if !self.parent.borrow().vars().contains(self.var) {
            0
        } else {
            self.parent.borrow().degree().saturating_sub(1)
        }
    }
}

impl<OwnedParent, Coeff> PolyIntoCoeffsIter for PartialDeriv<OwnedParent, OwnedParent>
where
    OwnedParent: Poly + PolyIntoCoeffsIter,
    OwnedParent::Coeff: IntegerMultiple<Output = Coeff> + Sized,
    Coeff: Zero,
{
    type IntoCoeffsIter = PartialDerivCoeffsIter<OwnedParent::IntoCoeffsIter>;

    fn into_coeffs_iter(self) -> Self::IntoCoeffsIter {
        if self.parent.vars().contains(self.var) {
            // If the degree of `self.parent` is zero, the iterator we return
            // here will be empty, which violates the requirement of `Poly`:
            // the number of coefficients for a polynomial of degree zero is
            // one. However, since the `Poly::vars()` should be empty for a
            // polynomial of degree zero, this situation cannot occur.
            PartialDerivCoeffsIter::NonZero(self.parent.into_coeffs_iter_with_powers(), self.var)
        } else {
            PartialDerivCoeffsIter::Zero(iter::once(Self::Coeff::zero()))
        }
    }
}

impl<'parent, OwnedParent, Coeff> PolyIntoCoeffsIter
    for PartialDeriv<&'parent OwnedParent, OwnedParent>
where
    OwnedParent: Poly + PolyCoeffsIter,
    OwnedParent::Coeff: IntegerMultiple<Output = Coeff>,
    for<'coeff> &'coeff OwnedParent::Coeff: IntegerMultiple<Output = Coeff>,
    Coeff: Zero,
{
    type IntoCoeffsIter = PartialDerivCoeffsIter<OwnedParent::CoeffsIter<'parent>>;

    fn into_coeffs_iter(self) -> Self::IntoCoeffsIter {
        if self.parent.vars().contains(self.var) {
            // If the degree of `self.parent` is zero, the iterator we return
            // here will be empty, which violates the requirement of `Poly`:
            // the number of coefficients for a polynomial of degree zero is
            // one. However, since the `Poly::vars()` should be empty for a
            // polynomial of degree zero, this situation cannot occur.
            PartialDerivCoeffsIter::NonZero(self.parent.coeffs_iter_with_powers(), self.var)
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
            Self::NonZero(iter, var) => iter
                .filter(|(powers, _)| powers[*var] > 0)
                .map(|(powers, coeff)| coeff.integer_multiple(powers[*var]))
                .next(),
        }
    }
}

impl<P> PolyAssign for P
where
    P: PolyIntoCoeffsIter,
    P::Coeff: Sized,
{
    fn assign_to<Target>(self, target: &mut Target) -> Result<(), Error>
    where
        Self::Coeff: Zero,
        Target: Poly<Coeff = Self::Coeff> + PolyCoeffsMut + PolyCoeffsIterMut,
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
            for (t, s) in iter::zip(target.coeffs_iter_mut(), self.into_coeffs_iter()) {
                *t = s;
            }
            Ok(())
        } else {
            target
                .coeffs_iter_mut()
                .for_each(|c| *c = Self::Coeff::zero());
            for (spower, scoeff) in self.into_coeffs_iter_with_powers() {
                if let Some(index) = spower.to_index(tvars, tdegree) {
                    if let Some(tcoeff) = target.coeff_mut(index) {
                        *tcoeff = scoeff;
                    }
                }
            }
            Ok(())
        }
    }

    fn add_to<Target>(self, target: &mut Target) -> Result<(), Error>
    where
        Target: PolyCoeffsMut + PolyCoeffsIterMut,
        Target::Coeff: ops::AddAssign<Self::Coeff>,
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
            for (t, s) in iter::zip(target.coeffs_iter_mut(), self.into_coeffs_iter()) {
                *t += s;
            }
            Ok(())
        } else {
            for (spower, scoeff) in self.into_coeffs_iter_with_powers() {
                if let Some(index) = spower.to_index(tvars, tdegree) {
                    if let Some(tcoeff) = target.coeff_mut(index) {
                        *tcoeff += scoeff;
                    }
                }
            }
            Ok(())
        }
    }
}

impl<P: PolyCoeffsIter> PolyAssignRef for P {
    fn assign_clone_to<Target>(&self, target: &mut Target) -> Result<(), Error>
    where
        Self::Coeff: Zero + Clone,
        Target: Poly<Coeff = Self::Coeff> + PolyCoeffsMut + PolyCoeffsIterMut,
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
            for (t, s) in iter::zip(target.coeffs_iter_mut(), self.coeffs_iter()) {
                *t = s.clone();
            }
            Ok(())
        } else {
            target
                .coeffs_iter_mut()
                .for_each(|c| *c = Self::Coeff::zero());
            for (spower, scoeff) in self.coeffs_iter_with_powers() {
                if let Some(index) = spower.to_index(tvars, tdegree) {
                    if let Some(tcoeff) = target.coeff_mut(index) {
                        *tcoeff = scoeff.clone();
                    }
                }
            }
            Ok(())
        }
    }

    fn add_ref_to<'a, Target>(&'a self, target: &mut Target) -> Result<(), Error>
    where
        Target: PolyCoeffsMut + PolyCoeffsIterMut,
        Target::Coeff: ops::AddAssign<&'a Self::Coeff>,
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
            for (t, s) in iter::zip(target.coeffs_iter_mut(), self.coeffs_iter()) {
                *t += s;
            }
            Ok(())
        } else {
            for (spower, scoeff) in self.coeffs_iter_with_powers() {
                if let Some(index) = spower.to_index(tvars, tdegree) {
                    if let Some(tcoeff) = target.coeff_mut(index) {
                        *tcoeff += scoeff;
                    }
                }
            }
            Ok(())
        }
    }
}

trait EvalCoeffsIter<Value, Coeff, Output> {
    fn init_acc_coeff(&self, coeff: Coeff) -> Output;
    fn init_acc(&self) -> Output;
    fn update_acc_coeff(&self, acc: &mut Output, coeff: Coeff, value: &Value);
    fn update_acc_inner(&self, acc: &mut Output, inner: Output, value: &Value);

    //#[inline]
    //fn eval<P, Values>(&self, poly: &P, values: &Values) -> Output
    //where
    //    P: Poly<Coeff = Coeff> + PolyCoeffsIter,
    //    Values: Sequence<Item = Value> + ?Sized,
    //{
    //    assert!(values.len() >= poly.nvars());
    //    let degree = poly.degree();
    //    self.eval_iter(&mut poly.coeffs_iter(), degree, values)
    //}

    #[inline]
    fn eval_iter<Coeffs, Values>(
        &self,
        coeffs: &mut Coeffs,
        degree: Power,
        values: &Values,
    ) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: SequenceRef<OwnedItem = Value> + IndexableSequence + ?Sized,
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
        coeffs
            .next()
            .map_or_else(|| self.init_acc(), |coeff| self.init_acc_coeff(coeff))
    }

    #[inline]
    fn eval_1d<Coeffs, Values>(&self, coeffs: &mut Coeffs, degree: Power, values: &Values) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: SequenceRef<OwnedItem = Value> + IndexableSequence + ?Sized,
    {
        let mut acc = self.init_acc();
        if let Some(value) = values.get(0) {
            for coeff in coeffs.take(degree as usize + 1) {
                self.update_acc_coeff(&mut acc, coeff, value);
            }
        }
        acc
    }

    #[inline]
    fn eval_2d<Coeffs, Values>(&self, coeffs: &mut Coeffs, degree: Power, values: &Values) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: SequenceRef<OwnedItem = Value> + IndexableSequence + ?Sized,
    {
        let mut acc = self.init_acc();
        if let Some(value) = values.get(1) {
            for p in 0..=degree {
                let inner = self.eval_1d(coeffs, p, values);
                self.update_acc_inner(&mut acc, inner, value);
            }
        }
        acc
    }

    #[inline]
    fn eval_3d<Coeffs, Values>(&self, coeffs: &mut Coeffs, degree: Power, values: &Values) -> Output
    where
        Coeffs: Iterator<Item = Coeff>,
        Values: SequenceRef<OwnedItem = Value> + IndexableSequence + ?Sized,
    {
        let mut acc = self.init_acc();
        if let Some(value) = values.get(2) {
            for p in 0..=degree {
                let inner = self.eval_2d(coeffs, p, values);
                self.update_acc_inner(&mut acc, inner, value);
            }
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
        Values: SequenceRef<OwnedItem = Value> + IndexableSequence + ?Sized,
    {
        if nvars == 3 {
            self.eval_3d(coeffs, degree, values)
        } else {
            let mut acc = self.init_acc();
            if let Some(value) = values.get(nvars - 1) {
                for p in 0..=degree {
                    let inner = self.eval_nd(coeffs, p, values, nvars - 1);
                    self.update_acc_inner(&mut acc, inner, value);
                }
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
    fn init_acc_coeff(&self, coeff: Coeff) -> Value {
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

type PolyVec<Coeff> = ExplicitPoly<sqnc::Wrapper<Vec<Coeff>, ((),)>>;

impl<Value, Coeff, OCoeff> EvalCoeffsIter<Value, Coeff, PolyVec<OCoeff>>
    for EvalCompositionCoeffsIter
where
    Value: PolyCoeffsIter,
    Value::Coeff: Zero,
    for<'a> &'a OCoeff: ops::Mul<&'a Value::Coeff, Output = OCoeff>,
    OCoeff: Zero + ops::AddAssign + ops::AddAssign<Coeff>,
{
    #[inline]
    fn init_acc_coeff(&self, coeff: Coeff) -> PolyVec<OCoeff> {
        let mut acc: PolyVec<OCoeff> = ExplicitPoly::zeros(Variables::none(), 0);
        if let Some(acc_coeff) = acc.coeffs.last_mut() {
            *acc_coeff += coeff;
        }
        acc
    }
    #[inline]
    fn init_acc(&self) -> PolyVec<OCoeff> {
        ExplicitPoly::zeros(Variables::none(), 0)
    }
    #[inline]
    fn update_acc_coeff(&self, acc: &mut PolyVec<OCoeff>, coeff: Coeff, value: &Value) {
        if acc.degree() == 0 {
            if let Some(acc_coeff) = acc.coeffs.last_mut() {
                if acc_coeff.is_zero() {
                    *acc_coeff += coeff;
                    return;
                }
            }
        }
        let mut old_acc = ExplicitPoly::zeros(acc.vars() | value.vars(), acc.degree() + value.degree());
        std::mem::swap(acc, &mut old_acc);
        let _ = (&old_acc * value).add_to(acc);
        if let Some(acc_coeff) = acc.coeffs.last_mut() {
            *acc_coeff += coeff;
        }
    }
    #[inline]
    fn update_acc_inner(
        &self,
        acc: &mut PolyVec<OCoeff>,
        mut inner: PolyVec<OCoeff>,
        value: &Value,
    ) {
        if acc.degree() == 0 && acc.coeffs.get(0).map_or(false, |c| c.is_zero()) {
            std::mem::swap(acc, &mut inner)
        } else {
            let mut old_acc = ExplicitPoly::zeros(
                acc.vars() | inner.vars() | value.vars(),
                std::cmp::max(acc.degree() + value.degree(), inner.degree()),
            );
            std::mem::swap(acc, &mut old_acc);
            let _ = (&old_acc * value).add_to(acc);
            let _ = inner.add_to(acc);
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

    let inner_vars = Variables::from(..from_nvars);
    let outer_vars = Variables::from(..to_nvars);
    let transform_polys: Vec<_> = transform_coeffs
        .chunks_exact(transform_ncoeffs)
        .map(|c| ExplicitPoly::<sqnc::Wrapper<&[f64], ((),)>>::new_unchecked(c, inner_vars, transform_degree))
        .collect();

    let nrows = ncoeffs(from_nvars, row_degree);
    let ncols = ncoeffs(to_nvars, degree);
    let mut matrix: Vec<f64> = Vec::new();
    matrix.resize(nrows * ncols, 0.0);

    for (i, col) in matrix.chunks_exact_mut(nrows).enumerate() {
        let mut col = ExplicitPoly::new_unchecked(col, inner_vars, row_degree);
        let mut outer: PolyVec<f64> = ExplicitPoly::zeros(outer_vars, degree);
        *outer.coeffs.get_mut(i).unwrap() = 1.0;
        EvalCompositionCoeffsIter
            .eval_iter(
                &mut outer.coeffs_iter(),
                outer.degree(),
                transform_polys.as_slice(),
            )
            .assign_to(&mut col)
            .unwrap();
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::{ExplicitPoly, Powers, Variable, Variables};
    use crate::traits::*;
    use approx::assert_abs_diff_eq;
    use std::iter;

    macro_rules! v {
        ($i:literal) => { Variable::new($i).unwrap() };
        [] => { Variables::none() };
        [$i:literal,] => { Variables::from(Variable::new($i).unwrap()) };
        [$i:literal, $($tt:tt)*] => { v![$($tt)*] | v!($i) }
    }

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
        let vars = v![2, 4, 5];
        assert_eq!(vars.index(v!(0)), None);
        assert_eq!(vars.index(v!(1)), None);
        assert_eq!(vars.index(v!(2)), Some(0));
        assert_eq!(vars.index(v!(3)), None);
        assert_eq!(vars.index(v!(4)), Some(1));
        assert_eq!(vars.index(v!(5)), Some(2));
        assert_eq!(vars.index(v!(6)), None);
    }

    #[test]
    fn powers_to_index_to_powers() {
        macro_rules! assert_index_powers {
            ($degree:literal, $powers:tt) => {
                let vars = Variables::from(..$powers[0].len());
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
        assert_eq!(ExplicitPoly::new([1], v![], 0).unwrap().eval(&[] as &[usize]), 1);
    }

    #[test]
    fn eval_1d() {
        assert_eq!(ExplicitPoly::new([1], v![0], 0).unwrap().eval(&[5]), 1);
        assert_eq!(ExplicitPoly::new([2, 1], v![0], 1).unwrap().eval(&[5]), 11);
        assert_eq!(ExplicitPoly::new([3, 2, 1], v![0], 2).unwrap().eval(&[5]), 86);
    }

    #[test]
    fn eval_2d() {
        assert_eq!(ExplicitPoly::new([1], v![0, 1], 0).unwrap().eval(&[5, 3]), 1);
        assert_eq!(ExplicitPoly::new([0, 0, 1], v![0, 1], 1).unwrap().eval(&[5, 3]), 1);
        assert_eq!(ExplicitPoly::new([0, 1, 0], v![0, 1], 1).unwrap().eval(&[5, 3]), 5);
        assert_eq!(ExplicitPoly::new([1, 0, 0], v![0, 1], 1).unwrap().eval(&[5, 3]), 3);
        assert_eq!(ExplicitPoly::new([3, 2, 1], v![0, 1], 1).unwrap().eval(&[5, 3]), 20);
        assert_eq!(
            ExplicitPoly::new([6, 5, 4, 3, 2, 1], v![0, 1], 2)
                .unwrap()
                .eval(&[5, 3]),
            227
        );
    }

    #[test]
    fn eval_3d() {
        assert_eq!(ExplicitPoly::new([1], v![0, 1, 2], 0).unwrap().eval(&[5, 3, 2]), 1);
        assert_eq!(
            ExplicitPoly::new([0, 0, 0, 1], v![0, 1, 2], 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            1
        );
        assert_eq!(
            ExplicitPoly::new([0, 0, 1, 0], v![0, 1, 2], 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            5
        );
        assert_eq!(
            ExplicitPoly::new([0, 1, 0, 0], v![0, 1, 2], 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            3
        );
        assert_eq!(
            ExplicitPoly::new([1, 0, 0, 0], v![0, 1, 2], 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            2
        );
        assert_eq!(
            ExplicitPoly::new([4, 3, 2, 1], v![0, 1, 2], 1)
                .unwrap()
                .eval(&[5, 3, 2]),
            28
        );
        assert_eq!(
            ExplicitPoly::new([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], v![0, 1, 2], 2)
                .unwrap()
                .eval(&[5, 3, 2]),
            415,
        );
    }

    #[test]
    fn eval_ref() {
        let coeffs: Vec<usize> = vec![3, 2, 1];
        let poly = ExplicitPoly::new(&coeffs, v![0], 2).unwrap();
        assert_eq!(poly.eval(&[5]), 86);
    }

    #[test]
    fn partial_deriv() {
        assert_eq!(
            ExplicitPoly::from_assignable(ExplicitPoly::new([1], v![0], 0).unwrap().partial_deriv(v!(0))),
            ExplicitPoly::new(vec![0], v![0], 0).unwrap(),
        );
        assert_eq!(
            ExplicitPoly::from_assignable(ExplicitPoly::new([2, 1], v![0], 1).unwrap().partial_deriv(v!(0))),
            ExplicitPoly::new(vec![2], v![0], 0).unwrap(),
        );
        assert_eq!(
            ExplicitPoly::from_assignable(
                ExplicitPoly::new([4, 3, 2, 1], v![0], 3)
                    .unwrap()
                    .partial_deriv(v!(0))
            ),
            ExplicitPoly::new(vec![12, 6, 2], v![0], 2).unwrap(),
        );
        assert_eq!(
            ExplicitPoly::from_assignable(
                ExplicitPoly::new([6, 5, 4, 3, 2, 1], v![0, 1], 2)
                    .unwrap()
                    .partial_deriv(v!(0))
            ),
            ExplicitPoly::new(vec![5, 6, 2], v![0, 1], 1).unwrap(),
        );
        assert_eq!(
            ExplicitPoly::from_assignable(
                ExplicitPoly::new([6, 5, 4, 3, 2, 1], v![0, 1], 2)
                    .unwrap()
                    .partial_deriv(v!(1))
            ),
            ExplicitPoly::new(vec![12, 5, 4], v![0, 1], 1).unwrap(),
        );
    }

    #[test]
    fn mul() {
        let l = ExplicitPoly::new([2], v![0], 0).unwrap();
        let r = ExplicitPoly::new([3], v![0], 0).unwrap();
        assert_eq!(
            ExplicitPoly::from_assignable(&l * &r),
            ExplicitPoly::new(vec![6], v![0], 0).unwrap(),
        );

        let l = ExplicitPoly::new([2, 1], v![0], 1).unwrap();
        let r = ExplicitPoly::new([4, 3], v![0], 1).unwrap();
        assert_eq!(
            ExplicitPoly::from_assignable(&l * &r),
            ExplicitPoly::new(vec![8, 10, 3], v![0], 2).unwrap(),
        );

        let l = ExplicitPoly::new([3, 2, 1], v![0, 1], 1).unwrap();
        let r = ExplicitPoly::new([6, 5, 4], v![0, 1], 1).unwrap();
        assert_eq!(
            ExplicitPoly::from_assignable(&l * &r),
            ExplicitPoly::new(vec![18, 27, 18, 10, 13, 4], v![0, 1], 2).unwrap(),
        );
    }

    //#[test]
    //fn composition() {
    //    let p = PolySequence::new([2, 1], 0..1, 1).unwrap();
    //    let q = PolySequence::new([4, 3], 1..2, 1).unwrap();
    //    let desired = PolySequence::new(vec![8, 7], 1..2, 1).unwrap();
    //    assert_eq!(EvalCompositionCoeffsIter.eval(p, &[q]), desired,);

    //    let p = PolySequence::new([3, 2, 1], 0..1, 2).unwrap();
    //    let q = PolySequence::new([4, 3], 1..2, 1).unwrap();
    //    let desired = PolySequence::new(vec![48, 80, 34], 1..2, 2).unwrap();
    //    assert_eq!(EvalCompositionCoeffsIter.eval(p, &[q]), desired,);

    //    let p = PolySequence::new([3, 2, 1], 0..2, 1).unwrap();
    //    let q = PolySequence::new([4, 3], 2..3, 1).unwrap();
    //    let r = PolySequence::new([2, 1], 3..4, 1).unwrap();
    //    let desired = PolySequence::new(vec![6, 8, 10], 2..4, 1).unwrap();
    //    assert_eq!(EvalCompositionCoeffsIter.eval(p, &[q, r]), desired,);
    //}

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
    use super::{traits::*, ExplicitPoly, Variable, Variables};

    macro_rules! mk_bench_eval {
        ($name:ident, $degree:literal, $nvars:literal) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let coeffs: Vec<_> = (1..=super::ncoeffs($nvars, $degree))
                    .map(|i| i as f64)
                    .collect();
                let values: Vec<_> = (1..=$nvars).map(|x| x as f64).collect();
                let vars = Variables::from(..$nvars);
                let poly = ExplicitPoly::new(coeffs.as_slice(), vars, $degree).unwrap();
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
        let vars = Variables::from(..3);
        let l = ExplicitPoly::new(
            (0..super::ncoeffs(3, 4))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            vars,
            4,
        )
        .unwrap();
        let r = ExplicitPoly::new(
            (0..super::ncoeffs(3, 2))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            vars,
            2,
        )
        .unwrap();
        let mut target: ExplicitPoly<sqnc::Wrapper<Vec<f64>, ((),)>> = ExplicitPoly::zeros(vars, 6);
        b.iter(|| (test::black_box(&l) * test::black_box(&r)).assign_to(&mut target));
    }

    #[bench]
    fn mul_different_vars_1d_degree4_2d_degree2(b: &mut Bencher) {
        let lvars = Variables::from(..1);
        let rvars = Variables::from(1..3);
        let l = ExplicitPoly::new(
            (0..super::ncoeffs(1, 4))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            lvars,
            4,
        )
        .unwrap();
        let r = ExplicitPoly::new(
            (0..super::ncoeffs(2, 2))
                .into_iter()
                .map(|i| i as f64)
                .collect::<Vec<_>>(),
            rvars,
            2,
        )
        .unwrap();
        let mut target: ExplicitPoly<sqnc::Wrapper<Vec<f64>, ((),)>> = ExplicitPoly::zeros(lvars | rvars, 6);
        b.iter(|| (test::black_box(&l) * test::black_box(&r)).assign_to(&mut target));
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
