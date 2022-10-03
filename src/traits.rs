use crate::{
    ncoeffs, DefaultEvalCoeffsIter, Error, EvalCoeffsIter, Power, Powers, PowersIter, Variable,
    Variables,
};
use num_traits::Zero;
use sqnc::RandomAccessSequence;
use std::iter;
use std::ops;

pub trait PolyMeta {
    type Coeff;

    fn vars(&self) -> Variables;
    fn degree(&self) -> Power;

    #[inline]
    fn nvars(&self) -> usize {
        self.vars().len()
    }

    #[inline]
    fn ncoeffs(&self) -> usize {
        ncoeffs(self.nvars(), self.degree())
    }
}

pub trait PolyCoeffs: PolyMeta {
    fn coeff(&self, index: usize) -> Option<&Self::Coeff>;
}

pub trait PolyCoeffsMut: PolyMeta {
    fn coeff_mut(&mut self, index: usize) -> Option<&mut Self::Coeff>;
}

pub trait PolyCoeffsIter: PolyMeta {
    type CoeffsIter<'a>: Iterator<Item = &'a Self::Coeff>
    where
        Self: 'a;

    fn coeffs_iter(&self) -> Self::CoeffsIter<'_>;

    #[inline]
    fn coeffs_iter_with_powers(&self) -> iter::Zip<PowersIter, Self::CoeffsIter<'_>> {
        iter::zip(
            Powers::iter_all(self.vars(), self.degree()),
            self.coeffs_iter(),
        )
    }
}

pub trait PolyCoeffsIterMut: PolyMeta {
    type CoeffsIterMut<'a>: Iterator<Item = &'a mut Self::Coeff>
    where
        Self: 'a;

    fn coeffs_iter_mut(&mut self) -> Self::CoeffsIterMut<'_>;

    #[inline]
    fn coeffs_iter_mut_with_powers(&mut self) -> iter::Zip<PowersIter, Self::CoeffsIterMut<'_>> {
        iter::zip(
            Powers::iter_all(self.vars(), self.degree()),
            self.coeffs_iter_mut(),
        )
    }
}

pub trait PolyIntoCoeffsIter: PolyMeta
where
    Self: Sized,
{
    type IntoCoeffsIter: Iterator<Item = Self::Coeff>;

    fn into_coeffs_iter(self) -> Self::IntoCoeffsIter;

    #[inline]
    fn into_coeffs_iter_with_powers(self) -> iter::Zip<PowersIter, Self::IntoCoeffsIter> {
        iter::zip(
            Powers::iter_all(self.vars(), self.degree()),
            self.into_coeffs_iter(),
        )
    }
}

pub trait PolyAssign: PolyMeta
where
    Self: Sized,
{
    /// Assign the coefficients to the target.
    fn assign_to<Target>(self, target: &mut Target) -> Result<(), Error>
    where
        Self::Coeff: Zero,
        Target: PolyMeta<Coeff = Self::Coeff> + PolyCoeffsMut + PolyCoeffsIterMut;

    /// Add the coefficients to the target.
    fn add_to<Target>(self, target: &mut Target) -> Result<(), Error>
    where
        Target: PolyCoeffsMut + PolyCoeffsIterMut,
        Target::Coeff: ops::AddAssign<Self::Coeff>;
}

pub trait PolyAssignRef: PolyMeta {
    /// Assign cloned coefficients to the target.
    fn assign_clone_to<Target>(&self, target: &mut Target) -> Result<(), Error>
    where
        Self::Coeff: Zero + Clone,
        Target: PolyMeta<Coeff = Self::Coeff> + PolyCoeffsMut + PolyCoeffsIterMut;

    /// Add coefficients to the target by reference.
    fn add_ref_to<'a, Target>(&'a self, target: &mut Target) -> Result<(), Error>
    where
        Target: PolyCoeffsMut + PolyCoeffsIterMut,
        Target::Coeff: ops::AddAssign<&'a Self::Coeff>;
}

pub trait PolyEval<Value>: PolyMeta {
    type Output;

    fn eval<Values>(&self, values: &Values) -> Self::Output
    where
        Values: RandomAccessSequence<Item = Value> + ?Sized;
}

pub trait PolyPartialDeriv: PolyMeta {
    type PartialDeriv<'a>: PolyMeta + PolyAssign
    where
        Self: 'a;

    fn partial_deriv(&self, var: Variable) -> Self::PartialDeriv<'_>;
}

impl<P, Value> PolyEval<Value> for P
where
    P: PolyCoeffsIter,
    Value: Zero + ops::AddAssign + for<'c> ops::AddAssign<&'c P::Coeff>,
    for<'v> Value: ops::MulAssign<&'v Value>,
{
    type Output = Value;

    fn eval<Values>(&self, values: &Values) -> Value
    where
        Values: RandomAccessSequence<Item = Value> + ?Sized,
    {
        // TODO: If vars != 0..self.nvars(), wrap `values` in a `Sequence` that
        // gets the appropriate elements.
        assert!(values.len() >= self.nvars());
        DefaultEvalCoeffsIter.eval_iter(&mut self.coeffs_iter(), self.degree(), values)
    }
}
