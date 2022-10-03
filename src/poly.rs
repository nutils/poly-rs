use crate::traits::*;
use crate::{ncoeffs, Error, IntegerMultiple, Mul, PartialDeriv, Power, Variable, Variables};
use num_traits::Zero;
use sqnc::traits::*;
use sqnc::SequenceWrapper;
use std::iter;
use std::ops;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poly<Coeffs, CoeffsN> {
    pub(crate) coeffs: SequenceWrapper<Coeffs, CoeffsN>,
    vars: Variables,
    degree: Power,
}

impl<Coeffs, CoeffsN> Poly<Coeffs, CoeffsN>
where
    Coeffs: AsSequence<CoeffsN>,
{
    pub fn new<IntoVars>(coeffs: Coeffs, vars: IntoVars, mut degree: Power) -> Result<Self, Error>
    where
        IntoVars: Into<Variables>,
    {
        let mut vars = vars.into();
        // Normalize degree and vars.
        if degree == 0 || vars.is_empty() {
            degree = 0;
            vars = Variables::none();
        }
        if coeffs.as_sequence().len() != ncoeffs(vars.len(), degree) {
            Err(Error::NCoeffsNVarsDegreeMismatch)
        } else {
            Ok(Self::new_unchecked(coeffs, vars, degree))
        }
    }

    #[inline]
    pub(crate) fn new_unchecked(coeffs: Coeffs, vars: Variables, degree: Power) -> Self {
        // TODO: debug asserts
        Self {
            coeffs: coeffs.into(),
            vars,
            degree,
        }
    }
}

impl<Coeff, Coeffs, CoeffsN> Poly<Coeffs, CoeffsN>
where
    Coeff: Zero,
    Coeffs: AsSequence<CoeffsN, Item = Coeff> + FromIterator<Coeff>,
{
    pub fn zeros<IntoVars>(vars: IntoVars, degree: Power) -> Self
    where
        IntoVars: Into<Variables>,
    {
        let vars = vars.into();
        let coeffs = iter::repeat_with(Coeff::zero)
            .take(ncoeffs(vars.len(), degree))
            .collect();
        Self::new_unchecked(coeffs, vars, degree)
    }
}

impl<Coeff, Coeffs, CoeffsN> Poly<Coeffs, CoeffsN>
where
    Coeffs: AsSequence<CoeffsN, Item = Coeff> + FromIterator<Coeff>,
{
    #[inline]
    pub fn from_iter<Source>(source: Source) -> Self
    where
        Source: PolyMeta<Coeff = Coeff> + PolyIntoCoeffsIter,
    {
        let vars = source.vars();
        let degree = source.degree();
        Self::new_unchecked(source.into_coeffs_iter().collect(), vars, degree)
    }
}

impl<Coeff, Coeffs, CoeffsN> Poly<Coeffs, CoeffsN>
where
    Coeff: Zero + ops::AddAssign,
    Coeffs: AsMutSequence<CoeffsN, Item = Coeff> + FromIterator<Coeff>,
    Coeffs::Sequence: RandomAccessSequenceMut + IterableMutSequence,
{
    pub fn from_assignable<Source>(source: Source) -> Self
    where
        Source: PolyMeta<Coeff = Coeff> + PolyAssign,
    {
        let mut result = Self::zeros(source.vars(), source.degree());
        let _ = source.add_to(&mut result);
        result
    }
}

impl<Coeffs, CoeffsN> PolyMeta for Poly<Coeffs, CoeffsN>
where
    Coeffs: AsSequence<CoeffsN>,
{
    type Coeff = Coeffs::Item;

    #[inline]
    fn vars(&self) -> Variables {
        self.vars
    }

    #[inline]
    fn degree(&self) -> Power {
        self.degree
    }
}

impl<Coeffs, CoeffsN> PolyCoeffs for Poly<Coeffs, CoeffsN>
where
    Coeffs: AsSequence<CoeffsN>,
    Coeffs::Sequence: RandomAccessSequence,
{
    #[inline]
    fn coeff(&self, index: usize) -> Option<&Self::Coeff> {
        self.coeffs.get(index)
    }
}

impl<Coeffs, CoeffsN> PolyCoeffsMut for Poly<Coeffs, CoeffsN>
where
    Coeffs: AsMutSequence<CoeffsN>,
    Coeffs::Sequence: RandomAccessSequenceMut,
{
    #[inline]
    fn coeff_mut(&mut self, index: usize) -> Option<&mut Self::Coeff> {
        self.coeffs.get_mut(index)
    }
}

impl<Coeffs, CoeffsN> PolyCoeffsIter for Poly<Coeffs, CoeffsN>
where
    Coeffs: AsSequence<CoeffsN>,
    Coeffs::Sequence: RandomAccessSequence + IterableSequence,
{
    type CoeffsIter<'a> = <Coeffs::Sequence as IterableSequence>::Iter<'a> where Self: 'a;

    #[inline]
    fn coeffs_iter(&self) -> Self::CoeffsIter<'_> {
        self.coeffs.iter()
    }
}

impl<Coeffs, CoeffsN> PolyCoeffsIterMut for Poly<Coeffs, CoeffsN>
where
    Coeffs: AsMutSequence<CoeffsN>,
    Coeffs::Sequence: RandomAccessSequenceMut + IterableMutSequence,
{
    type CoeffsIterMut<'a> = <Coeffs::Sequence as IterableMutSequence>::IterMut<'a> where Self: 'a;

    #[inline]
    fn coeffs_iter_mut(&mut self) -> Self::CoeffsIterMut<'_> {
        self.coeffs.iter_mut()
    }
}

impl<Coeff, Coeffs, CoeffsN> PolyIntoCoeffsIter for Poly<Coeffs, CoeffsN>
where
    Coeffs: AsSequence<CoeffsN, Item = Coeff> + IntoIterator<Item = Coeff>,
    Coeffs::Sequence: RandomAccessSequence,
{
    type IntoCoeffsIter = Coeffs::IntoIter;

    #[inline]
    fn into_coeffs_iter(self) -> Self::IntoCoeffsIter {
        self.coeffs.into_inner().into_iter()
    }
}

impl<Coeff, Coeffs, CoeffsN, PDCoeff> PolyPartialDeriv for Poly<Coeffs, CoeffsN>
where
    Coeffs: AsSequence<CoeffsN, Item = Coeff>,
    Coeffs::Sequence: RandomAccessSequence + IterableSequence,
    Coeff: IntegerMultiple<Output = PDCoeff>,
    for<'coeff> &'coeff Coeff: IntegerMultiple<Output = PDCoeff>,
    PDCoeff: Zero,
{
    type PartialDeriv<'a> = PartialDeriv<&'a Self, Self> where Self: 'a;

    #[inline]
    fn partial_deriv(&self, var: Variable) -> Self::PartialDeriv<'_> {
        PartialDeriv::new(self, var)
    }
}

impl<'l, 'r, LCoeffs, LCoeffsN, RPoly> ops::Mul<&'r RPoly> for &'l Poly<LCoeffs, LCoeffsN>
where
    LCoeffs: AsSequence<LCoeffsN>,
    LCoeffs::Sequence: RandomAccessSequence + IterableSequence,
    RPoly: PolyCoeffsIter,
{
    type Output = Mul<Self, &'r RPoly>;

    fn mul(self, rhs: &'r RPoly) -> Self::Output {
        // NOTE: we don't check if `self.degree() + rhs.degree()` overflows.
        // TODO: impl `Poly::checked_mul()`.
        Mul(self, rhs)
    }
}
