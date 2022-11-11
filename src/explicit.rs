use crate::traits::*;
use crate::{ncoeffs, Error, IntegerMultiple, Mul, PartialDeriv, Power, Variable, Variables};
use num_traits::Zero;
use sqnc;
use sqnc::traits::*;
use std::iter;
use std::ops;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitPoly<Coeffs> {
    pub(crate) coeffs: Coeffs,
    vars: Variables,
    degree: Power,
}

impl<Coeff, Coeffs, CoeffsN> ExplicitPoly<sqnc::Wrapper<Coeffs, CoeffsN>>
where
    Coeffs: DerefSequence<CoeffsN>,
    Coeffs::Sequence: SequenceRef<OwnedItem = Coeff>,
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
        if coeffs.deref_sqnc().len() != ncoeffs(vars.len(), degree) {
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

    pub fn zeros<IntoVars>(vars: IntoVars, degree: Power) -> Self
    where
        IntoVars: Into<Variables>,
        Coeff: Zero,
        Coeffs: FromIterator<Coeff>,
    {
        let vars = vars.into();
        let coeffs = iter::repeat_with(Coeff::zero)
            .take(ncoeffs(vars.len(), degree))
            .collect();
        Self::new_unchecked(coeffs, vars, degree)
    }

    #[inline]
    pub fn from_iter<Source>(source: Source) -> Self
    where
        Coeffs: FromIterator<Coeff>,
        Source: Poly<Coeff = Coeff> + PolyIntoCoeffsIter,
    {
        let vars = source.vars();
        let degree = source.degree();
        Self::new_unchecked(source.into_coeffs_iter().collect(), vars, degree)
    }

    pub fn from_assignable<Source>(source: Source) -> Self
    where
        Coeff: Zero + ops::AddAssign,
        Coeffs: FromIterator<Coeff> + DerefMutSequence<CoeffsN>,
        Coeffs::Sequence: SequenceRef<OwnedItem = Coeff> + SequenceRefMut + IndexableMutSequence + IterableMutSequence,
        sqnc::Wrapper<Coeffs, CoeffsN>: SequenceRef<OwnedItem = Coeff>,
        Source: Poly<Coeff = Coeff> + PolyAssign,
    {
        let mut result = Self::zeros(source.vars(), source.degree());
        let _ = source.add_to(&mut result);
        result
    }
}

impl<Coeffs> Poly for ExplicitPoly<Coeffs>
where
    Coeffs: SequenceRef,
{
    type Coeff = Coeffs::OwnedItem;

    #[inline]
    fn vars(&self) -> Variables {
        self.vars
    }

    #[inline]
    fn degree(&self) -> Power {
        self.degree
    }
}

impl<Coeffs> PolyCoeffs for ExplicitPoly<Coeffs>
where
    Coeffs: SequenceRef + IndexableSequence,
{
    #[inline]
    fn coeff(&self, index: usize) -> Option<&Self::Coeff> {
        self.coeffs.get(index)
    }
}

impl<Coeffs> PolyCoeffsMut for ExplicitPoly<Coeffs>
where
    Coeffs: SequenceRefMut + IndexableMutSequence,
{
    #[inline]
    fn coeff_mut(&mut self, index: usize) -> Option<&mut Self::Coeff> {
        self.coeffs.get_mut(index)
    }
}

impl<Coeffs> PolyCoeffsIter for ExplicitPoly<Coeffs>
where
    Coeffs: SequenceRef + IterableSequence,
{
    type CoeffsIter<'a> = <Coeffs as SequenceIter<'a>>::Iter where Self: 'a;

    #[inline]
    fn coeffs_iter(&self) -> Self::CoeffsIter<'_> {
        self.coeffs.iter()
    }
}

impl<Coeffs> PolyCoeffsIterMut for ExplicitPoly<Coeffs>
where
    Coeffs: SequenceRefMut + IterableMutSequence,
{
    type CoeffsIterMut<'a> = <Coeffs as SequenceIterMut<'a>>::IterMut where Self: 'a;

    #[inline]
    fn coeffs_iter_mut(&mut self) -> Self::CoeffsIterMut<'_> {
        self.coeffs.iter_mut()
    }
}

impl<Coeff, Coeffs> PolyIntoCoeffsIter for ExplicitPoly<Coeffs>
where
    Coeffs: SequenceRef<OwnedItem = Coeff> + IntoIterator<Item = Coeff>,
{
    type IntoCoeffsIter = Coeffs::IntoIter;

    #[inline]
    fn into_coeffs_iter(self) -> Self::IntoCoeffsIter {
        self.coeffs.into_iter()
    }
}

impl<Coeff, Coeffs, PDCoeff> PolyPartialDeriv for ExplicitPoly<Coeffs>
where
    Coeff: IntegerMultiple<Output = PDCoeff>,
    for<'coeff> &'coeff Coeff: IntegerMultiple<Output = PDCoeff>,
    PDCoeff: Zero,
    Coeffs: SequenceRef<OwnedItem = Coeff> + IndexableSequence + IterableSequence,
{
    type PartialDeriv<'a> = PartialDeriv<&'a Self, Self> where Self: 'a;

    #[inline]
    fn partial_deriv(&self, var: Variable) -> Self::PartialDeriv<'_> {
        PartialDeriv::new(self, var)
    }
}

impl<'l, 'r, LCoeffs, RPoly> ops::Mul<&'r RPoly> for &'l ExplicitPoly<LCoeffs> {
    type Output = Mul<Self, &'r RPoly>;

    fn mul(self, rhs: &'r RPoly) -> Self::Output {
        // NOTE: we don't check if `self.degree() + rhs.degree()` overflows.
        // TODO: impl `Poly::checked_mul()`.
        Mul(self, rhs)
    }
}
