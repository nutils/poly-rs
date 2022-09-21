use crate::variable::{Variable, Variables, NVARIABLES};
use std::iter;
use std::ops::{Index, IndexMut};

pub type Power = u8;

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
            len: crate::ncoeffs(vars.len(), degree),
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
                #[allow(clippy::mut_range_bound)]
                for i in 0..=degree {
                    let n = crate::ncoeffs(nvars, i);
                    if index < n {
                        result[var] += degree - i;
                        degree = i;
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
        for (var, power) in iter::zip(Variable::iter_all(), self.iter()).rev() {
            if vars.contains(var) {
                nvars -= 1;
                degree = degree.checked_sub(power)?;
                index += crate::ncoeffs_sum(nvars, degree);
            } else if power > 0 {
                return None;
            }
        }
        Some(index)
    }
    #[inline]
    pub fn iter(self) -> std::array::IntoIter<Power, NVARIABLES> {
        unsafe { self.vec.into_iter() }
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
            Self {
                int: self.int.overflowing_add(rhs.int).0,
            }
        }
    }
}

impl std::fmt::Debug for Powers {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe { self.vec.fmt(f) }
    }
}

impl std::cmp::PartialEq for Powers {
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.int == other.int }
    }
}

impl Index<Variable> for Powers {
    type Output = Power;

    #[inline]
    fn index(&self, var: Variable) -> &Power {
        unsafe { &self.vec[var.index() as usize] }
    }
}

impl IndexMut<Variable> for Powers {
    #[inline]
    fn index_mut(&mut self, var: Variable) -> &mut Power {
        unsafe { &mut self.vec[var.index() as usize] }
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
