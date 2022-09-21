use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Range, RangeFrom,
    RangeFull, RangeInclusive, RangeTo, RangeToInclusive, Sub, SubAssign,
};

type VariableData = u8;

type VariablesData = u8;

pub const NVARIABLES: usize = VariablesData::BITS as usize;

/// Variable of a polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Variable(VariableData);

impl Variable {
    #[inline]
    pub fn iter_all() -> impl Iterator<Item = Self> + DoubleEndedIterator + ExactSizeIterator {
        (0..NVARIABLES as VariablesData).map(Variable)
    }
    #[inline]
    pub fn new(index: usize) -> Option<Self> {
        (index < NVARIABLES).then_some(Variable(index as VariableData))
    }
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<Variable> for usize {
    #[inline]
    fn from(var: Variable) -> usize {
        var.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variables(VariablesData);

impl Variables {
    pub fn new_first_nth(n: usize) -> Option<Self> {
        (n <= NVARIABLES).then_some(Variables(!(!0 << n)))
    }

    /// Returns the number of variables in the set.
    #[inline]
    pub fn len(self) -> usize {
        let mut len = 0;
        let mut v = self.0 as usize;
        for _ in 0..NVARIABLES {
            len += v & 1;
            v >>= 1;
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
    pub const fn none() -> Self {
        Self(0)
    }

    /// Creates a set containing all variables.
    #[inline]
    pub const fn all() -> Self {
        Self(!0)
    }

    /// Returns `true` if the variable is in the set.
    #[inline]
    pub const fn contains(self, var: Variable) -> bool {
        (self.0 >> var.0) & 1 == 1
    }

    /// Returns the index of the variable in the set or `None` if the variable is not in the set.
    #[inline]
    pub fn index(self, var: Variable) -> Option<usize> {
        if self.contains(var) {
            Some(Variables(self.0 & !(!0 << var.0)).len())
        } else {
            None
        }
    }

    #[inline]
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

    #[inline]
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

    #[inline]
    pub fn iter(self) -> impl Iterator<Item = Variable> + DoubleEndedIterator {
        Variable::iter_all().filter(move |v| self.contains(*v))
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

impl From<Range<Variable>> for Variables {
    #[inline]
    fn from(range: Range<Variable>) -> Self {
        let h = !0 << range.end.index();
        let l = !0 << range.start.index();
        Self((h ^ l) & !h)
    }
}

impl From<RangeFrom<Variable>> for Variables {
    #[inline]
    fn from(range: RangeFrom<Variable>) -> Self {
        Self(!0 << range.start.index())
    }
}

impl From<RangeInclusive<Variable>> for Variables {
    #[inline]
    fn from(range: RangeInclusive<Variable>) -> Self {
        let (start, end) = range.into_inner();
        if end.index() + 1 < NVARIABLES {
            let h = !0 << (end.index() + 1);
            let l = !0 << start.index();
            Self((h ^ l) & !h)
        } else {
            Self(!0 << start.index())
        }
    }
}

impl From<RangeTo<Variable>> for Variables {
    #[inline]
    fn from(range: RangeTo<Variable>) -> Self {
        Self(!(!0 << range.end.index()))
    }
}

impl From<RangeToInclusive<Variable>> for Variables {
    #[inline]
    fn from(range: RangeToInclusive<Variable>) -> Self {
        if range.end.index() + 1 < NVARIABLES {
            Self(!(!0 << (range.end.index() + 1)))
        } else {
            Self(!0)
        }
    }
}

impl From<Range<usize>> for Variables {
    #[inline]
    fn from(range: Range<usize>) -> Self {
        Self::from(range.start..) & Self::from(..range.end)
    }
}

impl From<RangeInclusive<usize>> for Variables {
    #[inline]
    fn from(range: RangeInclusive<usize>) -> Self {
        let (start, end) = range.into_inner();
        Self::from(start..) & Self::from(..=end)
    }
}

impl From<RangeFrom<usize>> for Variables {
    #[inline]
    fn from(range: RangeFrom<usize>) -> Self {
        if range.start < NVARIABLES {
            Self(!0 << range.start)
        } else {
            Self::none()
        }
    }
}

impl From<RangeTo<usize>> for Variables {
    #[inline]
    fn from(range: RangeTo<usize>) -> Self {
        !Self::from(range.end..)
    }
}

impl From<RangeToInclusive<usize>> for Variables {
    #[inline]
    fn from(range: RangeToInclusive<usize>) -> Self {
        if let Some(end) = range.end.checked_add(1) {
            Self::from(..end)
        } else {
            Self::all()
        }
    }
}

impl From<RangeFull> for Variables {
    #[inline]
    fn from(_range: RangeFull) -> Self {
        Self::all()
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

impl Not for Variables {
    type Output = Self;

    #[inline]
    fn not(self) -> Self {
        Self(!self.0 & Self::all().0)
    }
}

impl BitAnd for Variables {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Variables {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Variables {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Variables {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Variables {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Variables {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Sub for Variables {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 & !rhs.0)
    }
}

impl SubAssign for Variables {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 &= !rhs.0;
    }
}

impl From<Variable> for Variables {
    fn from(var: Variable) -> Self {
        Variables(1 << var.0)
    }
}

impl BitAnd<Variable> for Variable {
    type Output = Variables;

    #[inline]
    fn bitand(self, rhs: Variable) -> Variables {
        Variables::from(self) & Variables::from(rhs)
    }
}

impl BitAnd<Variable> for Variables {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Variable) -> Self {
        self & Self::from(rhs)
    }
}

impl BitAndAssign<Variable> for Variables {
    #[inline]
    fn bitand_assign(&mut self, rhs: Variable) {
        *self &= Self::from(rhs)
    }
}

impl BitOr<Variable> for Variable {
    type Output = Variables;

    #[inline]
    fn bitor(self, rhs: Variable) -> Variables {
        Variables::from(self) | Variables::from(rhs)
    }
}

impl BitOr<Variable> for Variables {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Variable) -> Self {
        self | Self::from(rhs)
    }
}

impl BitOrAssign<Variable> for Variables {
    #[inline]
    fn bitor_assign(&mut self, rhs: Variable) {
        *self |= Self::from(rhs)
    }
}

impl Sub<Variable> for Variables {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Variable) -> Self {
        self - Self::from(rhs)
    }
}

impl SubAssign<Variable> for Variables {
    #[inline]
    fn sub_assign(&mut self, rhs: Variable) {
        *self -= Self::from(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::{Variable, Variables, NVARIABLES};

    const V: [Variable; NVARIABLES] = [
        Variable(0),
        Variable(1),
        Variable(2),
        Variable(3),
        Variable(4),
        Variable(5),
        Variable(6),
        Variable(7),
    ];

    #[test]
    fn variables_from_range() {
        assert_eq!(Variables::from(V[2]..V[1]), Variables(0b00000000));
        assert_eq!(Variables::from(V[0]..V[0]), Variables(0b00000000));
        assert_eq!(Variables::from(V[0]..V[1]), Variables(0b00000001));
        assert_eq!(Variables::from(V[1]..V[1]), Variables(0b00000000));
        assert_eq!(Variables::from(V[2]..V[5]), Variables(0b00011100));
        assert_eq!(Variables::from(V[4]..V[7]), Variables(0b01110000));
    }

    #[test]
    fn variables_from_range_from() {
        assert_eq!(Variables::from(V[0]..), Variables(0b11111111));
        assert_eq!(Variables::from(V[2]..), Variables(0b11111100));
        assert_eq!(Variables::from(V[7]..), Variables(0b10000000));
    }

    #[test]
    fn variables_from_range_inclusive() {
        assert_eq!(Variables::from(V[2]..=V[1]), Variables(0b00000000));
        assert_eq!(Variables::from(V[0]..=V[0]), Variables(0b00000001));
        assert_eq!(Variables::from(V[0]..=V[1]), Variables(0b00000011));
        assert_eq!(Variables::from(V[1]..=V[1]), Variables(0b00000010));
        assert_eq!(Variables::from(V[2]..=V[5]), Variables(0b00111100));
        assert_eq!(Variables::from(V[4]..=V[7]), Variables(0b11110000));
    }

    #[test]
    fn variables_from_range_to() {
        assert_eq!(Variables::from(..V[0]), Variables(0b00000000));
        assert_eq!(Variables::from(..V[1]), Variables(0b00000001));
        assert_eq!(Variables::from(..V[5]), Variables(0b00011111));
        assert_eq!(Variables::from(..V[7]), Variables(0b01111111));
    }

    #[test]
    fn variables_from_range_to_inclusive() {
        assert_eq!(Variables::from(..=V[0]), Variables(0b00000001));
        assert_eq!(Variables::from(..=V[1]), Variables(0b00000011));
        assert_eq!(Variables::from(..=V[5]), Variables(0b00111111));
        assert_eq!(Variables::from(..=V[7]), Variables(0b11111111));
    }

    #[test]
    fn variables_from_range_usize() {
        assert_eq!(Variables::from(2..1), Variables(0b00000000));
        assert_eq!(Variables::from(9..8), Variables(0b00000000));
        assert_eq!(Variables::from(0..1), Variables(0b00000001));
        assert_eq!(Variables::from(0..7), Variables(0b01111111));
        assert_eq!(Variables::from(0..8), Variables(0b11111111));
        assert_eq!(Variables::from(0..9), Variables(0b11111111));
        assert_eq!(Variables::from(1..1), Variables(0b00000000));
        assert_eq!(Variables::from(2..5), Variables(0b00011100));
        assert_eq!(Variables::from(4..7), Variables(0b01110000));
    }

    #[test]
    fn variables_from_range_inclusive_usize() {
        assert_eq!(Variables::from(2..=1), Variables(0b00000000));
        assert_eq!(Variables::from(9..=8), Variables(0b00000000));
        assert_eq!(Variables::from(0..=0), Variables(0b00000001));
        assert_eq!(Variables::from(0..=6), Variables(0b01111111));
        assert_eq!(Variables::from(0..=7), Variables(0b11111111));
        assert_eq!(Variables::from(0..=8), Variables(0b11111111));
        assert_eq!(Variables::from(1..=0), Variables(0b00000000));
        assert_eq!(Variables::from(2..=4), Variables(0b00011100));
        assert_eq!(Variables::from(4..=6), Variables(0b01110000));
    }

    #[test]
    fn variables_from_range_from_usize() {
        assert_eq!(Variables::from(0..), Variables(0b11111111));
        assert_eq!(Variables::from(2..), Variables(0b11111100));
        assert_eq!(Variables::from(7..), Variables(0b10000000));
        assert_eq!(Variables::from(8..), Variables(0b00000000));
    }

    #[test]
    fn variables_from_range_to_usize() {
        assert_eq!(Variables::from(..0), Variables(0b00000000));
        assert_eq!(Variables::from(..1), Variables(0b00000001));
        assert_eq!(Variables::from(..5), Variables(0b00011111));
        assert_eq!(Variables::from(..7), Variables(0b01111111));
        assert_eq!(Variables::from(..8), Variables(0b11111111));
        assert_eq!(Variables::from(..9), Variables(0b11111111));
    }

    #[test]
    fn variables_from_range_to_inclusive_usize() {
        assert_eq!(Variables::from(..=0), Variables(0b00000001));
        assert_eq!(Variables::from(..=1), Variables(0b00000011));
        assert_eq!(Variables::from(..=5), Variables(0b00111111));
        assert_eq!(Variables::from(..=7), Variables(0b11111111));
        assert_eq!(Variables::from(..=8), Variables(0b11111111));
    }

    #[test]
    fn variables_from_range_full() {
        assert_eq!(Variables::from(..), Variables(0b11111111));
    }
}
