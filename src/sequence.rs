//! Traits for sequences.
//!
//! The [`Sequence`] and [`SequenceMut`] traits define interfaces for sequences
//! of uniform elements. The traits are implemented for
//!
//! *   [`slice`]
//! *   [`array`],
//! *   [`Vec`],
//! *   [`std::collections::VecDeque`],
//! *   [`Box`], [`std::rc::Rc`] and [`std::sync::Arc`] of a sequence,
//! *   [`ndarray::Array1`] (requires feature `ndarray`),
//! *   [`smallvec::SmallVec`] (requires feature `smallvec`).
//!
//! # Example
//!
//! ```
//! use nutils_poly::sequence::{Sequence, SequenceMut};
//!
//! fn take<Seq, Idx, Out>(seq: &Seq, idx: &Idx) -> Option<Out>
//! where
//!     Seq: Sequence,
//!     Idx: Sequence<Item = usize>,
//!     Seq::Item: Clone,
//!     Out: FromIterator<Seq::Item>,
//! {
//!     idx.iter().map(|i| seq.get(*i).cloned()).collect()
//! }
//!
//! assert_eq!(
//!     take(&['a', 'e', 'k', 'n', 't', '!'], &[4, 0, 2, 1, 3, 5]),
//!     Some(String::from("taken!")),
//! );
//! ```

use std::collections::vec_deque;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::slice;
use std::sync::Arc;

/// An interface for sequences.
pub trait Sequence {
    /// The type of the elements of the sequence.
    type Item;

    /// The return type of [`Sequence::iter()`].
    type Iter<'a>: Iterator<Item = &'a Self::Item>
    where
        Self: 'a;

    /// Returns the number of elements in the sequence.
    fn len(&self) -> usize;

    /// Returns `true` if the sequence is empty.
    #[inline]
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
    fn iter(&self) -> Self::Iter<'_>;
}

/// An interface for sequences with mutable elements.
pub trait SequenceMut: Sequence {
    /// The return type of [`Sequence::iter()`].
    type IterMut<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

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

    /// Fills `self` with elements by cloning `value`.
    #[inline]
    fn fill(&mut self, value: Self::Item)
    where
        Self::Item: Clone,
    {
        self.fill_with(|| value.clone());
    }

    /// Fills `self` with elements returned by calling a closure repeatedly.
    #[inline]
    fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> Self::Item,
    {
        self.iter_mut().for_each(|v| *v = f());
    }

    /// Returns an iterator over the sequence that allows modifying each element.
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

macro_rules! test_impl {
    {$mod:ident, $S:ty, |$s:ident| $from_slice_body:expr $(; $($body:tt)*)?} => {
        #[cfg(test)]
        mod $mod {
            $($($body)*)?

            use $crate::{Sequence, SequenceMut};

            fn from_slice($s: &[usize]) -> $S { $from_slice_body }

            fn to_vec(s: $S) -> Vec<usize> {
                Sequence::iter(&s).copied().collect()
            }

            #[test]
            fn len() {
                let x = from_slice(&[3, 4, 5]);
                assert_eq!(Sequence::len(&x), 3);
            }

            #[test]
            fn is_empty() {
                let x = from_slice(&[3, 4, 5]);
                assert_eq!(Sequence::is_empty(&x), false);
                let y = from_slice(&[]);
                assert_eq!(Sequence::is_empty(&y), true);
            }

            #[test]
            fn get() {
                let x = from_slice(&[3, 4, 5]);
                assert_eq!(Sequence::get(&x, 1), Some(&4));
                assert_eq!(Sequence::get(&x, 3), None);
            }

            #[test]
            fn first() {
                let x = from_slice(&[3, 4, 5]);
                assert_eq!(Sequence::first(&x), Some(&3));
                let y = from_slice(&[]);
                assert_eq!(Sequence::first(&y), None);
            }

            #[test]
            fn last() {
                let x = from_slice(&[3, 4, 5]);
                assert_eq!(Sequence::last(&x), Some(&5));
                let y = from_slice(&[]);
                assert_eq!(Sequence::last(&y), None);
            }

            #[test]
            fn iter() {
                let x = from_slice(&[3, 4, 5]);
                let mut iter = Sequence::iter(&x);
                assert_eq!(iter.next(), Some(&3));
                assert_eq!(iter.next(), Some(&4));
                assert_eq!(iter.next(), Some(&5));
                assert_eq!(iter.next(), None);
            }

            #[test]
            fn get_mut() {
                let mut x = from_slice(&[3, 4, 5]);
                *SequenceMut::get_mut(&mut x, 1).unwrap() = 6;
                assert_eq!(to_vec(x), vec![3, 6, 5]);
            }

            #[test]
            fn first_mut() {
                let mut x = from_slice(&[3, 4, 5]);
                *SequenceMut::first_mut(&mut x).unwrap() = 6;
                assert_eq!(to_vec(x), vec![6, 4, 5]);
                let mut y = from_slice(&[]);
                assert_eq!(SequenceMut::first_mut(&mut y), None);
            }

            #[test]
            fn last_mut() {
                let mut x = from_slice(&[3, 4, 5]);
                *SequenceMut::last_mut(&mut x).unwrap() = 6;
                assert_eq!(to_vec(x), vec![3, 4, 6]);
                let mut y = from_slice(&[]);
                assert_eq!(SequenceMut::last_mut(&mut y), None);
            }

            #[test]
            fn fill() {
                let mut x = from_slice(&[3, 4, 5]);
                SequenceMut::fill(&mut x, 6);
                assert_eq!(to_vec(x), vec![6, 6, 6]);
            }

            #[test]
            fn fill_with() {
                let mut x = from_slice(&[3, 4, 5]);
                let mut v = 5;
                SequenceMut::fill_with(&mut x, || { v += 1; v });
                assert_eq!(to_vec(x), vec![6, 7, 8]);
            }

            #[test]
            fn iter_mut() {
                let mut x = from_slice(&[3, 4, 5]);
                SequenceMut::iter_mut(&mut x).for_each(|v| *v += 3);
                assert_eq!(to_vec(x), vec![6, 7, 8]);
            }
        }
    };
}

test_impl! {test_defaults, Minimal, |s| Minimal(s.to_vec());
    use std::slice;

    struct Minimal(Vec<usize>);

    impl Sequence for Minimal {
        type Item = usize;
        type Iter<'a> = slice::Iter<'a, usize> where Self: 'a;

        fn len(&self) -> usize {
            self.0.len()
        }

        fn get(&self, index: usize) -> Option<&Self::Item> {
            self.0.get(index)
        }

        fn iter(&self) -> Self::Iter<'_> {
            self.0.iter()
        }
    }

    impl SequenceMut for Minimal {
        type IterMut<'a> = slice::IterMut<'a, usize> where Self: 'a;

        fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item> {
            self.0.get_mut(index)
        }

        fn iter_mut(&mut self) -> Self::IterMut<'_> {
            self.0.iter_mut()
        }
    }
}

macro_rules! redir {
    ($Self:ident, $self:ident, $as_sequence:expr) => {
        #[inline]
        fn len(&$self) -> usize {
            $as_sequence.len()
        }
        #[inline]
        fn is_empty(&$self) -> bool {
            $as_sequence.is_empty()
        }
        #[inline]
        fn get(&$self, index: usize) -> Option<&$Self::Item> {
            $as_sequence.get(index)
        }
        #[inline]
        fn first(&$self) -> Option<&$Self::Item> {
            $as_sequence.first()
        }
        #[inline]
        fn last(&$self) -> Option<&$Self::Item> {
            $as_sequence.last()
        }
        #[inline]
        fn iter(&$self) -> $Self::Iter<'_> {
            $as_sequence.iter()
        }
    };
}

macro_rules! redir_mut {
    ($Self:ident, $self:ident, $as_sequence_mut:expr) => {
        #[inline]
        fn get_mut(&mut $self, index: usize) -> Option<&mut $Self::Item> {
            $as_sequence_mut.get_mut(index)
        }
        #[inline]
        fn first_mut(&mut $self) -> Option<&mut $Self::Item> {
            $as_sequence_mut.first_mut()
        }
        #[inline]
        fn last_mut(&mut $self) -> Option<&mut $Self::Item> {
            $as_sequence_mut.last_mut()
        }
        #[inline]
        fn fill(&mut $self, value: $Self::Item)
        where
            $Self::Item: Clone,
        {
            $as_sequence_mut.fill(value);
        }
        #[inline]
        fn fill_with<F>(&mut $self, f: F)
        where
            F: FnMut() -> $Self::Item,
        {
            $as_sequence_mut.fill_with(f);
        }
        #[inline]
        fn iter_mut(&mut $self) -> $Self::IterMut<'_> {
            $as_sequence_mut.iter_mut()
        }
    };
}

impl<T> Sequence for [T] {
    type Item = T;
    type Iter<'a> = slice::Iter<'a, T> where Self: 'a;
    redir! {Self, self, self}
}

impl<T> SequenceMut for [T] {
    type IterMut<'a> = slice::IterMut<'a, T> where Self: 'a;
    redir_mut! {Self, self, self}
}

impl<T, const N: usize> Sequence for [T; N] {
    type Item = T;
    type Iter<'a> = slice::Iter<'a, T> where Self: 'a;
    redir! {Self, self, self.as_slice()}
}

impl<T, const N: usize> SequenceMut for [T; N] {
    type IterMut<'a> = slice::IterMut<'a, T> where Self: 'a;
    redir_mut! {Self, self, self.as_mut_slice()}
}

impl<T> Sequence for Vec<T> {
    type Item = T;
    type Iter<'a> = slice::Iter<'a, T> where Self: 'a;
    redir! {Self, self, self.as_slice()}
}

impl<T> SequenceMut for Vec<T> {
    type IterMut<'a> = slice::IterMut<'a, T> where Self: 'a;
    redir_mut! {Self, self, self.as_mut_slice()}
}

test_impl! {test_vec, Vec<usize>, |s| s.to_vec()}

impl<S: Sequence + ?Sized> Sequence for Box<S> {
    type Item = S::Item;
    type Iter<'a> = S::Iter<'a> where Self: 'a;
    redir! {Self, self, self.deref()}
}

impl<S: SequenceMut + ?Sized> SequenceMut for Box<S> {
    type IterMut<'a> = S::IterMut<'a> where Self: 'a;
    redir_mut! {Self, self, self.deref_mut()}
}

test_impl! {test_rc, Box<[usize]>, |s| s.into()}

impl<S: Sequence + ?Sized> Sequence for Rc<S> {
    type Item = S::Item;
    type Iter<'a> = S::Iter<'a> where Self: 'a;
    redir! {Self, self, self.deref()}
}

impl<S: Sequence + ?Sized> Sequence for Arc<S> {
    type Item = S::Item;
    type Iter<'a> = S::Iter<'a> where Self: 'a;
    redir! {Self, self, self.deref()}
}

impl<T> Sequence for vec_deque::VecDeque<T> {
    type Item = T;
    type Iter<'a> = vec_deque::Iter<'a, T> where Self: 'a;

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Item> {
        self.get(index)
    }

    #[inline]
    fn first(&self) -> Option<&Self::Item> {
        self.front()
    }

    #[inline]
    fn last(&self) -> Option<&Self::Item> {
        self.back()
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}

impl<T> SequenceMut for vec_deque::VecDeque<T> {
    type IterMut<'a> = vec_deque::IterMut<'a, T> where Self: 'a;

    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item> {
        self.get_mut(index)
    }

    #[inline]
    fn first_mut(&mut self) -> Option<&mut Self::Item> {
        self.front_mut()
    }

    #[inline]
    fn last_mut(&mut self) -> Option<&mut Self::Item> {
        self.back_mut()
    }

    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.iter_mut()
    }
}

test_impl! {test_vec_deque, super::vec_deque::VecDeque<usize>, |s| s.to_vec().into()}

#[cfg(feature = "ndarray")]
mod impl_ndarray {
    use super::{Sequence, SequenceMut};
    use ndarray::iter::{Iter, IterMut};
    use ndarray::{ArrayBase, Data, DataMut, Ix1};

    impl<S: Data> Sequence for ArrayBase<S, Ix1> {
        type Item = S::Elem;
        type Iter<'a> = Iter<'a, S::Elem, Ix1> where Self: 'a;

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }
        #[inline]
        fn get(&self, index: usize) -> Option<&Self::Item> {
            self.get(index)
        }
        #[inline]
        fn iter(&self) -> Self::Iter<'_> {
            self.iter()
        }
    }

    impl<S: Data + DataMut> SequenceMut for ArrayBase<S, Ix1> {
        type IterMut<'a> = IterMut<'a, S::Elem, Ix1> where Self: 'a;

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
        fn iter_mut(&mut self) -> Self::IterMut<'_> {
            self.iter_mut()
        }
    }

    test_impl! {test, ndarray::Array1<usize>, |s| ndarray::Array1::from_iter(s.iter().cloned())}
}

#[cfg(feature = "smallvec")]
mod impl_smallvec {
    use super::{Sequence, SequenceMut};
    use smallvec::{Array, SmallVec};
    use std::slice;

    impl<A: Array> Sequence for SmallVec<A> {
        type Item = A::Item;
        type Iter<'a> = slice::Iter<'a, A::Item> where Self: 'a;
        redir! {Self, self, self.as_slice()}
    }

    impl<A: Array> SequenceMut for SmallVec<A> {
        type IterMut<'a> = slice::IterMut<'a, A::Item> where Self: 'a;
        redir_mut! {Self, self, self.as_mut_slice()}
    }

    test_impl! {test, smallvec::SmallVec<[usize; 4]>, |s| smallvec::SmallVec::from_slice(s)}
}
