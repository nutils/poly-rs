//! Traits for dynamically sized sequences.

pub trait SequenceIter {
    /// The type of the elements of the sequence.
    type Item;
    type Iter<'a>: Iterator<Item = &'a Self::Item> where <Self as SequenceIter>::Item: 'a;
}

/// An interface for dynamically sized sequences.
pub trait Sequence: SequenceIter {

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
    fn iter(&self) -> Self::Iter<'_>;
}

pub trait SequenceIterMut: Sequence {
    type IterMut<'a>: Iterator<Item = &'a mut Self::Item> where <Self as SequenceIter>::Item: 'a;
}

/// An interface for dynamically sized sequences with mutable elements.
pub trait SequenceMut: Sequence + SequenceIterMut {

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

macro_rules! impl_sequence_for_as_ref_slice {
    ($T:ident, $ty:ty, <$($params:tt)*) => {
        impl<$($params)* SequenceIter for $ty {
            type Item = $T;
            type Iter<'a> = std::slice::Iter<'a, $T> where Self::Item: 'a;
        }

        impl<$($params)* Sequence for $ty {
            #[inline]
            fn len(&self) -> usize {
                <Self as AsRef::<[$T]>>::as_ref(self).len()
            }
            #[inline]
            fn get(&self, index: usize) -> Option<&$T> {
                <Self as AsRef::<[$T]>>::as_ref(self).get(index)
            }
            #[inline]
            fn iter(&self) -> Self::Iter<'_> {
                <Self as AsRef::<[$T]>>::as_ref(self).iter()
            }
        }

        impl<$($params)* SequenceIterMut for $ty {
            type IterMut<'a> = std::slice::IterMut<'a, $T> where Self::Item: 'a;
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
            fn iter_mut(&mut self) -> Self::IterMut<'_> {
                <Self as AsMut::<[$T]>>::as_mut(self).iter_mut()
            }
        }
    };
}

impl_sequence_for_as_ref_slice! {T, [T], <T>}
impl_sequence_for_as_ref_slice! {T, [T; N], <T, const N: usize>}
impl_sequence_for_as_ref_slice! {T, Vec<T>, <T>}
impl_sequence_for_as_ref_slice! {T, Box<[T]>, <T>}

impl<S: SequenceIter + ?Sized> SequenceIter for &S {
    type Item = S::Item;
    type Iter<'a> = S::Iter<'a> where Self::Item: 'a;
}

impl<S: Sequence + ?Sized> Sequence for &S {

    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Item> {
        (**self).get(index)
    }
    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        (**self).iter()
    }
}

impl<S: SequenceIter + ?Sized> SequenceIter for &mut S {
    type Item = S::Item;
    type Iter<'a> = S::Iter<'a> where Self::Item: 'a;
}

impl<S: Sequence + ?Sized> Sequence for &mut S {

    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }
    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Item> {
        (**self).get(index)
    }
    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        (**self).iter()
    }
}

impl<S: SequenceMut + ?Sized> SequenceIterMut for &mut S {
    type IterMut<'a> = S::IterMut<'a> where Self::Item: 'a;
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
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        (**self).iter_mut()
    }
}

#[cfg(feature = "ndarray")]
mod impl_ndarray {
    use super::{Sequence, SequenceIterMut, SequenceIter, SequenceMut};
    use ndarray::iter::{Iter, IterMut};
    use ndarray::{ArrayBase, Data, DataMut, Ix1};

    impl<S: Data> SequenceIter for ArrayBase<S, Ix1> {
        type Item = S::Elem;
        type Iter<'a> = Iter<'a, S::Elem, Ix1> where Self::Item: 'a;
    }

    impl<S: Data> Sequence for ArrayBase<S, Ix1> {
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

    impl<S: Data + DataMut> SequenceIterMut for ArrayBase<S, Ix1> {
        type IterMut<'a> = IterMut<'a, S::Elem, Ix1> where Self::Item: 'a;
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
        fn iter_mut(&mut self) -> Self::IterMut<'_> {
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
