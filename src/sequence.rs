//! Traits for dynamically sized sequences.

// Workaround for associated type `Iter<'a>: Iterator<Item = &'a T>` of
// `Sequence` from
// https://web.archive.org/web/20220530082425/https://sabrinajewson.org/blog/the-better-alternative-to-lifetime-gats#the-better-gats

/// An interface for dynamically sized sequences.
pub trait Sequence
where
    Self: for<'me> SequenceIterType<'me, &'me Self::Item>,
{
    /// The type of the elements of the sequence.
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

/// An interface for dynamically sized sequences with mutable elements.
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
