pub fn np_argmax<T>(slice: &[T]) -> usize
where
  T: PartialOrd,
{
  let (idx, _) = slice
    .iter()
    .enumerate()
    .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
    .unwrap();

  idx
}
