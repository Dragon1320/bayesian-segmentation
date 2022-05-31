use crate::math::binomial;

pub fn pri_num_brks(num_brks: usize) -> f64 {
  0.5 / (num_brks + 1) as f64
}

// TODO: apply len - 1 here
pub fn pri_segmentation(seq_len: usize, num_brks: usize) -> f64 {
  1.0 / binomial(seq_len as u64, num_brks as u64) as f64
}
