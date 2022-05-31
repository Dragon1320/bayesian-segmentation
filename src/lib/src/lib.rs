#[macro_use]
extern crate lazy_static;

use std::{ptr, slice, mem::MaybeUninit};

use math::gamma;
use ndarray::Array3;
use rug::Float;
use prior::{pri_num_brks, pri_segmentation};
use util::np_argmax;

mod math;
mod prior;
mod util;

const PRI_A: f64 = 1.0;
const PRI_B: f64 = 1.0;

// TODO: probably more safety, assertions, etc
// TODO: can i optimise this more? compute shaders?

/// # Safety
/// its not
#[no_mangle]
pub unsafe extern "C" fn find_breakpoints(
  seq: *const usize,
  seq_len: usize,
  buf: *mut usize,
  buf_len: usize,
  kmax: usize,
) -> usize {
  let seq = slice::from_raw_parts(seq, seq_len);

  // validate input once at the start
  let ht_count = seq.iter().filter(|&n| *n == 0 || *n == 1).count();

  assert!(seq.len() == ht_count);

  // allocate dp array withing initialising values
  // less than half of this will actually be filled with our algo
  let mut dp = Array3::<Float>::uninit((kmax + 1, seq.len() + 1, seq.len() + 1));

  // precalculate any constant parts
  let pri_part = gamma(PRI_A + PRI_B) / (gamma(PRI_A) * gamma(PRI_B));

  // precalculate all gamma fn values
  // im very much assuming that PRI_A == PRI_B == 1.0
  let mut pre_gamma = Vec::with_capacity(seq.len() + 3);

  for i in 0..seq.len() + 3 {
    pre_gamma.push(Float::with_val(24, i as f64).gamma());
  }

  for j in 0..seq.len() {
    // we can track the number of heads by observing the values we loop through
    // this way we dont need to re-calculate this every sequence
    let mut h = 0;

    for (v, _) in seq.iter().enumerate().skip(j) {
      let slice_len = (v + 1) - j;

      if seq[v] == 1 {
        h += 1;
      }

      // again, assuming PRI_A == PRI_B == 1.0
      let t1 = &pre_gamma[h + PRI_A as usize];
      let t2 = &pre_gamma[slice_len - h + PRI_B as usize];
      let t3 = &pre_gamma[slice_len + (PRI_A + PRI_B) as usize];

      let mut p = Float::with_val(24, t1 * t2);
      p /= t3;
      p *= &pri_part;

      dp[[0, j, v + 1]] = MaybeUninit::new(p);
    }
  }

  // calculate all other values of k until kmax
  for k in 1..kmax + 1 {
    for j in 0..seq.len() - k {
      let mut p = Float::new(24);

      for s in j + 1..seq.len() - (k - 1) {
        let left = dp[[0, j, s]].assume_init_ref();
        let right = dp[[k - 1, s, seq.len()]].assume_init_ref();

        p += left * right;
      }

      dp[[k, j, seq.len()]] = MaybeUninit::new(p);
    }
  }

  // we dont care abour P(R) here - its only used for normalisation
  let mut p_brks = vec![];

  for k in 0..kmax + 1 {
    let p_calc = dp[[k, 0, seq.len()]].assume_init_ref() * pri_segmentation(seq.len() - 1, k);
    let pk = pri_num_brks(k);

    let p_brk = Float::with_val(24, p_calc) * pk;

    p_brks.push(p_brk);
  }

  let num_brks = np_argmax(&p_brks);

  // calculate brk locations
  let mut m = 0;
  let brk_locs = slice::from_raw_parts_mut(buf, buf_len);

  for q in (1..num_brks + 1).rev() {
    let mut p_seg = vec![];

    for j in m + 1..seq.len() - q {
      let seq_len_right = seq.len() - j;
      let pri_segmentation_right = pri_segmentation(seq_len_right - 1, q - 1);

      let left = dp[[0, m, j]].assume_init_ref();
      let right = dp[[q - 1, j, seq.len()]].assume_init_ref() * pri_segmentation_right;

      let p = Float::with_val(24, right) * left;

      p_seg.push(p);
    }

    let idx = np_argmax(&p_seg);

    m += idx;
    brk_locs[num_brks - q] = m;
  }

  // free memory
  for j in 0..seq.len() {
    for (v, _) in seq.iter().enumerate().skip(j) {
      ptr::drop_in_place(dp[[0, j, v + 1]].as_mut_ptr());
    }
  }

  for k in 1..kmax + 1 {
    for j in 0..seq.len() - k {
      ptr::drop_in_place(dp[[k, j, seq.len()]].as_mut_ptr());
    }
  }

  num_brks
}
