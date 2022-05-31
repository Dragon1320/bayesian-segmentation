from cffi import FFI
from os import path

LIB_DIR = "lib/target/release"

ffi = FFI()

ffi.cdef("""
  size_t find_breakpoints(size_t* seq, size_t seq_len, size_t* buf, size_t buf_len, size_t kmax);
""")

lib_path = path.join(LIB_DIR, "libseqseg.so")
lib = ffi.dlopen(lib_path)

kmax = 10

def find_breakpoints(seq):
  seq_len = len(seq)
  seq_buf = ffi.new("size_t[]", seq)

  brk_len = kmax
  brk_buf = ffi.new("size_t[]", brk_len)

  num_brks = lib.find_breakpoints(seq_buf, seq_len, brk_buf, brk_len, kmax)
  brk_locs = list(brk_buf)[:num_brks]

  return num_brks, brk_locs
