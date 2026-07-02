//! Fast scalar pretokenizer for the GPT-2 (r50k_base) regex:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! Uses SWAR (u64) for letter runs + arithmetic predicates. The hot path
//! (space + letters / bare letters) is fully inlined in `advance`.

use super::{
    decode_non_ascii, is_ascii_ws, is_digit, is_letter, scan_digits_from, scan_letters_from,
    scan_other_from,
};
use crate::pretokenize::{Pretoken, unicode};

// -----------------------------------------------------------------------
// FastR50kPretokenizer
// -----------------------------------------------------------------------

pub struct FastR50kPretokenizer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FastR50kPretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    /// Used by the Python bindings, which re-borrow the underlying buffer on
    /// every `__next__` call.
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline(always)]
    fn scan_letters(&mut self) {
        self.pos = scan_letters_from(self.bytes, self.pos);
    }

    #[inline(always)]
    fn scan_digits(&mut self) {
        self.pos = scan_digits_from(self.bytes, self.pos);
    }

    #[inline(always)]
    fn scan_other(&mut self) {
        self.pos = scan_other_from(self.bytes, self.pos);
    }

    #[inline(always)]
    fn advance_whitespace(&mut self, start: usize) {
        self.pos = advance_ws(self.bytes, self.pos, start);
    }

    /// Advance past one token. self.pos must be < self.bytes.len().
    /// Uses direct comparison chains instead of LUT + jump table to avoid
    /// GOT indirection and improve branch prediction on common patterns.
    #[inline(always)]
    fn advance(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();
        let start = self.pos;
        let b0 = unsafe { *bytes.get_unchecked(start) };

        // Hot path 1: ASCII letter (~40% of tokens)
        if is_letter(b0) {
            self.pos = start + 1;
            self.scan_letters();
            return;
        }

        // Hot path 2: space before content (~25% of tokens)
        if b0 == b' ' {
            if start + 1 < len {
                let b1 = unsafe { *bytes.get_unchecked(start + 1) };
                if is_letter(b1) {
                    self.pos = start + 2;
                    self.scan_letters();
                } else if is_digit(b1) {
                    self.pos = start + 2;
                    self.scan_digits();
                } else if b1 >= 0x80 {
                    self.pos = start + 1;
                    let c = unsafe { decode_non_ascii(&bytes[self.pos..]) };
                    self.pos += c.len_utf8();
                    if unicode::is_letter(c) {
                        self.scan_letters();
                    } else if unicode::is_number(c) {
                        self.scan_digits();
                    } else if unicode::is_whitespace(c) {
                        self.advance_whitespace(start);
                    } else {
                        self.scan_other();
                    }
                } else if is_ascii_ws(b1) {
                    self.pos = start + 1;
                    self.advance_whitespace(start);
                } else {
                    self.pos = start + 2;
                    self.scan_other();
                }
            } else {
                self.pos = start + 1;
            }
            return;
        }

        // Non-ASCII
        if b0 >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[start..]) };
            self.pos = start + c.len_utf8();
            if unicode::is_letter(c) {
                self.scan_letters();
            } else if unicode::is_number(c) {
                self.scan_digits();
            } else if unicode::is_whitespace(c) {
                self.advance_whitespace(start);
            } else {
                self.scan_other();
            }
            return;
        }

        // Digit
        if is_digit(b0) {
            self.pos = start + 1;
            self.scan_digits();
            return;
        }

        // Apostrophe / contraction
        if b0 == b'\'' {
            match bytes.get(start + 1) {
                Some(b's' | b'd' | b'm' | b't') => {
                    self.pos = start + 2;
                }
                Some(b'l') if bytes.get(start + 2) == Some(&b'l') => {
                    self.pos = start + 3;
                }
                Some(b'v') if bytes.get(start + 2) == Some(&b'e') => {
                    self.pos = start + 3;
                }
                Some(b'r') if bytes.get(start + 2) == Some(&b'e') => {
                    self.pos = start + 3;
                }
                _ => {
                    self.pos = start + 1;
                    self.scan_other();
                }
            }
            return;
        }

        // Whitespace (tab, newline, etc.)
        if b0.wrapping_sub(9) < 5 {
            self.pos = start + 1;
            self.advance_whitespace(start);
            return;
        }

        // Other (punctuation, symbols)
        self.pos = start + 1;
        self.scan_other();
    }
}

impl<'a> Iterator for FastR50kPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        self.advance();
        Some(Pretoken(&self.bytes[start..self.pos]))
    }
}

/// Advance through whitespace. `scan_pos` is where to continue scanning,
/// `token_start` is where the token began (for the split-off-last-char logic).
#[inline(always)]
fn advance_ws(bytes: &[u8], scan_pos: usize, token_start: usize) -> usize {
    let len = bytes.len();
    let mut p = scan_pos;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if is_ascii_ws(b) {
            p += 1;
        } else if b >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_whitespace(c) {
                p += c.len_utf8();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if p < len {
        let ws_bytes = p - token_start;
        if ws_bytes >= 2 {
            let mut last = p - 1;
            while last > token_start && unsafe { *bytes.get_unchecked(last) } & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > token_start {
                return last;
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_matches_state_machine_owt() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes = std::fs::read(data_dir.join("owt_train.txt"))
            .expect("Could not read ~/data/owt_train.txt");
        let max = 5_000_000.min(all_bytes.len());
        let mut end = max;
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        let input = &all_bytes[..end];

        let mut sm = crate::pretokenize::PretokenizerIter::new(input);
        let mut fast = FastR50kPretokenizer::new(input);
        let mut idx = 0usize;

        loop {
            match (sm.next(), fast.next()) {
                (Some(a), Some(b)) => {
                    assert_eq!(
                        a.0, b.0,
                        "Mismatch at token {idx}: sm={:?} fast={:?}",
                        String::from_utf8_lossy(a.0),
                        String::from_utf8_lossy(b.0),
                    );
                }
                (None, None) => break,
                (Some(a), None) => panic!("SM extra at {idx}: {:?}", String::from_utf8_lossy(a.0)),
                (None, Some(b)) => {
                    panic!("Fast extra at {idx}: {:?}", String::from_utf8_lossy(b.0))
                }
            }
            idx += 1;
        }
        eprintln!("All {idx} tokens match.");
    }
}
