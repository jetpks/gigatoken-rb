//! Fast scalar pretokenizers, one submodule per pretokenization scheme.
//!
//! Each scheme implements an advance function that consumes exactly one
//! pretoken, wrapped in a thin iterator struct. The byte predicates and
//! SWAR scans below are shared; a new scheme (e.g. o200k) should slot in
//! as another submodule reusing these primitives where its character
//! classes line up.

pub mod cl100k;
pub mod r50k;

pub use cl100k::FastCl100kPretokenizer;
pub use r50k::FastR50kPretokenizer;

use crate::pretokenize::unicode;

// -----------------------------------------------------------------------
// Branchless byte predicates
// -----------------------------------------------------------------------

#[inline(always)]
pub(crate) fn is_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
pub(crate) fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline(always)]
pub(crate) fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

#[inline(always)]
pub(crate) unsafe fn decode_non_ascii(bytes: &[u8]) -> char {
    unsafe {
        std::str::from_utf8_unchecked(bytes)
            .chars()
            .next()
            .unwrap_unchecked()
    }
}

// -----------------------------------------------------------------------
// SWAR
// -----------------------------------------------------------------------

pub(crate) const HI: u64 = 0x8080_8080_8080_8080;

#[inline(always)]
pub(crate) fn swar64_letter_mask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lowered);
    ge_a & le_z & HI
}

/// Returns the high bit set in each lane that is NOT an ASCII letter.
/// Equivalent to `!swar64_letter_mask(word) & HI` but computed directly so
/// the scan loop can branch on `!= 0` and reuse the value for `trailing_zeros`.
#[inline(always)]
pub(crate) fn swar64_letter_nonmask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lowered);
    !(ge_a & le_z) & HI
}

/// SWAR letter scan: advances `pos` past ASCII letters.
/// Returns the updated pos.
#[inline(always)]
pub(crate) fn swar_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    // SWAR: 8 bytes at a time
    while pos + 8 <= len {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HI != 0 {
            break;
        }
        let nonletter = swar64_letter_nonmask(word);
        if nonletter != 0 {
            return pos + nonletter.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    // Scalar tail
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_letter(b) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// SWAR digit scan: advances `pos` past ASCII digits.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn swar_scan_digits(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    while pos + 8 <= len {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HI != 0 {
            break;
        }
        let ge_0 = (word | HI).wrapping_sub(0x3030_3030_3030_3030) & HI;
        let le_9 = (0x3939_3939_3939_3939 | HI).wrapping_sub(word) & HI;
        let nondigit = !(ge_0 & le_9) & HI;
        if nondigit != 0 {
            return pos + nondigit.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    while pos < len && is_digit(unsafe { *bytes.get_unchecked(pos) }) {
        pos += 1;
    }
    pos
}

/// SWAR "other" scan: advances `pos` past bytes that are NOT letter, digit,
/// whitespace, or high (>=0x80). Returns position of first non-"other" byte.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn swar_scan_other(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    while pos + 8 <= len {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        // Any high byte means non-ASCII — stop immediately
        if word & HI != 0 {
            break;
        }
        // Detect bytes that are NOT "other": letter OR digit OR whitespace
        let lowered = word | 0x2020_2020_2020_2020;
        let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161) & HI;
        let le_z = (0x7A7A_7A7A_7A7A_7A7A | HI).wrapping_sub(lowered) & HI;
        let is_letter = ge_a & le_z & HI;

        let ge_0 = (word | HI).wrapping_sub(0x3030_3030_3030_3030) & HI;
        let le_9 = (0x3939_3939_3939_3939 | HI).wrapping_sub(word) & HI;
        let is_digit = ge_0 & le_9 & HI;

        let ge_9 = (word | HI).wrapping_sub(0x0909_0909_0909_0909) & HI;
        let le_13 = (0x0D0D_0D0D_0D0D_0D0D | HI).wrapping_sub(word) & HI;
        let is_ws_ctrl = ge_9 & le_13 & HI;
        let xor_space = word ^ 0x2020_2020_2020_2020;
        let is_space = (xor_space.wrapping_sub(0x0101_0101_0101_0101)) & !xor_space & HI;

        let not_other = is_letter | is_digit | is_ws_ctrl | is_space;
        if not_other != 0 {
            return pos + not_other.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if b >= 0x80 || is_letter(b) || is_digit(b) || is_ascii_ws(b) {
            break;
        }
        pos += 1;
    }
    pos
}

// -----------------------------------------------------------------------
// Shared run scans (`\p{L}+`, `\p{N}+`, `[^\s\p{L}\p{N}]+`)
// -----------------------------------------------------------------------

#[inline(always)]
pub(crate) fn scan_letters_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_letter(c) {
                p += c.len_utf8();
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_digits_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len && is_digit(unsafe { *bytes.get_unchecked(p) }) {
            p += 1;
        }
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_number(c) {
                p += c.len_utf8();
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 { break; }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) { return p; }
            p += 1;
        }
        if p < len {
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_other_complete(c) {
                p += c.len_utf8();
                continue;
            }
        }
        return p;
    }
}

