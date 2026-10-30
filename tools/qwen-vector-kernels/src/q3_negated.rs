/// Negate Q3 values after applying their signed six-bit subgroup scale.
/// Each code byte is in 0..=7; scale is in -32..=31. Negated products fit S8.
#[inline(always)]
pub(super) fn negated_q3(code: u32, scale: i32) -> u32 {
    let magnitude = scale.unsigned_abs();
    let (unsigned, offset) = if scale < 0 {
        (code, 128 - 4 * magnitude)
    } else {
        (0x07070707 - code, 128 - 3 * magnitude)
    };
    (unsigned * magnitude + offset * 0x01010101) ^ 0x80808080
}

#[cfg(test)]
mod tests {
    #[test]
    fn all_packed_codes_and_scales_match_independent_signed_bytes() {
        for scale in -32..=31 {
            for pattern in 0u32..4096 {
                let codes = core::array::from_fn::<_, 4, _>(|i| (pattern >> (3 * i)) & 7);
                let packed = codes.iter().enumerate().fold(0, |word, (i, code)| word | (code << (8 * i)));
                let actual = super::negated_q3(packed, scale).to_le_bytes();
                for i in 0..4 {
                    let q = codes[i] as i32 - 4;
                    assert_eq!(i32::from(actual[i] as i8), -(q * scale));
                }
            }
        }
    }
}
