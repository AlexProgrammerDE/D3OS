use d3os_nvidia::dma::copy_owned;

#[test]
fn owned_dma_copies_preserve_unaligned_ranges_and_guards() {
    for length in [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 4095, 4096, 4097, 1 << 20] {
        let source: Vec<u8> = (0..length + 48).map(|index| (index as u8).wrapping_mul(73).wrapping_add(19)).collect();
        for source_offset in [0, 1, 3, 7, 15] {
            for destination_offset in [16, 17, 19, 23, 31] {
                let mut destination = vec![0xd3; length + 64];
                unsafe {
                    copy_owned(destination.as_mut_ptr().add(destination_offset), source.as_ptr().add(source_offset), length);
                }
                assert_eq!(
                    &destination[destination_offset..destination_offset + length],
                    &source[source_offset..source_offset + length]
                );
                assert!(destination[..destination_offset].iter().all(|&byte| byte == 0xd3));
                assert!(destination[destination_offset + length..].iter().all(|&byte| byte == 0xd3));
            }
        }
    }
}
