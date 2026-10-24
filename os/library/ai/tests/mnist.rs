#![cfg(all(feature = "cpu", feature = "training", feature = "native"))]
use d3os_ai::{
    burn::{module::Module, tensor::Device},
    mnist::{
        artifact,
        data::{Dataset, SampleOrder},
        training::{Config, Trainer, evaluate},
    },
};

fn fixture(count: u32) -> (Vec<u8>, Vec<u8>) {
    let mut images = Vec::new();
    for word in [2051u32, count, 28, 28] {
        images.extend_from_slice(&word.to_be_bytes());
    }
    images.extend((0..count as usize * 784).map(|i| (i.wrapping_mul(37) ^ (i / 784 * 79)) as u8));
    let mut labels = Vec::new();
    for word in [2049u32, count] {
        labels.extend_from_slice(&word.to_be_bytes());
    }
    labels.extend((0..count).map(|i| (i % 10) as u8));
    (images, labels)
}

#[test]
fn idx_rejects_malformed_shapes_labels_and_framing() {
    let (images, labels) = fixture(3);
    let data = Dataset::parse(&images, &labels).unwrap();
    assert_eq!(data.len(), 3);
    for (offset, byte) in [(0, 1), (7, 4), (11, 27), (15, 29)] {
        let mut invalid = images.clone();
        invalid[offset] = byte;
        assert!(Dataset::parse(&invalid, &labels).is_err());
    }
    for end in [0, 7, 15, images.len() - 1] {
        assert!(Dataset::parse(&images[..end], &labels).is_err());
    }
    let mut invalid = labels.clone();
    invalid[9] = 10;
    assert!(Dataset::parse(&images, &invalid).is_err());
    let mut trailing = images.clone();
    trailing.push(0);
    assert!(Dataset::parse(&trailing, &labels).is_err());
    let device = Device::flex();
    assert!(data.batch(&[], &device).is_err());
    assert!(data.batch(&[3], &device).is_err());
    let (batch, target) = data.batch(&[2, 0], &device).unwrap();
    assert_eq!(batch.dims(), [2, 1, 28, 28]);
    assert_eq!(target.try_into_vec_as::<i32>().unwrap(), [2, 0]);
}

#[test]
fn epoch_shuffle_matches_independent_splitmix_reference_and_covers_the_tail() {
    let order = SampleOrder::new(7, 42, 0);
    assert_eq!(order.batch(0, 7).unwrap(), [2, 4, 6, 0, 3, 1, 5]);
    assert_eq!(order.state(), 13_064_056_694_810_536_104);
    assert_eq!(order.batch(6, 3).unwrap(), [5]);
    assert!(order.batch(7, 1).is_err());
    assert!(order.batch(0, 0).is_err());
    let order = SampleOrder::new(7, 42, 1);
    assert_eq!(order.batch(0, 7).unwrap(), [5, 4, 0, 2, 6, 3, 1]);
    assert_eq!(order.state(), 9_692_027_447_243_036_733);
}

#[test]
fn convolutional_adam_resume_preserves_progress_updates_and_exported_predictions() {
    let (images, labels) = fixture(5);
    let data = Dataset::parse(&images, &labels).unwrap();
    let device = Device::flex().autodiff();
    let mut trainer = Trainer::new(
        &data,
        &device,
        Config {
            batch_size: 2,
            ..Config::default()
        },
    )
    .unwrap();
    for _ in 0..2 {
        trainer = trainer.step(&data).unwrap().0;
    }
    let checkpoint = trainer.checkpoint().unwrap();
    let mut resumed = Trainer::restore(&checkpoint, &data, &device).unwrap();
    assert_eq!(resumed.progress(), trainer.progress());
    for _ in 0..3 {
        let (next, expected) = trainer.step(&data).unwrap();
        trainer = next;
        let (next, actual) = resumed.step(&data).unwrap();
        resumed = next;
        assert_eq!(trainer.progress(), resumed.progress());
        assert_eq!(expected.samples, actual.samples);
        assert_eq!(expected.completed_epoch, actual.completed_epoch);
        assert!((expected.loss - actual.loss).abs() < 2e-6);
        let (input, _) = data.batch(&[0, 1], &device.clone().without_autodiff()).unwrap();
        let expected = trainer.model().valid().forward(input.clone()).try_into_vec_as::<f32>().unwrap();
        let actual = resumed.model().valid().forward(input).try_into_vec_as::<f32>().unwrap();
        assert!(expected.iter().zip(actual).all(|(a, b)| (a - b).abs() < 2e-6));
    }
    assert_eq!((trainer.progress().epoch, trainer.progress().offset, trainer.progress().steps), (1, 4, 5));
    let progress = trainer.progress();
    assert!(trainer.set_learning_rate(f64::NAN).is_err());
    assert_eq!(trainer.config().learning_rate, 0.001);
    trainer.set_learning_rate(0.0001).unwrap();
    let lower_rate = Trainer::restore(&trainer.checkpoint().unwrap(), &data, &device).unwrap();
    assert_eq!(lower_rate.config().learning_rate, 0.0001);
    assert_eq!(lower_rate.progress(), progress);
    let exported = artifact::export(trainer.model()).unwrap();
    let reloaded = artifact::load(&exported, &device).unwrap();
    let expected = evaluate(trainer.model(), &data, 2).unwrap();
    let actual = evaluate(&reloaded, &data, 2).unwrap();
    assert_eq!(
        (expected.samples, expected.correct, expected.predictions_crc32),
        (actual.samples, actual.correct, actual.predictions_crc32)
    );
    for offset in [0, 12, 32, exported.len() - 5] {
        let mut invalid = exported.clone();
        invalid[offset] ^= 1;
        assert!(artifact::load(&invalid, &device).is_err());
    }
    assert!(artifact::load(&checkpoint, &device).is_err());
    let (other_images, other_labels) = fixture(6);
    let other = Dataset::parse(&other_images, &other_labels).unwrap();
    assert!(Trainer::restore(&checkpoint, &other, &device).is_err());
    let mut changed_images = images.clone();
    changed_images[31] ^= 1;
    let changed_data = Dataset::parse(&changed_images, &labels).unwrap();
    assert!(Trainer::restore(&checkpoint, &changed_data, &device).is_err());
    // These payloads retain a valid container checksum, so validation must
    // reject their training state rather than stopping at transport framing.
    for (word, value) in [(2usize, f64::NAN.to_bits()), (4, 1), (8, 0)] {
        let mut invalid = checkpoint.clone();
        let offset = 32 + 36 + word * 8;
        invalid[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        let end = invalid.len() - 4;
        let checksum = crc32fast::hash(&invalid[..end]);
        invalid[end..].copy_from_slice(&checksum.to_le_bytes());
        assert!(Trainer::restore(&invalid, &data, &device).is_err());
    }
}
