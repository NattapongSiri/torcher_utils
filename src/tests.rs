use torcher::{populate_tensor, shape};
use torcher::tensor::{BasicManipulateOp, ViewOp};
use super::*;

#[test]
fn batch_tensor() {
    let tensor = populate_tensor!(f32, 40, |(i, v)| {
        *v = i as f32;
    });
    // simulate lstm
    let tv = tensor.view(shape!([-1, 5, 2])).unwrap();
    let mut i = 0;
    let batchs = BatchTensor::new(IncompleteOp::AsIs, 0f32, 2, move || {
        unsafe {
            if i < 4 {
                let result = Some(tv.unsafe_narrow(&[i..(i+1), 0..5, 0..2]).unsafe_squeeze());
                i += 1;
                result
            } else {
                None
            }
        }
    });

    for (i, batch) in batchs.enumerate() {
        println!("Batch#{}", i);
        println!("Tensor shape={:?}", batch.shape().0);
        println!("Tensor data={:?}", batch.data());
    }
}

#[test]
fn shuffle_tensor() {
    let tensor = populate_tensor!(f32, 10, |(i, v)| *v = i as f32);
    let ts = IndicedTensor::from(tensor);
    let ts = ts.shuffle(0).unwrap();
    let range = vec![0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut shuffled = Vec::new();
    ts.iter().for_each(|v| {
        shuffled.push(v);
    });

    assert_ne!(shuffled, range);
    assert_eq!(shuffled.len(), range.len());
}

#[test]
fn unfold_tensor() {
    let tensor = populate_tensor!(f32, 10, |(i, v)| *v = i as f32);
    let ts = IndicedTensor::from(tensor);
    let ts = ts.unfold(0, 5, 1).unwrap();
    let expected_shape: &[usize] = &[6, 5];
    /* 
     * shape is now [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6],
     * [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]
     */
    let mut ts_iter = ts.iter();
    for i in 0..expected_shape[0] {
        for j in i..(i + expected_shape[1]) {
            assert_eq!(ts_iter.next().unwrap(), j as f32);
        }
    }
}

#[test]
fn transpose_tensor() {
    let mut tensor = populate_tensor!(f32, 10, |(i, v)| *v = i as f32);
    tensor.resize_nd(&[5, 2], &[2]);
    let ts = IndicedTensor::from(tensor);
    let tp_ts = ts.transpose(0, 1).unwrap().as_tensor();
    let expected_shape : &[usize] = &[2, 5];
    let expected_stride: &[usize] = &[1, 2];
    assert_eq!((expected_shape, expected_stride), tp_ts.1.shape());
    let expected_value: &[f32] = &[0f32, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 9.0];
    /*
     * [[0, 1],[2, 3],[4, 5],[6, 7],[8, 9]]
     */
    tp_ts.1.iter().enumerate().for_each(|(i, v)| {
        assert_eq!(v, expected_value[i]);
    });
}

#[test]
fn as_batch_incompat() {
    let tensor = populate_tensor!(f32, 20, |(i, v)| *v = i as f32);
    let ts = IndicedTensor::from(tensor);
    let batch = ts.unfold(0, 2, 1).unwrap().as_batch(5, false);
    let expected : Vec<[f32; 2]> = (0..19).map(|v| [v as f32, (v + 1) as f32]).collect();
    let mut i = 0;
    let mut j = 0;
    let mut batch_num = 0;
    batch.for_each(|b| {
        batch_num += 1;
        b.iter().for_each(|v| {
            assert_eq!(v, expected[i][j]);
            j += 1;

            if j == 2 {
                i += 1;
                j = 0;
            }
        });
    });

    assert_eq!(batch_num, 4);
}

#[test]
fn as_batch_compat() {
    let tensor = populate_tensor!(f32, 16, |(i, v)| *v = i as f32);
    let ts = IndicedTensor::from(tensor);
    /*
     * shape = [3, 5, 2]
     * 5 * 3 = 15
     * (x - 2) / 1 + 1 = 15
     * x - 2 + 1 = 15
     * x = 16 
     */
    let batch = ts.unfold(0, 2, 1).unwrap().as_batch(5, false);
    let expected : Vec<[f32; 2]> = (0..15).map(|v| [v as f32, (v + 1) as f32]).collect();
    let mut i = 0;
    let mut j = 0;
    let mut batch_num = 1;
    let expected_shape : &[usize] = &[5, 2];
    let expected_stride: &[usize] = &[1, 1];
    batch.for_each(|b| {
        assert_eq!(b.shape(), (expected_shape, expected_stride));
        batch_num += 1;
        b.iter().for_each(|v| {
            assert_eq!(v, expected[i][j]);
            j += 1;

            if j == 2 {
                i += 1;
                j = 0;
            }
        });
    });

    assert_eq!(batch_num, 4);
}