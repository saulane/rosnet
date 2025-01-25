use rosnet::Tensor;

#[test]
fn test_basic_add() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t3 = t1 + t2;
    assert_eq!(t3, Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2],));
}

#[test]
fn test_basic_mul() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t3 = t1 * t2;
    assert_eq!(t3, Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2],));
}

#[test]
fn test_basic_dot() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t3 = t1.dot(t2);
    println!("{:?}", t3);
    assert_eq!(t3, Tensor::new(vec![7.0, 10.0, 15.0, 22.0], vec![2, 2]));
}

#[test]
fn test_basic_sum() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t3 = t1.sum(1);
    println!("{:?}", t3);
    assert_eq!(t3, Tensor::new(vec![3.0, 7.0], vec![2, 2]));
}

// #[test]
// fn test_basic_add() {
//     let t1 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
//     let t2 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
//     let t3 = t1 + t2;
//     assert_eq!(t3, Tensor::new_f32(vec![2, 2], vec![2.0, 4.0, 6.0, 8.0]));
// }

// #[test]
// fn test_basic_mul() {
//     let t1 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
//     let t2 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
//     let t3 = t1 * t2;
//     assert_eq!(t3, Tensor::new_f32(vec![2, 2], vec![1.0, 4.0, 9.0, 16.0]));
// }
