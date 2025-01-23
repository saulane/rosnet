use rosnet::Tensor;

#[test]
fn test_basic_add() {
    let t1 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t2 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t3 = t1 + t2;
    assert_eq!(t3, Tensor::new_f32(vec![2, 2], vec![2.0, 4.0, 6.0, 8.0]));
}

#[test]
fn test_basic_mul() {
    let t1 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t2 = Tensor::new_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t3 = t1 * t2;
    assert_eq!(t3, Tensor::new_f32(vec![2, 2], vec![1.0, 4.0, 9.0, 16.0]));
}
