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
    assert_eq!(t1.sum(1), Tensor::new(vec![3.0, 7.0], vec![2]));

    let t2 = Tensor::arange(1, 9, 1).reshape(vec![2, 2, 2]);
    assert_eq!(t2.sum(0), Tensor::new(vec![6, 8, 10, 12], vec![2, 2]));
    let t3 = Tensor::arange(1, 9, 1).reshape(vec![2, 2, 2]);
    assert_eq!(t3.sum(1), Tensor::new(vec![4, 6, 12, 14], vec![2, 2]));
    let t4 = Tensor::arange(1, 9, 1).reshape(vec![2, 2, 2]);
    assert_eq!(t4.sum(2), Tensor::new(vec![3, 7, 11, 15], vec![2, 2]));
}
