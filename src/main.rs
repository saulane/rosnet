use rosnet::Tensor;

fn main() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t3 = t1.dot(t2);
    println!("{}", t3);
    // println!("{}", t3);
}
