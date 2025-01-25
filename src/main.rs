use rosnet::Tensor;

fn main() {
    let t1 = Tensor::arange(0, 10, 1).reshape(vec![2, 5]);
    println!("{}", t1);
    let t3 = t1.sum(1);
    println!("{}", t3);
}
