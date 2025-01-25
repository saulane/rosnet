// use ndarray::prelude::*;
use rosnet::Tensor;

fn main() {
    // let t2 = Tensor::arange(1, 9, 1).reshape(vec![2, 2, 2]);
    // let t3 = Tensor::arange(1, 9, 1).reshape(vec![2, 2, 2]);
    // let t4 = Tensor::arange(1, 9, 1).reshape(vec![2, 2, 2]);

    // println!("{}", t2.sum(0));
    // println!("{}", t3.sum(1));
    // println!("{}", t4.sum(2));

    let t = Tensor::arange(0_i64, 1_000_000_000_i64, 1).reshape(vec![10, 1000, 100000]);
    let start = std::time::Instant::now();
    let res1 = t.sum(2);
    let elapsed = start.elapsed();
    println!("Time elapsed: {:?}", elapsed);

    // let arr = Array::range(0., 1_000_000_000., 1.);
    // let c = arr.to_shape((10, 1000, 100000)).unwrap();

    // let start2 = std::time::Instant::now();
    // let res2 = &c.sum_axis(Axis(2));
    // let elapsed2 = start2.elapsed();
    // // println!("Time elapsed: {:?}", elapsed2);
}
