use std::{
    fmt::{self, Display},
    iter::Step,
    ops::{Add, Div, Mul, Sub},
};

use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> Tensor<T> {
    /// Create a new tensor with the given shape and data.
    /// Infers the dtype from `T::DTYPE`.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len = shape.iter().product::<usize>();
        assert_eq!(
            expected_len,
            data.len(),
            "Shape does not match number of data elements."
        );
        Tensor { data, shape }
    }

    pub fn arange(start: T, end: T, step: usize) -> Self
    where
        T: Add<Output = T> + Copy + Default + Step,
    {
        let data: Vec<T> = (start..end).step_by(step).collect();
        let shape = vec![data.len()];
        Tensor { data, shape }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn reshape(self, shape: Vec<usize>) -> Self {
        assert_eq!(self.data.len(), shape.iter().product());
        Tensor {
            data: self.data,
            shape,
        }
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // First, ensure the data length matches the product of the shape dimensions.
        let expected_len: usize = self.shape.iter().product();
        assert_eq!(
            self.data.len(),
            expected_len,
            "Data length does not match product of shape dimensions."
        );

        // This helper function recursively prints slices of data according to the shape.
        fn recurse<T: Display>(data: &[T], shape: &[usize], indent: usize) {
            // If there's no dimension, there's nothing to print.
            if shape.is_empty() {
                return;
            }

            // If there's only one dimension left, just print a flat "row" of that dimension.
            if shape.len() == 1 {
                print!("[");
                for i in 0..shape[0] {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{}", data[i]);
                }
                print!("]");
            } else {
                // Otherwise, break the data into chunks for each sub-dimension
                let chunk_size: usize = shape[1..].iter().product();

                print!("[");
                for i in 0..shape[0] {
                    // For a nicer layout, if we're not in the first chunk,
                    // print a comma and a newline with indentation.
                    if i > 0 {
                        print!(",\n{}", " ".repeat(indent + 1));
                    }
                    let start = i * chunk_size;
                    let end = start + chunk_size;
                    recurse(&data[start..end], &shape[1..], indent + 1);
                }
                print!("]");
            }
        }

        // Call the recursive helper starting at the top level.
        recurse(&self.data, &self.shape, 0);
        println!(); // Print a final newline at the end
        Ok(())
    }
}

impl<T: Add> Add for Tensor<T>
where
    Vec<T>: FromIterator<<T as Add>::Output>,
{
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .into_iter()
            .zip(other.data.into_iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape)
    }
}

impl<T: Mul> Mul for Tensor<T>
where
    Vec<T>: FromIterator<<T as Mul>::Output>,
{
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .into_iter()
            .zip(other.data.into_iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape)
    }
}

impl<T> Tensor<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default,
{
    pub fn dot(self, other: Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape.last(), other.shape.first());
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);

        if (self.shape[0] == 0) || (self.shape[1] == 0) || (other.shape[1] == 0) {
            return Tensor::new(vec![], vec![0, 0]);
        }

        if (self.shape.len() == 2) && (other.shape.len() == 2) {
            return dot2d(self, other);
        } else {
            panic!("Not implemented");
        }
    }
}

fn flattened_index(index: &[usize], shape: &[usize]) -> usize {
    // for (i, el) in index.iter().enumerate() {
    //     assert!(*el < shape[i]);
    // }
    index
        .iter()
        .zip(shape.iter())
        .fold(0, |acc, (i, s)| acc * s + i)
}

fn unflattened_index(index: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(shape.len());
    let mut index = index;
    for s in shape.iter().rev() {
        result.push(index % s);
        index /= s;
    }
    result.reverse();
    result
}

pub fn all_coords_of_shape(shape: &[usize]) -> Vec<Vec<usize>> {
    if shape.is_empty() {
        // No dimensions => one "coordinate": the empty one
        return vec![vec![]];
    }
    // For shape = [d0, d1, ...], we iterate i in [0..d0] and then
    // do all coords in [d1, ...], appending i in front.
    let (first_dim, rest) = shape.split_first().unwrap();
    let tail_coords = all_coords_of_shape(rest);

    let mut coords = Vec::new();
    for i in 0..*first_dim {
        for tail in &tail_coords {
            let mut c = Vec::with_capacity(1 + tail.len());
            c.push(i);
            c.extend_from_slice(tail);
            coords.push(c);
        }
    }
    coords
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1;
    // Work from the last dimension backwards
    for &dim in shape.iter().rev() {
        strides.push(acc);
        acc *= dim;
    }
    // We built them in reverse order, so reverse them back
    strides.reverse();
    strides
}

impl<T> Tensor<T>
where
    T: Add<Output = T>
        + Mul<Output = T>
        + Copy
        + Default
        + PartialOrd
        + Send
        + Sync
        + std::fmt::Debug,
{
    fn reduction<F>(self, operator: F, axis: usize) -> Tensor<T>
    where
        F: Fn(&[T]) -> T + Send + Sync,
    {
        let rank = self.shape.len();
        let axis_len = self.shape[axis];
        assert!(axis < self.shape.len());
        let mut out_shape = self.shape.clone();
        out_shape.remove(axis);

        let out_size: usize = out_shape.iter().product();
        let mut out_data = vec![T::default(); out_size];
        if axis == self.shape.len() - 1 {
            out_data.par_iter_mut().enumerate().for_each(|(i, out)| {
                *out = operator(&self.data[i * axis_len..i * axis_len + axis_len]);
            });
            return Tensor::new(out_data, out_shape);
        } else if axis == 0 {
            let shift: usize = self.shape[1..].iter().product();
            out_data.par_iter_mut().enumerate().for_each(|(i, out)| {
                let mut data = Vec::with_capacity(axis_len);
                for j in 0..self.shape[axis] {
                    data.push(self.data[i + j * shift]);
                }
                *out = operator(&data);
            });
            return Tensor::new(out_data, out_shape);
        }

        let strides = compute_strides(&self.shape);
        let axis_stride = strides[axis];

        let mut dims_no_axis = Vec::with_capacity(rank - 1);
        for i in 0..rank {
            if i != axis {
                dims_no_axis.push(i);
            }
        }

        fn next_coord(coord: &mut [usize], shape: &[usize]) -> bool {
            // shape.len() == coord.len()
            for i in (0..coord.len()).rev() {
                coord[i] += 1;
                if coord[i] < shape[i] {
                    return false; // still valid, not done
                } else {
                    coord[i] = 0;
                }
            }
            true // all done
        }

        // We'll track the current coordinate for all dims except `axis`
        let mut coord = vec![0; rank - 1]; // all zero initially
        let mut out_idx = 0; // linear index into out_data

        // Main loop: for each "reduced" coordinate...
        loop {
            // 1) Compute base offset for these coordinates in the input `data`
            //    offset = sum_{j} [coord[j] * strides[dims_no_axis[j]]].
            let mut offset = 0;
            for (j, &val) in coord.iter().enumerate() {
                let dim_j = dims_no_axis[j];
                offset += val * strides[dim_j];
            }

            // 2) Accumulate the sum along the chosen axis by stepping
            //    in increments of `axis_stride`.
            let mut s = T::default();
            let mut cur = offset;
            for _ in 0..axis_len {
                s = s + self.data[cur];
                cur += axis_stride;
            }

            // 3) Store in out_data
            out_data[out_idx] = s;
            out_idx += 1;

            // 4) Move to the next coordinate
            if next_coord(&mut coord, &out_shape) {
                // If we've overflowed, we're done
                break;
            }
        }

        Tensor::new(out_data, out_shape)
    }

    pub fn sum(self, axis: usize) -> Tensor<T>
    where
        T: std::iter::Sum<T> + Copy,
    {
        self.reduction(&|a: &[T]| a.iter().copied().sum(), axis)
    }

    // pub fn product(self, axis: usize) -> Tensor<T> {
    //     self.reduction(&|a, b| a * b, axis)
    // }

    // pub fn max(self, axis: usize) -> Tensor<T> {
    //     self.reduction(&|a, b| if a > b { a } else { b }, axis)
    // }

    // pub fn min(self, axis: usize) -> Tensor<T> {
    //     self.reduction(&|a, b| if a < b { a } else { b }, axis)
    // }
}

fn dot2d<T>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default,
{
    let mut data = Vec::with_capacity(a.shape[0] * b.shape[1]);
    for i in 0..a.shape[0] {
        for j in 0..b.shape[1] {
            data.push(T::default());
            for k in 0..a.shape[1] {
                data[i * b.shape[1] + j] = data[i * a.shape[1] + j]
                    + a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
            }
        }
    }
    Tensor::new(data, vec![a.shape[0], b.shape[1]])
}
