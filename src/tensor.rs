use std::{
    fmt::{self, Display},
    ops::{Add, Mul},
};

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

    pub fn as_slice(&self) -> &[T] {
        &self.data
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

impl<T> Tensor<T>
where
    T: Add<Output = T> + Copy + Default + std::iter::Sum,
{
    pub fn sum(self, axis: usize) -> Tensor<T> {
        assert!(axis < self.shape.len());
        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        let mut new_data = Vec::with_capacity(new_shape.iter().product());
        for i in 0..self.shape[axis] {
            let start = i * self.shape[axis + 1];
            let end = start + self.shape[axis + 1];
            let sum = self.data[start..end].iter().copied().sum();
            new_data.push(sum);
        }
        Tensor::new(new_data, new_shape)
    }
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
