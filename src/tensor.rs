use std::ops::{Add, Mul};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    // Int32,
    // UInt32,
    // ...
}

/// A trait to associate a Rust numeric type with a `DType`.
pub trait NumericDType {
    const DTYPE: DType;
}

// Implement it for the types we care about:
// impl NumericDType for i32 {
//     const DTYPE: DType = DType::I32;
// }

impl NumericDType for f32 {
    const DTYPE: DType = DType::F32;
}

impl NumericDType for f64 {
    const DTYPE: DType = DType::F64;
}

#[macro_export]
macro_rules! tensor {
    // 2D pattern: [[...], [...], ...]
    ( [ $( [ $( $elem:expr ),* $(,)? ] ),+ $(,)? ] ) => {{
        // Create a Vec<Vec<_>> from the nested array
        let data_2d = vec![
            $(
                vec![
                    $( $elem ),*
                ],
            )*
        ];
        // The outer dimension
        let outer = data_2d.len();
        // Inner dimension (assume at least 1 row)
        let inner = if outer > 0 {
            data_2d[0].len()
        } else {
            0
        };

        // Flatten into a single Vec<_>
        let flattened = data_2d.into_iter().flat_map(|row| row.into_iter()).collect::<Vec<_>>();

        // Let Rust infer T from the elements. T must implement NumericDType.
        $crate::Tensor::new(vec![outer, inner], flattened)
    }};

    // 1D pattern: [elem, elem, ...]
    ( [ $( $elem:expr ),* $(,)? ] ) => {{
        let data_1d = vec![$( $elem ),*];
        let len = data_1d.len();
        $crate::Tensor::new(vec![len], data_1d)
    }};

    // Optionally, you could add more arms for 3D, etc.
}

impl DType {
    /// Returns the size in bytes for each data type.
    fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
}

// #[derive(Debug, Clone, PartialEq)]
// pub struct Tensor {
//     pub shape: Vec<usize>,
//     pub data: Vec<u8>,
//     pub dtype: DType,
// }

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
    pub dtype: DType,
}

impl<T: NumericDType> Tensor<T> {
    /// Create a new tensor with the given shape and data.
    /// Infers the dtype from `T::DTYPE`.
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        let expected_len = shape.iter().product::<usize>();
        assert_eq!(
            expected_len,
            data.len(),
            "Shape does not match number of data elements."
        );
        Tensor {
            shape,
            dtype: T::DTYPE,
            data,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

// impl Tensor {
//     /// Construct a Tensor with `dtype = F32` from a `Vec<f32>`.
//     pub fn new_f32(shape: Vec<usize>, data_f32: Vec<f32>) -> Self {
//         // 1. Check shape consistency
//         let expected_len = shape.iter().product::<usize>();
//         assert_eq!(
//             expected_len,
//             data_f32.len(),
//             "Shape vs data length mismatch"
//         );

//         // 2. Convert f32 data into raw bytes
//         let mut bytes = Vec::with_capacity(expected_len * 4);
//         for &val in &data_f32 {
//             bytes.extend_from_slice(&val.to_ne_bytes());
//         }

//         Tensor {
//             shape,
//             dtype: DType::F32,
//             data: bytes,
//         }
//     }

//     /// Construct a Tensor with `dtype = F64` from a `Vec<f64>`.
//     pub fn new_f64(shape: Vec<usize>, data_f64: Vec<f64>) -> Self {
//         let expected_len = shape.iter().product::<usize>();
//         assert_eq!(
//             expected_len,
//             data_f64.len(),
//             "Shape vs data length mismatch"
//         );

//         let mut bytes = Vec::with_capacity(expected_len * 8);
//         for &val in &data_f64 {
//             bytes.extend_from_slice(&val.to_ne_bytes());
//         }

//         Tensor {
//             shape,
//             dtype: DType::F64,
//             data: bytes,
//         }
//     }

//     fn as_slice_f32(&self) -> &[f32] {
//         assert_eq!(self.dtype, DType::F32, "Tensor is not f32");
//         let ptr = self.data.as_ptr() as *const f32;
//         let len = self.data.len() / 4;
//         unsafe { std::slice::from_raw_parts(ptr, len) }
//     }

//     /// Interpret the internal data as a slice of f64.
//     /// Panics if the dtype is not F64.
//     fn as_slice_f64(&self) -> &[f64] {
//         assert_eq!(self.dtype, DType::F64, "Tensor is not f64");
//         let ptr = self.data.as_ptr() as *const f64;
//         let len = self.data.len() / 8;
//         unsafe { std::slice::from_raw_parts(ptr, len) }
//     }
// }

impl<T: NumericDType> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Tensor<T> {
        // 1. Check that shapes match
        assert_eq!(self.shape, rhs.shape, "Shapes must match for add.");

        // 2. Check that dtypes match
        assert_eq!(self.dtype, rhs.dtype, "DTypes must match for add.");

        match T::DTYPE {
            DType::F32 => {
                // Reinterpret as &[f32]
                let a: &[f32] = self.as_slice();
                let b = rhs.as_slice();

                // We'll produce a new Vec<f32> with the results
                let mut out = vec![0f32; a.len()];

                // match T::DTYPE {
                //     DType::F32 => {
                //         // Dispatch to an f32-add kernel (which may use AVX)
                //         cpu_add_f32(a, b, &mut out);
                //     }
                //     DType::F64 => {
                //         // Dispatch to an f64-add kernel (similar idea, possibly AVX for double)
                //         cpu_add_f64(a, b, &mut out);
                //     }
                // }
                // Dispatch to an f32-add kernel (which may use AVX)
                // cpu_add_f32(a, b, &mut out);

                // Convert `out` into bytes to build a new Tensor
                let mut out_bytes = Vec::with_capacity(out.len() * 4);
                for &val in &out {
                    out_bytes.extend_from_slice(&val.to_ne_bytes());
                }

                Tensor {
                    shape: self.shape.clone(),
                    dtype: DType::F32,
                    data: out_bytes,
                }
            }
            DType::F64 => {
                let a = self.as_slice_f64();
                let b = rhs.as_slice_f64();
                let mut out = vec![0f64; a.len()];

                // Dispatch to an f64-add kernel (similar idea, possibly AVX for double)
                cpu_add_f64(a, b, &mut out);

                // Convert the f64 result into bytes
                let mut out_bytes = Vec::with_capacity(out.len() * 8);
                for &val in &out {
                    out_bytes.extend_from_slice(&val.to_ne_bytes());
                }

                Tensor {
                    shape: self.shape.clone(),
                    dtype: DType::F64,
                    data: out_bytes,
                }
            }
        }
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        // 1. Check that shapes match
        assert_eq!(self.shape, rhs.shape, "Shapes must match for add.");

        // 2. Check that dtypes match
        assert_eq!(self.dtype, rhs.dtype, "DTypes must match for add.");

        match self.dtype {
            DType::F32 => {
                // Reinterpret as &[f32]
                let a = self.as_slice_f32();
                let b = rhs.as_slice_f32();

                // We'll produce a new Vec<f32> with the results
                let mut out = vec![0f32; a.len()];

                // Dispatch to an f32-add kernel (which may use AVX)
                cpu_mul_f32(a, b, &mut out);

                // Convert `out` into bytes to build a new Tensor
                let mut out_bytes = Vec::with_capacity(out.len() * 4);
                for &val in &out {
                    out_bytes.extend_from_slice(&val.to_ne_bytes());
                }

                Tensor {
                    shape: self.shape.clone(),
                    dtype: DType::F32,
                    data: out_bytes,
                }
            }
            DType::F64 => {
                let a = self.as_slice_f64();
                let b = rhs.as_slice_f64();
                let mut out = vec![0f64; a.len()];

                // Dispatch to an f64-add kernel (similar idea, possibly AVX for double)
                cpu_mul_f64(a, b, &mut out);

                // Convert the f64 result into bytes
                let mut out_bytes = Vec::with_capacity(out.len() * 8);
                for &val in &out {
                    out_bytes.extend_from_slice(&val.to_ne_bytes());
                }

                Tensor {
                    shape: self.shape.clone(),
                    dtype: DType::F64,
                    data: out_bytes,
                }
            }
        }
    }
}

fn cpu_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    if std::arch::is_aarch64_feature_detected!("neon") {
        mul_f32_neon(a, b, out);
    } else {
        mul_f32_scalar(a, b, out);
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let len = a.len();
        let chunks = len / 4 * 4;
        for i in (0..chunks).step_by(4) {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vsum = vmulq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(i), vsum);
        }
        for i in chunks..len {
            out[i] = a[i] * b[i];
        }
    }
}

fn mul_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

fn cpu_mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    if std::arch::is_aarch64_feature_detected!("neon") {
        mul_f64_neon(a, b, out);
    } else {
        mul_f64_scalar(a, b, out);
    }
}

fn mul_f64_scalar(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
fn mul_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
    use std::arch::aarch64::*;
    unsafe {
        let len = a.len();
        let chunks = len / 2 * 2;
        for i in (0..chunks).step_by(2) {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vsum = vmulq_f64(va, vb);
            vst1q_f64(out.as_mut_ptr().add(i), vsum);
        }
        for i in chunks..len {
            out[i] = a[i] * b[i];
        }
    }
}

// --- Add Kernel for f64 ---
// # x86_64
// --------------------------
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        if std::arch::is_x86_feature_detected!("avx") {
            add_f32_avx(a, b, out);
        } else {
            add_f32_scalar(a, b, out);
        }
    }
}

fn add_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// AVX version for f32
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn add_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    let len = a.len();
    let chunked = len / 8 * 8;
    for i in (0..chunked).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vsum = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), vsum);
    }
    // handle remainder
    for i in chunked..len {
        out[i] = a[i] + b[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    unsafe {
        if std::arch::is_x86_feature_detected!("avx") {
            add_f64_avx(a, b, out);
        } else {
            add_f64_scalar(a, b, out);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn add_f64_avx(a: &[f64], b: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    let len = a.len();
    let chunked = len / 4 * 4; // 4 doubles per 256-bit register
    for i in (0..chunked).step_by(4) {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        let vsum = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), vsum);
    }
    for i in chunked..len {
        out[i] = a[i] + b[i];
    }
}

fn add_f64_scalar(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let len = a.len();
        let chunks = len / 4 * 4;
        for i in (0..chunks).step_by(4) {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vsum = vaddq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(i), vsum);
        }
        for i in chunks..len {
            out[i] = a[i] + b[i];
        }
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub fn cpu_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    // If you want to do NEON, implement that similarly, e.g.
    if std::arch::is_aarch64_feature_detected!("neon") {
        add_f32_neon(a, b, out);
    } else {
        add_f32_scalar(a, b, out);
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub fn cpu_add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    // If you want to do NEON, implement that similarly, e.g.
    if std::arch::is_aarch64_feature_detected!("neon") {
        add_f64_neon(a, b, out);
    } else {
        add_f64_scalar(a, b, out);
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
fn add_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
    use std::arch::aarch64::*;
    unsafe {
        let len = a.len();
        let chunks = len / 2 * 2;
        for i in (0..chunks).step_by(2) {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vsum = vaddq_f64(va, vb);
            vst1q_f64(out.as_mut_ptr().add(i), vsum);
        }
        for i in chunks..len {
            out[i] = a[i] + b[i];
        }
    }
}
