use std::time::Instant;
use rand::Rng;
use std::fs::File;
use std::io::Write;

#[target_feature(enable = "neon")]
unsafe fn add_arrays_simd(a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::aarch64::*;

    let chunks = a.len() / 4;
    for i in 0..chunks {
        let a_chunk = vld1q_f32(a.as_ptr().add(i * 4));
        let b_chunk = vld1q_f32(b.as_ptr().add(i * 4));
        let c_chunk = vaddq_f32(a_chunk, b_chunk);
        vst1q_f32(c.as_mut_ptr().add(i * 4), c_chunk);
    }

    for i in chunks * 4..a.len() {
        c[i] = a[i] + b[i];
    }
}

fn add_arrays(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}


fn benchmark_add() {
    let sizes = [10_000, 100_000, 1_000_000, 10_000_000];
    let mut results = String::from("input_size,normal,simd\n");

    for &size in &sizes {
        let a = generate_random_array(size);
        let b = generate_random_array(size);
        let mut c = vec![0.0; size];
        let mut d = vec![0.0; size];

        let start = Instant::now();
        add_arrays(&a, &b, &mut c);
        let duration_normal = start.elapsed().as_nanos();

        let start = Instant::now();
        unsafe {
            add_arrays_simd(&a, &b, &mut d);
        }
        let duration_simd = start.elapsed().as_nanos();

        results.push_str(&format!("{},{},{}\n", size, duration_normal, duration_simd));
    }

    let mut file = File::create("benchmark_add_results.csv").unwrap();
    file.write_all(results.as_bytes()).unwrap();
}


fn multiply_matrices_normal(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn multiply_matrices_simd(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    use core::arch::aarch64::*;
    for i in 0..n {
        for j in 0..n {
            let mut sum = vdupq_n_f32(0.0);
            for k in (0..n).step_by(4) {
                let a_vec = vld1q_f32(a.as_ptr().add(i * n + k));
                // Load elements individually to form a column vector from matrix B
                let b0 = *b.get_unchecked(k * n + j);
                let b1 = *b.get_unchecked((k + 1) * n + j);
                let b2 = *b.get_unchecked((k + 2) * n + j);
                let b3 = *b.get_unchecked((k + 3) * n + j);
                let b_vec = vsetq_lane_f32::<0>(b0, vdupq_n_f32(0.0));
                let b_vec = vsetq_lane_f32::<1>(b1, b_vec);
                let b_vec = vsetq_lane_f32::<2>(b2, b_vec);
                let b_vec = vsetq_lane_f32::<3>(b3, b_vec);
                sum = vfmaq_f32(sum, a_vec, b_vec);
            }
            let result = vaddvq_f32(sum);
            *c.get_unchecked_mut(i * n + j) = result;
        }
    }
}

fn arrays_equal(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let epsilon = 0.0001; // Define the tolerance level for comparison
    for (i, &val) in a.iter().enumerate() {
        if (val - b[i]).abs() > epsilon {
            println!("{} not equal to {}", val, b[i]);
            return false;
        }
    }
    true
}

fn generate_random_array(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0.0..10.0)).collect()
}


fn benchmark_multiple() {
    let n = 4_usize.pow(5);
    let a = generate_random_array(n * n);
    let b = generate_random_array(n * n);
    let mut c = vec![0.0; n * n];
    let mut d = vec![0.0; n * n];

    // Measure time for normal matrix multiplication
    let start = Instant::now();
    multiply_matrices_normal(&a, &b, &mut c, n);
    let duration_normal = start.elapsed();

    // Measure time for SIMD matrix multiplication
    let start = Instant::now();
    unsafe {
        multiply_matrices_simd(&a, &b, &mut d, n);
    }
    let duration_simd = start.elapsed();

    println!("Normal matrix multiplication took: {:?}", duration_normal);
    println!("SIMD matrix multiplication took: {:?}", duration_simd);

    assert!(arrays_equal(&c, &d));
}

fn main() {
    println!("Running benchmarks");
    println!("Benchmarking array addition");
    benchmark_add();
    println!("Benchmarking matrix multiplication");
    benchmark_multiple();
    println!("Benchmarks completed");
}

