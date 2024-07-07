use std::time::Instant;
use rand::Rng;
use std::fs::File;
use std::io::Write;

#[target_feature(enable = "neon")]
unsafe fn add_arrays_simd(a: &[f32], b: &[f32], c: &mut [f32]) {
    // NEON intrinsics for ARM architecture
    use core::arch::aarch64::*;

    let chunks = a.len() / 4;
    for i in 0..chunks {
        // Load 4 elements from each array into a NEON register
        let a_chunk = vld1q_f32(a.as_ptr().add(i * 4));
        let b_chunk = vld1q_f32(b.as_ptr().add(i * 4));
        let c_chunk = vaddq_f32(a_chunk, b_chunk);
        // Store the result back to memory
        vst1q_f32(c.as_mut_ptr().add(i * 4), c_chunk);
    }

    // Handle the remaining elements that do not fit into a 128-bit register
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
    let sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];
    let mut results = String::from("input_size,normal,simd\n");

    for &size in &sizes {
        let mut avg_duration_normal = 0;
        let mut avg_duration_simd = 0;
        let num_iterations = 10;

        for _ in 0..num_iterations {
            let a = generate_random_array(size);
            let b = generate_random_array(size);
            let mut c = vec![0.0; size];
            let mut d = vec![0.0; size];

            let start = Instant::now();
            add_arrays(&a, &b, &mut c);
            let duration_normal = start.elapsed().as_nanos();
            avg_duration_normal += duration_normal;

            let start = Instant::now();
            unsafe {
                add_arrays_simd(&a, &b, &mut d);
            }
            let duration_simd = start.elapsed().as_nanos();
            avg_duration_simd += duration_simd;
            assert!(arrays_equal(&c, &d));
        }
        avg_duration_normal /= num_iterations;
        avg_duration_simd /= num_iterations;

        results.push_str(&format!("{},{},{}\n", size, avg_duration_normal, avg_duration_simd));
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
    // NEON intrinsics for ARM architecture
    use core::arch::aarch64::*;
    for i in 0..n {
        for j in 0..n {
            // Initialize a register to hold the sum
            let mut sum = vdupq_n_f32(0.0);

            for k in (0..n).step_by(4) {
                // Load 4 elements from matrix A into a NEON register
                let a_vec = vld1q_f32(a.as_ptr().add(i * n + k));
                // Load elements individually to form a column vector from matrix B
                let b0 = *b.get_unchecked(k * n + j);
                let b1 = *b.get_unchecked((k + 1) * n + j);
                let b2 = *b.get_unchecked((k + 2) * n + j);
                let b3 = *b.get_unchecked((k + 3) * n + j);

                // Form a NEON register from the column vector
                let b_vec = vsetq_lane_f32::<0>(b0, vdupq_n_f32(0.0));
                let b_vec = vsetq_lane_f32::<1>(b1, b_vec);
                let b_vec = vsetq_lane_f32::<2>(b2, b_vec);
                let b_vec = vsetq_lane_f32::<3>(b3, b_vec);

                // Intrinsic to perform (a * b) + c
                sum = vfmaq_f32(sum, a_vec, b_vec);
            }
            // Horizontal add the elements in the sum register
            let result = vaddvq_f32(sum);
            // Store the result in the output matrix
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


fn benchmark_multiply() {
    let powers = [1, 2, 3, 4];
    let mut results = String::from("input_size,normal,simd\n");

    for &n in &powers {
        let size = 4_usize.pow(n as u32);
        let mut avg_duration_normal = 0;
        let mut avg_duration_simd = 0;
        let num_iterations = 10;

        for _ in 0..num_iterations {
            let a = generate_random_array(size * size);
            let b = generate_random_array(size * size);
            let mut c = vec![0.0; size * size];
            let mut d = vec![0.0; size * size];

            let start = Instant::now();
            multiply_matrices_normal(&a, &b, &mut c, size);
            let duration_normal = start.elapsed().as_nanos();
            avg_duration_normal += duration_normal;

            let start = Instant::now();
            unsafe {
                multiply_matrices_simd(&a, &b, &mut d, size);
            }
            let duration_simd = start.elapsed().as_nanos();
            avg_duration_simd += duration_simd;
        }
        avg_duration_normal /= num_iterations;
        avg_duration_simd /= num_iterations;

        results.push_str(&format!("{},{},{}\n", size, avg_duration_normal, avg_duration_simd));
    }

    let mut file = File::create("benchmark_multiply_results.csv").unwrap();
    file.write_all(results.as_bytes()).unwrap();

}

fn main() {
    println!("Running benchmarks");
    println!("Benchmarking array addition");
    benchmark_add();
    println!("Benchmarking matrix multiplication");
    benchmark_multiply();
    println!("Benchmarks completed");
}

