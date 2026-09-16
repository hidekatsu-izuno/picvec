//! Explicit benchmark for scheduling repeated paint-sized color batches.
use picvec::{
    color::{rgb_to_oklab, Oklab},
    edge::oklab_values,
};
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};

fn main() {
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for size in [32, 256, 768, 2048, 4096, 16384] {
                let pixels: Vec<_> = (0..size).map(|i| [((i*73)%257) as f32 / 256.0, ((i*31)%257) as f32 / 256.0, ((i*17)%257) as f32 / 256.0]).collect();
                for nested in [false, true] {
                    for mode in ["parallel", "serial", "dispatch"] {
                        let run = || {
                            let p = black_box(&pixels);
                            let values: Vec<Oklab> = match mode {
                                "parallel" => p.par_iter().copied().map(rgb_to_oklab).collect(),
                                "serial" => p.iter().copied().map(rgb_to_oklab).collect(),
                                _ => oklab_values(p),
                            };
                            black_box(values);
                        };
                        let start = Instant::now();
                        if nested { (0..256).into_par_iter().for_each(|_| run()); }
                        else { for _ in 0..256 { run(); } }
                        eprintln!("threads={threads} size={size} nested={nested} mode={mode} seconds={:.6}", start.elapsed().as_secs_f64());
                    }
                }
            }
        });
    }
}
