//! Whole-document multithreaded encode benchmark, mirroring
//! `BPETokenizer.encode_files` on a single plain-text file: the file is read
//! by path, the entire input is ONE document, and it is split at
//! pretoken-safe boundaries (token-identical to a serial pass) and encoded
//! by a persistent worker pool, then gathered into one flat id buffer.
//!
//! Run with: cargo bench --bench encode_doc              (2 GB default)
//!           ENCODE_MB=500 cargo bench --bench encode_doc
//!           TOKENIZER_JSON=data/qwen3_5_tokenizer.json cargo bench --bench encode_doc

use jeton_rs::load_tokenizer::hf::load_hf_bpe;
use jeton_rs::pretokenize::safe_split_ranges;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

/// Same chunking policy as the Python bindings in lib.rs.
const MIN_CHUNK_BYTES: usize = 1 << 20;
const DEFAULT_MB: usize = 2000;

fn chunk_target_bytes(total_bytes: usize) -> usize {
    (total_bytes / (4 * rayon::current_num_threads())).max(MIN_CHUNK_BYTES)
}

fn load_input() -> Vec<u8> {
    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    eprintln!("Reading {owt_path:?}...");
    let t0 = Instant::now();

    use std::io::Read;
    let mb = std::env::var("ENCODE_MB")
        .map(|mb| mb.trim().parse::<usize>().expect("ENCODE_MB must be an integer"))
        .unwrap_or(DEFAULT_MB);
    let max_bytes = mb * 1_000_000;
    let file = std::fs::File::open(&owt_path).expect("Could not open ~/data/owt_train.txt");
    let mut data = Vec::with_capacity(max_bytes);
    file.take(max_bytes as u64)
        .read_to_end(&mut data)
        .expect("read failed");
    // Back up to a UTF-8 character boundary.
    let mut end = data.len();
    while end > 0 && std::str::from_utf8(&data[..end]).is_err() {
        end -= 1;
    }
    data.truncate(end);

    eprintln!(
        "Read {:.2} GB in {:.1}s (ENCODE_MB={mb})",
        data.len() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    data
}

fn main() {
    let tokenizer_json = std::env::var("TOKENIZER_JSON")
        .unwrap_or_else(|_| "data/olmo3_tokenizer.json".to_string());
    let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&tokenizer_json);
    eprintln!("Loading tokenizer from {tokenizer_path:?}...");
    let tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load tokenizer");

    let input = load_input();
    let size_mb = input.len() as f64 / 1e6;

    // The whole input is one document, split at pretoken-safe boundaries.
    let added = tokenizer.added_token_contents();
    let ranges = safe_split_ranges(&input, chunk_target_bytes(input.len()), &added);
    eprintln!(
        "1 document, {} pretoken-safe fragments, {} threads\n",
        ranges.len(),
        rayon::current_num_threads()
    );

    // Persistent worker pool, retained across rounds like the binding's
    // WorkerPool — pretoken caches stay warm after round 0.
    let workers: Vec<Mutex<_>> = (0..rayon::current_num_threads())
        .map(|_| Mutex::new(tokenizer.fork()))
        .collect();

    for round in 0..5 {
        let t0 = Instant::now();
        let fragments: Vec<Vec<u32>> = ranges
            .par_iter()
            .map(|range| {
                let mut tok = 'acquire: loop {
                    for w in &workers {
                        if let Ok(guard) = w.try_lock() {
                            break 'acquire guard;
                        }
                    }
                    std::thread::yield_now();
                };
                let mut ids: Vec<u32> = vec![];
                tok.encode_with_added_tokens(&input[range.clone()], |tokens| {
                    for &e in tokens {
                        ids.push(e.into())
                    }
                });
                ids
            })
            .collect();

        // Gather into one flat buffer, like the binding's ragged assembly.
        let total: usize = fragments.iter().map(|f| f.len()).sum();
        let mut flat = vec![0u32; total];
        let mut rest: &mut [u32] = &mut flat;
        let mut slices = Vec::with_capacity(fragments.len());
        for f in &fragments {
            let (head, tail) = rest.split_at_mut(f.len());
            slices.push(head);
            rest = tail;
        }
        slices
            .into_par_iter()
            .zip(fragments.par_iter())
            .for_each(|(dst, f)| dst.copy_from_slice(f));

        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!(
            "round {round}: {} tokens in {elapsed:.3}s — {:.0} MB/s",
            flat.len(),
            size_mb / elapsed
        );
    }
}
