//! Per-scheme variant of `pretokenize_profile`: same single-pass loop over
//! OWT with every yielded pretoken black_boxed, scheme selected via the
//! SCHEME env var (r50k | cl100k | olmo3 | qwen2 | qwen3_5). Used for
//! interleaved A/B runs of the mask-scanner schemes.

use gigatok_rs::pretokenize::{
    FastCl100kPretokenizer, FastOlmo3Pretokenizer, FastQwen2Pretokenizer,
    FastQwen35Pretokenizer, FastR50kPretokenizer,
};
use std::hint::black_box;
use std::time::Instant;

fn load_input() -> Vec<u8> {
    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    eprintln!("Reading {owt_path:?}...");
    let t0 = Instant::now();

    let input = match std::env::var("ENCODE_MB") {
        Ok(mb) => {
            use std::io::Read;
            let max_bytes = mb
                .trim()
                .parse::<usize>()
                .expect("ENCODE_MB must be an integer")
                * 1_000_000;
            let file = std::fs::File::open(&owt_path).expect("Could not open ~/data/owt_train.txt");
            let mut data = Vec::with_capacity(max_bytes);
            file.take(max_bytes as u64)
                .read_to_end(&mut data)
                .expect("read failed");
            let mut end = data.len();
            while end > 0 && std::str::from_utf8(&data[..end]).is_err() {
                end -= 1;
            }
            data.truncate(end);
            eprintln!(
                "Capped input to {} MB (ENCODE_MB={mb})",
                data.len() / 1_000_000
            );
            data
        }
        Err(_) => std::fs::read(&owt_path).expect("Could not read ~/data/owt_train.txt"),
    };

    let size_gb = input.len() as f64 / 1e9;
    eprintln!(
        "Read {:.2} GB in {:.1}s",
        size_gb,
        t0.elapsed().as_secs_f64()
    );
    input
}

macro_rules! drive {
    ($ty:ty, $buf:expr) => {{
        let mut total_tokens: usize = 0;
        let mut iter = <$ty>::new($buf);
        while let Some(pretoken) = iter.next() {
            black_box(pretoken);
            total_tokens += 1;
        }
        total_tokens
    }};
}

fn main() {
    let input = load_input();
    let size_gb = input.len() as f64 / 1e9;
    let buf: &[u8] = &input;
    let scheme = std::env::var("SCHEME").unwrap_or_else(|_| "r50k".to_string());

    eprintln!("Pretokenizing ({scheme}, single-threaded, whole buffer)...");
    let start = Instant::now();
    let total_tokens = match scheme.as_str() {
        "r50k" => drive!(FastR50kPretokenizer, buf),
        "cl100k" => drive!(FastCl100kPretokenizer, buf),
        "olmo3" => drive!(FastOlmo3Pretokenizer, buf),
        "qwen2" => drive!(FastQwen2Pretokenizer, buf),
        "qwen3_5" => drive!(FastQwen35Pretokenizer, buf),
        other => panic!("unknown SCHEME {other:?}"),
    };
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gb = size_gb / elapsed;

    eprintln!(
        "{total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
