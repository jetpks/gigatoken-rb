use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use toker_rs::pretokenize::{PretokenizerIter, pretoken_combinator::pretokens_iterator};

fn state_machine_pretokenize(input: &[u8]) -> Vec<&[u8]> {
    let iter = PretokenizerIter::new(input);
    iter.map(|pretoken| pretoken.0).collect()
}

fn winnow_pretokenize(input: &[u8]) -> Vec<&[u8]> {
    let mut iter = pretokens_iterator(unsafe { std::str::from_utf8_unchecked(input) });
    iter.map(|pretoken| pretoken.0).collect()
}

fn regex_pretokenize<'a>(re: &fancy_regex::Regex, input: &'a [u8]) -> Vec<&'a [u8]> {
    let text = unsafe { std::str::from_utf8_unchecked(input) };
    re.find_iter(text)
        .map(|m| {
            let m = m.unwrap();
            &input[m.start()..m.end()]
        })
        .collect()
}

const TARGET_BENCH_SIZE: usize = 100_000_000; // ~100 MB

/// Load OWT data, truncated to a UTF-8-safe boundary near `max_bytes`.
fn load_owt(max_bytes: usize) -> Vec<u8> {
    let data_dir = std::env::home_dir().unwrap().join("data");
    let all_bytes =
        std::fs::read(data_dir.join("owt_train.txt")).expect("Could not read ~/data/owt_train.txt");
    let mut end = max_bytes.min(all_bytes.len());
    // Back up to a UTF-8 character boundary
    while end > 0 && !std::str::from_utf8(&all_bytes[..end]).is_ok() {
        end -= 1;
    }
    all_bytes[..end].to_vec()
}

fn pretokenize_benches(c: &mut Criterion) {
    let input = load_owt(TARGET_BENCH_SIZE);
    let input_len = input.len() as u64;
    eprintln!("Benchmark input size: {:.1} MB", input_len as f64 / 1e6);

    let mut group = c.benchmark_group("pretokenize");
    group.throughput(Throughput::Bytes(input_len));
    group.sample_size(10);

    group.bench_function("state_machine", |b| {
        b.iter(|| {
            let count = PretokenizerIter::new(&input).count();
            black_box(count);
        });
    });

    group.bench_function("winnow", |b| {
        b.iter(|| {
            let count =
                pretokens_iterator(unsafe { std::str::from_utf8_unchecked(&input) }).count();
            black_box(count);
        });
    });

    let re = fancy_regex::Regex::new(
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    )
    .unwrap();

    group.bench_function("regex", |b| {
        b.iter(|| {
            let text = unsafe { std::str::from_utf8_unchecked(&input) };
            let count = re.find_iter(text).count();
            black_box(count);
        });
    });

    group.finish();
}

criterion_group!(benches, pretokenize_benches);
criterion_main!(benches);
