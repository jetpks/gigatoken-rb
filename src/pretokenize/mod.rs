//! The pretokenizer is responsible for taking a single document and producing an iterator of
//! pretokens.
use crate::bpe_train::PretokenizeableSpec;
pub(crate) use crate::pretokenize::pretoken::Pretoken;
use crate::pretokenize::pretokenize_traits::{
    ParallelMergeCounts, ParallelPretokenCountable, PretokenCountable,
};
use itertools::Itertools;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::HashMap;

mod options;
mod pretoken;
mod pretoken_chunks;
pub mod pretoken_combinator;
pub mod pretoken_state_machine;
mod pretokenize_traits;
mod unicode;

pub use options::PretokenizerType;
pub use pretoken_state_machine::{PretokenizerIter, pretokenize_as_iter};

pub fn find_boundaries(bytes: &[u8]) -> Vec<usize> {
    fn advance_to_boundary(input: &[u8]) -> usize {
        for (i, (first, second)) in input.iter().tuple_windows().enumerate() {
            if matches!((first, second), (b'.', b' ')) {
                return i + 1;
            }
        }
        panic!("No boundary found in input");
    }

    let n_threads = rayon::current_num_threads();
    eprintln!("Using {n_threads} threads for pretokenization");
    let chunk_size = bytes.len().div_ceil(n_threads);
    let mut boundaries: Vec<usize> = (0..=n_threads)
        .map(|i| min(i * chunk_size, bytes.len()))
        .collect();
    for b in boundaries[1..n_threads].iter_mut() {
        *b += advance_to_boundary(&bytes[*b..]);
    }
    boundaries
}

pub fn pretokenize_par_bytes(
    bytes: &[u8],
) -> HashMap<Pretoken<'_>, usize, rustc_hash::FxBuildHasher> {
    let start_time = std::time::Instant::now();
    let boundaries = find_boundaries(bytes);
    let merged_counts = boundaries
        .par_windows(2)
        .map(|window| {
            let start = window[0];
            let end = window[1];
            pretokenize_count(&bytes[start..end])
        })
        .par_merge_counts();

    let time_elapsed = start_time.elapsed();
    eprintln!("Pretokenization took {time_elapsed:?}");

    merged_counts
    // merged_counts
    //     .into_iter()
    //     .map(|(k, v)| (k.to_owned(), v))
    //     .collect()
}

pub fn pretokenize_par(
    pretokenizeable: PretokenizeableSpec,
) -> HashMap<Pretoken, usize, rustc_hash::FxBuildHasher> {
    match pretokenizeable {
        PretokenizeableSpec::Bytes(s) => pretokenize_par_bytes(s),
        #[cfg(feature = "parquet")]
        PretokenizeableSpec::Parquet(path) => pretokenize_par_parquet(&path),
    }
}

// Only when the "parquet" feature is enabled
#[cfg(feature = "parquet")]
pub fn pretokenize_par_parquet(
    parquet_path: &Path,
) -> HashMap<Vec<u8>, usize, rustc_hash::FxBuildHasher> {
    use indicatif::{ProgressBar, ProgressIterator};
    use polars::prelude::*;
    let parquet_path = PlPath::Local(Arc::from(parquet_path.to_owned()));

    let df = LazyFrame::scan_parquet(parquet_path.clone(), ScanArgsParquet::default()).unwrap();

    let length = df.select([len()]).collect().unwrap();
    let length_value = length.get(0).unwrap();
    let length_value = length_value.first().unwrap();
    let length_value = match length_value {
        AnyValue::UInt32(v) => *v,
        _ => panic!("Unexpected length value type"),
    };

    eprintln!("Dataframe length: {:?}", length_value);

    let n_chunks = rayon::current_num_threads();
    let chunk_size = (length_value as usize).div_ceil(n_chunks);
    let total_counts = (0..n_chunks)
        .par_bridge()
        .map(|i| {
            let df =
                LazyFrame::scan_parquet(parquet_path.clone(), ScanArgsParquet::default())
                    .unwrap();
            let mut thread_counts = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
            let start = i * chunk_size;
            let end = min((i + 1) * chunk_size, length_value as usize);
            let m_chunks = 1024;
            let inner_chunk_size = (end - start).div_ceil(1024);
            for j in (0..m_chunks).progress_with(if i == 0 {
                ProgressBar::new(m_chunks as u64)
                    .with_finish(indicatif::ProgressFinish::AndLeave)
                    .with_style(
                        indicatif::ProgressStyle::default_bar()
                            .template(
                                "Pretokenizing and counting [{elapsed_precise}/{duration_precise}, ({per_sec})] [{wide_bar}] {pos}/{len} ({eta_precise} remaining)",
                            )
                            .unwrap(),
                    )
            } else {
                ProgressBar::hidden()
            }) {
                let inner_start = start + j * inner_chunk_size;
                let inner_end = min(start + (j + 1) * inner_chunk_size, end);
                let chunk = df.clone().slice(inner_start as i64, (inner_end - inner_start) as u32);
                let loaded = chunk.collect().unwrap();

                let col = loaded.column("text").unwrap();
                let strings = col.str().expect("Didn't find strings");
                let freqs = loaded.column("frequency").unwrap();
                let freqs = freqs.i64().expect("Didn't find frequencies");


                strings.iter().zip(freqs.iter()).flat_map(|(s, f)| match (s, f) {
                    (Some(s), Some(f)) => Some((s.as_bytes(), f as usize)),
                    (Some(s), None) => Some((s.as_bytes(), 1)),
                    _ => None,
                }).for_each(|(s, f)| {
                    pretokenize_as_iter(s).for_each(|pretoken| {
                        thread_counts
                            .entry(pretoken.to_owned())
                            .and_modify(|e| *e += f)
                            .or_insert(f);
                    })
                });
            }
            thread_counts
        })
        .reduce(
            || HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            |mut acc, counts| {
                if acc.is_empty() {
                    return counts;
                }

                for (k, v) in counts {
                    *acc.entry(k).or_insert(0) += v;
                }
                acc
            },
        );

    total_counts
}

/// Return counts of all pretokens.
pub fn pretokenize_count(bytes: &[u8]) -> HashMap<Pretoken<'_>, usize, rustc_hash::FxBuildHasher> {
    let string = unsafe { std::str::from_utf8_unchecked(bytes) };
    string
        .split("<|endoftext|>")
        .flat_map(|s| pretokenize_as_iter(s.as_bytes().into()))
        .pretoken_count()
}

// pub fn count_pretokens<'a>(
//     pretoken_iter: impl Iterator<Item = &'a [u8]>,
// ) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher> {
//     let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
//     pretoken_iter.for_each(|token| {
//         hashmap.entry(token).and_modify(|e| *e += 1).or_insert(1);
//     });
//     hashmap
// }

pub fn count_pretokens_weighted<'a>(
    pretoken_weight_iter: impl Iterator<Item = (&'a [u8], usize)>,
) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher> {
    let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
    pretoken_weight_iter.for_each(|(token, weight)| {
        hashmap
            .entry(token)
            .and_modify(|e| *e += weight)
            .or_insert(weight);
    });
    hashmap
}

pub fn pretokenize_doc_iterable<'a>(
    docs: impl Iterator<Item = &'a [u8]>,
) -> impl Iterator<Item = Pretoken<'a>> {
    docs.flat_map(|doc| pretokenize_as_iter(doc.into()))
}

pub fn pretokenize_with_endoftext(
    bytes: &[u8],
) -> HashMap<Pretoken<'_>, usize, rustc_hash::FxBuildHasher> {
    let string = unsafe { std::str::from_utf8_unchecked(bytes) };

    let mut hashmap = HashMap::default();

    string
        .split("<|endoftext|>")
        .flat_map(|part| pretokenize_as_iter(part.as_bytes().into()))
        .for_each(|token| {
            hashmap.entry(token).and_modify(|e| *e += 1).or_insert(1);
        });

    hashmap
}

struct Pretokenizer {
    special_tokens: Vec<(Vec<u8>, u32)>, // Split on these tokens, keep them in the stream
}

impl Pretokenizer {
    pub fn new(special_tokens: Vec<(Vec<u8>, u32)>) -> Self {
        Pretokenizer { special_tokens }
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use std::fs;

    use super::*;

    const GPT2_REGEX: &str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    /// Load the first `max_bytes` of ~/data/owt_train.txt, truncated to a UTF-8 boundary.
    fn load_owt(max_bytes: usize) -> Vec<u8> {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes =
            fs::read(data_dir.join("owt_train.txt")).expect("Could not read ~/data/owt_train.txt");
        let mut end = max_bytes.min(all_bytes.len());
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        all_bytes[..end].to_vec()
    }

    /// Compare the state-machine pretokenizer against the GPT-2 reference regex
    /// on ~5 MB of OWT data, token by token.
    #[test]
    fn test_pretokenizer_matches_regex_owt() {
        const SIZE: usize = 5_000_000;
        let input = load_owt(SIZE);
        eprintln!(
            "Testing pretokenizer vs regex on {:.1} MB of OWT",
            input.len() as f64 / 1e6
        );

        let re = fancy_regex::Regex::new(GPT2_REGEX).unwrap();
        let text = std::str::from_utf8(&input).unwrap();

        let mut sm_iter = pretokenize_as_iter(&input);
        let mut re_iter = re.find_iter(text);
        let mut token_idx: usize = 0;
        let mut recent: Vec<(String, String)> = Vec::new();

        loop {
            match (sm_iter.next(), re_iter.next()) {
                (Some(sm_tok), Some(re_match)) => {
                    let re_match = re_match.expect("regex match error");
                    let sm_str = String::from_utf8_lossy(sm_tok.0);
                    let re_str = &text[re_match.start()..re_match.end()];
                    recent.push((sm_str.to_string(), re_str.to_string()));
                    if recent.len() > 10 {
                        recent.remove(0);
                    }
                    assert_eq!(
                        sm_str, re_str,
                        "Mismatch at token {token_idx} (byte ~{}).\n  state machine: {:?}\n  regex:         {:?}\n  recent tokens: {:?}",
                        re_match.start(), sm_str, re_str, recent
                    );
                }
                (None, None) => break,
                (Some(sm_tok), None) => {
                    panic!(
                        "State machine produced extra token at index {token_idx}: {:?}\n  recent: {:?}",
                        String::from_utf8_lossy(sm_tok.0),
                        recent
                    );
                }
                (None, Some(re_match)) => {
                    let re_match = re_match.expect("regex match error");
                    panic!(
                        "Regex produced extra token at index {token_idx}: {:?}\n  recent: {:?}",
                        &text[re_match.start()..re_match.end()],
                        recent
                    );
                }
            }
            token_idx += 1;
        }
        eprintln!("All {token_idx} tokens match.");
    }

    #[test]
    fn test_pretokenizer_ts() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let file_bytes = fs::read(data_dir.join("TinyStoriesV2-GPT4-train.txt")).unwrap();

        let pretokenized_counts = pretokenize_as_iter(&file_bytes).counts();
        eprintln!("Pretokenized {} unique tokens", pretokenized_counts.len());

        let mut sorted_counts: Vec<_> = pretokenized_counts.iter().collect();
        sorted_counts.sort_by_key(|&(_, &v)| v);
        sorted_counts.reverse();
        for &(&token, &count) in sorted_counts.iter().take(100) {
            eprintln!("{1}: {0}", String::from_utf8_lossy(&token), count);
        }
    }

    #[test]
    fn test_pretokenizer_owt_length() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let file_bytes = fs::read(data_dir.join("owt_train.txt")).unwrap();

        let pretokens_count = pretokenize_as_iter(&file_bytes).count();
        eprintln!("Pretokenized {pretokens_count} tokens");
    }
}
