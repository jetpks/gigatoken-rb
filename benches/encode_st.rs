use gigatoken_rs::load_tokenizer::hf::{HfTokenizer, load_hf_slice};
use gigatoken_rs::pretokenize::FastR50kPretokenizer;
use std::path::PathBuf;
use std::time::Instant;

mod common;
fn main() {
    common::allow_thp();
    // ENCODE_TOKENIZER overrides the tokenizer.json: a local path (e.g.
    // data/qwen3_tokenizer.json to bench the qwen2-scheme encode path) or a
    // HuggingFace repo id (e.g. Qwen/Qwen2-1.5B-Instruct), served from the
    // standard HF cache and downloaded into it on a miss. Encoding then runs
    // through the scheme dispatch instead of the hardcoded r50k pretokenizer.
    let tokenizer_override = std::env::var("ENCODE_TOKENIZER").ok().map(|value| {
        let path = PathBuf::from(&value);
        if !path.exists() && gigatoken_rs::load_tokenizer::hub::looks_like_repo_id(&value) {
            gigatoken_rs::load_tokenizer::hub::hub_file(&value, "tokenizer.json", "main")
                .expect("Could not fetch tokenizer.json from the HuggingFace Hub")
        } else {
            path
        }
    });
    let tokenizer_path = tokenizer_override.clone().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2_tokenizer.json")
    });
    eprintln!("Loading tokenizer from {tokenizer_path:?}...");
    let data = std::fs::read(&tokenizer_path).expect("Could not read tokenizer.json");
    // byte_fallback configs (Llama/gemma-style SentencePiece) dispatch to the
    // SP encode path; everything else to ByteLevel BPE, as in benches/encode.
    let tokenizer = load_hf_slice(&data).expect("Could not load tokenizer");

    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;
    // Encode the whole buffer in one pass (matches real usage; the pretokenizer
    // handles newlines itself, so pre-splitting into lines is unnecessary).
    let buf: &[u8] = &input;

    // ENCODE_PASSES=N re-encodes the same buffer N times; passes after the
    // first run with a fully warm pretoken cache, isolating the hit path.
    let passes: usize = std::env::var("ENCODE_PASSES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1);
    eprintln!("Encoding (single-threaded)...");
    // Flat output buffer, reserved once from the batch engine's
    // bytes-per-token estimate (see batch::encode_chunk) and reused across
    // passes; the token count is its length.
    let mut out: Vec<u32> = Vec::with_capacity(input.len() / 4 + 16);
    common::madvise_hugepage_capacity(&mut out);
    // ENCODE_DUMP=path writes the final pass's tokens as little-endian u32s,
    // for byte-exact A/B comparison across encode-path changes.
    let dump_path = std::env::var("ENCODE_DUMP").ok();
    match tokenizer {
        HfTokenizer::Bpe(mut tokenizer) => {
            for pass in 0..passes {
                out.clear();
                let start = Instant::now();
                if tokenizer_override.is_some() {
                    let pretokens = tokenizer.pretokenizer_type().pretokenize(buf);
                    tokenizer.memoized_encode_flat(pretokens, &mut out);
                } else {
                    tokenizer.memoized_encode_flat(FastR50kPretokenizer::new(buf), &mut out);
                }
                report_pass(pass, out.len(), size_gb, start);
            }
            let (sl, sc, ll, lc, lkb, al, ac) = tokenizer.cache_mem_stats();
            eprintln!(
                "cache: short {sl} entries (cap {sc}), long {ll} (cap {lc}, {lkb} key bytes), arena {al} tokens (cap {ac})"
            );
        }
        HfTokenizer::SentencePiece(tokenizer) => {
            // Trailing partial UTF-8 char (from the ENCODE_MB byte cap) is
            // dropped; SP encoding takes str input.
            let text = match std::str::from_utf8(buf) {
                Ok(text) => text,
                Err(e) => std::str::from_utf8(&buf[..e.valid_up_to()]).unwrap(),
            };
            let mut state = gigatoken_rs::EncodeState::new();
            for pass in 0..passes {
                out.clear();
                let start = Instant::now();
                let out_ref = &mut out;
                tokenizer.encode_raw_cb(&mut state, text, &mut |tokens| {
                    out_ref.extend(tokens.iter().map(|t| t.0))
                });
                report_pass(pass, out.len(), size_gb, start);
            }
            eprintln!("cache: {} units", state.cache_size());
        }
    }
    if let Some(path) = dump_path {
        let bytes: Vec<u8> = out.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("Could not write ENCODE_DUMP");
        eprintln!("dumped {} tokens to {path}", out.len());
    }
}

fn report_pass(pass: usize, total_tokens: usize, size_gb: f64, start: Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gb = size_gb / elapsed;
    eprintln!(
        "pass {pass}: {total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
