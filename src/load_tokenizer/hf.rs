//! Load HuggingFace tokenizer.json files.
//!
//! Supports SentencePiece-style BPE tokenizers (e.g. Llama 2) that use
//! `byte_fallback=true`.

use crate::bpe::SentencePieceBPE;
use crate::token::TokenId;
use eyre::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// JSON schema (only the fields we need)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizerJson {
    model: Model,
}

#[derive(Deserialize)]
struct Model {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    merges: Vec<[String; 2]>,
    #[serde(default)]
    byte_fallback: bool,
}

// ---------------------------------------------------------------------------
// Token string → raw bytes conversion
// ---------------------------------------------------------------------------

/// Convert a HuggingFace vocab string to raw bytes.
///
/// - Byte-fallback tokens `<0xHH>` → the single byte.
/// - Everything else → its UTF-8 bytes (▁ is kept as-is).
fn token_str_to_bytes(s: &str) -> Vec<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }
    s.as_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a HuggingFace `tokenizer.json` that uses SentencePiece-style BPE with
/// byte fallback (e.g. Llama 2 / TinyLlama).
///
/// Returns a [`SentencePieceBPE`] that preserves the original HF token IDs.
pub fn load_hf_tokenizer(path: impl AsRef<Path>) -> Result<SentencePieceBPE> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let tj: TokenizerJson = sonic_rs::from_slice(&data)
        .with_context(|| format!("Failed to parse {}", path.display()))?;

    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        tj.model.byte_fallback,
        "Only byte_fallback tokenizers are supported"
    );

    let hf_vocab = &tj.model.vocab;
    let hf_merges = &tj.model.merges;

    // --- Build vocab (preserving original HF IDs) ----------------------------

    let max_id = hf_vocab.values().max().copied().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    let mut vocab_inv: HashMap<Arc<[u8]>, TokenId> = HashMap::with_capacity(hf_vocab.len());

    // Insert byte-fallback tokens first, then character tokens, so that
    // character tokens win in vocab_inv when both map to the same bytes.
    let mut byte_fallback_entries = Vec::new();
    let mut other_entries = Vec::new();
    for (tok_str, &id) in hf_vocab {
        if tok_str.len() == 6 && tok_str.starts_with("<0x") && tok_str.ends_with('>') {
            byte_fallback_entries.push((tok_str, id));
        } else {
            other_entries.push((tok_str, id));
        }
    }
    for (tok_str, id) in byte_fallback_entries {
        let bytes: Arc<[u8]> = token_str_to_bytes(tok_str).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }
    for (tok_str, id) in other_entries {
        let bytes: Arc<[u8]> = token_str_to_bytes(tok_str).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }

    // --- Extract byte-fallback token IDs -------------------------------------

    let mut byte_fallback_ids = [TokenId::from(0u32); 256];
    for byte_val in 0u16..=255 {
        let key = format!("<0x{:02X}>", byte_val);
        let &id = hf_vocab
            .get(&key)
            .ok_or_else(|| eyre::eyre!("Missing byte fallback token {key}"))?;
        byte_fallback_ids[byte_val as usize] = TokenId::from(id);
    }

    // --- Build merge table (with explicit ranks) -----------------------------

    let mut merges: HashMap<(TokenId, TokenId), (TokenId, u32)> =
        HashMap::with_capacity(hf_merges.len());

    let hf_str_to_id = |s: &str| -> Option<TokenId> {
        let bytes = token_str_to_bytes(s);
        vocab_inv.get(bytes.as_slice()).copied()
    };

    for (rank, [str_a, str_b]) in hf_merges.iter().enumerate() {
        let id_a = match hf_str_to_id(str_a) {
            Some(id) => id,
            None => continue,
        };
        let id_b = match hf_str_to_id(str_b) {
            Some(id) => id,
            None => continue,
        };

        let merged_str = format!("{str_a}{str_b}");
        let id_merged = match hf_str_to_id(&merged_str) {
            Some(id) => id,
            None => continue,
        };

        merges.entry((id_a, id_b)).or_insert((id_merged, rank as u32));
    }

    Ok(SentencePieceBPE {
        merges,
        vocab,
        vocab_inv,
        byte_fallback_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_str_to_bytes() {
        assert_eq!(token_str_to_bytes("<0x00>"), vec![0x00]);
        assert_eq!(token_str_to_bytes("<0xFF>"), vec![0xFF]);
        assert_eq!(token_str_to_bytes("<0x0A>"), vec![0x0A]);
        assert_eq!(token_str_to_bytes("hello"), b"hello".to_vec());
        assert_eq!(token_str_to_bytes("▁the"), "▁the".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("▁"), "▁".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("<unk>"), b"<unk>".to_vec());
        assert_eq!(token_str_to_bytes("<s>"), b"<s>".to_vec());
    }

    #[test]
    fn test_load_tinyllama() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/scripts/tinyllama_tokenizer.json"
        );
        let tokenizer = load_hf_tokenizer(path).unwrap();
        eprintln!("{:?}", tokenizer);
    }

    #[test]
    fn test_encode_hello() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/scripts/tinyllama_tokenizer.json"
        );
        let tokenizer = load_hf_tokenizer(path).unwrap();
        let normalized = SentencePieceBPE::normalize("Hello world");
        let ids = tokenizer.encode(&normalized);
        eprintln!("Encoded: {:?}", ids);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, b"Hello world");
    }
}
