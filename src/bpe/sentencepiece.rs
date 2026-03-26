use crate::bpe::bpe_merge_symbols_ranked;
use crate::token::TokenId;
use std::collections::HashMap;
use std::sync::Arc;

/// SentencePiece uses U+2581 (▁) as a space marker.
const SENTENCEPIECE_SPACE: char = '\u{2581}';

/// A tokenizer that mirrors SentencePiece BPE with `byte_fallback`.
///
/// Encoding works in two phases:
/// 1. **Character-level initialisation**: each Unicode character is looked up
///    in the vocab.  If not found, its UTF-8 bytes are used via byte-fallback
///    tokens.
/// 2. **BPE merges**: the resulting token sequence is merged using the learned
///    merge rules (priority by explicit merge rank, not token ID).
pub struct SentencePieceBPE {
    /// Merges with explicit rank: `(a, b) → (merged, rank)`.
    pub(crate) merges: HashMap<(TokenId, TokenId), (TokenId, u32)>,
    pub(crate) vocab: Vec<Arc<[u8]>>,
    /// Maps byte sequences → token IDs (for character lookup).
    pub(crate) vocab_inv: HashMap<Arc<[u8]>, TokenId>,
    /// Token ID for each byte value (0x00–0xFF) via `<0xHH>` fallback tokens.
    pub(crate) byte_fallback_ids: [TokenId; 256],
}

impl std::fmt::Debug for SentencePieceBPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SentencePieceBPE {{ vocab_size: {}, merges_count: {} }}",
            self.vocab.len(),
            self.merges.len(),
        )
    }
}

impl SentencePieceBPE {
    /// Normalise input the way the Llama tokenizer does:
    /// prepend ▁ and replace all spaces with ▁.
    pub fn normalize(input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(input.len() + 3);
        out.push(SENTENCEPIECE_SPACE);
        for ch in input.chars() {
            if ch == ' ' {
                out.push(SENTENCEPIECE_SPACE);
            } else {
                out.push(ch);
            }
        }
        out
    }

    /// Encode a text chunk.  The input should already be normalised.
    /// Each Unicode character is looked up in the vocab; unknown
    /// characters fall back to byte tokens.  Then BPE merges are applied.
    pub fn encode(&self, input: &str) -> Vec<TokenId> {
        let mut symbols = Vec::new();
        for ch in input.chars() {
            let mut buf = [0u8; 4];
            let ch_bytes = ch.encode_utf8(&mut buf).as_bytes();
            if let Some(&id) = self.vocab_inv.get(ch_bytes) {
                symbols.push(id);
            } else {
                // Byte fallback: decompose character to UTF-8 bytes.
                for &b in ch_bytes {
                    symbols.push(self.byte_fallback_ids[b as usize]);
                }
            }
        }
        bpe_merge_symbols_ranked(&self.merges, &mut symbols);
        symbols
    }

    /// Decode token IDs back to a UTF-8 string.
    /// Reverses the normalisation: ▁ → space, then strips the leading space
    /// that was prepended during normalisation.
    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        let mut raw = Vec::new();
        for &t in tokens {
            let idx: usize = t.into();
            if idx < self.vocab.len() {
                raw.extend_from_slice(&self.vocab[idx]);
            }
        }
        // Replace ▁ (3 UTF-8 bytes: E2 96 81) with ASCII space.
        let text = String::from_utf8_lossy(&raw);
        let mut out: Vec<u8> = text.replace(SENTENCEPIECE_SPACE, " ").into_bytes();
        // Strip the leading space that normalize() prepended.
        if out.first() == Some(&b' ') {
            out.remove(0);
        }
        out
    }
}
