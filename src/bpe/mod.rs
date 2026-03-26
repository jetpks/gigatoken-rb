pub mod sentencepiece;
pub mod tiktoken;

use crate::token::TokenId;
use eyre::{Result, anyhow};
use itertools::Itertools;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ByteRemapping — shared between tokenizer types
// ---------------------------------------------------------------------------

pub struct ByteRemapping {
    mapping: Vec<u8>, // Maps string byte to symbol byte
    unmap: Vec<u8>,   // Maps symbol byte to string byte
}

impl ByteRemapping {
    pub fn from_byte_vocab(vocab: &[impl AsRef<[u8]>]) -> Result<Option<Self>> {
        let byte_remapping = vocab[..256]
            .iter()
            .map(|b| {
                let b = b.as_ref();
                if b.len() != 1 {
                    anyhow!(
                        "Byte remapping failed because vocab entry for byte is not length 1: {:?}",
                        b
                    );
                }
                Ok(b[0])
            })
            .collect::<Result<Vec<u8>>>()?;

        // Only use the byte remapping if it's not the identity mapping
        let byte_remapping = byte_remapping
            .iter()
            .enumerate()
            .any(|(i, &b)| i != b as usize)
            .then_some(byte_remapping)
            .map(|mapping| {
                let mut unmap = vec![0_u8; 256];
                for (i, &b) in mapping.iter().enumerate() {
                    unmap[b as usize] = i as u8;
                }
                ByteRemapping {
                    unmap: mapping,
                    mapping: unmap,
                }
            });
        Ok(byte_remapping)
    }
    pub fn remap_bytes(&self, bytes: &[u8]) -> Vec<u8> {
        bytes.iter().map(|&b| self.mapping[b as usize]).collect()
    }
    pub fn unmap_bytes(&self, bytes: &[u8]) -> Vec<u8> {
        bytes.iter().map(|&b| self.unmap[b as usize]).collect()
    }
}

// ---------------------------------------------------------------------------
// Shared BPE merge functions
// ---------------------------------------------------------------------------

/// Apply BPE merges to an already-initialized symbol sequence.
/// Priority is determined by the merged token's ID (lower = first).
/// This is correct for tiktoken-style tokenizers where vocab ID equals merge rank.
pub fn bpe_merge_symbols(
    merges: &HashMap<(TokenId, TokenId), TokenId>,
    symbols: &mut Vec<TokenId>,
) {
    loop {
        let candidate_merges = symbols
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
            .filter_map(|(i, (a, b))| merges.get(&(a, b)).map(|&v| (i, v)));

        let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token);

        if let Some((merge_index, merge_token)) = best_merge {
            symbols[merge_index] = merge_token;
            symbols.remove(merge_index + 1);
        } else {
            break;
        }
    }
}

/// Apply BPE merges using explicit merge ranks for priority (lower rank = first).
/// The merge table maps `(token_a, token_b) → (merged_token, rank)`.
/// This is needed for HF/SentencePiece tokenizers where merge order differs
/// from vocab ID order.
pub fn bpe_merge_symbols_ranked(
    merges: &HashMap<(TokenId, TokenId), (TokenId, u32)>,
    symbols: &mut Vec<TokenId>,
) {
    loop {
        let best_merge = symbols
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
            .filter_map(|(i, (a, b))| merges.get(&(a, b)).map(|&(tok, rank)| (i, tok, rank)))
            .min_by_key(|&(_, _, rank)| rank);

        if let Some((merge_index, merge_token, _)) = best_merge {
            symbols[merge_index] = merge_token;
            symbols.remove(merge_index + 1);
        } else {
            break;
        }
    }
}

/// Tokenize a single pretoken by mapping each byte to TokenId(byte_value)
/// then applying BPE merges (priority by merged token ID).
pub fn simple_bpe_merge(
    merges: &HashMap<(TokenId, TokenId), TokenId>,
    pre_token: &[u8],
) -> Vec<TokenId> {
    let mut symbols: Vec<TokenId> = pre_token.iter().map(|&b| TokenId::from(b as u32)).collect();
    bpe_merge_symbols(merges, &mut symbols);
    symbols
}

// Re-export the main types so existing `use crate::bpe::Tokenizer` still works.
pub use sentencepiece::SentencePieceBPE;
pub use tiktoken::Tokenizer;
