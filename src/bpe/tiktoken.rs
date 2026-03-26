use bumpalo::collections::CollectIn;
use bumpalo::collections::Vec as BumpVec;
use itertools::Itertools;

use crate::bpe::{ByteRemapping, simple_bpe_merge};
use crate::pretokenize::Pretoken;
use crate::token::TokenId;
use eyre::Result;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

/// Byte-level BPE tokenizer (tiktoken / GPT-2 style).
///
/// Initial symbols are individual bytes (0–255).  Merge priority is
/// determined by the merged token's vocab ID (lower = first), which
/// equals the merge rank for tiktoken vocabularies.
pub struct Tokenizer {
    pub(crate) merges: HashMap<(TokenId, TokenId), TokenId>,
    pub(crate) vocab: Vec<Arc<[u8]>>,
    pub(crate) vocab_inv: HashMap<Arc<[u8]>, TokenId>,
    pub(crate) byte_remapping: Option<ByteRemapping>,
    pretoken_cache: HashMap<Arc<[u8]>, Arc<[TokenId]>, rustc_hash::FxBuildHasher>,
}

/// Tokenize a single pretoken by repeatedly applying BPE merges in order.
pub fn simple_bpe_merge_in_arena<'a>(
    merges: &HashMap<(TokenId, TokenId), TokenId>,
    pre_token: &[u8],
    merge_arena: &'a bumpalo::Bump,
) -> BumpVec<'a, TokenId> {
    let mut symbols: BumpVec<TokenId> = pre_token
        .iter()
        .map(|&b| TokenId::from(b as u32))
        .collect_in(merge_arena);

    loop {
        let candidate_merges = symbols
            .iter()
            .tuple_windows()
            .enumerate()
            .filter_map(|(i, (a, b))| merges.get(&(*a, *b)).map(|v| (i, *v)));

        let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token);

        if let Some((merge_index, merge_token)) = best_merge {
            symbols[merge_index] = merge_token;
            symbols.remove(merge_index + 1);
        } else {
            break;
        }
    }
    symbols.shrink_to_fit();
    symbols
}

impl Tokenizer {
    pub fn new(
        merges: HashMap<(TokenId, TokenId), TokenId>,
        vocab: Vec<Vec<u8>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect::<Vec<Arc<_>>>();
        let vocab_inv = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect();
        Tokenizer {
            merges,
            vocab_inv,
            vocab,
            byte_remapping,
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
        }
    }

    /// Given a list of tokens in rank order (by merge order), reconstructs the
    /// merges map and returns a Tokenizer.
    ///
    /// This process is necessary to load some tokenizers found in tiktoken.
    pub fn from_ranks(vocab: Vec<Vec<u8>>) -> Result<Self> {
        let mut merges = HashMap::new();
        let vocab = vocab
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Arc<[u8]>>>();
        let vocab_inv = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect::<HashMap<_, TokenId>>();

        for (token_idx, token_bytes) in vocab.iter().cloned().enumerate() {
            if token_bytes.len() < 2 {
                continue;
            }
            let byte_symbols: Vec<u8> = token_bytes
                .iter()
                .map(|b| vocab_inv.get(std::slice::from_ref(b)).unwrap().0 as u8)
                .collect();
            let tokenized = simple_bpe_merge(&merges, &byte_symbols);
            assert_eq!(tokenized.len(), 2);
            merges.insert((tokenized[0], tokenized[1]), TokenId::from(token_idx));
        }

        Ok(Tokenizer {
            merges,
            byte_remapping: ByteRemapping::from_byte_vocab(&vocab)?,
            vocab,
            vocab_inv,
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
        })
    }

    pub fn encode_pretoken(
        byte_remapping: Option<&ByteRemapping>,
        merges: &HashMap<(TokenId, TokenId), TokenId>,
        pretoken: Pretoken,
    ) -> Vec<TokenId> {
        let pretoken: Cow<[u8]> = if let Some(byte_remapping) = byte_remapping {
            pretoken
                .iter()
                .map(|&b| byte_remapping.mapping[b as usize])
                .collect::<Cow<[u8]>>()
        } else {
            Cow::Borrowed(pretoken.0)
        };
        simple_bpe_merge(merges, &pretoken)
    }

    /// For each pretoken in the input iterator, looks up the string in the
    /// cache, and if not found, encodes it and inserts it into the cache.
    pub fn memoized_encode<'a, 'i>(
        &'a mut self,
        pretoken_iter: impl Iterator<Item = Pretoken<'i>>,
    ) -> impl Iterator<Item = Arc<[TokenId]>> {
        let pretoken_cache = &mut self.pretoken_cache;
        pretoken_iter.map(|pretoken: Pretoken| {
            let found_value = pretoken_cache.get(pretoken.as_ref());
            if let Some(v) = found_value {
                return v.clone();
            }
            let inserted_value: Arc<[TokenId]> =
                Self::encode_pretoken(self.byte_remapping.as_ref(), &self.merges, pretoken).into();
            pretoken_cache.insert(pretoken.as_ref().into(), inserted_value.clone());
            inserted_value
        })
    }

    pub fn decode(&self, v: &[TokenId]) -> impl Iterator<Item = u8> {
        v.iter()
            .flat_map(|&token| self.vocab[token.0 as usize].as_ref())
            .copied()
    }

    /// Get the number of pretokens currently in the cache.
    pub fn pretoken_cache_size(&self) -> usize {
        self.pretoken_cache.len()
    }
}

impl Debug for Tokenizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab.len())
            .field("merges_count", &self.merges.len())
            .field("byte_remapping", &self.byte_remapping.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_tokenizer::tiktoken::load_tiktoken;
    use std::io::Read;

    #[test]
    fn test_merges_from_vocab() {
        use base64::prelude::*;
        let mut buf = String::new();
        let data_dir = std::env::home_dir().unwrap().join("data");
        let tiktoken_path = data_dir.join("tokenizers/r50k_base.tiktoken");
        std::fs::File::open(tiktoken_path)
            .expect("Didn't find file")
            .read_to_string(&mut buf)
            .unwrap();
        let vocab: Vec<Vec<u8>> = buf
            .lines()
            .enumerate()
            .map(|(i, line)| {
                let (base64_token, id_str) = line.split_once(' ').unwrap();
                let id = id_str.trim().parse::<u32>().unwrap();
                assert!(id == i as u32);
                let token_bytes = BASE64_STANDARD.decode(base64_token).unwrap();
                token_bytes
            })
            .collect();
        for (i, token) in vocab.iter().enumerate().skip(256).take(20) {
            eprintln!("{i}: {:?}", String::from_utf8_lossy(token));
        }
        let tokenizer = Tokenizer::from_ranks(vocab).unwrap();

        let merges_inv = tokenizer
            .merges
            .iter()
            .map(|((a, b), c)| (*c, (*a, *b)))
            .collect::<HashMap<TokenId, (TokenId, TokenId)>>();

        let decode_token = |token_id: TokenId| -> String {
            String::from_utf8_lossy(&tokenizer.vocab[token_id.0 as usize]).into_owned()
        };

        eprintln!("Merges:");
        for i in 256..=300 {
            let (a, b) = *merges_inv.get(&i.into()).unwrap();
            eprintln!(
                "Merge {i}: \"{}\" + \"{}\" -> \"{}\"",
                decode_token(a),
                decode_token(b),
                decode_token(i.into()),
            )
        }
    }

    #[test]
    fn basic_tokenization() {
        let text = "This is a test string. Please tokenize it!";
        let data_dir = std::env::home_dir().unwrap().join("data");
        let tiktoken_path = data_dir.join("tokenizers/r50k_base.tiktoken");
        let mut tokenizer = load_tiktoken(tiktoken_path).expect("Failed to load tokenizer");
        let pretokenize_iter = crate::pretokenize::pretokenize_as_iter(text.as_bytes());
        let mut output = vec![];
        tokenizer
            .memoized_encode(pretokenize_iter)
            .for_each(|pretoken| output.extend_from_slice(&pretoken));
        assert!(tokenizer.byte_remapping.is_some());
        println!("Encoded: {:?}", output);
        let decoded = tokenizer.decode(&output).collect::<Vec<u8>>();
        println!("Decoded: {:?}", String::from_utf8_lossy(&decoded));
    }
}
