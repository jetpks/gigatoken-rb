use crate::bpe::Tokenizer;
use eyre::{Context, Result};
use std::path::Path;

pub fn load_tiktoken(file_path: impl AsRef<Path>) -> Result<Tokenizer> {
    use crate::bpe::Tokenizer;
    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use base64::prelude::*;
    use std::io::Read;
    let mut buf = String::new();
    std::fs::File::open(&file_path)
        .with_context(|| format!("Failed to read {}", file_path.as_ref().display()))?
        .read_to_string(&mut buf)?;

    // Tiktoken vocabs contain a list of the vocab in order, meaning one needs to reconstruct the merges from this list.

    let rank_vocab: Vec<Vec<u8>> = buf
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let (base64_token, id_str) = line.split_once(' ').unwrap();
            let id = id_str.trim().parse::<u32>().unwrap();
            assert_eq!(id, i as u32);
            let token_bytes: Vec<u8> = BASE64_STANDARD.decode(base64_token).unwrap();
            token_bytes
        })
        .collect();

    let n_ranks = rank_vocab.len() as u32;
    let mut tokenizer = Tokenizer::from_ranks(rank_vocab)?;
    // Tiktoken vocab files carry no special tokens; GPT-2-family vocabs
    // (gpt2/r50k) place <|endoftext|> at the id right after the mergeable
    // ranks. Register it so tiktoken- and tokenizer.json-loaded tokenizers
    // encode and decode identically.
    tokenizer.add_special_token(b"<|endoftext|>".to_vec(), n_ranks.into());
    Ok(tokenizer)
}
