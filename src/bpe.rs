use itertools::Itertools;
use onig::Regex;
use std::collections::HashMap;


pub fn encode(
    re: &Regex,                        // Regex to pre-tokenize
    vocab_inv_bytes: &[Option<u16>],   // Mapping from byte to token for initial vocab
    merges: &HashMap<(u16, u16), u16>, // Tuple of tokens to merged token
    text: &str,
) -> Vec<u16> {
    let words: Vec<&str> = re.find_iter(text).map(|m| &text[m.0..m.1]).collect();
    words
        .into_iter()
        .flat_map(|word| {
            let word_bytes = word.as_bytes();
            let mut symbols: Vec<u16> = word_bytes
                .iter()
                .map(|c| vocab_inv_bytes[*c as usize].unwrap())
                .collect();

            loop {
                let candidate_merges = symbols
                    .iter()
                    .tuple_windows()
                    .enumerate()
                    .filter_map(|(i, (a, b))| merges.get(&(*a, *b)).map(|v| (i, *v)));
                let best_merge =
                    candidate_merges.min_by_key(|(_index, merged_token)| *merged_token); // Earliest merge in list of merges

                if let Some((merge_index, merge_token)) = best_merge {
                    symbols[merge_index] = merge_token;
                    symbols.remove(merge_index + 1); // O(n) worst case
                } else {
                    break;
                }
            }
            // println!("Merged {:?} into {:?}", word, word_bytes, symbols)

            symbols
        })
        .collect()
}

pub fn decode(v: &[u16], vocab: &HashMap<u16, Vec<u8>>) -> Vec<u8> {
    v.iter()
        .flat_map(|&token| vocab.get(&token).unwrap())
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode() {
        let re =
            Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
                .unwrap();
        let vocab = HashMap::from([
            (0_u16, &b" "[..]),
            (1, b"a"),
            (2, b"c"),
            (3, b"e"),
            (4, b"h"),
            (5, b"t"),
            (6, b"th"),
            (7, b" c"),
            (8, b" a"),
            (9, b"the"),
            (10, b" at"),
        ]);
        let mut vocab_inv_bytes = vec![None; 256];
        vocab.iter().for_each(|(&k, &v)| {
            if v.len() == 1 {
                vocab_inv_bytes[v[0] as usize] = Some(k);
            }
        });

        let merges: HashMap<(u16, u16), u16> = vec![
            (&b"t"[..], &b"h"[..]),
            (b" ", b"c"),
            (b" ", b"a"),
            (b"th", b"e"),
            (b" a", b"t"),
        ]
        .into_iter()
        .map(|(e1, e2)| {
            let mut merged = Vec::from(e1);
            merged.append(&mut Vec::from(e2));
            let e1_token = vocab.iter().find(|&(_, &v)| v == e1).unwrap().0;
            let e2_token = vocab.iter().find(|&(_, &v)| v == e2).unwrap().0;
            let merged_token = vocab.iter().find(|&(_, &v)| v == merged).unwrap().0;
            ((*e1_token, *e2_token), *merged_token)
        })
        .collect();
        println!("{merges:?}");

        //let special_tokens = HashMap::new();

        let text = "the cat ate";
        let encoded = encode(&re, &vocab_inv_bytes, &merges, text);
        assert_eq!(encoded, vec![9, 7, 1, 5, 10, 3]);
    }
}
