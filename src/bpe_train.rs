use dashmap::DashMap;
use indicatif::ProgressBar;
use itertools::Itertools;
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};

use crate::pretokenize::pretokenize_par;

#[derive(Clone)]
pub struct Word {
    pub symbols: Vec<u16>,
    pub word_count: isize,
}

type Pair = (u16, u16);

fn count_pairs(words: &[Word]) -> HashMap<Pair, isize> {
    let mut symbol_counts: HashMap<Pair, isize> = HashMap::new();
    for word in words.iter() {
        for i in 0..word.symbols.len() - 1 {
            let pair = (word.symbols[i], word.symbols[i + 1]);
            let count = symbol_counts.entry(pair).or_insert(0);
            *count += word.word_count;
        }
    }
    symbol_counts
}

fn update_word(w: &mut Word, pair: Pair, new_symbol: u16) -> Vec<(Pair, isize)> {
    let mut i = 0;
    let mut count_changes = vec![];
    while i < w.symbols.len() - 1 {
        if w.symbols[i] == pair.0 && w.symbols[i + 1] == pair.1 {
            // Perform the merge
            // count_changes.push((pair, -1)); // This one was removed from the priority queue
            if i >= 1 {
                count_changes.push(((w.symbols[i - 1], pair.0), -w.word_count));
                count_changes.push(((w.symbols[i - 1], new_symbol), w.word_count));
            }
            if w.symbols.len() >= 3 && i <= w.symbols.len() - 3 {
                count_changes.push(((pair.1, w.symbols[i + 2]), -w.word_count));
                count_changes.push(((new_symbol, w.symbols[i + 2]), w.word_count));
            }
            w.symbols[i] = new_symbol;
            w.symbols.remove(i + 1);
        }
        i += 1;
    }
    count_changes
}

#[derive(Clone)]
struct PtrHolder {
    ptr: *mut Word,
}

unsafe impl Sync for PtrHolder {}
unsafe impl Send for PtrHolder {}

fn update_words(
    words: &mut [Word],
    contained_in_words: &mut HashMap<(u16, u16), BTreeSet<u16>>,
    pair: Pair,
    new_symbol: u16,
) -> DashMap<(u16, u16), isize> {
    let count_changes: DashMap<(u16, u16), isize> = DashMap::new();

    let n_threads = rayon::current_num_threads();
    // let chunk_size = std::cmp::min(words.len().div_ceil(n_threads), 5_000);
    // let n_threads = 32;
    // let n_threads = 1;

    // Iterate through all words containing first or second
    let word_idcs = &contained_in_words[&(pair.0, pair.1)];
    let words_ptr = PtrHolder {
        ptr: words.as_mut_ptr(),
    };
    let contained_updates: DashMap<(u16, u16), BTreeSet<u16>> = DashMap::new();

    word_idcs.iter().copied().collect::<Vec<_>>()
        .par_chunks(word_idcs.len().div_ceil(n_threads))
        .for_each(|idcs_chunk| {
            for &i in idcs_chunk {
                // Smuggle in a mutable reference to the word
                let local_words_ptr = words_ptr.clone();
                let word = unsafe { &mut *local_words_ptr.ptr.add(i as usize) };
                let count_changes_word = update_word(word, pair, new_symbol);
                for (pair, change) in count_changes_word {
                    if change > 0 { // Was added to the word, need to track this
                        contained_updates.entry(pair).or_default().insert(i);
                    }
                    *count_changes.entry(pair).or_insert(0) += change;
                }
            }
        });
    
    for (pair, mut word_idcs) in contained_updates.into_iter() {
        let set = contained_in_words.entry(pair).or_default();
        set.append(&mut word_idcs);
    }
    
    // word_idcs.iter().copied().for_each(|i| {
    //     contained_in_words[new_symbol as usize].insert(i);
    // });
    // .for_each(|word| {
    //     let count_changes_word = update_word(word, pair, new_symbol);
    //     for (pair, change) in count_changes_word {
    //         *count_changes.entry(pair).or_insert(0) += change;
    //     }
    // });

    // words.par_chunks_mut(words.len().div_ceil(n_threads)).for_each(|chunk| {
    //     for word in chunk {
    //         let count_changes_word = update_word(word, pair, new_symbol);
    //         for (pair, change) in count_changes_word {
    //             *count_changes.entry(pair).or_insert(0) += change;
    //         }
    //     }
    // });

    // count_changes.into_iter()
    //     .map(|(pair, change)| (pair, change.into_inner()))
    //     .collect()
    count_changes
}

pub fn assemble_token(token: u16, symbols: &[Vec<u8>]) -> String {
    symbols[token as usize]
        .iter()
        .map(|x| *x as char)
        .collect::<String>()
}

pub struct BPEResult {
    pub vocab: HashMap<u16, Vec<u8>>,
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

pub fn train_bpe(
    in_string: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> BPEResult {
    let counts = pretokenize_par(in_string.as_bytes());

    // println!("Gathering to a single vector");

    // let words = parallel_concat(&words);
    // let words = words.into_iter().flatten().collect::<Vec<&str>>();

    println!("Placing into Word struct");
    // let mut words = count_words(&words);

    // Indicates which word indices contain a given symbol
    let mut contained_in_words: HashMap<(u16, u16), BTreeSet<u16>> = HashMap::new();
    let mut contained_in_words_arr = vec![vec![vec![]; 256]; 256];
    let mut words: Vec<Word> = counts
        .into_iter()
        .enumerate()
        .map(|(word_i, (word, count))| {
            // At first we have only bytes, so we won't need to hash the u16 pairs
            let word_symbols: Vec<u16> = word.iter().map(|&b| b as u16).collect();
            for c in word_symbols.iter().copied().tuple_windows::<(u16, u16)>() {
                contained_in_words_arr[c.0 as usize][c.1 as usize].push(word_i as u16);
            }
            Word {
                symbols: word_symbols,
                word_count: count as isize,
            }
        })
        .collect();

    for (i, j) in (0..256).cartesian_product(0..256) {
        if !contained_in_words_arr[i][j].is_empty() {
            contained_in_words.insert((i as u16, j as u16), BTreeSet::from_iter(contained_in_words_arr[i][j].iter().copied()));
        }
    }
    drop(contained_in_words_arr);
        
    println!("{} unique words", words.len());
    let max_symbols = vocab_size;

    let symbol_counts = count_pairs(&words);

    // Symbols 0 through 255 are unicode characters
    let mut symbols: Vec<Vec<u8>> = (0..=255).map(|x| vec![x]).collect();
    symbols.extend(
        special_tokens
            .into_iter()
            .map(|x| x.bytes().collect::<Vec<u8>>()),
    );

    let mut pq = PriorityQueue::new();
    symbol_counts.into_iter().for_each(|(pair, count)| {
        pq.push(pair, count);
    });

    let mut merges = vec![];

    println!("Starting merges");
    let bar = ProgressBar::new(max_symbols as u64).with_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar}] {pos}/{len} ({eta})")
            .unwrap(),
    );
    while !pq.is_empty() && symbols.len() < max_symbols {
        bar.set_position(symbols.len() as u64);
        let pair = {
            let (first_pair, first_count) = pq.pop().unwrap();
            let mut tied_pairs = vec![first_pair];
            while let Some((_next_pair, &next_count)) = pq.peek() {
                if next_count != first_count {
                    break;
                }
                tied_pairs.push(pq.pop().unwrap().0);
            }
            // Find the greatest pair lexicographically
            let mut greatest_pair = first_pair;
            let assemble_pair =
                |(p0, p1)| (assemble_token(p0, &symbols), assemble_token(p1, &symbols));

            for pair in tied_pairs.iter().copied() {
                if assemble_pair(pair) > assemble_pair(greatest_pair) {
                    greatest_pair = pair;
                }
            }

            // println!("Tied pairs");
            for pair in tied_pairs {
                // println!("{:?}", assemble_pair(pair));
                if pair != greatest_pair {
                    pq.push(pair, first_count);
                }
            }

            greatest_pair
        };

        // Merge the pair
        let new_symbol: Vec<u8> = [&symbols[pair.0 as usize], &symbols[pair.1 as usize]]
            .into_iter()
            .flatten()
            .copied()
            .collect();

        merges.push((
            symbols[pair.0 as usize].clone(),
            symbols[pair.1 as usize].clone(),
        ));

        symbols.push(new_symbol);

        let count_changes = update_words(
            &mut words,
            &mut contained_in_words,
            pair,
            symbols.len() as u16 - 1,
        );

        for (pair, change) in count_changes.into_iter() {
            let found_item = pq.change_priority_by(&pair, |p| *p += change);
            if !found_item {
                pq.push(pair, change);
            }
        }
    }
    bar.finish();

    let vocab: HashMap<_, _> = symbols
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u16, v))
        .collect();

    BPEResult { vocab, merges }
}
