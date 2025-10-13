//! Once we have a document, we can pretokenize it (potentially in parallel)

// use std::borrow::Cow;

#[derive(Debug, Clone, Copy)]
struct Pretoken<'a>(&'a [u8]);

// #[derive(Debug, Clone)]
// struct DocumentPretokenIter<'a> {
//     bytes: Document<'a>,
//     position: usize,
// }

// impl<'a> Iterator for DocumentPretokenIter<'a> {
//     type Item = &'a [u8];
// }
