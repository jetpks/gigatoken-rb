//! Shared input loading for the bench targets. Lives in a subdirectory so
//! cargo does not treat it as a bench target; each bench pulls it in with
//! `mod common;`.

use std::time::Instant;

/// Re-enable transparent huge pages for this process. The encode paths
/// madvise their big tables and buffers to 2 MiB pages (they far exceed
/// 4 KiB dTLB coverage, and Zen drops software prefetches that miss the
/// TLB), but some session managers launch children with
/// PR_SET_THP_DISABLE, which silently vetoes MADV_HUGEPAGE; clear it so
/// the bench measures the tokenizer, not the launcher's memory policy.
/// No-op off Linux.
pub fn allow_thp() {
    #[cfg(target_os = "linux")]
    // SAFETY: prctl(PR_SET_THP_DISABLE, 0) only clears a per-process flag.
    unsafe {
        libc::prctl(libc::PR_SET_THP_DISABLE, 0, 0, 0, 0);
    }
}

/// Hint 2 MiB pages for a buffer's reserved capacity, BEFORE it is first
/// written (the ordering is what makes the hint effective: pages fault in
/// huge only if the madvise precedes the first touch). Multi-GB bench
/// buffers otherwise saturate the dTLB alongside the encode's own tables.
/// No-op off Linux.
#[allow(unused_variables, clippy::missing_safety_doc)]
pub fn madvise_hugepage_capacity<T>(v: &mut Vec<T>) {
    #[cfg(target_os = "linux")]
    if v.capacity() > 0 {
        // Align the start inward: malloc's mmap chunks carry a 16-byte
        // header, and an unaligned madvise start is EINVAL (silent no-op).
        const PAGE: usize = 4096;
        let addr = v.as_mut_ptr() as usize;
        let start = (addr + PAGE - 1) & !(PAGE - 1);
        let end = addr + v.capacity() * std::mem::size_of::<T>();
        if end > start {
            // SAFETY: the range is one live allocation; the hint neither
            // reads nor writes it.
            unsafe {
                libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
            }
        }
    }
}

/// Load the benchmark input from `~/data/owt_train.txt`, truncated to a
/// UTF-8 character boundary.
///
/// ENCODE_MB caps the input for fast profiling iterations (only that many
/// bytes are read from disk, so the read does not dominate a profile of the
/// encode loop). When it is unset, `default_mb` applies; `None` reads the
/// whole file.
pub fn load_owt_input(default_mb: Option<usize>) -> Vec<u8> {
    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    eprintln!("Reading {owt_path:?}...");
    let t0 = Instant::now();

    let cap_mb = std::env::var("ENCODE_MB")
        .ok()
        .map(|mb| {
            mb.trim()
                .parse::<usize>()
                .expect("ENCODE_MB must be an integer")
        })
        .or(default_mb);
    let mut data = match cap_mb {
        Some(mb) => {
            use std::io::Read;
            let max_bytes = mb * 1_000_000;
            let file =
                std::fs::File::open(&owt_path).expect("Could not open ~/data/owt_train.txt");
            let mut data = Vec::with_capacity(max_bytes);
            madvise_hugepage_capacity(&mut data);
            file.take(max_bytes as u64)
                .read_to_end(&mut data)
                .expect("read failed");
            data
        }
        None => {
            let len = std::fs::metadata(&owt_path)
                .expect("Could not stat ~/data/owt_train.txt")
                .len() as usize;
            use std::io::Read;
            let mut data = Vec::with_capacity(len + 1);
            madvise_hugepage_capacity(&mut data);
            std::fs::File::open(&owt_path)
                .expect("Could not open ~/data/owt_train.txt")
                .read_to_end(&mut data)
                .expect("read failed");
            data
        }
    };
    // Back up to a UTF-8 character boundary (a byte cap can split a
    // multibyte character).
    if let Err(e) = std::str::from_utf8(&data) {
        data.truncate(e.valid_up_to());
    }
    eprintln!(
        "Read {:.2} GB in {:.1}s",
        data.len() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    data
}
