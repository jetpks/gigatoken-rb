use rayon::prelude::*;

struct SendPtr<T>(*mut T);

unsafe impl<T> Sync for SendPtr<T> {}
// unsafe impl<T> Send for SendPtr<T> {}

pub fn parallel_concat<T: Send + Sync>(arrs: &[impl AsRef<[T]> + Send + Sync]) -> Vec<T> {
    let lens = arrs
        .iter()
        .map(|e| e.as_ref().len())
        .collect::<Vec<usize>>();
    let start_idcs = lens
        .iter()
        .scan(0, |acc, &x| {
            let old = *acc;
            *acc += x;
            Some(old)
        })
        .collect::<Vec<usize>>();

    let total_len = arrs.iter().map(|e| e.as_ref().len()).sum();
    let mut result = Vec::with_capacity(total_len);

    let result_ptr = SendPtr(result.as_mut_ptr());

    unsafe {
        result.set_len(total_len);
    }

    arrs.par_iter()
        .zip(start_idcs)
        .for_each(|(arr, start_idx)| unsafe {
            let sent = &result_ptr;
            let out_ptr: *mut T = sent.0;
            let result_ptr = out_ptr.add(start_idx);
            std::ptr::copy_nonoverlapping(arr.as_ref().as_ptr(), result_ptr, arr.as_ref().len())
        });

    result
}
