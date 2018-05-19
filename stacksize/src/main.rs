// Note: this program relies on `taint` not being optimized away.
//       this may happen in future Rust versions. Ultimately, using
//       `black_box` would be better.

use std::usize;

const INCREMENT: usize = 65536;

#[inline(never)]
fn taint(s: &mut [u8]) {
    s[0] = 0;
}

#[inline(never)]
fn print_stack_use(size: usize) {
    println!("Got a stack of at least: {}k", size / 1024);    
}

fn overflow(mut size: usize) {
    let mut x = [0u8; INCREMENT];

    size += INCREMENT;

    // println! uses stack space, so do a non-inlined call instead.
    print_stack_use(size);

    if size == usize::MAX {
        // A stack size larger than the maximum machine word is impossible.
    } else {
        taint(&mut x);
        overflow(size)
    }
}

fn main() {
    overflow(0);
}
