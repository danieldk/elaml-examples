use std::io::{stdin, BufRead};

fn main() {
    let input = stdin();
    let lines: Vec<_> = input.lock().lines().map(|l| l.unwrap()).collect();

    for line in lines.iter().rev() {
        println!("{}", line);
    }
}
