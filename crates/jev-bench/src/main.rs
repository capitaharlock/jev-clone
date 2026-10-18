use jev_bench::{bench_device, bench_state_cache};

fn main() {
    let state_cache = std::env::args().any(|a| a == "--state-cache");
    if state_cache {
        let r = bench_state_cache(100);
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
    } else {
        let r = bench_device(200);
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
    }
}
