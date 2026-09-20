use jev_bench::bench_device;

fn main() {
    let r = bench_device(200);
    println!("{}", serde_json::to_string_pretty(&r).unwrap());
}
