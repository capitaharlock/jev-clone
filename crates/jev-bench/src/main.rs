use jev_bench::bench_latency;

fn main() {
    let r = bench_latency(200);
    println!(
        "{{\"iters\": {}, \"p50_ms\": {:.4}, \"p95_ms\": {:.4}}}",
        r.iters, r.p50_ms, r.p95_ms
    );
}
