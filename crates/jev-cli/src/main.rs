// `jevclone` CLI: serve / verify / gate / devices (#T-rust-skel, #T-local-infer).
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "jevclone", about = "Local decision-model runtime")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve the Axum inference server.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        /// Backend to run on: auto picks the best available one.
        #[arg(long, default_value = "auto")]
        device: String,
        /// Optional artifact directory: verified by hash before serving.
        #[arg(long)]
        artifact: Option<PathBuf>,
    },
    /// Verify an artifact directory against its manifest hashes.
    Verify { dir: PathBuf },
    /// Run the verification gate for a task id.
    Gate { task: String },
    /// Print the device inventory (available + not_available with reasons).
    Devices,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve {
            listen,
            device,
            artifact,
        } => {
            let (resolved, _) = match jev_model::backend::select(&device) {
                Ok(found) => found,
                Err(e) => fail(&format!("cannot serve: {e}")),
            };
            if let Some(dir) = artifact {
                match jev_core::load(&dir) {
                    Ok(art) => println!(
                        "artifact {} (format {}) verified",
                        art.manifest.name, art.manifest.format
                    ),
                    Err(e) => fail(&format!("VERIFY FAILED: {e}")),
                }
            }
            println!(
                "jevclone serving on {listen} (device {})",
                resolved.as_str()
            );
            if let Err(e) = jev_server::serve(&listen).await {
                fail(&e);
            }
        }
        Cmd::Verify { dir } => match jev_core::load(&dir) {
            Ok(art) => {
                println!("OK {} (format {})", art.manifest.name, art.manifest.format);
            }
            Err(e) => fail(&format!("VERIFY FAILED: {e}")),
        },
        Cmd::Gate { task } => {
            let root = std::env::current_dir().unwrap_or_else(|e| fail(&format!("no cwd: {e}")));
            let script = root.join("training/python/tools/gate.py");
            let status = std::process::Command::new("python3")
                .arg(script)
                .arg("--task")
                .arg(&task)
                .status()
                .unwrap_or_else(|e| fail(&format!("cannot run gate.py: {e}")));
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
        }
        Cmd::Devices => {
            let inv = jev_model::backend::inventory();
            println!("{}", serde_json::to_string_pretty(&to_json(&inv)).unwrap());
        }
    }
}

fn to_json(inv: &[jev_model::backend::InventoryEntry]) -> serde_json::Value {
    serde_json::json!({
        "devices": inv.iter().map(|e| serde_json::json!({
            "device": e.device,
            "available": e.available,
            "reason": e.reason,
        })).collect::<Vec<_>>(),
    })
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
