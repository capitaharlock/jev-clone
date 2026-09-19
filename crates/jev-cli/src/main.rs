// `jevclone` CLI: serve / verify / gate (#T-rust-skel).
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
    },
    /// Verify an artifact directory against its manifest hashes.
    Verify { dir: PathBuf },
    /// Run the verification gate for a task id.
    Gate { task: String },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve { listen } => {
            println!("jevclone serving on {listen}");
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
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
