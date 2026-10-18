// Artifact contract for jevclone bundles (#T-rust-skel).
//
// A valid artifact directory contains exactly:
//   config.json, tokenizer.json, model.safetensors,
//   calibration.json, manifest.json, LICENSE
// `load` refuses to open the bundle when any file is missing or when a
// recorded SHA-256 mismatches the bytes on disk.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Files every artifact directory must contain.
pub const REQUIRED_FILES: [&str; 6] = [
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "calibration.json",
    "manifest.json",
    "LICENSE",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub name: String,
    pub format: u32,
    pub files: BTreeMap<String, String>,
    pub dataset_manifest: Option<String>,
}

impl Manifest {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("bad manifest JSON: {e}"))
    }
}

#[derive(Debug)]
pub struct Artifact {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

#[derive(Debug, PartialEq)]
pub enum LoadError {
    MissingFile(String),
    HashMismatch { file: String },
    Manifest(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::MissingFile(n) => write!(f, "missing artifact file: {n}"),
            LoadError::HashMismatch { file } => {
                write!(f, "SHA-256 mismatch (refusing to load): {file}")
            }
            LoadError::Manifest(e) => write!(f, "manifest error: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

/// Verify + open an artifact directory. Any mismatch refuses the load.
pub fn load(dir: &Path) -> Result<Artifact, LoadError> {
    for name in REQUIRED_FILES {
        if !dir.join(name).is_file() {
            return Err(LoadError::MissingFile(name.to_string()));
        }
    }
    let manifest = Manifest::from_file(&dir.join("manifest.json")).map_err(LoadError::Manifest)?;
    for (name, want) in &manifest.files {
        let path = dir.join(name);
        if !path.is_file() {
            return Err(LoadError::MissingFile(name.clone()));
        }
        let got = sha256_file(&path).map_err(LoadError::Manifest)?;
        if &got != want {
            return Err(LoadError::HashMismatch { file: name.clone() });
        }
    }
    Ok(Artifact {
        dir: dir.to_path_buf(),
        manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/fixtures")
            .join(name)
    }

    #[test]
    fn tiny_artifact_loads() {
        let art = load(&fixture("tiny")).expect("tiny fixture must load");
        assert_eq!(art.manifest.format, 1);
    }

    #[test]
    fn missing_file_refuses() {
        let err = load(&fixture("missing-file")).unwrap_err();
        assert!(matches!(err, LoadError::MissingFile(_)), "got {err:?}");
    }

    #[test]
    fn hash_mismatch_refuses() {
        let err = load(&fixture("tampered")).unwrap_err();
        assert!(matches!(err, LoadError::HashMismatch { .. }), "got {err:?}");
    }
}
