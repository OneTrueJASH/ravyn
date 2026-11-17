//! Test data for the integration tests: the dictionary and leave table
//! from the workspace root. A missing file or a Git LFS pointer stub means
//! "skip": the loaders return `None` after saying why.

#![allow(dead_code)] // each test file uses part of this

use std::path::{Path, PathBuf};

use ravyn::dictionary::Dictionary;
use ravyn::leaves::LeaveValues;

/// The data root: this crate's directory when it holds `data/` (the
/// standalone repository), else the workspace root two levels up.
pub fn root() -> PathBuf {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    if here.join("data").is_dir() {
        here.to_path_buf()
    } else {
        here.join("../..")
    }
}

/// True for a real data file: present and not a Git LFS pointer stub.
pub fn is_data(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > 1024 => true,
        Ok(_) => !std::fs::read(path)
            .unwrap_or_default()
            .starts_with(b"version https://git-lfs"),
        Err(_) => false,
    }
}

/// The dictionary, with the leave table attached when it is real data.
pub fn dictionary() -> Option<Dictionary> {
    let words = root().join("data/words.fst");
    if !is_data(&words) {
        eprintln!("skipping: {} is missing or an LFS pointer", words.display());
        return None;
    }
    let dict = Dictionary::load_from_fst(&words).expect("dictionary");
    let leaves = root().join("data/leaves.rlv");
    Some(if is_data(&leaves) {
        dict.with_leaves(LeaveValues::load(&leaves).expect("leave table"))
    } else {
        dict
    })
}

/// The dictionary only if the leave table is real data too.
pub fn dictionary_with_leaves() -> Option<Dictionary> {
    let dict = dictionary()?;
    if dict.has_leave_table() {
        Some(dict)
    } else {
        eprintln!("skipping: the leave table is missing or an LFS pointer");
        None
    }
}
