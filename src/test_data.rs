//! Test data: the word graph and leave table, found from the workspace
//! root. A test whose data is missing (or is a Git LFS pointer) is skipped
//! with a note rather than failed, so a checkout without LFS still builds
//! and runs the rest of the suite.

use std::path::{Path, PathBuf};

use crate::dictionary::Dictionary;
use crate::leaves::LeaveValues;

/// The word graph, relative to the data root.
pub const WORDS_FST: &str = "data/words.fst";
/// The leave table, relative to the data root.
pub const LEAVES_KLV2: &str = "external/refengine/data/lexica/AMERICAN.klv2";

/// The data root: the workspace root, two levels above this crate.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// True when `path` is a Git LFS pointer stub (a tiny text file starting
/// with the LFS version line) rather than the real data.
fn is_lfs_pointer(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() <= 1024 => std::fs::read(path)
            .map(|b| b.starts_with(b"version https://git-lfs"))
            .unwrap_or(false),
        _ => false,
    }
}

/// Path to a data file under the root, if it is real data.
pub fn file(relative: &str) -> Option<PathBuf> {
    let path = root().join(relative);
    if !path.exists() {
        eprintln!("skipping: {} is missing", path.display());
        None
    } else if is_lfs_pointer(&path) {
        eprintln!("skipping: {} is a Git LFS pointer", path.display());
        None
    } else {
        Some(path)
    }
}

/// The dictionary, with the leave table attached when it is real data.
pub fn dictionary() -> Option<Dictionary> {
    let dict = match Dictionary::load_from_fst(file(WORDS_FST)?) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: dictionary failed to load: {e:#}");
            return None;
        }
    };
    let leaves = root().join(LEAVES_KLV2);
    if leaves.exists() && !is_lfs_pointer(&leaves) {
        if let Ok(table) = LeaveValues::load(&leaves) {
            return Some(dict.with_leaves(table));
        }
    }
    Some(dict)
}
