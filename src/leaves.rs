//! Leave values from a RefEngine `.klv2` file.
//!
//! A KLV2 file is a KWG (a DAWG of every rack leave, tiles sorted with the
//! blank first) followed by one 32-bit float per leave in DAWG order:
//!
//! ```text
//! u32 LE  node count
//! u32 LE  nodes[node count]   bits 0-21 arc index, 22 is-end, 23 accepts,
//!                             24-31 tile (0 = blank, 1-26 = A-Z)
//! u32 LE  leave count
//! f32 LE  values[leave count]
//! ```
//!
//! Node 0 points at the DAWG root's first child. A node's children start at
//! its arc index and run until a node with the is-end bit. The value index
//! of a leave is the number of accepting nodes that precede it in depth-first
//! order, which we compute from per-node subtree word counts.
//!
//! The values were fitted for standard English Scrabble (2 blanks, 4 S's,
//! 50-point bingos). Crossplay has 3 blanks, 5 S's and 40-point bingos, so
//! blank- and bingo-heavy leaves are worth somewhat less than the table
//! says; the table is still far better calibrated than per-letter guesses.

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::tile::Tile;

const ARC_MASK: u32 = 0x3F_FFFF;
const IS_END: u32 = 1 << 22;
const ACCEPTS: u32 = 1 << 23;

/// Leave value table backed by a KLV2 DAWG.
pub struct LeaveValues {
    nodes: Vec<u32>,
    /// Words in the subtree of node `i` plus those of its following siblings.
    counts: Vec<u32>,
    values: Vec<f32>,
}

impl LeaveValues {
    /// Load a `.klv2` file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).context("Failed to read leave values")?;
        Self::parse(&bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Self> {
        let u32_at = |off: usize| -> Result<u32> {
            bytes
                .get(off..off + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .context("truncated leave file")
        };

        let node_count = u32_at(0)? as usize;
        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            nodes.push(u32_at(4 + i * 4)?);
        }
        let values_off = 4 + node_count * 4;
        let value_count = u32_at(values_off)? as usize;
        let data_off = values_off + 4;
        if bytes.len() < data_off + value_count * 4 {
            bail!("leave file is shorter than its declared value count");
        }
        let values: Vec<f32> = (0..value_count)
            .map(|i| {
                let b = &bytes[data_off + i * 4..data_off + i * 4 + 4];
                f32::from_le_bytes([b[0], b[1], b[2], b[3]])
            })
            .collect();

        if nodes.is_empty() {
            bail!("leave file has no nodes");
        }
        for &n in &nodes {
            if (n & ARC_MASK) as usize >= node_count {
                bail!("leave file node points outside the node table");
            }
        }

        // Subtree word counts via an explicit depth-first walk (the table
        // has thousands of nodes and shared subtrees, so recursion depth
        // and memoisation both matter). A node depends on its first child
        // (arc) and its next sibling (p + 1). State: 0 unvisited, 1 on the
        // current path, 2 done; meeting a node in state 1 means a cycle.
        let mut counts = vec![0u32; node_count];
        let mut state = vec![0u8; node_count];
        let mut stack: Vec<usize> = Vec::new();
        for start in 0..node_count {
            if state[start] == 2 {
                continue;
            }
            stack.push(start);
            while let Some(&p) = stack.last() {
                if state[p] == 2 {
                    stack.pop();
                    continue;
                }
                state[p] = 1;
                let node = nodes[p];
                let arc = (node & ARC_MASK) as usize;
                let next = if node & IS_END == 0 {
                    Some(p + 1)
                } else {
                    None
                };
                if let Some(n) = next {
                    if n >= node_count {
                        bail!("leave file sibling list runs off the node table");
                    }
                }
                let mut pending = None;
                for dep in [if arc != 0 { Some(arc) } else { None }, next]
                    .into_iter()
                    .flatten()
                {
                    match state[dep] {
                        2 => {}
                        1 => bail!("leave file DAWG contains a cycle"),
                        _ => {
                            pending = Some(dep);
                            break;
                        }
                    }
                }
                if let Some(dep) = pending {
                    stack.push(dep);
                    continue;
                }
                let mut c = if node & ACCEPTS != 0 { 1 } else { 0 };
                if arc != 0 {
                    c += counts[arc];
                }
                if let Some(n) = next {
                    c += counts[n];
                }
                counts[p] = c;
                state[p] = 2;
                stack.pop();
            }
        }

        let root = (nodes[0] & ARC_MASK) as usize;
        let total = if root == 0 { 0 } else { counts[root] as usize };
        if total != value_count {
            bail!(
                "leave file DAWG holds {} leaves but {} values",
                total,
                value_count
            );
        }

        Ok(LeaveValues {
            nodes,
            counts,
            values,
        })
    }

    /// Number of leaves in the table.
    #[cfg_attr(not(test), allow(dead_code))] // used by tests and library consumers
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[allow(dead_code)] // library API (clippy::len_without_is_empty)
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Value of a leave given as tile codes sorted ascending (0 = blank,
    /// 1-26 = A-Z). Returns `None` for the empty leave or a leave not in
    /// the table (more than 6 tiles, or one the fitting never saw).
    pub fn value_of_codes(&self, codes: &[u8]) -> Option<f32> {
        if codes.is_empty() {
            return None;
        }
        let mut p = (self.nodes[0] & ARC_MASK) as usize;
        if p == 0 {
            return None;
        }
        let mut idx = 0u32;
        for (i, &tile) in codes.iter().enumerate() {
            loop {
                let node = self.nodes[p];
                if (node >> 24) as u8 == tile {
                    if i + 1 == codes.len() {
                        return if node & ACCEPTS != 0 {
                            self.values.get(idx as usize).copied()
                        } else {
                            None
                        };
                    }
                    if node & ACCEPTS != 0 {
                        idx += 1;
                    }
                    p = (node & ARC_MASK) as usize;
                    if p == 0 {
                        return None;
                    }
                    break;
                }
                // Skip this sibling's whole subtree.
                let own = if node & IS_END == 0 {
                    self.counts[p] - self.counts[p + 1]
                } else {
                    self.counts[p]
                };
                idx += own;
                if node & IS_END != 0 {
                    return None;
                }
                p += 1;
            }
        }
        None
    }

    /// Value of a leave given as tiles (any order).
    pub fn value(&self, leave: &[Tile]) -> Option<f32> {
        let mut codes: Vec<u8> = leave
            .iter()
            .map(|t| {
                if t.is_blank {
                    0
                } else {
                    (t.letter as u8).saturating_sub(b'A') + 1
                }
            })
            .collect();
        codes.sort_unstable();
        self.value_of_codes(&codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KLV_PATH: &str = "external/refengine/data/lexica/NWL23.klv2";

    #[test]
    fn loads_and_indexes_consistently() {
        if !std::path::Path::new(KLV_PATH).exists() {
            eprintln!("KLV2 file not present; skipping");
            return;
        }
        let lv = LeaveValues::load(KLV_PATH).expect("leave table parses");
        assert_eq!(lv.len(), 914_624);

        // Well-known orderings from any leave table: a blank beats an S
        // beats a V; the empty leave is not in the table.
        let blank = lv.value(&[Tile::blank(' ')]).unwrap();
        let s = lv.value(&[Tile::new('S')]).unwrap();
        let v = lv.value(&[Tile::new('V')]).unwrap();
        let q = lv.value(&[Tile::new('Q')]).unwrap();
        assert!(
            blank > s && s > 0.0 && v < 0.0 && q < v,
            "{blank} {s} {v} {q}"
        );
        assert!(lv.value(&[]).is_none());

        // Order of tiles must not matter.
        let a = lv.value(&[Tile::new('E'), Tile::new('R'), Tile::new('S')]);
        let b = lv.value(&[Tile::new('S'), Tile::new('E'), Tile::new('R')]);
        assert_eq!(a, b);
        assert!(a.is_some());

        // Synergy the per-letter heuristic cannot express: AEINRS is worth
        // more than the sum of its parts would suggest relative to AEIINR.
        let aeinrs = lv
            .value(&"AEINRS".chars().map(Tile::new).collect::<Vec<_>>())
            .unwrap();
        let aeiinr = lv
            .value(&"AEIINR".chars().map(Tile::new).collect::<Vec<_>>())
            .unwrap();
        assert!(aeinrs > aeiinr, "{aeinrs} vs {aeiinr}");

        // Seven-tile leaves are not leaves.
        assert!(lv
            .value(&"AEINRST".chars().map(Tile::new).collect::<Vec<_>>())
            .is_none());
    }

    #[test]
    fn rejects_garbage() {
        assert!(LeaveValues::parse(&[]).is_err());
        assert!(LeaveValues::parse(&[1, 0, 0, 0]).is_err());
        // One node pointing at itself with an accepting bit: a cycle.
        let mut bytes = vec![2, 0, 0, 0];
        bytes.extend_from_slice(&(1u32 | IS_END).to_le_bytes()); // root -> node 1
        bytes.extend_from_slice(&(1u32 | IS_END | ACCEPTS).to_le_bytes()); // node 1 -> itself
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        assert!(LeaveValues::parse(&bytes).is_err());
    }
}
