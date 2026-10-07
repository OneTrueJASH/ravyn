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
//! 50-point bingos). Crossplay has 3 blanks, 5 S's and 40-point bingos. The
//! bonus difference is corrected for here ([`bingo_bonus_correction`]): a
//! leave that makes a bingo more likely is worth 10 points less per unit of
//! probability than the table says. The remaining differences (tile counts,
//! board) are second order; the table is still far better calibrated than
//! per-letter guesses.

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::tile::Tile;

/// Scrabble's bingo bonus, the one the leave table was fitted under.
const TABLE_BINGO_BONUS: f32 = 50.0;

/// Logistic model of the probability that a rack keeping a given leave
/// finds a bingo on its next turn, in Crossplay self-play positions:
/// `p = sigmoid(bias + sum(weight[letter] * count))`, blank at index 26.
///
/// Fitted 2026-08-20 by `cargo test --release study_bingo_probability --
/// --ignored --nocapture` in `mcts.rs`: 24,549 sampled leaves (0-6 tiles)
/// from 400 self-play games, log-loss 0.425 against 0.485 for a constant
/// 18.9%. Single-tile probabilities: blank 0.587, S 0.273, R 0.226,
/// E 0.207, Q 0.014, J 0.025. The fit is linear in counts, so multi-tile
/// synergy beyond additive log-odds is not captured.
const BINGO_MODEL_BIAS: f32 = -1.481;
const BINGO_MODEL_WEIGHTS: [f32; 27] = [
    0.008, -0.473, -0.020, -0.077, 0.140, -0.859, -0.377, -0.391, 0.008, -2.168, -1.034, 0.092,
    -0.124, 0.142, -0.171, -0.351, -2.770, 0.248, 0.501, 0.207, -0.182, -0.848, -1.052, -1.767,
    -1.018, -1.423, 1.831,
];

/// Probability of a bingo next turn for a rack keeping `leave` (see
/// [`BINGO_MODEL_WEIGHTS`]).
pub fn bingo_probability(leave: &[Tile]) -> f32 {
    let mut z = BINGO_MODEL_BIAS;
    for t in leave {
        let idx = if t.is_blank {
            26
        } else {
            (t.letter as u8).wrapping_sub(b'A') as usize
        };
        if let Some(w) = BINGO_MODEL_WEIGHTS.get(idx) {
            z += w;
        }
    }
    1.0 / (1.0 + (-z).exp())
}

/// Adjustment that converts a table value fitted under a 50-point bingo
/// bonus to Crossplay's 40: minus 10 points per unit of extra next-turn
/// bingo probability relative to drawing seven fresh tiles. First order
/// only (the following turn's bingo chances are not adjusted).
pub fn bingo_bonus_correction(leave: &[Tile], bingo_bonus: u32) -> f32 {
    let p = bingo_probability(leave);
    let p0 = bingo_probability(&[]);
    (bingo_bonus as f32 - TABLE_BINGO_BONUS) * (p - p0)
}

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
        // The count comes from the file: check it against the file's
        // length before reserving memory for it.
        let nodes_end = node_count
            .checked_mul(4)
            .and_then(|n| n.checked_add(4))
            .filter(|&end| end <= bytes.len())
            .context("leave file declares more nodes than it holds")?;
        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            nodes.push(u32_at(4 + i * 4)?);
        }
        let values_off = nodes_end;
        let value_count = u32_at(values_off)? as usize;
        let data_off = values_off + 4;
        let fits = value_count
            .checked_mul(4)
            .and_then(|n| n.checked_add(data_off))
            .is_some_and(|end| end <= bytes.len());
        if !fits {
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

    #[test]
    fn loads_and_indexes_consistently() {
        let Some(klv_path) = crate::datafiles::test_data::file(crate::datafiles::LEAVES_KLV2)
        else {
            return;
        };
        let lv = LeaveValues::load(klv_path).expect("leave table parses");
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
    fn bingo_correction_direction_and_size() {
        let blank = [Tile::blank(' ')];
        let q = [Tile::new('Q')];
        let p0 = bingo_probability(&[]);
        assert!((p0 - 0.185).abs() < 0.01, "{p0}");
        assert!(bingo_probability(&blank) > 0.5);
        assert!(bingo_probability(&q) < 0.03);
        // Blank: roughly -4 points; Q: roughly +1.7; empty leave: exactly 0.
        let c_blank = bingo_bonus_correction(&blank, 40);
        let c_q = bingo_bonus_correction(&q, 40);
        assert!((-4.5..-3.5).contains(&c_blank), "{c_blank}");
        assert!((1.4..2.0).contains(&c_q), "{c_q}");
        assert_eq!(bingo_bonus_correction(&[], 40), 0.0);
        // Under Scrabble's own bonus there is nothing to correct.
        assert_eq!(bingo_bonus_correction(&blank, 50), 0.0);
    }

    #[test]
    fn rejects_garbage() {
        assert!(LeaveValues::parse(&[]).is_err());
        assert!(LeaveValues::parse(&[1, 0, 0, 0]).is_err());
        // A huge declared node count is rejected before any allocation.
        assert!(LeaveValues::parse(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]).is_err());
        // So is a huge declared value count.
        let mut huge_values = vec![1, 0, 0, 0];
        huge_values.extend_from_slice(&IS_END.to_le_bytes());
        huge_values.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(LeaveValues::parse(&huge_values).is_err());
        // One node pointing at itself with an accepting bit: a cycle.
        let mut bytes = vec![2, 0, 0, 0];
        bytes.extend_from_slice(&(1u32 | IS_END).to_le_bytes()); // root -> node 1
        bytes.extend_from_slice(&(1u32 | IS_END | ACCEPTS).to_le_bytes()); // node 1 -> itself
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        assert!(LeaveValues::parse(&bytes).is_err());
    }
}
