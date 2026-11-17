use std::path::Path;

use std::collections::HashMap;

use anyhow::{Context, Result};
use fst::raw::{CompiledAddr, Fst};
use fst::Set;

use crate::leaves::{bingo_bonus_correction, LeaveValues};
use crate::rules::{Rules, CROSSPLAY};
use crate::solver::leave_value_with_phase;
use crate::tile::Tile;

/// A node of the flattened word graph: where its outgoing arcs start, how
/// many there are, and whether a word ends here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DawgNode {
    first: u32,
    len: u8,
    is_final: bool,
}

impl DawgNode {
    /// True when the letters walked so far spell a word.
    #[inline]
    pub fn is_final(&self) -> bool {
        self.is_final
    }
}

/// An outgoing arc: the letter (0-25 for A-Z) and the node it leads to.
#[derive(Debug, Clone, Copy)]
pub struct DawgArc {
    /// Letter index, 0 = A.
    pub letter: u8,
    child_len: u8,
    child_final: bool,
    child_first: u32,
}

impl DawgArc {
    /// The node this arc leads to.
    #[inline]
    pub fn node(&self) -> DawgNode {
        DawgNode {
            first: self.child_first,
            len: self.child_len,
            is_final: self.child_final,
        }
    }
}

/// The dictionary's word graph flattened into one array of arcs, each
/// node's arcs contiguous and sorted by letter. Built once from the FST at
/// load time; the move generator walks it instead of decoding FST nodes,
/// which dominated its run time. Shared FST states stay shared, so the
/// graph has the FST's size.
struct Dawg {
    arcs: Vec<DawgArc>,
    root: DawgNode,
}

impl Dawg {
    fn from_fst(fst: &Fst<Vec<u8>>) -> Self {
        let mut arcs = Vec::new();
        let mut memo: HashMap<CompiledAddr, (u32, u8)> = HashMap::new();
        let root = fst.root();
        let (first, len) = Self::flatten(fst, root.addr(), &mut memo, &mut arcs);
        Dawg {
            arcs,
            root: DawgNode {
                first,
                len,
                is_final: root.is_final(),
            },
        }
    }

    /// Lay out the arcs of the node at `addr` (and, recursively, of every
    /// node below it not laid out yet). Returns where they start and how
    /// many there are. Recursion depth is bounded by the longest word.
    fn flatten(
        fst: &Fst<Vec<u8>>,
        addr: CompiledAddr,
        memo: &mut HashMap<CompiledAddr, (u32, u8)>,
        arcs: &mut Vec<DawgArc>,
    ) -> (u32, u8) {
        if let Some(&done) = memo.get(&addr) {
            return done;
        }
        let node = fst.node(addr);
        let letters: Vec<(u8, CompiledAddr)> = node
            .transitions()
            .filter(|t| t.inp.is_ascii_uppercase())
            .map(|t| (t.inp - b'A', t.addr))
            .collect();
        let first = arcs.len() as u32;
        let len = letters.len() as u8;
        memo.insert(addr, (first, len));
        arcs.resize(
            arcs.len() + letters.len(),
            DawgArc {
                letter: 0,
                child_len: 0,
                child_final: false,
                child_first: 0,
            },
        );
        for (k, (letter, child_addr)) in letters.into_iter().enumerate() {
            let child_final = fst.node(child_addr).is_final();
            let (child_first, child_len) = Self::flatten(fst, child_addr, memo, arcs);
            arcs[first as usize + k] = DawgArc {
                letter,
                child_len,
                child_final,
                child_first,
            };
        }
        (first, len)
    }
}

/// Dictionary loaded from a pre-compiled FST for fast lookups
pub struct Dictionary {
    /// FST set for O(1) word validation and prefix matching
    fst: Set<Vec<u8>>,
    /// The same words as a flat graph for the move generator.
    dawg: Dawg,
    word_count: usize,
    /// Leave values from RefEngine's table, when the file was found.
    leaves: Option<LeaveValues>,
    /// How leave values are weighted by game phase.
    phase: PhaseScaling,
    /// Bingo bonus of the rules being played, for the leave table's
    /// bingo-bonus correction.
    bingo_bonus: u32,
}

/// How leave values are weighted as the bag runs out (see
/// [`PhaseScaling::scale`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhaseScaling {
    /// 1.0 above 14 unseen tiles, 0.6 at 8-14, 0.25 at 7 or fewer.
    Stepped,
    /// The same anchor values joined linearly: 1.0 from 21 unseen down to
    /// 0.6 at 14 and 0.25 at 7, flat below. The default: in 4,000 paired
    /// greedy self-play games (`examples/selfplay.rs phase 2000`) it beat
    /// the stepped weights by +0.49 +- 0.33 points of spread per game
    /// (score 0.501 +- 0.003).
    #[default]
    Smoothed,
}

impl PhaseScaling {
    /// Leave weight with `unseen` tiles unseen (bag + opponent rack);
    /// `None` means full weight.
    pub fn scale(self, unseen: Option<usize>) -> f32 {
        let Some(n) = unseen else {
            return 1.0;
        };
        match self {
            PhaseScaling::Stepped => phase_scale(Some(n)),
            PhaseScaling::Smoothed => {
                let n = n as f32;
                if n <= 7.0 {
                    0.25
                } else if n <= 14.0 {
                    0.25 + (n - 7.0) / 7.0 * 0.35
                } else if n <= 21.0 {
                    0.6 + (n - 14.0) / 7.0 * 0.4
                } else {
                    1.0
                }
            }
        }
    }
}

impl Dictionary {
    /// Load dictionary from a pre-compiled FST file
    pub fn load_from_fst<P: AsRef<Path>>(path: P) -> Result<Self> {
        let fst_bytes = std::fs::read(path.as_ref()).context("Failed to read FST file")?;

        // Count words by iterating once
        let fst = Set::new(fst_bytes).context("Failed to parse FST data")?;

        use fst::Streamer;
        let mut count = 0;
        let mut stream = fst.stream();
        while stream.next().is_some() {
            count += 1;
        }

        let dawg = Dawg::from_fst(fst.as_fst());
        Ok(Dictionary {
            fst,
            dawg,
            word_count: count,
            leaves: None,
            phase: PhaseScaling::default(),
            bingo_bonus: CROSSPLAY.bingo_bonus,
        })
    }

    /// Attach a leave-value table; static equity uses it instead of the
    /// per-letter heuristic.
    pub fn with_leaves(mut self, leaves: LeaveValues) -> Self {
        self.leaves = Some(leaves);
        self
    }

    /// Value leaves for `rules` (their bingo bonus). A loaded dictionary
    /// starts out set for [`CROSSPLAY`].
    pub fn for_rules(mut self, rules: &Rules) -> Self {
        self.bingo_bonus = rules.bingo_bonus;
        self
    }

    /// The bingo bonus leaves are valued for.
    pub fn bingo_bonus(&self) -> u32 {
        self.bingo_bonus
    }

    /// Use `phase` to weight leave values by game phase.
    pub fn with_phase_scaling(mut self, phase: PhaseScaling) -> Self {
        self.phase = phase;
        self
    }

    /// The phase weighting in use.
    pub fn phase_scaling(&self) -> PhaseScaling {
        self.phase
    }

    /// True when leave values come from a fitted table rather than the
    /// per-letter heuristic.
    pub fn has_leave_table(&self) -> bool {
        self.leaves.is_some()
    }

    /// Strategic value of keeping `leave` on the rack, scaled by game phase
    /// (`unseen` = bag + opponent rack; `None` means full weight).
    ///
    /// Uses the fitted leave table when loaded and the leave is in it (the
    /// table covers 1–6 tiles), corrected from the table's 50-point bingo
    /// bonus to the rules' (see [`bingo_bonus_correction`]); otherwise
    /// the per-letter heuristic.
    ///
    /// A full 7-tile rack (the racks at a playout cut-off) is valued on the
    /// same scale as the mean of its seven 6-tile sub-leaves, each looked up
    /// in the table, so the cut-off valuation and static equity agree.
    pub fn leave_value(&self, leave: &[Tile], unseen: Option<usize>) -> f32 {
        if leave.is_empty() {
            return 0.0;
        }
        let scale = self.phase.scale(unseen);
        if let Some(table) = &self.leaves {
            if let Some(v) = self.table_value(table, leave) {
                return v * scale;
            }
            if leave.len() == crate::tile::RACK_SIZE {
                let mut total = 0.0;
                let mut found = 0;
                for skip in 0..leave.len() {
                    let sub: Vec<Tile> = leave
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != skip)
                        .map(|(_, t)| *t)
                        .collect();
                    if let Some(v) = self.table_value(table, &sub) {
                        total += v;
                        found += 1;
                    }
                }
                if found == leave.len() {
                    return total / found as f32 * scale;
                }
            }
        }
        // Heuristic fallback, scaled the same way.
        leave_value_with_phase(leave, None) * scale
    }

    /// Table value of a leave with the bingo-bonus correction, extended to
    /// leaves the Scrabble-fitted table cannot hold: Crossplay has three
    /// blanks and the table at most two, so each blank beyond the second
    /// is valued at what the second adds over the first. Without this a
    /// three-blank leave fell back to the per-letter heuristic, on a
    /// different scale from every other leave.
    fn table_value(&self, table: &LeaveValues, leave: &[Tile]) -> Option<f32> {
        let correction = bingo_bonus_correction(leave, self.bingo_bonus);
        if let Some(v) = table.value(leave) {
            return Some(v + correction);
        }
        let blanks = leave.iter().filter(|t| t.is_blank).count();
        if blanks <= 2 {
            return None;
        }
        let mut reduced = Vec::with_capacity(leave.len());
        let mut kept_blanks = 0;
        for t in leave {
            if t.is_blank {
                if kept_blanks == 2 {
                    continue;
                }
                kept_blanks += 1;
            }
            reduced.push(*t);
        }
        let base = table.value(&reduced)?;
        let one = table.value(&[Tile::blank(' ')])?;
        let two = table.value(&[Tile::blank(' '), Tile::blank(' ')])?;
        Some(base + (blanks - 2) as f32 * (two - one) + correction)
    }

    /// Check if a word is valid
    pub fn is_valid(&self, word: &str) -> bool {
        let upper = word.to_ascii_uppercase();
        self.fst.contains(upper.as_bytes())
    }

    /// Check if a prefix could lead to a valid word
    pub fn has_prefix(&self, prefix: &str) -> bool {
        if prefix.is_empty() {
            return true;
        }

        let upper = prefix.to_ascii_uppercase();
        let bytes = upper.as_bytes();

        // Use FST range query to check for prefix
        use fst::{IntoStreamer, Streamer};
        let mut stream = self.fst.range().ge(bytes).into_stream();

        if let Some(word) = stream.next() {
            word.starts_with(bytes)
        } else {
            false
        }
    }

    /// Root of the word graph, for incremental prefix walking.
    ///
    /// The move generator descends one letter at a time with [`Self::child`]
    /// or [`Self::arcs`] instead of re-validating whole prefixes.
    #[inline]
    pub fn root(&self) -> DawgNode {
        self.dawg.root
    }

    /// Follow the arc for `byte` (an uppercase ASCII letter) out of `node`,
    /// returning the child node if the extended prefix exists.
    #[inline]
    pub fn child(&self, node: DawgNode, byte: u8) -> Option<DawgNode> {
        let letter = byte.wrapping_sub(b'A');
        self.arcs(node)
            .iter()
            .find(|a| a.letter == letter)
            .map(DawgArc::node)
    }

    /// The arcs out of `node`, sorted by letter.
    #[inline]
    pub fn arcs(&self, node: DawgNode) -> &[DawgArc] {
        let start = node.first as usize;
        &self.dawg.arcs[start..start + node.len as usize]
    }

    /// Get the total number of words in the dictionary
    pub fn len(&self) -> usize {
        self.word_count
    }

    /// Returns true if the dictionary contains no words
    pub fn is_empty(&self) -> bool {
        self.word_count == 0
    }
}

/// Leave weight by game phase: leaves matter less as redraws run out
/// ([`PhaseScaling::Stepped`]).
pub fn phase_scale(unseen: Option<usize>) -> f32 {
    match unseen {
        Some(n) if n <= 7 => 0.25, // Endgame: no redraws; the 2-ply solver decides
        Some(n) if n <= 14 => 0.6, // Pre-endgame: one draw left at most
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::PhaseScaling;
    use crate::test_data::{self, LEAVES_KLV2};

    #[test]
    fn test_load_dictionary() {
        let Some(dict) = test_data::dictionary() else {
            return;
        };
        assert!(dict.len() > 100000);
        // The leave table is attached whenever the real file is present
        // (an LFS pointer stub falls back to heuristic leaves).
        let klv_real = test_data::file(LEAVES_KLV2).is_some();
        assert_eq!(dict.has_leave_table(), klv_real);
    }

    /// Leaves the Scrabble table cannot hold (three blanks) are valued on
    /// the table's scale, not by the per-letter heuristic: the third blank
    /// adds what the second did, and a full rack with three blanks is the
    /// mean of its sub-leaves like any other.
    #[test]
    fn three_blank_leaves_stay_on_the_table_scale() {
        use crate::tile::Tile;
        let Some(dict) = test_data::dictionary() else {
            return;
        };
        if !dict.has_leave_table() {
            return;
        }
        let blanks = |n: usize, rest: &str| -> Vec<Tile> {
            let mut v: Vec<Tile> = (0..n).map(|_| Tile::blank(' ')).collect();
            v.extend(rest.chars().map(Tile::new));
            v
        };
        let one = dict.leave_value(&blanks(1, ""), None);
        let two = dict.leave_value(&blanks(2, ""), None);
        let three = dict.leave_value(&blanks(3, ""), None);
        assert!(two > one && three > two, "{one} {two} {three}");
        // Far below the heuristic stacking a flat 25 per blank (75+).
        assert!(three < 70.0, "{three}");

        let with_letters = dict.leave_value(&blanks(3, "ERS"), None);
        let two_letters = dict.leave_value(&blanks(2, "ERS"), None);
        assert!(with_letters > two_letters && with_letters < two_letters + 30.0);

        // A seven-tile rack with three blanks averages its sub-leaves.
        let rack = blanks(3, "AEIN");
        let v = dict.leave_value(&rack, None);
        assert!(v.is_finite() && v > two_letters - 30.0 && v < 80.0, "{v}");
    }

    /// Smoothed phase weights pass through the stepped anchors and are
    /// monotone in the unseen count.
    #[test]
    fn smoothed_phase_scaling() {
        let s = PhaseScaling::Smoothed;
        assert_eq!(s.scale(None), 1.0);
        assert_eq!(s.scale(Some(3)), 0.25);
        assert_eq!(s.scale(Some(7)), 0.25);
        assert!((s.scale(Some(14)) - 0.6).abs() < 1e-6);
        assert_eq!(s.scale(Some(21)), 1.0);
        assert_eq!(s.scale(Some(80)), 1.0);
        let mut prev = 0.0;
        for n in 0..=100 {
            let v = s.scale(Some(n));
            assert!(v >= prev);
            prev = v;
        }
        // The stepped weights jump at the same anchors.
        assert_eq!(PhaseScaling::Stepped.scale(Some(8)), 0.6);
        assert_eq!(PhaseScaling::Stepped.scale(Some(15)), 1.0);
    }

    #[test]
    fn test_word_validation() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };

        // Valid words
        assert!(dict.is_valid("WORD"));
        assert!(dict.is_valid("word"));
        assert!(dict.is_valid("QUARTZ"));
        assert!(dict.is_valid("AA")); // Valid 2-letter word

        // Invalid words
        assert!(!dict.is_valid("XYZZY"));
        assert!(!dict.is_valid("ASDFGH"));
    }

    #[test]
    fn test_prefix_matching() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };

        assert!(dict.has_prefix("QU"));
        assert!(dict.has_prefix("QUAR"));
        assert!(dict.has_prefix(""));

        // QX is not a valid prefix
        assert!(!dict.has_prefix("QX"));
    }
}
