//! Leave values: what keeping a set of tiles is worth, in points of
//! equity, relative to keeping nothing.
//!
//! The values are trained by self-play ([`crate::leavegen`]) for one game's
//! rules and saved as an `RLV2` file ([`LeaveValues::save`]): the bag they
//! were trained with, then one 16-bit value per possible leave of 1-6
//! tiles in a fixed order ([`all_leave_keys`]). A table played under a
//! different bingo bonus than it was fitted for is corrected
//! ([`bingo_bonus_correction`]).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::tile::Tile;

/// Logistic model of the probability that a rack keeping a given leave
/// finds a bingo on its next turn, in Crossplay self-play positions:
/// `p = sigmoid(bias + sum(weight[letter] * count))`, blank at index 26.
///
/// Fitted 2026-08-20 by `cargo test --release study_bingo_probability --
/// --ignored --nocapture` in `mcts/studies.rs`: 24,549 sampled leaves (0-6 tiles)
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

/// Adjustment that converts a leave value fitted under a `fitted` bingo
/// bonus to one worth `bingo_bonus`: the difference in bonus per unit of
/// extra next-turn bingo probability relative to drawing seven fresh tiles
/// (RefEngine's table, fitted at 50, loses 10 points per unit in Crossplay).
/// First order only (the following turn's bingo chances are not adjusted).
pub fn bingo_bonus_correction(leave: &[Tile], fitted: u32, bingo_bonus: u32) -> f32 {
    if fitted == bingo_bonus {
        return 0.0;
    }
    // The empty leave's probability is a constant of the model.
    static P0: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| bingo_probability(&[]));
    let p = bingo_probability(leave);
    let p0 = *P0;
    (bingo_bonus as f32 - fitted as f32) * (p - p0)
}

/// Key of a leave in a trained table: its tile codes sorted ascending (0 =
/// blank, 1-26 = A-Z), each plus one, in five bits apiece (up to six).
pub fn leave_key(codes: &[u8]) -> u32 {
    codes.iter().fold(0, |k, &c| (k << 5) | (c as u32 + 1))
}

/// Magic bytes of a leave table file (see [`LeaveValues::save`]).
const RLV2_MAGIC: &[u8; 4] = b"RLV2";

/// Every leave of 1-6 tiles a bag with `distribution` (A-Z, then blanks)
/// allows, as [`leave_key`]s, in a fixed order: by size, then tile codes
/// ascending. Compact tables store their values in this order.
pub fn all_leave_keys(distribution: &[u8; 27]) -> Vec<u32> {
    // Codes 0 (blank) then 1-26 (A-Z), with their counts.
    let mut avail = [0u8; 27];
    avail[0] = distribution[26];
    avail[1..].copy_from_slice(&distribution[..26]);
    fn go(
        avail: &[u8; 27],
        size: usize,
        from: usize,
        codes: &mut Vec<u8>,
        used: &mut [u8; 27],
        out: &mut Vec<u32>,
    ) {
        if size == 0 {
            out.push(leave_key(codes));
            return;
        }
        for c in from..27 {
            if used[c] < avail[c] {
                used[c] += 1;
                codes.push(c as u8);
                go(avail, size - 1, c, codes, used, out);
                codes.pop();
                used[c] -= 1;
            }
        }
    }
    let mut out = Vec::new();
    for size in 1..=6 {
        go(
            &avail,
            size,
            0,
            &mut Vec::with_capacity(6),
            &mut [0u8; 27],
            &mut out,
        );
    }
    out
}

/// A table of leave values by [`leave_key`], for leaves of 1-6 tiles.
pub struct LeaveValues {
    values: HashMap<u32, f32>,
    /// The bingo bonus the values were fitted under.
    fitted_bingo_bonus: u32,
}

impl LeaveValues {
    /// Load an `RLV2` leave table.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).context("Failed to read leave values")?;
        Self::parse_rlv2(&bytes)
    }

    /// A table of `values` by [`leave_key`], fitted under `fitted_bingo_bonus`.
    pub fn from_table(values: HashMap<u32, f32>, fitted_bingo_bonus: u32) -> Self {
        LeaveValues {
            values,
            fitted_bingo_bonus,
        }
    }

    /// The bingo bonus the values were fitted under.
    pub fn fitted_bingo_bonus(&self) -> u32 {
        self.fitted_bingo_bonus
    }

    /// Save a trained table for `distribution` (the bag it was trained
    /// with, A-Z then blanks) as an `RLV2` file: the magic, the fitted
    /// bingo bonus (u32), the distribution (27 bytes), the leave count
    /// (u32), then each leave's value in hundredths of a point (i16), in
    /// [`all_leave_keys`] order; little-endian. About 1.8 MB for
    /// Crossplay. Leaves missing from the table are written as 0.
    pub fn save<P: AsRef<Path>>(&self, path: P, distribution: &[u8; 27]) -> Result<()> {
        let map = &self.values;
        let keys = all_leave_keys(distribution);
        let mut out = Vec::with_capacity(4 + 4 + 27 + 4 + keys.len() * 2);
        out.extend_from_slice(RLV2_MAGIC);
        out.extend_from_slice(&self.fitted_bingo_bonus.to_le_bytes());
        out.extend_from_slice(distribution);
        out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        for k in keys {
            let v = map.get(&k).copied().unwrap_or(0.0);
            let hundredths = (v * 100.0).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            out.extend_from_slice(&hundredths.to_le_bytes());
        }
        std::fs::write(path.as_ref(), out).context("Failed to write leave values")
    }

    /// Parse an `RLV2` file (untrusted: sizes are checked).
    pub fn parse_rlv2(bytes: &[u8]) -> Result<Self> {
        if !bytes.starts_with(RLV2_MAGIC) || bytes.len() < 4 + 4 + 27 + 4 {
            bail!("not an RLV2 leave table");
        }
        let fitted = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
        let distribution: [u8; 27] = bytes[8..35].try_into().expect("27 bytes");
        if distribution.iter().any(|&n| n > 30) {
            bail!("leave table distribution is implausible");
        }
        let count = u32::from_le_bytes(bytes[35..39].try_into().expect("4 bytes")) as usize;
        // The size check first: enumerating the leaves is the costly part.
        if count.checked_mul(2).and_then(|n| n.checked_add(39)) != Some(bytes.len()) {
            bail!("leave table size does not match its leave count");
        }
        let keys = all_leave_keys(&distribution);
        if count != keys.len() {
            bail!("leave table size does not match its distribution");
        }
        let map = keys
            .into_iter()
            .enumerate()
            .map(|(i, k)| {
                let off = 39 + i * 2;
                let v = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
                (k, v as f32 / 100.0)
            })
            .collect();
        Ok(Self::from_table(map, fitted))
    }

    /// Number of leaves in the table.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Value of a leave given as tile codes sorted ascending (0 = blank,
    /// 1-26 = A-Z). `None` for the empty leave or one not in the table
    /// (more than six tiles, or more of a tile than the bag holds).
    pub fn value_of_codes(&self, codes: &[u8]) -> Option<f32> {
        if codes.is_empty() || codes.len() > 6 {
            return None;
        }
        self.values.get(&leave_key(codes)).copied()
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

    /// The shipped table covers every leave and shows the well-known
    /// orderings: a blank beats an S, which beats a V, which beats a Q.
    #[test]
    fn the_shipped_table_loads_and_orders_sensibly() {
        let Some(path) = crate::test_data::file(crate::test_data::LEAVES) else {
            return;
        };
        let lv = LeaveValues::load(path).expect("leave table parses");
        assert_eq!(
            lv.len(),
            all_leave_keys(&crate::rules::CROSSPLAY.distribution).len()
        );
        assert_eq!(lv.fitted_bingo_bonus(), crate::rules::CROSSPLAY.bingo_bonus);
        let v = |s: &str| {
            let t: Vec<Tile> = s
                .chars()
                .map(|c| {
                    if c == '?' {
                        Tile::blank(' ')
                    } else {
                        Tile::new(c)
                    }
                })
                .collect();
            lv.value(&t).unwrap()
        };
        assert!(v("?") > v("S") && v("S") > 0.0 && v("V") < 0.0 && v("Q") < v("V"));
        // Order does not matter; seven tiles and nothing are not leaves.
        assert_eq!(v("ERS"), v("SER"));
        assert!(lv.value(&[]).is_none());
        assert!(lv
            .value(&"AEINRST".chars().map(Tile::new).collect::<Vec<_>>())
            .is_none());
        // Synergy beyond the letters: AEINRS beats AEIINR.
        assert!(v("AEINRS") > v("AEIINR"));
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
        let c_blank = bingo_bonus_correction(&blank, 50, 40);
        let c_q = bingo_bonus_correction(&q, 50, 40);
        assert!((-4.5..-3.5).contains(&c_blank), "{c_blank}");
        assert!((1.4..2.0).contains(&c_q), "{c_q}");
        assert_eq!(bingo_bonus_correction(&[], 50, 40), 0.0);
        // Under Scrabble's own bonus there is nothing to correct.
        assert_eq!(bingo_bonus_correction(&blank, 50, 50), 0.0);
    }

    #[test]
    fn rejects_garbage() {
        assert!(LeaveValues::parse_rlv2(&[]).is_err());
        assert!(LeaveValues::parse_rlv2(b"RLV1....").is_err());
        let mut bytes = b"RLV2".to_vec();
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&[200u8; 27]); // implausible bag
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert!(LeaveValues::parse_rlv2(&bytes).is_err());
    }

    /// A trained table survives saving and loading as RLV2, to the
    /// hundredth of a point, and corrupt files are refused.
    #[test]
    fn rlv2_round_trip() {
        let dist = crate::rules::CROSSPLAY.distribution;
        let keys = all_leave_keys(&dist);
        let map: std::collections::HashMap<u32, f32> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, (i % 4001) as f32 / 37.0 - 50.0))
            .collect();
        let table = LeaveValues::from_table(map.clone(), 40);
        let path = std::env::temp_dir().join(format!("ravyn-rlv2-{}.rlv", std::process::id()));
        table.save(&path, &dist).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(bytes.len(), 39 + keys.len() * 2);
        let back = LeaveValues::parse_rlv2(&bytes).unwrap();
        assert_eq!(back.fitted_bingo_bonus(), 40);
        assert_eq!(back.len(), map.len());
        for (k, v) in map.iter().take(5000) {
            let got = back.value_of_codes(&codes_of_key(*k)).unwrap();
            assert!((got - v).abs() <= 0.005 + 1e-6, "{k}: {got} vs {v}");
        }
        assert!(LeaveValues::parse_rlv2(&bytes[..bytes.len() - 1]).is_err());
        assert!(LeaveValues::parse_rlv2(b"RLV2").is_err());
    }

    /// Sorted tile codes of a leave key (test helper).
    fn codes_of_key(mut key: u32) -> Vec<u8> {
        let mut codes = Vec::new();
        while key != 0 {
            codes.push((key & 31) as u8 - 1);
            key >>= 5;
        }
        codes.reverse();
        codes
    }
}
