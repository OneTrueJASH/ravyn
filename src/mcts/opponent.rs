//! The opponent model: inference-weighted deals and the measured
//! softmax temperature of the simulated opponent.

use fastrand::Rng;

use crate::tile::{Tile, RACK_SIZE};

use super::pool::{Fnv, TilePool};

/// Model of the opponent for playouts: how their hidden rack is dealt and
/// how strongly they play. The default deals uniformly and plays perfect
/// greedy, which is exactly the behaviour without a model.
#[derive(Debug, Clone)]
pub struct OpponentModel {
    /// Per-letter sampling weight for the tiles the opponent kept after
    /// their last play, indexed by letter (26 = blank). The weight is the
    /// inferred expected count relative to a uniform deal's, so a value
    /// above 1 tilts the deal toward putting that letter in their rack.
    /// `None` deals uniformly.
    pub(super) kept_weights: Option<[f32; 27]>,
    /// How many tiles they kept (7 minus the tiles they just played).
    pub(super) kept_size: usize,
    /// Equity spread of the opponent's simulated move choice, in points:
    /// 0 plays the best move every turn, larger values err more, roughly
    /// matching a player who gives up `temperature` points per turn.
    pub temperature: f32,
}

impl OpponentModel {
    pub fn new(temperature: f32) -> Self {
        OpponentModel {
            kept_weights: None,
            kept_size: 0,
            temperature,
        }
    }

    /// Attach per-letter marginals from rack inference: `(letter,
    /// expected)` is the inferred expected count of each letter among the
    /// `kept_size` tiles the opponent kept ('?' for the blank). Weights
    /// are clamped so a noisy or stale inference can only tilt the deal,
    /// never make a holding impossible or certain.
    ///
    /// Sequential weighted sampling without replacement undershoots the
    /// target marginals (achieved expectations land between uniform and
    /// inferred); the tilt direction is what matters, exact fidelity would
    /// need a conditional-Poisson weight solve.
    pub fn with_marginals(
        mut self,
        marginals: &[(char, f32)],
        kept_size: usize,
        pool: &TilePool,
    ) -> Self {
        // Kept tiles beyond a rack would spill past the opponent's region
        // of the deal; no real caller keeps more than 6.
        let kept_size = kept_size.min(RACK_SIZE - 1);
        if marginals.is_empty() || kept_size == 0 || pool.is_empty() {
            return self;
        }
        let mut counts = [0u32; 27];
        for tile in &pool.tiles {
            counts[Self::index(tile)] += 1;
        }
        let mut weights = [1.0f32; 27];
        for &(letter, expected) in marginals {
            let idx = if letter == '?' {
                26
            } else if letter.is_ascii_uppercase() {
                (letter as u8 - b'A') as usize
            } else {
                continue;
            };
            if counts[idx] == 0 {
                continue;
            }
            let uniform = kept_size as f32 * counts[idx] as f32 / pool.len() as f32;
            weights[idx] = (expected / uniform).clamp(0.05, 20.0);
        }
        self.kept_weights = Some(weights);
        self.kept_size = kept_size;
        self
    }

    pub fn has_marginals(&self) -> bool {
        self.kept_weights.is_some()
    }

    pub(super) fn index(tile: &Tile) -> usize {
        if tile.is_blank {
            26
        } else {
            (tile.letter as u8 - b'A') as usize
        }
    }

    /// Deal the pool for one playout, laid out as `[opponent rack | bag]`
    /// like a uniform shuffle; our draw comes off the top (the end) of the
    /// bag. The opponent's kept tiles are sampled first, by weight and
    /// without replacement, and open their rack, so they can never appear
    /// in our draw or the bag; everything else is dealt uniformly (their
    /// fresh draws are exchangeable with ours). The layout does not depend
    /// on the candidate, so every candidate in an iteration faces the same
    /// opponent rack.
    pub(super) fn dealt(&self, pool: &TilePool, rng: &mut Rng) -> Vec<Tile> {
        let Some(weights) = &self.kept_weights else {
            return pool.shuffled(rng);
        };
        let mut rest = pool.tiles.clone();
        let mut kept = Vec::with_capacity(self.kept_size);
        for _ in 0..self.kept_size.min(rest.len()) {
            let total: f32 = rest.iter().map(|t| weights[Self::index(t)]).sum();
            let picked = if total > 0.0 {
                let mut roll = rng.f32() * total;
                let mut picked = rest.len() - 1;
                for (i, t) in rest.iter().enumerate() {
                    roll -= weights[Self::index(t)];
                    if roll <= 0.0 {
                        picked = i;
                        break;
                    }
                }
                picked
            } else {
                rng.usize(..rest.len())
            };
            kept.push(rest.swap_remove(picked));
        }
        rng.shuffle(&mut rest);
        kept.append(&mut rest);
        kept
    }

    /// Hash of the model's parameters, for keying cached simulations.
    pub fn fingerprint(&self) -> u64 {
        let mut h = Fnv::new();
        h.write_u64(self.temperature.to_bits() as u64);
        h.write_u64(self.kept_size as u64);
        if let Some(w) = &self.kept_weights {
            for x in w {
                h.write_u64(x.to_bits() as u64);
            }
        }
        h.finish()
    }
}
