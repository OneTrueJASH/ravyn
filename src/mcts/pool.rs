//! The unseen-tile pool, fingerprints of positions, and the seeds that
//! make playout `i` the same deal for every candidate.

use super::*;

/// Pool of available tiles for random drawing
pub struct TilePool {
    pub(super) tiles: Vec<Tile>,
}

impl TilePool {
    /// Create a tile pool from remaining tile counts
    pub fn from_remaining(remaining: &std::collections::HashMap<char, u32>) -> Self {
        let mut tiles = Vec::with_capacity(100);
        for (&letter, &count) in remaining {
            for _ in 0..count {
                if letter == '?' {
                    tiles.push(Tile::blank(' '));
                } else {
                    tiles.push(Tile::new(letter));
                }
            }
        }
        // Canonical order, so the same seed gives the same draw sequence
        // regardless of HashMap iteration order.
        tiles.sort_by_key(|t| (t.is_blank, t.letter));
        TilePool { tiles }
    }

    /// Get the number of tiles in the pool
    #[inline]
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Check if pool is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// A shuffled copy of the pool's tiles.
    pub(super) fn shuffled(&self, rng: &mut Rng) -> Vec<Tile> {
        let mut tiles = self.tiles.clone();
        rng.shuffle(&mut tiles);
        tiles
    }
}

/// FNV-1a, for small deterministic fingerprints (position and model keys,
/// playout seeds). Not for hash tables.
pub(crate) struct Fnv(u64);

impl Fnv {
    pub(crate) fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    pub(crate) fn write_u64(&mut self, x: u64) {
        for b in x.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0
    }
}

/// Fingerprint of a position: board tiles (with blank flags) and our rack.
/// Playout seeds derive from it, so a position replays the same deals in
/// every run, which lets a stopped simulation resume where it left off.
pub fn position_fingerprint(state: &SolverState) -> u64 {
    let mut h = Fnv::new();
    for r in 0..crate::board::BOARD_SIZE {
        for c in 0..crate::board::BOARD_SIZE {
            let code = match state.board.get_tile(r, c) {
                Some(t) => (t.letter as u64) << 1 | t.is_blank as u64,
                None => 0,
            };
            h.write_u64(code);
        }
    }
    let mut rack: Vec<(bool, char)> = state
        .rack
        .tiles()
        .iter()
        .map(|t| (t.is_blank, t.letter))
        .collect();
    rack.sort_unstable();
    for (b, l) in rack {
        h.write_u64((l as u64) << 1 | b as u64);
    }
    h.finish()
}

/// Base seed of the playout generators (see [`CandidateSim::run_batch`]).
pub(super) const PLAYOUT_SEED: u64 = 0x5EED_C0DE;

/// SplitMix64 finaliser: a well-mixed seed for iteration `i`.
pub(super) fn iteration_seed(base: u64, i: u64) -> u64 {
    let mut z = base ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
