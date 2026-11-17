//! Training leave values by self-play.
//!
//! A leave's value is what keeping those tiles is worth on the following
//! turns, relative to keeping nothing (playing all seven, worth 0 by
//! definition). Self-play games measure it: whenever a player keeps a
//! leave with enough tiles still in the bag, the sample's target is what
//! that player scores on their next turn plus the value of the leave they
//! keep then (one step of temporal-difference learning, bootstrapping from
//! the current values). Averaging a leave's targets and subtracting the
//! empty leave's average gives its value.
//!
//! Most of the ~900,000 possible leaves are rarely or never kept, so a
//! smooth model is fitted to every sample (ridge regression on single
//! tiles, tile pairs including duplicates, and size by vowel count), and
//! each leave's value moves from the model's toward its own average in
//! proportion to how often it was seen.
//!
//! Each generation plays with the values the previous one fitted (the
//! first with the per-letter heuristic), so the values and the play
//! improve together.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::board::Board;
use crate::dictionary::Dictionary;
use crate::game::best_exchange;
use crate::leaves::{all_leave_keys, leave_key, LeaveValues};
use crate::rules::Rules;
use crate::solver::{Move, MoveKind, Solver, EXCHANGE_MIN_BAG};
use crate::tile::{Tile, RACK_SIZE};

/// How a generation of training is run and fitted.
#[derive(Debug, Clone, Copy)]
pub struct TrainingSettings {
    /// Self-play games.
    pub games: usize,
    /// Seed of the games' shuffles.
    pub seed: u64,
    /// Leaves are sampled only while at least this many tiles are left in
    /// the bag after drawing, so the next turn is an ordinary one.
    pub min_bag: usize,
    /// Ridge penalty of the smooth model.
    pub ridge: f64,
    /// Samples at which a leave's own average weighs as much as the
    /// model's value.
    pub prior_samples: f64,
}

impl Default for TrainingSettings {
    fn default() -> Self {
        TrainingSettings {
            games: 100_000,
            seed: 1,
            min_bag: 14,
            ridge: 1.0,
            prior_samples: 20.0,
        }
    }
}

/// What a generation measured, besides the fitted table.
#[derive(Debug, Clone, Copy)]
pub struct TrainingReport {
    pub games: usize,
    pub samples: u64,
    /// Distinct leaves seen at least once.
    pub leaves_seen: usize,
    /// Average target of the empty leave (the zero point).
    pub empty_mean: f64,
}

const VOWELS: [u8; 5] = [0, 4, 8, 14, 20]; // A E I O U as indices
const BLANK: usize = 26;
/// Single tiles, pairs (with duplicates), size by vowel count.
const SINGLES: usize = 27;
const PAIRS: usize = 27 * 28 / 2;
const SIZE_VOWELS: usize = 27; // sizes 1-6, vowels 0..=size
const FEATURES: usize = SINGLES + PAIRS + SIZE_VOWELS;

/// Tile counts of a leave, blank at 26.
fn counts_of(tiles: &[Tile]) -> [u8; 27] {
    let mut c = [0u8; 27];
    for t in tiles {
        let i = if t.is_blank {
            BLANK
        } else {
            (t.letter as u8 - b'A') as usize
        };
        c[i] += 1;
    }
    c
}

/// Sorted tile codes (0 = blank, 1-26 = A-Z) of a counts array, as
/// [`leave_key`] wants them.
fn codes_of(counts: &[u8; 27]) -> Vec<u8> {
    let mut codes = Vec::with_capacity(RACK_SIZE);
    codes.extend(std::iter::repeat_n(0, counts[BLANK] as usize));
    for (i, &n) in counts[..26].iter().enumerate() {
        codes.extend(std::iter::repeat_n(i as u8 + 1, n as usize));
    }
    codes
}

/// Tile counts of a leave from its [`leave_key`].
fn counts_of_key(mut key: u32) -> [u8; 27] {
    let mut c = [0u8; 27];
    while key != 0 {
        let code = (key & 31) as usize - 1;
        c[if code == 0 { BLANK } else { code - 1 }] += 1;
        key >>= 5;
    }
    c
}

/// The sparse features of a leave.
fn features(counts: &[u8; 27]) -> Vec<(usize, f64)> {
    let mut f = Vec::with_capacity(32);
    let size: usize = counts.iter().map(|&n| n as usize).sum();
    if size == 0 {
        return f;
    }
    for (i, &n) in counts.iter().enumerate() {
        if n > 0 {
            f.push((i, n as f64));
        }
    }
    let mut pair = SINGLES;
    for i in 0..27 {
        for j in i..27 {
            let v = if i == j {
                (counts[i] as f64) * (counts[i] as f64 - 1.0) / 2.0
            } else {
                counts[i] as f64 * counts[j] as f64
            };
            if v > 0.0 {
                f.push((pair, v));
            }
            pair += 1;
        }
    }
    let vowels: usize = VOWELS.iter().map(|&v| counts[v as usize] as usize).sum();
    // Size s has s + 1 slots (0 to s vowels), so it starts after the
    // 2 + 3 + ... + s slots of the smaller sizes (none for size 1).
    let start = SINGLES + PAIRS + (2..=size).sum::<usize>();
    f.push((start + vowels, 1.0));
    f
}

/// What one generation's games measured.
#[derive(Default)]
struct Accumulator {
    xtx: Vec<f64>,
    xty: Vec<f64>,
    /// Sum of each feature, to centre the targets afterwards.
    xsum: Vec<f64>,
    by_leave: HashMap<u32, (f64, u64)>,
    samples: u64,
}

impl Accumulator {
    fn new() -> Self {
        Accumulator {
            xtx: vec![0.0; FEATURES * FEATURES],
            xty: vec![0.0; FEATURES],
            xsum: vec![0.0; FEATURES],
            by_leave: HashMap::new(),
            samples: 0,
        }
    }

    fn add(&mut self, counts: &[u8; 27], y: f64) {
        let f = features(counts);
        for &(i, a) in &f {
            self.xty[i] += a * y;
            self.xsum[i] += a;
            for &(j, b) in &f {
                self.xtx[i * FEATURES + j] += a * b;
            }
        }
        let e = self
            .by_leave
            .entry(leave_key(&codes_of(counts)))
            .or_insert((0.0, 0));
        e.0 += y;
        e.1 += 1;
        self.samples += 1;
    }

    fn merge(mut self, other: Accumulator) -> Accumulator {
        for (a, b) in self.xtx.iter_mut().zip(other.xtx) {
            *a += b;
        }
        for (a, b) in self.xty.iter_mut().zip(other.xty) {
            *a += b;
        }
        for (a, b) in self.xsum.iter_mut().zip(other.xsum) {
            *a += b;
        }
        for (k, (s, n)) in other.by_leave {
            let e = self.by_leave.entry(k).or_insert((0.0, 0));
            e.0 += s;
            e.1 += n;
        }
        self.samples += other.samples;
        self
    }
}

/// `rack` without one tile matching each of `used` (blanks match blanks).
fn without(rack: &[Tile], used: &[Tile]) -> Vec<Tile> {
    let mut left = rack.to_vec();
    for u in used {
        if let Some(i) = left
            .iter()
            .position(|t| t.is_blank == u.is_blank && (u.is_blank || t.letter == u.letter))
        {
            left.swap_remove(i);
        }
    }
    left
}

/// Play one greedy self-play game, recording leave samples.
fn play_game(
    dict: &Dictionary,
    rules: &'static Rules,
    seed: u64,
    min_bag: usize,
    acc: &mut Accumulator,
) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut bag = rules.full_tile_set();
    rng.shuffle(&mut bag);
    let mut racks = [
        bag.split_off(bag.len() - RACK_SIZE),
        bag.split_off(bag.len() - RACK_SIZE),
    ];
    let mut board = Board::with_rules(rules);
    // Each player's last sampled leave, awaiting their next move.
    let mut pending: [Option<[u8; 27]>; 2] = [None, None];
    let mut to_move = 0;
    let mut scoreless = 0;
    while !bag.is_empty() && scoreless < 6 {
        let rack = racks[to_move].clone();
        let tiles_on_board = (0..crate::board::BOARD_SIZE)
            .flat_map(|r| (0..crate::board::BOARD_SIZE).map(move |c| (r, c)))
            .filter(|&(r, c)| board.get_tile(r, c).is_some())
            .count();
        let unseen = rules
            .total_tiles()
            .saturating_sub(tiles_on_board + rack.len());
        let play = Solver::new(&board, dict).best_move_by_equity(&rack);
        let exchange = (bag.len() >= EXCHANGE_MIN_BAG)
            .then(|| best_exchange(&rack, dict, Some(unseen)))
            .flatten();
        let mv = match (play, exchange) {
            (Some(p), Some(e)) => {
                if e.equity > p.equity {
                    e
                } else {
                    p
                }
            }
            (Some(p), None) => p,
            (None, Some(e)) => e,
            (None, None) => Move::pass(&rack),
        };

        let kept = without(&rack, &mv.tiles_from_rack());
        // The previous leave's target: this move's score plus what is kept now.
        if let Some(leave) = pending[to_move].take() {
            let y = mv.score as f64 + dict.leave_value(&kept, None) as f64;
            acc.add(&leave, y);
        }

        for (r, c, t) in &mv.tiles_used {
            board.place_tile(*r, *c, *t);
        }
        scoreless = if mv.score == 0 { scoreless + 1 } else { 0 };
        let mut new_rack = kept.clone();
        let draw = (RACK_SIZE - new_rack.len()).min(bag.len());
        new_rack.extend(bag.split_off(bag.len() - draw));
        if let MoveKind::Exchange(returned) = &mv.kind {
            bag.extend_from_slice(returned);
            rng.shuffle(&mut bag);
        }
        if bag.len() >= min_bag {
            pending[to_move] = Some(counts_of(&kept));
        }
        racks[to_move] = new_rack;
        to_move = 1 - to_move;
    }
}

/// Solve `a x = b` for symmetric positive definite `a` (n x n, row-major)
/// by Cholesky decomposition.
fn solve_spd(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        let d = d.max(1e-12).sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    b
}

/// Play one generation of self-play with `dict`'s leave values and fit a
/// new table from it.
pub fn train_generation(
    dict: &Dictionary,
    rules: &'static Rules,
    settings: &TrainingSettings,
) -> (LeaveValues, TrainingReport) {
    let acc = (0..settings.games as u64)
        .into_par_iter()
        .fold(Accumulator::new, |mut acc, g| {
            let seed = settings
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(g);
            play_game(dict, rules, seed, settings.min_bag, &mut acc);
            acc
        })
        .reduce(Accumulator::new, Accumulator::merge);

    // The zero point: the empty leave's average target.
    let (empty_sum, empty_n) = acc
        .by_leave
        .get(&leave_key(&[]))
        .copied()
        .unwrap_or((0.0, 0));
    let empty_mean = if empty_n > 0 {
        empty_sum / empty_n as f64
    } else {
        0.0
    };

    // Ridge regression on the centred targets: X'X w = X'(y - mean).
    let mut xtx = acc.xtx.clone();
    for i in 0..FEATURES {
        xtx[i * FEATURES + i] += settings.ridge;
    }
    let xty: Vec<f64> = acc
        .xty
        .iter()
        .zip(&acc.xsum)
        .map(|(xy, x)| xy - empty_mean * x)
        .collect();
    let w = solve_spd(xtx, xty, FEATURES);
    let model = |counts: &[u8; 27]| features(counts).iter().map(|&(i, a)| w[i] * a).sum::<f64>();

    let table: HashMap<u32, f32> = all_leave_keys(&rules.distribution)
        .into_par_iter()
        .map(|key| {
            let counts = counts_of_key(key);
            let m = model(&counts);
            let value = match acc.by_leave.get(&key) {
                Some(&(sum, n)) if n > 0 => {
                    let mean = sum / n as f64 - empty_mean;
                    let weight = n as f64 / (n as f64 + settings.prior_samples);
                    m + weight * (mean - m)
                }
                _ => m,
            };
            (key, value as f32)
        })
        .collect();
    let report = TrainingReport {
        games: settings.games,
        samples: acc.samples,
        leaves_seen: acc.by_leave.len(),
        empty_mean,
    };
    (LeaveValues::from_table(table, rules.bingo_bonus), report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::CROSSPLAY;

    #[test]
    fn features_cover_singles_pairs_and_shape() {
        let leave = counts_of(&[Tile::new('E'), Tile::new('E'), Tile::new('S')]);
        let f = features(&leave);
        // E (2) and S (1), the EE pair (1) and ES pairs (2), size 3 with 2
        // vowels.
        let get = |i: usize| f.iter().find(|&&(j, _)| j == i).map(|&(_, v)| v);
        assert_eq!(get(4), Some(2.0));
        assert_eq!(get(18), Some(1.0));
        assert_eq!(f.len(), 2 + 2 + 1);
        // Size 3 starts after size 1's two slots and size 2's three.
        assert_eq!(get(SINGLES + PAIRS + 5 + 2), Some(1.0));
        let one = features(&counts_of(&[Tile::new('Q')]));
        assert!(one.contains(&(SINGLES + PAIRS, 1.0)), "{one:?}");
        let six = features(&counts_of(
            &"AEIOUS".chars().map(Tile::new).collect::<Vec<_>>(),
        ));
        assert!(six.contains(&(FEATURES - 2, 1.0)), "{six:?}");
        assert!(f.iter().all(|&(i, _)| i < FEATURES));
        assert!(features(&[0; 27]).is_empty());
    }

    #[test]
    fn every_leave_is_enumerated_once() {
        let keys = all_leave_keys(&CROSSPLAY.distribution);
        let unique: std::collections::HashSet<u32> = keys.iter().copied().collect();
        assert_eq!(unique.len(), keys.len());
        assert!(keys.len() > 800_000, "{}", keys.len());
        // Keys round-trip through counts.
        for &k in keys.iter().step_by(997) {
            assert_eq!(leave_key(&codes_of(&counts_of_key(k))), k);
        }
    }

    #[test]
    fn cholesky_solves() {
        // [[4, 2], [2, 3]] x = [2, 5] -> x = [-0.5, 2]
        let x = solve_spd(vec![4.0, 2.0, 2.0, 3.0], vec![2.0, 5.0], 2);
        assert!((x[0] + 0.5).abs() < 1e-9 && (x[1] - 2.0).abs() < 1e-9);
    }

    /// A small generation fits a table covering every leave, and the
    /// classic orderings already show: a blank and an S beat a Q.
    #[test]
    fn a_small_generation_learns_the_basics() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        // Start from the heuristic, as training does.
        let words = crate::test_data::file(crate::test_data::WORDS_FST).unwrap();
        let plain = Dictionary::load_from_fst(words).unwrap();
        drop(dict);
        let settings = TrainingSettings {
            games: 2_000,
            ..TrainingSettings::default()
        };
        // On a pool of its own: 2,000 games would otherwise crowd out a
        // timing test running alongside.
        let (table, report) = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool")
            .install(|| train_generation(&plain, &CROSSPLAY, &settings));
        assert!(report.samples > 10_000, "{report:?}");
        assert_eq!(table.len(), all_leave_keys(&CROSSPLAY.distribution).len());
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
            table.value(&t).unwrap()
        };
        assert!(v("?") > v("Q"), "blank {} vs Q {}", v("?"), v("Q"));
        assert!(v("S") > v("Q"), "S {} vs Q {}", v("S"), v("Q"));
        assert!(v("ERS") > v("UUV"), "ERS {} vs UUV {}", v("ERS"), v("UUV"));
    }
}
