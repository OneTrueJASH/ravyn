//! Rack inference: what the opponent probably kept, given the play they
//! made.
//!
//! Every leave they could have kept is tried: the tiles they played plus
//! that leave make a rack, and the leave stays in the running if their
//! play was within [`InferenceSettings::equity_margin`] of the best play
//! that rack had (an exchange included, when the bag allowed one). A
//! player rarely chooses a play far below their best, so racks with a much
//! better play are unlikely.
//!
//! Each surviving leave counts as often as it could have been drawn from
//! the unseen tiles (the product over letters of `C(unseen, kept)`), and
//! the result is per-letter marginals over the kept tiles, which the
//! simulation uses to bias the opponent's deals.
//!
//! The search for each leave stops at the first play that beats the cutoff
//! ([`Solver::any_play_beats`]), and the board is analyzed once for all
//! leaves ([`Solver::prepared`]); leaves are checked in parallel.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::board::{Board, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::game::best_exchange;
use crate::solver::{Solver, EXCHANGE_MIN_BAG};
use crate::tile::{Tile, RACK_SIZE};

/// Index of the blank in a 27-entry count array.
const BLANK: usize = 26;

/// How inference judges a rack.
#[derive(Debug, Clone, Copy)]
pub struct InferenceSettings {
    /// A rack stays in the running if the observed play's equity is within
    /// this many points of the rack's best play.
    pub equity_margin: f32,
    /// How many of the most likely leaves to report.
    pub top_leaves: usize,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        InferenceSettings {
            equity_margin: 5.0,
            top_leaves: 10,
        }
    }
}

/// What inference says about one letter in the opponent's kept tiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterMarginal {
    /// The letter, with `'?'` for the blank.
    pub letter: char,
    /// Expected number of copies in the kept tiles.
    pub expected: f32,
    /// Probability (0-100) that they kept at least one copy.
    pub p_at_least_one: f32,
}

/// The result of inferring the opponent's leave.
#[derive(Debug, Clone, PartialEq)]
pub struct RackInference {
    /// The tiles the opponent played, with `'?'` for blanks.
    pub played_tiles: String,
    /// The most likely leaves (letters A-Z, then `'?'`), with their
    /// probability in percent, most likely first.
    pub top_leaves: Vec<(String, f32)>,
    /// Per-letter marginals over the kept tiles, for every letter that may
    /// have been kept (A-Z, then `'?'`).
    pub letter_marginals: Vec<LetterMarginal>,
    /// Racks consistent with the play, counting draw multiplicity.
    pub total_racks: Option<u64>,
}

/// Infer what the opponent kept after playing `played` (the tiles from
/// their rack, blanks as blank tiles) for `score` points on `board` (the
/// board before their play). `our_rack` holds tiles they cannot have.
///
/// Returns `None` if no rack is consistent with the play (or the search
/// was cancelled): nothing can be said about their leave.
pub fn infer_leave(
    board: &Board,
    dictionary: &Dictionary,
    our_rack: &[Tile],
    played: &[Tile],
    score: u32,
    settings: &InferenceSettings,
    cancel: Option<&AtomicBool>,
) -> Option<RackInference> {
    if played.is_empty() || played.len() > RACK_SIZE {
        return None;
    }
    let rules = board.rules();

    // Tiles the opponent could have held besides the ones they played:
    // everything not on the board before their play and not on our rack.
    let mut pool = [0u8; 27];
    for (i, n) in pool.iter_mut().enumerate().take(26) {
        *n = rules.count((b'A' + i as u8) as char) as u8;
    }
    pool[BLANK] = rules.blank_count() as u8;
    let mut on_board = 0usize;
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if let Some(t) = board.get_tile(r, c) {
                take(&mut pool, t);
                on_board += 1;
            }
        }
    }
    for t in our_rack {
        take(&mut pool, t);
    }
    for t in played {
        if !take(&mut pool, t) {
            // They played a tile that cannot have been unseen.
            return None;
        }
    }
    let pool_len: usize = pool.iter().map(|&n| n as usize).sum();

    // They kept a full rack's worth, or everything left if the bag ran
    // short. With no bag left after their rack, every leave is consistent
    // (there is nothing to infer from: the rack is all that is unseen).
    let keep = (RACK_SIZE - played.len()).min(pool_len);
    let bag_after = pool_len - keep;
    let rack_len = played.len() + keep;
    let unseen_for_them = rules.total_tiles().saturating_sub(on_board + rack_len);
    let can_exchange = pool_len >= EXCHANGE_MIN_BAG + keep;

    let mut leaves = Vec::new();
    enumerate_leaves(&pool, keep, 0, &mut [0u8; 27], &mut leaves);

    let solver = Solver::new(board, dictionary).prepared();
    let played_tiles: Vec<Tile> = played
        .iter()
        .map(|t| if t.is_blank { Tile::blank(' ') } else { *t })
        .collect();
    let stopped = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));

    let consistent: Vec<([u8; 27], u64)> = leaves
        .par_iter()
        .filter(|leave| {
            if stopped() {
                return false;
            }
            if bag_after == 0 {
                return true;
            }
            let leave_tiles = tiles_of(leave);
            let cutoff = score as f32
                + dictionary.leave_value(&leave_tiles, Some(unseen_for_them))
                + settings.equity_margin;
            let mut rack = played_tiles.clone();
            rack.extend_from_slice(&leave_tiles);
            if solver.any_play_beats(&rack, cutoff) {
                return false;
            }
            !(can_exchange
                && best_exchange(&rack, dictionary, Some(unseen_for_them))
                    .is_some_and(|ex| ex.equity > cutoff))
        })
        .map(|leave| (*leave, draws(&pool, leave)))
        .collect();
    if stopped() {
        return None;
    }

    let total: u64 = consistent.iter().map(|(_, w)| w).sum();
    if total == 0 {
        return None;
    }
    let total_f = total as f64;

    let mut expected = [0f64; 27];
    let mut at_least_one = [0f64; 27];
    for (leave, w) in &consistent {
        for i in 0..27 {
            if leave[i] > 0 {
                expected[i] += *w as f64 * leave[i] as f64;
                at_least_one[i] += *w as f64;
            }
        }
    }
    let letter_marginals = (0..27)
        .filter(|&i| expected[i] > 0.0)
        .map(|i| LetterMarginal {
            letter: letter_of(i),
            expected: (expected[i] / total_f) as f32,
            p_at_least_one: (100.0 * at_least_one[i] / total_f) as f32,
        })
        .collect();

    let mut ranked: Vec<(String, u64)> = consistent
        .iter()
        .map(|(leave, w)| (leave_string(leave), *w))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top_leaves = ranked
        .into_iter()
        .take(settings.top_leaves)
        .map(|(leave, w)| (leave, (100.0 * w as f64 / total_f) as f32))
        .collect();

    Some(RackInference {
        played_tiles: played
            .iter()
            .map(|t| if t.is_blank { '?' } else { t.letter })
            .collect(),
        top_leaves,
        letter_marginals,
        total_racks: Some(total),
    })
}

/// Remove one `tile` from `pool`; false if there was none to remove.
fn take(pool: &mut [u8; 27], tile: &Tile) -> bool {
    let i = if tile.is_blank {
        BLANK
    } else {
        match tile.letter {
            c @ 'A'..='Z' => (c as u8 - b'A') as usize,
            _ => return false,
        }
    };
    if pool[i] == 0 {
        return false;
    }
    pool[i] -= 1;
    true
}

/// Every multiset of `size` tiles from `pool`, as count arrays, choosing
/// letters from index `from` on (so each multiset appears once).
fn enumerate_leaves(
    pool: &[u8; 27],
    size: usize,
    from: usize,
    current: &mut [u8; 27],
    out: &mut Vec<[u8; 27]>,
) {
    if size == 0 {
        out.push(*current);
        return;
    }
    for i in from..27 {
        if current[i] < pool[i] {
            current[i] += 1;
            enumerate_leaves(pool, size - 1, i, current, out);
            current[i] -= 1;
        }
    }
}

/// Ways to draw `leave` from `pool`: the product of `C(pool, leave)`.
fn draws(pool: &[u8; 27], leave: &[u8; 27]) -> u64 {
    pool.iter()
        .zip(leave)
        .map(|(&n, &k)| choose(n as u64, k as u64))
        .product()
}

fn choose(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

fn letter_of(i: usize) -> char {
    if i == BLANK {
        '?'
    } else {
        (b'A' + i as u8) as char
    }
}

/// The tiles of a count array (blanks undesignated).
fn tiles_of(leave: &[u8; 27]) -> Vec<Tile> {
    let mut tiles = Vec::with_capacity(RACK_SIZE);
    for (i, &n) in leave.iter().enumerate() {
        for _ in 0..n {
            tiles.push(if i == BLANK {
                Tile::blank(' ')
            } else {
                Tile::new(letter_of(i))
            });
        }
    }
    tiles
}

/// A count array as letters, A-Z then `'?'`.
fn leave_string(leave: &[u8; 27]) -> String {
    tiles_of(leave)
        .iter()
        .map(|t| if t.is_blank { '?' } else { t.letter })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::SolverState;
    use crate::solver::MoveKind;

    const BEECH: &str = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15";

    #[test]
    fn choose_and_draws() {
        assert_eq!(choose(5, 0), 1);
        assert_eq!(choose(5, 2), 10);
        assert_eq!(choose(12, 6), 924);
        assert_eq!(choose(2, 3), 0);
        let mut pool = [0u8; 27];
        pool[0] = 9; // A
        pool[BLANK] = 3;
        let mut leave = [0u8; 27];
        leave[0] = 2;
        leave[BLANK] = 1;
        assert_eq!(draws(&pool, &leave), 36 * 3);
    }

    #[test]
    fn leaves_are_enumerated_once_each() {
        let mut pool = [0u8; 27];
        pool[0] = 2; // AA
        pool[1] = 1; // B
        pool[BLANK] = 1;
        let mut out = Vec::new();
        enumerate_leaves(&pool, 2, 0, &mut [0u8; 27], &mut out);
        let names: Vec<String> = out.iter().map(leave_string).collect();
        assert_eq!(names, vec!["AA", "AB", "A?", "B?"]);
    }

    /// The opponent holds DIOQTUZ and plays its best four-tile word.
    /// Inference responds to the score: no rack makes a 0-point play its
    /// choice, the real score leaves some racks, and an inflated score
    /// accepts every rack. The kept tiles add up to three.
    #[test]
    fn inference_responds_to_the_move_score() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let pre = SolverState::from_cgp(&format!("{BEECH} DIOQTUZ")).unwrap();
        let mv = pre
            .get_suggestions(&dict)
            .into_iter()
            .filter(|m| m.kind == MoveKind::Play && m.tiles_used.len() == 4)
            .max_by_key(|m| m.score)
            .unwrap();
        let played = mv.tiles_from_rack();
        let ours: Vec<Tile> = "AEILNOS".chars().map(Tile::new).collect();
        let settings = InferenceSettings::default();
        let infer =
            |score: u32| infer_leave(&pre.board, &dict, &ours, &played, score, &settings, None);

        assert!(infer(0).is_none(), "no rack plays {} for 0 points", mv.word);
        let real = infer(mv.score).expect("racks consistent with the real score");
        let all = infer(mv.score + 40).expect("every rack is consistent");
        assert!(real.total_racks.unwrap() < all.total_racks.unwrap());

        let kept: f32 = real.letter_marginals.iter().map(|m| m.expected).sum();
        assert!((kept - 3.0).abs() < 1e-3, "kept {kept}");
        let total_pct: f32 = all.top_leaves.iter().map(|(_, p)| p).sum();
        assert!(total_pct <= 100.0 + 1e-3);
        for m in &real.letter_marginals {
            assert!((0.0..=100.0 + 1e-3).contains(&m.p_at_least_one));
        }
    }

    #[test]
    fn a_cancelled_inference_says_nothing() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let pre = SolverState::from_cgp(&format!("{BEECH} DIOQTUZ")).unwrap();
        let played = [Tile::new('Q'), Tile::new('U')];
        let cancel = AtomicBool::new(true);
        let settings = InferenceSettings::default();
        assert!(infer_leave(
            &pre.board,
            &dict,
            &[],
            &played,
            20,
            &settings,
            Some(&cancel)
        )
        .is_none());
    }
}
