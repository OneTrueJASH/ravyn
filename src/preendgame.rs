//! Exact pre-endgame: one tile left in the bag.
//!
//! With one tile in the bag, every play empties it: we draw the last tile,
//! the opponent takes one more turn (Crossplay's equal-turn rule), and we
//! play the last move. What we cannot see is which unseen tile is in the
//! bag; the other unseen tiles are the opponent's rack. So there are at most
//! as many cases as distinct unseen tiles, each as likely as its copies.
//!
//! Each case is solved exactly, as a game of perfect information: the
//! opponent's best turn given our best final reply
//! ([`EndgameSolver::best_value`]). That is the usual pre-endgame
//! assumption, and close to the truth here: with one tile in the bag, the
//! opponent knows our rack but for one tile.
//!
//! A full evaluation of every play would take minutes with a blank on the
//! rack (thousands of plays), so plays are first bounded: whatever the
//! opponent does, they can at least take their highest-scoring play or
//! pass, so either option (followed by our best final play) caps what our
//! play is worth. Plays are then solved exactly from the highest cap down
//! (win probability, then expected spread), and solving stops when no
//! remaining cap can beat the best exact result. The best play is always
//! exact; plays never solved keep their caps and are marked as such.
//!
//! Passing leaves the bag for the opponent: if they play, they draw the
//! last tile, we get one turn and they play last. They may also pass back;
//! the rules for two passes in a row are not modelled, so that is counted
//! as ending the game at the current score. That only adds an option for
//! the opponent, so a pass is never rated higher than it deserves. Their
//! replies are capped (after any of them we can at least pass and let them
//! play last) and solved from the highest cap down.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::board::Board;
use crate::dictionary::Dictionary;
use crate::endgame::EndgameSolver;
use crate::solver::{Move, Solver};
use crate::tile::{Tile, RACK_SIZE};

/// One possible last tile and what a play is worth if it is the one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawOutcome {
    /// The tile in the bag (`'?'` for a blank).
    pub tile: char,
    /// How likely it is to be the one in the bag.
    pub probability: f32,
    /// The play's score minus the opponent's best reply value: the change
    /// in spread (ours minus theirs) from now to the end of the game.
    pub value: i32,
}

/// A play (or the pass) with its pre-endgame evaluation.
#[derive(Debug, Clone)]
pub struct PreEndgameMove {
    pub mv: Move,
    /// Probability of winning (a tie counts half), given the spread before
    /// the play.
    pub win: f32,
    /// Expected change in spread to the end of the game.
    pub spread: f32,
    /// The outcome for each possible last tile.
    pub outcomes: Vec<DrawOutcome>,
    /// False when the play was not solved because it cannot be the best:
    /// `win`, `spread` and the outcome values are then upper bounds.
    pub exact: bool,
}

/// Evaluate every play from `our_rack` on `board`, with one tile in the bag
/// and the rest of `unseen` on the opponent's rack. `score_diff` is our
/// score minus theirs before the play.
///
/// Returns `None` unless exactly one tile is in the bag (`unseen` holds a
/// full rack plus one), or if `cancel` is set. Best first: highest win
/// probability, then expected spread.
pub fn solve_one_in_bag(
    board: &Board,
    dictionary: &Dictionary,
    our_rack: &[Tile],
    unseen: &[Tile],
    score_diff: i32,
    cancel: Option<&AtomicBool>,
) -> Option<Vec<PreEndgameMove>> {
    if unseen.len() != RACK_SIZE + 1 || our_rack.is_empty() {
        return None;
    }
    let stopped = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));

    // The distinct tiles that may be in the bag, with their probability.
    let mut cases: Vec<(Tile, usize)> = Vec::new();
    for t in unseen {
        let t = undesignated(*t);
        match cases.iter_mut().find(|(c, _)| *c == t) {
            Some((_, n)) => *n += 1,
            None => cases.push((t, 1)),
        }
    }
    cases.sort_by_key(|(t, _)| (t.is_blank, t.letter));
    let total = unseen.len() as f32;

    let endgame = EndgameSolver::new(dictionary);
    let plays = Solver::new(board, dictionary).find_moves_for(our_rack);
    let summarize = |mv: Move, values: Vec<i32>, exact: bool| {
        let (mut win, mut spread) = (0.0f32, 0.0f32);
        let outcomes = cases
            .iter()
            .zip(values)
            .map(|(&(tile, copies), value)| {
                let p = copies as f32 / total;
                win += p * match (score_diff + value).cmp(&0) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Less => 0.0,
                };
                spread += p * value as f32;
                DrawOutcome {
                    tile: if tile.is_blank { '?' } else { tile.letter },
                    probability: p,
                    value,
                }
            })
            .collect();
        PreEndgameMove {
            mv,
            win,
            spread,
            outcomes,
            exact,
        }
    };
    // The position after a play and a draw: board, our rack, theirs.
    let case = |mv: &Move, tile: Tile| {
        let mut after = board.clone();
        for (r, c, t) in &mv.tiles_used {
            after.place_tile(*r, *c, *t);
        }
        let mut ours = remove(our_rack, &mv.tiles_from_rack());
        ours.push(tile);
        (after, ours, remove(unseen, &[tile]))
    };

    // Caps: the opponent gets at least their top-scoring play or a pass,
    // each followed by our best final play.
    let mut bounded: Vec<PreEndgameMove> = plays
        .into_par_iter()
        .map(|mv| {
            if stopped() {
                return None;
            }
            let values = cases
                .iter()
                .map(|&(tile, _)| {
                    let (after, ours, theirs) = case(&mv, tile);
                    let final_score =
                        |b: &Board| Solver::new(b, dictionary).best_score(&ours) as i32;
                    let mut opp_floor = -final_score(&after);
                    if let Some(top) = Solver::new(&after, dictionary).best_move_by_score(&theirs) {
                        let mut b2 = after.clone();
                        for (r, c, t) in &top.tiles_used {
                            b2.place_tile(*r, *c, *t);
                        }
                        opp_floor = opp_floor.max(top.score as i32 - final_score(&b2));
                    }
                    mv.score as i32 - opp_floor
                })
                .collect();
            Some(summarize(mv, values, false))
        })
        .collect::<Option<Vec<_>>>()?;
    let rank = |a: &PreEndgameMove, b: &PreEndgameMove| {
        b.win
            .total_cmp(&a.win)
            .then(b.spread.total_cmp(&a.spread))
            .then(b.mv.score.cmp(&a.mv.score))
    };
    bounded.sort_by(rank);

    // Solve from the highest cap down, a batch at a time, until no cap
    // left can beat the best exact result.
    let batch = rayon::current_num_threads().max(1) * 2;
    // The pass, solved exactly; as a known result it also prunes plays.
    let pass_values = cases
        .par_iter()
        .map(|&(tile, _)| {
            if stopped() {
                return None;
            }
            let theirs = remove(unseen, &[tile]);
            Some(pass_value(
                &endgame, board, dictionary, our_rack, &theirs, tile,
            ))
        })
        .collect::<Option<Vec<i32>>>()?;
    let mut solved: Vec<PreEndgameMove> = vec![summarize(Move::pass(our_rack), pass_values, true)];
    let mut rest = bounded.into_iter().peekable();
    loop {
        if stopped() {
            return None;
        }
        let best = solved.iter().min_by(|a, b| rank(a, b));
        let can_beat =
            |cap: &PreEndgameMove| best.is_none_or(|b| (cap.win, cap.spread) >= (b.win, b.spread));
        let mut next = Vec::with_capacity(batch);
        while next.len() < batch {
            match rest.next_if(|cap| can_beat(cap)) {
                Some(cap) => next.push(cap.mv),
                None => break,
            }
        }
        if next.is_empty() {
            break;
        }
        let exact: Vec<PreEndgameMove> = next
            .into_par_iter()
            .map(|mv| {
                let values = cases
                    .iter()
                    .map(|&(tile, _)| {
                        let (after, ours, theirs) = case(&mv, tile);
                        mv.score as i32 - endgame.best_value(&after, &theirs, &ours)
                    })
                    .collect();
                summarize(mv, values, true)
            })
            .collect();
        solved.extend(exact);
    }
    if stopped() {
        return None;
    }
    solved.sort_by(rank);
    solved.extend(rest);
    Some(solved)
}

/// What passing is worth to us (spread change to the end of the game)
/// when the opponent holds `theirs` and `last` is the tile in the bag.
/// They either play, drawing `last`, after which we get one turn and they
/// play the final move, or pass back, counted as ending the game (0).
fn pass_value(
    endgame: &EndgameSolver,
    board: &Board,
    dictionary: &Dictionary,
    ours: &[Tile],
    theirs: &[Tile],
    last: Tile,
) -> i32 {
    // Each reply with its cap. After it we get one turn and they play
    // last, so we get at least the better of passing and our top-scoring
    // play (each followed by their best final play); the reply is worth at
    // most its score minus that.
    let mut replies: Vec<(i32, Move, Board, Vec<Tile>)> = Solver::new(board, dictionary)
        .find_moves_for(theirs)
        .into_par_iter()
        .map(|y| {
            let mut after = board.clone();
            for (r, c, t) in &y.tiles_used {
                after.place_tile(*r, *c, *t);
            }
            let mut drawn = remove(theirs, &y.tiles_from_rack());
            drawn.push(last);
            let their_final = |b: &Board| Solver::new(b, dictionary).best_score(&drawn) as i32;
            let mut ours_floor = -their_final(&after);
            if let Some(z) = Solver::new(&after, dictionary).best_move_by_score(ours) {
                let mut b2 = after.clone();
                for (r, c, t) in &z.tiles_used {
                    b2.place_tile(*r, *c, *t);
                }
                ours_floor = ours_floor.max(z.score as i32 - their_final(&b2));
            }
            let cap = y.score as i32 - ours_floor;
            (cap, y, after, drawn)
        })
        .collect();
    replies.sort_by_key(|r| std::cmp::Reverse(r.0));
    // Solve from the highest cap down, a parallel batch at a time, until no
    // cap can beat the best reply found.
    let batch = rayon::current_num_threads().max(1) * 2;
    let mut best = 0; // passing back
    let mut rest = replies.iter();
    loop {
        let next: Vec<_> = rest
            .by_ref()
            .take_while(|(cap, ..)| *cap > best)
            .take(batch)
            .collect();
        if next.is_empty() {
            break;
        }
        let found = next
            .par_iter()
            .map(|(_, y, after, drawn)| y.score as i32 - endgame.best_value(after, ours, drawn))
            .max()
            .unwrap_or(i32::MIN);
        best = best.max(found);
    }
    -best
}

/// A blank's designation is irrelevant on a rack or in the bag.
fn undesignated(t: Tile) -> Tile {
    if t.is_blank {
        Tile::blank(' ')
    } else {
        t
    }
}

/// `rack` without one tile matching each of `used` (blanks match blanks).
fn remove(rack: &[Tile], used: &[Tile]) -> Vec<Tile> {
    let mut left: Vec<Tile> = rack.iter().map(|t| undesignated(*t)).collect();
    for u in used {
        let u = undesignated(*u);
        if let Some(i) = left.iter().position(|t| *t == u) {
            left.swap_remove(i);
        }
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endgame::FinalTurn;
    use crate::rules::CROSSPLAY;

    /// A position with one tile in the bag, reached by greedy self-play
    /// from a bag shuffled with `seed`: the board, the rack of the side to
    /// move, and everything it cannot see. `None` if the game jumped past
    /// one tile in the bag.
    fn one_in_bag(dict: &Dictionary, seed: u64) -> Option<(Board, Vec<Tile>, Vec<Tile>)> {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut bag = CROSSPLAY.full_tile_set();
        rng.shuffle(&mut bag);
        let mut racks: [Vec<Tile>; 2] = [
            bag.split_off(bag.len() - RACK_SIZE),
            bag.split_off(bag.len() - RACK_SIZE),
        ];
        let mut board = Board::new();
        let mut to_move = 0;
        while bag.len() > 1 {
            let mv = Solver::new(&board, dict).best_move_by_equity(&racks[to_move])?;
            for (r, c, t) in &mv.tiles_used {
                board.place_tile(*r, *c, *t);
            }
            racks[to_move] = remove(&racks[to_move], &mv.tiles_from_rack());
            let draw = mv.tiles_used.len().min(bag.len());
            racks[to_move].extend(bag.split_off(bag.len() - draw));
            to_move = 1 - to_move;
        }
        (bag.len() == 1).then(|| {
            let mut unseen = racks[1 - to_move].clone();
            unseen.extend(bag);
            (board, racks[to_move].clone(), unseen)
        })
    }

    /// Run a heavy test on a pool of its own, so its long jobs do not
    /// queue ahead of other tests' work in the global pool (a timing test
    /// running alongside missed its deadline that way).
    fn isolated<R: Send>(f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool")
            .install(f)
    }

    fn positions(dict: &Dictionary, n: usize) -> Vec<(Board, Vec<Tile>, Vec<Tile>)> {
        (0..200)
            .filter_map(|s| one_in_bag(dict, s))
            .take(n)
            .collect()
    }

    /// Probabilities cover every case, the win probability matches the
    /// outcomes, and the list is ranked.
    #[test]
    fn outcomes_are_consistent() {
        isolated(|| {
            let Some(dict) = crate::test_data::dictionary() else {
                return;
            };
            for (board, ours, unseen) in positions(&dict, 3) {
                let results = solve_one_in_bag(&board, &dict, &ours, &unseen, 0, None).unwrap();
                assert!(!results.is_empty());
                for r in &results {
                    let p: f32 = r.outcomes.iter().map(|o| o.probability).sum();
                    assert!((p - 1.0).abs() < 1e-5);
                    let win: f32 = r
                        .outcomes
                        .iter()
                        .map(|o| {
                            o.probability
                                * if o.value > 0 {
                                    1.0
                                } else if o.value == 0 {
                                    0.5
                                } else {
                                    0.0
                                }
                        })
                        .sum();
                    assert!((win - r.win).abs() < 1e-5);
                }
                let exact: Vec<_> = results.iter().filter(|r| r.exact).collect();
                for w in exact.windows(2) {
                    assert!(w[0].win >= w[1].win);
                }
                // No unsolved play's cap beats the best solved one.
                let best = exact[0];
                for r in results.iter().filter(|r| !r.exact) {
                    assert!((r.win, r.spread) < (best.win, best.spread), "{}", r.mv.word);
                }
            }
        })
    }

    /// Each case's value is what the full endgame search gives after the
    /// play and the draw: the play's score minus the opponent's best move
    /// as `EndgameSolver::solve` ranks it.
    #[test]
    fn values_match_the_full_endgame_search() {
        isolated(|| {
            let Some(dict) = crate::test_data::dictionary() else {
                return;
            };
            let endgame = EndgameSolver::new(&dict);
            for (board, ours, unseen) in positions(&dict, 2) {
                let results = solve_one_in_bag(&board, &dict, &ours, &unseen, 0, None).unwrap();
                assert!(results[0].exact, "the best play is solved");
                for r in results.iter().filter(|r| r.exact).take(3) {
                    let mut after = board.clone();
                    for (row, col, t) in &r.mv.tiles_used {
                        after.place_tile(*row, *col, *t);
                    }
                    for o in &r.outcomes {
                        let tile = if o.tile == '?' {
                            Tile::blank(' ')
                        } else {
                            Tile::new(o.tile)
                        };
                        let mut mine = remove(&ours, &r.mv.tiles_from_rack());
                        mine.push(tile);
                        let theirs = remove(&unseen, &[tile]);
                        let full = endgame.solve(&after, &theirs, &mine, FinalTurn::Opponent);
                        assert_eq!(
                            o.value,
                            r.mv.score as i32 - full[0].endgame_value,
                            "{}",
                            r.mv.word
                        );
                    }
                }
            }
        })
    }

    /// Pruning finds the same best play as solving every play exactly
    /// (on positions small enough to solve in full), at several score
    /// differences.
    #[test]
    fn pruning_keeps_the_best_play() {
        isolated(|| {
            let Some(dict) = crate::test_data::dictionary() else {
                return;
            };
            let endgame = EndgameSolver::new(&dict);
            let small = positions(&dict, 12)
                .into_iter()
                .filter(|(board, ours, _)| {
                    Solver::new(board, &dict).find_moves_for(ours).len() <= 300
                })
                .take(2)
                .collect::<Vec<_>>();
            assert!(!small.is_empty(), "no position small enough");
            for (board, ours, unseen) in small {
                // Every play solved exactly, by brute force: its value for
                // each unseen tile (one entry per copy, so each is equally
                // likely).
                let full: Vec<Vec<i32>> = Solver::new(&board, &dict)
                    .find_moves_for(&ours)
                    .par_iter()
                    .map(|mv| {
                        let mut after = board.clone();
                        for (r, c, t) in &mv.tiles_used {
                            after.place_tile(*r, *c, *t);
                        }
                        let kept = remove(&ours, &mv.tiles_from_rack());
                        unseen
                            .iter()
                            .map(|t| {
                                let tile = undesignated(*t);
                                let mut mine = kept.clone();
                                mine.push(tile);
                                mv.score as i32
                                    - endgame.best_value(&after, &remove(&unseen, &[tile]), &mine)
                            })
                            .collect()
                    })
                    .collect();
                for diff in [-20, 0, 15] {
                    let n = unseen.len() as f32;
                    let best_full = full
                        .iter()
                        .map(|values| {
                            let win: f32 = values
                                .iter()
                                .map(|v| match (diff + v).cmp(&0) {
                                    std::cmp::Ordering::Greater => 1.0,
                                    std::cmp::Ordering::Equal => 0.5,
                                    std::cmp::Ordering::Less => 0.0,
                                })
                                .sum::<f32>()
                                / n;
                            let spread = values.iter().map(|&v| v as f32).sum::<f32>() / n;
                            (win, spread)
                        })
                        .fold((f32::MIN, f32::MIN), |a, b| if b > a { b } else { a });
                    let pruned =
                        solve_one_in_bag(&board, &dict, &ours, &unseen, diff, None).unwrap();
                    assert!(pruned[0].exact);
                    assert!((pruned[0].win - best_full.0).abs() < 1e-4, "diff {diff}");
                    assert!((pruned[0].spread - best_full.1).abs() < 1e-3, "diff {diff}");
                }
            }
        })
    }

    /// The pass is always solved, and its value for each last tile is
    /// what trying every opponent reply gives (with passing back as 0).
    #[test]
    fn the_pass_matches_trying_every_reply() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        isolated(|| {
            let endgame = EndgameSolver::new(&dict);
            for (board, ours, unseen) in positions(&dict, 1) {
                let results = solve_one_in_bag(&board, &dict, &ours, &unseen, 0, None).unwrap();
                let pass = results
                    .iter()
                    .find(|r| r.mv.kind == crate::solver::MoveKind::Pass)
                    .expect("the pass is listed");
                assert!(pass.exact);
                for o in pass.outcomes.iter().take(3) {
                    let last = if o.tile == '?' {
                        Tile::blank(' ')
                    } else {
                        Tile::new(o.tile)
                    };
                    let theirs = remove(&unseen, &[last]);
                    let opp_best = Solver::new(&board, &dict)
                        .find_moves_for(&theirs)
                        .iter()
                        .map(|y| {
                            let mut after = board.clone();
                            for (r, c, t) in &y.tiles_used {
                                after.place_tile(*r, *c, *t);
                            }
                            let mut drawn = remove(&theirs, &y.tiles_from_rack());
                            drawn.push(last);
                            y.score as i32 - endgame.best_value(&after, &ours, &drawn)
                        })
                        .fold(0, i32::max);
                    assert_eq!(o.value, -opp_best, "last tile {}", o.tile);
                }
            }
        })
    }

    #[test]
    fn needs_exactly_one_tile_in_the_bag() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = Board::new();
        let rack: Vec<Tile> = "AEINRST".chars().map(Tile::new).collect();
        assert!(solve_one_in_bag(&board, &dict, &rack, &rack, 0, None).is_none());
    }

    /// How long a solve takes. Run with
    /// `cargo test --release -p ravyn preendgame_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn preendgame_timing() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        for (board, ours, unseen) in positions(&dict, 6) {
            let t = std::time::Instant::now();
            let results = solve_one_in_bag(&board, &dict, &ours, &unseen, 0, None).unwrap();
            let best = &results[0];
            println!(
                "{} plays in {:?}; best {} {} win {:.0}% spread {:+.1}",
                results.len(),
                t.elapsed(),
                best.mv.coords(),
                best.mv.word,
                best.win * 100.0,
                best.spread
            );
        }
    }
}
