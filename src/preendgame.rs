//! Exact pre-endgame: one or two tiles left in the bag.
//!
//! Two tiles: see [`solve_two_in_bag`]. The rest of this introduction is
//! about one.
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
use crate::solver::{BoardAnalysis, Move, Solver};
use crate::tile::{Tile, RACK_SIZE};

/// One possible content of the bag and what a play is worth if it holds
/// that.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawOutcome {
    /// The tiles in the bag (`?` for a blank): one with one in the bag;
    /// with two, both, and for a one-tile play the one we draw first.
    pub tiles: String,
    /// How likely the bag is to hold them.
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
                    tiles: letters(&[tile]),
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
                &endgame, board, dictionary, our_rack, &theirs, tile, None,
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

/// One way the last two tiles can leave the bag: a one-tile play draws
/// `first`, and the opponent's next play draws `second`.
#[derive(Debug, Clone, Copy)]
struct Draw {
    first: Tile,
    second: Tile,
    probability: f32,
}

/// Every ordered draw of two tiles from `unseen`, with its probability.
fn two_tile_draws(unseen: &[Tile]) -> Vec<Draw> {
    let mut counts: Vec<(Tile, usize)> = Vec::new();
    for t in unseen {
        let t = undesignated(*t);
        match counts.iter_mut().find(|(c, _)| *c == t) {
            Some((_, n)) => *n += 1,
            None => counts.push((t, 1)),
        }
    }
    counts.sort_by_key(|(t, _)| (t.is_blank, t.letter));
    let n = unseen.len() as f32;
    let mut draws = Vec::new();
    for &(a, ca) in &counts {
        for &(b, cb) in &counts {
            let ways = if a == b { ca * (ca - 1) } else { ca * cb };
            if ways > 0 {
                draws.push(Draw {
                    first: a,
                    second: b,
                    probability: ways as f32 / (n * (n - 1.0)),
                });
            }
        }
    }
    draws
}

/// The draws with the order forgotten: each pair of tiles the bag may
/// hold, once, with its probability.
fn two_tile_pairs(draws: &[Draw]) -> Vec<Draw> {
    let key = |t: Tile| (t.is_blank, t.letter);
    let mut pairs: Vec<Draw> = Vec::new();
    for d in draws {
        let (a, b) = if key(d.first) <= key(d.second) {
            (d.first, d.second)
        } else {
            (d.second, d.first)
        };
        match pairs.iter_mut().find(|p| p.first == a && p.second == b) {
            Some(p) => p.probability += d.probability,
            None => pairs.push(Draw {
                first: a,
                second: b,
                probability: d.probability,
            }),
        }
    }
    pairs
}

/// A play's evaluation from its value in each case.
fn summarize(mv: Move, outcomes: Vec<DrawOutcome>, exact: bool, score_diff: i32) -> PreEndgameMove {
    let (mut win, mut spread) = (0.0f32, 0.0f32);
    for o in &outcomes {
        win += o.probability
            * match (score_diff + o.value).cmp(&0) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            };
        spread += o.probability * o.value as f32;
    }
    PreEndgameMove {
        mv,
        win,
        spread,
        outcomes,
        exact,
    }
}

/// Best first: win probability, then expected spread, then score.
fn rank(a: &PreEndgameMove, b: &PreEndgameMove) -> std::cmp::Ordering {
    b.win
        .total_cmp(&a.win)
        .then(b.spread.total_cmp(&a.spread))
        .then(b.mv.score.cmp(&a.mv.score))
}

/// True when the list's first play is proven best: it is solved, and no
/// play left unsolved has a cap that could beat it.
pub fn proven(results: &[PreEndgameMove]) -> bool {
    let Some(best) = results.first().filter(|b| b.exact) else {
        return false;
    };
    results
        .iter()
        .filter(|r| !r.exact)
        .all(|r| (r.win, r.spread) < (best.win, best.spread))
}

/// Evaluate every play from `our_rack` on `board` with two tiles in the
/// bag and the rest of `unseen` (nine tiles) on the opponent's rack.
/// `score_diff` is our score minus theirs before the play.
///
/// A play of two or more tiles draws both and empties the bag: the
/// opponent takes one more turn and we play last, solved for each pair of
/// tiles the bag may hold. A one-tile play draws one of the two, leaving
/// the opponent to move with one in the bag: they play (drawing it, after
/// which we get one turn and they play last) or pass back, counted as
/// ending the game as in [`solve_one_in_bag`]'s pass; solved for each
/// order the two tiles may come out in. As there, each case is solved as a
/// game of perfect information.
///
/// Passing is not evaluated: the opponent would face two tiles in the bag
/// with every option open, a search of its own.
///
/// Plays are capped and solved from the highest cap down as in
/// [`solve_one_in_bag`]; a play is given up as soon as its solved cases,
/// with caps for the rest, cannot beat the best play so far. Every play is
/// capped whatever the `deadline`; past it no more are solved, and the list
/// holds what was solved, best first, then the rest with their caps (see
/// [`proven`]). Returns `None` unless nine tiles are unseen, or if `cancel`
/// is set.
pub fn solve_two_in_bag(
    board: &Board,
    dictionary: &Dictionary,
    our_rack: &[Tile],
    unseen: &[Tile],
    score_diff: i32,
    cancel: Option<&AtomicBool>,
    deadline: Option<std::time::Instant>,
) -> Option<Vec<PreEndgameMove>> {
    if unseen.len() != RACK_SIZE + 2 || our_rack.is_empty() {
        return None;
    }
    let stopped = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
    let expired = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
    let draws = two_tile_draws(unseen);
    let pairs = two_tile_pairs(&draws);
    let endgame = EndgameSolver::new(dictionary);
    let plays = Solver::new(board, dictionary).find_moves_for(our_rack);

    // Every case of a play: the board after it, the tiles the bag held
    // (in draw order), our rack and theirs after the draw, the tile still
    // in the bag for a one-tile play, and its probability.
    struct Case {
        bag: [Tile; 2],
        ours: Vec<Tile>,
        theirs: Vec<Tile>,
        last: Option<Tile>,
        probability: f32,
    }
    let cases = |mv: &Move| -> (Board, Vec<Case>) {
        let mut after = board.clone();
        for (r, c, t) in &mv.tiles_used {
            after.place_tile(*r, *c, *t);
        }
        let kept = remove(our_rack, &mv.tiles_from_rack());
        let one_tile = mv.tiles_used.len() == 1;
        let list = if one_tile { &draws } else { &pairs };
        let cases = list
            .iter()
            .map(|d| {
                let mut ours = kept.clone();
                ours.push(d.first);
                if !one_tile {
                    ours.push(d.second);
                }
                Case {
                    bag: [d.first, d.second],
                    ours,
                    theirs: remove(unseen, &[d.first, d.second]),
                    last: one_tile.then_some(d.second),
                    probability: d.probability,
                }
            })
            .collect();
        (after, cases)
    };
    let outcome = |c: &Case, value: i32| DrawOutcome {
        tiles: letters(&c.bag),
        probability: c.probability,
        value,
    };
    let place = |b: &Board, mv: &Move| {
        let mut b = b.clone();
        for (r, c, t) in &mv.tiles_used {
            b.place_tile(*r, *c, *t);
        }
        b
    };

    // Caps: what the opponent can surely get after the play.
    let mut bounded: Vec<PreEndgameMove> = plays
        .into_par_iter()
        .map(|mv| {
            if stopped() {
                return None;
            }
            let (after, cases) = cases(&mv);
            // Every case searches the same board: analyze it once.
            let here = Solver::new(&after, dictionary).prepared();
            let outcomes = cases
                .iter()
                .map(|c| {
                    let best_score = |b: &Board, rack: &[Tile]| {
                        Solver::new(b, dictionary).best_score(rack) as i32
                    };
                    let top = here.best_move_by_score(&c.theirs);
                    let opp_floor = match c.last {
                        // They play last: their top play, or passing, each
                        // followed by our best final play.
                        None => {
                            let mut floor = -(here.best_score(&c.ours) as i32);
                            if let Some(top) = &top {
                                floor = floor.max(
                                    top.score as i32 - best_score(&place(&after, top), &c.ours),
                                );
                            }
                            floor
                        }
                        // They draw the last tile: their top play less our
                        // best reply, or passing back (0).
                        Some(_) => top.as_ref().map_or(0, |top| {
                            (top.score as i32 - best_score(&place(&after, top), &c.ours)).max(0)
                        }),
                    };
                    outcome(c, mv.score as i32 - opp_floor)
                })
                .collect();
            Some(summarize(mv, outcomes, false, score_diff))
        })
        .collect::<Option<Vec<_>>>()?;
    bounded.sort_by(rank);

    let batch = rayon::current_num_threads().max(1) * 2;
    let mut solved: Vec<PreEndgameMove> = Vec::new();
    // Plays given up part way: their solved cases with caps for the rest,
    // which already fell below the best play.
    let mut abandoned: Vec<PreEndgameMove> = Vec::new();
    let mut rest = bounded.into_iter().peekable();
    loop {
        if stopped() {
            return None;
        }
        if expired() {
            break;
        }
        let best = solved
            .iter()
            .min_by(|a, b| rank(a, b))
            .map(|b| (b.win, b.spread));
        let can_beat = |cap: &PreEndgameMove| best.is_none_or(|b| (cap.win, cap.spread) >= b);
        // Until a play is solved nothing can be given up, so the first
        // batch is the single most promising play (its cases in parallel).
        let size = if best.is_some() { batch } else { 1 };
        let mut next = Vec::with_capacity(size);
        while next.len() < size {
            match rest.next_if(|cap| can_beat(cap)) {
                Some(cap) => next.push(cap),
                None => break,
            }
        }
        if next.is_empty() {
            break;
        }
        // Solve each play's cases, likeliest first, with the caps standing
        // in for those not solved yet; give the play up as soon as that
        // mix cannot beat the best play so far.
        let results: Vec<PreEndgameMove> = next
            .into_par_iter()
            .map(|cap| {
                let (after, cases) = cases(&cap.mv);
                let analysis = Solver::new(&after, dictionary).analysis();
                let score = cap.mv.score as i32;
                let mut outcomes = cap.outcomes;
                let replies = ReplyCache::default();
                let solve = |c: &Case| match c.last {
                    None => score - endgame.best_value_with(&after, &analysis, &c.theirs, &c.ours),
                    Some(last) => {
                        score
                            + pass_value(
                                &endgame,
                                &after,
                                dictionary,
                                &c.ours,
                                &c.theirs,
                                last,
                                Some(&replies),
                            )
                    }
                };
                let Some(b) = best else {
                    // Nothing to beat yet: solve every case, in parallel.
                    let values: Vec<i32> = cases.par_iter().map(solve).collect();
                    for (o, v) in outcomes.iter_mut().zip(values) {
                        o.value = v;
                    }
                    return summarize(cap.mv, outcomes, true, score_diff);
                };
                let below = |outcomes: &[DrawOutcome]| {
                    let mix = summarize(cap.mv.clone(), outcomes.to_vec(), false, score_diff);
                    ((mix.win, mix.spread) < b).then_some(mix)
                };
                // A one-tile play leaves the opponent one tile to draw:
                // their top-scoring reply, valued exactly, is a much
                // tighter cap than the first one, for one search each.
                if cap.mv.tiles_used.len() == 1 {
                    let here = Solver::with_analysis(&after, dictionary, &analysis);
                    for (o, c) in outcomes.iter_mut().zip(&cases) {
                        let (Some(last), Some(y)) = (c.last, here.best_move_by_score(&c.theirs))
                        else {
                            continue;
                        };
                        let after_y = place(&after, &y);
                        let squares: Vec<(usize, usize)> =
                            y.tiles_used.iter().map(|&(r, c, _)| (r, c)).collect();
                        let a_y = analysis.after_placing(&after_y, dictionary, &squares);
                        let mut drawn = remove(&c.theirs, &y.tiles_from_rack());
                        drawn.push(last);
                        let floor = y.score as i32
                            - endgame.best_value_with(&after_y, &a_y, &c.ours, &drawn);
                        o.value = o.value.min(score - floor.max(0));
                    }
                    if let Some(mix) = below(&outcomes) {
                        return mix;
                    }
                }
                // Solve the cases likeliest first, the caps standing in for
                // the rest; give the play up as soon as that mix cannot
                // beat the best play so far.
                let mut order: Vec<usize> = (0..cases.len()).collect();
                order.sort_by(|&x, &y| cases[y].probability.total_cmp(&cases[x].probability));
                for (done, &i) in order.iter().enumerate() {
                    outcomes[i].value = solve(&cases[i]);
                    if done + 1 < order.len() {
                        if let Some(mix) = below(&outcomes) {
                            return mix;
                        }
                    }
                }
                summarize(cap.mv, outcomes, true, score_diff)
            })
            .collect();
        for r in results {
            if r.exact {
                solved.push(r);
            } else {
                abandoned.push(r);
            }
        }
    }
    solved.sort_by(rank);
    let mut unsolved: Vec<PreEndgameMove> = rest.chain(abandoned).collect();
    unsolved.sort_by(rank);
    solved.extend(unsolved);
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
    cache: Option<&ReplyCache>,
) -> i32 {
    let remembered = |key: &[u8]| cache.and_then(|c| c.lock().ok()?.get(key).copied());
    let remember = |key: Vec<u8>, entry: (i32, Option<i32>)| {
        if let Some(mut c) = cache.and_then(|c| c.lock().ok()) {
            c.insert(key, entry);
        }
    };
    // Each reply with its cap. After it we get one turn and they play
    // last, so we get at least the better of passing and our top-scoring
    // play (each followed by their best final play); the reply is worth at
    // most its score minus that.
    let base = Solver::new(board, dictionary).analysis();
    #[allow(clippy::type_complexity)]
    let mut replies: Vec<(i32, Move, Board, BoardAnalysis, Vec<Tile>, Vec<u8>)> =
        Solver::with_analysis(board, dictionary, &base)
            .find_moves_for(theirs)
            .into_par_iter()
            .map(|y| {
                let mut after = board.clone();
                for (r, c, t) in &y.tiles_used {
                    after.place_tile(*r, *c, *t);
                }
                let squares: Vec<(usize, usize)> =
                    y.tiles_used.iter().map(|&(r, c, _)| (r, c)).collect();
                let analysis = base.after_placing(&after, dictionary, &squares);
                let mut drawn = remove(theirs, &y.tiles_from_rack());
                drawn.push(last);
                let key = reply_key(&y, &drawn, ours);
                if let Some((cap, _)) = remembered(&key) {
                    return (cap, y, after, analysis, drawn, key);
                }
                let their_final = |b: &Board| Solver::new(b, dictionary).best_score(&drawn) as i32;
                let here = Solver::with_analysis(&after, dictionary, &analysis);
                let mut ours_floor = -(here.best_score(&drawn) as i32);
                if let Some(z) = here.best_move_by_score(ours) {
                    let mut b2 = after.clone();
                    for (r, c, t) in &z.tiles_used {
                        b2.place_tile(*r, *c, *t);
                    }
                    ours_floor = ours_floor.max(z.score as i32 - their_final(&b2));
                }
                let cap = y.score as i32 - ours_floor;
                remember(key.clone(), (cap, None));
                (cap, y, after, analysis, drawn, key)
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
            .map(|(cap, y, after, analysis, drawn, key)| {
                if let Some((_, Some(value))) = remembered(key) {
                    return value;
                }
                let value = y.score as i32 - endgame.best_value_with(after, analysis, ours, drawn);
                remember(key.clone(), (*cap, Some(value)));
                value
            })
            .max()
            .unwrap_or(i32::MIN);
        best = best.max(found);
    }
    -best
}

/// What a one-tile play's cases know about the opponent's replies, shared
/// across the cases: a reply's cap and (once solved) value, by
/// [`reply_key`]. They depend only on the board (one play's), the reply,
/// and both racks after it, so cases that differ only in the tile left in
/// the bag share most of them.
type ReplyCache = std::sync::Mutex<std::collections::HashMap<Vec<u8>, (i32, Option<i32>)>>;

/// A reply with the racks after it, as a cache key: its tiles and squares,
/// then their rack and ours, each in a canonical order.
fn reply_key(y: &Move, theirs: &[Tile], ours: &[Tile]) -> Vec<u8> {
    let mut key = Vec::with_capacity(48);
    for &(r, c, t) in &y.tiles_used {
        key.extend([r as u8, c as u8, t.letter as u8, u8::from(t.is_blank)]);
    }
    for rack in [theirs, ours] {
        key.push(b'|');
        let mut letters: Vec<u8> = rack
            .iter()
            .map(|t| if t.is_blank { b'?' } else { t.letter as u8 })
            .collect();
        letters.sort_unstable();
        key.extend(letters);
    }
    key
}

/// Tiles as letters, `?` for a blank.
fn letters(tiles: &[Tile]) -> String {
    tiles
        .iter()
        .map(|t| if t.is_blank { '?' } else { t.letter })
        .collect()
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

    /// The tile a one-letter outcome names.
    fn tile_of(letters: &str) -> Tile {
        match letters.chars().next() {
            Some('?') => Tile::blank(' '),
            Some(c) => Tile::new(c),
            None => panic!("no tile"),
        }
    }

    /// A position with `n` tiles in the bag, reached by greedy self-play
    /// from a bag shuffled with `seed`: the board, the rack of the side to
    /// move, and everything it cannot see. `None` if the game jumped past
    /// `n` tiles in the bag.
    fn in_bag(dict: &Dictionary, seed: u64, n: usize) -> Option<(Board, Vec<Tile>, Vec<Tile>)> {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut bag = CROSSPLAY.full_tile_set();
        rng.shuffle(&mut bag);
        let mut racks: [Vec<Tile>; 2] = [
            bag.split_off(bag.len() - RACK_SIZE),
            bag.split_off(bag.len() - RACK_SIZE),
        ];
        let mut board = Board::new();
        let mut to_move = 0;
        while bag.len() > n {
            let mv = Solver::new(&board, dict).best_move_by_equity(&racks[to_move])?;
            for (r, c, t) in &mv.tiles_used {
                board.place_tile(*r, *c, *t);
            }
            racks[to_move] = remove(&racks[to_move], &mv.tiles_from_rack());
            let draw = mv.tiles_used.len().min(bag.len());
            racks[to_move].extend(bag.split_off(bag.len() - draw));
            to_move = 1 - to_move;
        }
        (bag.len() == n).then(|| {
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
        positions_in(dict, 1, n)
    }

    /// `count` positions with `bag` tiles in the bag.
    fn positions_in(
        dict: &Dictionary,
        bag: usize,
        count: usize,
    ) -> Vec<(Board, Vec<Tile>, Vec<Tile>)> {
        (0..400)
            .filter_map(|s| in_bag(dict, s, bag))
            .take(count)
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
                        let tile = tile_of(&o.tiles);
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
                    let last = tile_of(&o.tiles);
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
                    assert_eq!(o.value, -opp_best, "last tile {}", o.tiles);
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

    /// Two-tile positions small enough to test quickly.
    fn small_two_in_bag(dict: &Dictionary, n: usize) -> Vec<(Board, Vec<Tile>, Vec<Tile>)> {
        positions_in(dict, 2, 12)
            .into_iter()
            .filter(|(board, ours, _)| Solver::new(board, dict).find_moves_for(ours).len() <= 500)
            .take(n)
            .collect()
    }

    /// Two tiles in the bag: probabilities cover every case, win
    /// probabilities match the outcomes, the solved plays are ranked, and
    /// without a deadline the best play is proven.
    #[test]
    fn two_in_bag_outcomes_are_consistent() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        isolated(|| {
            let small = small_two_in_bag(&dict, 3);
            assert!(!small.is_empty(), "no small two-tile position");
            for (board, ours, unseen) in small {
                let results =
                    solve_two_in_bag(&board, &dict, &ours, &unseen, 0, None, None).unwrap();
                assert!(proven(&results), "{}", letters(&ours));
                for r in &results {
                    let p: f32 = r.outcomes.iter().map(|o| o.probability).sum();
                    assert!((p - 1.0).abs() < 1e-5, "{} sums to {p}", r.mv.word);
                    let cases = if r.mv.tiles_used.len() == 1 {
                        "ordered"
                    } else {
                        "pairs"
                    };
                    assert!(
                        r.outcomes.iter().all(|o| o.tiles.chars().count() == 2),
                        "{cases}"
                    );
                    let win: f32 = r
                        .outcomes
                        .iter()
                        .map(|o| o.probability * (o.value.signum() as f32 + 1.0) / 2.0)
                        .sum();
                    assert!((win - r.win).abs() < 1e-5);
                }
                let exact: Vec<_> = results.iter().filter(|r| r.exact).collect();
                for w in exact.windows(2) {
                    assert!((w[0].win, w[0].spread) >= (w[1].win, w[1].spread));
                }
                assert!(results
                    .iter()
                    .all(|r| r.mv.kind == crate::solver::MoveKind::Play));
                // With no time at all, every play still comes back, capped.
                let rushed = solve_two_in_bag(
                    &board,
                    &dict,
                    &ours,
                    &unseen,
                    0,
                    None,
                    Some(std::time::Instant::now()),
                )
                .unwrap();
                assert_eq!(rushed.len(), results.len());
            }
        })
    }

    /// A play's exact value in each two-tile case, the slow way: the
    /// endgame after a play that empties the bag, or every opponent reply
    /// after a one-tile play.
    fn brute_two_in_bag(
        dict: &Dictionary,
        board: &Board,
        ours: &[Tile],
        unseen: &[Tile],
        mv: &Move,
    ) -> Vec<(f32, i32)> {
        let endgame = EndgameSolver::new(dict);
        let mut after = board.clone();
        for (r, c, t) in &mv.tiles_used {
            after.place_tile(*r, *c, *t);
        }
        let kept = remove(ours, &mv.tiles_from_rack());
        let draws = two_tile_draws(unseen);
        draws
            .iter()
            .map(|d| {
                let theirs = remove(unseen, &[d.first, d.second]);
                let mut mine = kept.clone();
                mine.push(d.first);
                let value = if mv.tiles_used.len() == 1 {
                    let opp_best = Solver::new(&after, dict)
                        .find_moves_for(&theirs)
                        .iter()
                        .map(|y| {
                            let mut b = after.clone();
                            for (r, c, t) in &y.tiles_used {
                                b.place_tile(*r, *c, *t);
                            }
                            let mut drawn = remove(&theirs, &y.tiles_from_rack());
                            drawn.push(d.second);
                            y.score as i32 - endgame.best_value(&b, &mine, &drawn)
                        })
                        .fold(0, i32::max);
                    mv.score as i32 - opp_best
                } else {
                    mine.push(d.second);
                    mv.score as i32 - endgame.best_value(&after, &theirs, &mine)
                };
                (d.probability, value)
            })
            .collect()
    }

    /// Pruning, abandoning and the reply cache find the same best play as
    /// solving every play in every case the slow way, at several score
    /// differences. Takes about six minutes; run it after changing the
    /// two-tile solver with
    /// `cargo test --release -p ravyn two_in_bag_matches_brute_force -- --ignored`.
    #[test]
    #[ignore]
    fn two_in_bag_matches_brute_force() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        isolated(|| {
            let (board, ours, unseen) = positions_in(&dict, 2, 12)
                .into_iter()
                .min_by_key(|(board, ours, _)| Solver::new(board, &dict).find_moves_for(ours).len())
                .unwrap();
            let full: Vec<Vec<(f32, i32)>> = Solver::new(&board, &dict)
                .find_moves_for(&ours)
                .par_iter()
                .map(|mv| brute_two_in_bag(&dict, &board, &ours, &unseen, mv))
                .collect();
            for diff in [-10, 0, 10] {
                let best = full
                    .iter()
                    .map(|cases| {
                        let win: f32 = cases
                            .iter()
                            .map(|&(p, v)| p * ((diff + v).signum() as f32 + 1.0) / 2.0)
                            .sum();
                        let spread: f32 = cases.iter().map(|&(p, v)| p * v as f32).sum();
                        (win, spread)
                    })
                    .fold((f32::MIN, f32::MIN), |a, b| if b > a { b } else { a });
                let pruned =
                    solve_two_in_bag(&board, &dict, &ours, &unseen, diff, None, None).unwrap();
                assert!(proven(&pruned), "diff {diff}");
                assert!((pruned[0].win - best.0).abs() < 1e-4, "diff {diff}");
                assert!((pruned[0].spread - best.1).abs() < 1e-3, "diff {diff}");
                // The solved values agree with the slow way, case by case.
                for r in pruned.iter().filter(|r| r.exact).take(3) {
                    let slow = brute_two_in_bag(&dict, &board, &ours, &unseen, &r.mv);
                    let spread: f32 = slow.iter().map(|&(p, v)| p * v as f32).sum();
                    assert!(
                        (spread - r.spread).abs() < 1e-3,
                        "{} diff {diff}",
                        r.mv.word
                    );
                }
            }
        })
    }

    /// How long a two-tile solve takes. Run with
    /// `cargo test --release -p ravyn two_in_bag_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn two_in_bag_timing() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        for (board, ours, unseen) in positions_in(&dict, 2, 10) {
            let t = std::time::Instant::now();
            let results = solve_two_in_bag(&board, &dict, &ours, &unseen, 0, None, None).unwrap();
            let best = &results[0];
            println!(
                "{} {} plays, {} solved in {:?}; best {} {} win {:.0}% spread {:+.1}",
                letters(&ours),
                results.len(),
                results.iter().filter(|r| r.exact).count(),
                t.elapsed(),
                best.mv.coords(),
                best.mv.word,
                best.win * 100.0,
                best.spread
            );
        }
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
