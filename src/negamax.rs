//! Exact endgame search for the go-out ending ([`Ending::GoOut`]):
//! negamax with alpha-beta pruning and a transposition table.
//!
//! Once the bag is empty both racks are known, so the rest of the game is
//! a game of perfect information. A player who uses their last tile ends
//! it and gains twice the other's rack value in spread; six scoreless turns
//! in a row end it with each player losing their own rack's value.
//!
//! The search deepens one ply at a time within a time budget. A depth that
//! finishes without reaching its limit anywhere is exact, and the search
//! stops there; otherwise positions at the limit are valued as if the game
//! ended scoreless there (each side losing its rack), and the result is
//! marked inexact. Plays are tried highest-scoring first, plays that go out
//! first of all, and a transposition table keeps what each position is
//! worth across move orders and depths.
//!
//! [`Ending::GoOut`]: crate::rules::Ending::GoOut

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::board::{Board, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::rules::Ending;
use crate::solver::{Move, MoveKind, Solver};
use crate::tile::Tile;

/// A root move and what it is worth (spread change to the end of the game,
/// for the side to move).
#[derive(Debug, Clone)]
pub struct EndgameLine {
    pub mv: Move,
    pub value: i32,
    /// False when the move was cut off by a better one: `value` is then an
    /// upper bound.
    pub exact: bool,
}

/// The result of a go-out endgame search.
#[derive(Debug, Clone)]
pub struct GoOutResult {
    /// Every move, best first; the first is the best found.
    pub lines: Vec<EndgameLine>,
    /// True when the deepest finished search reached the end of the game on
    /// every line it looked at: the best value is then exact.
    pub solved: bool,
    /// Plies of the deepest finished search.
    pub depth: usize,
    /// Positions searched.
    pub nodes: u64,
}

/// How a stored value bounds the true one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct Entry {
    depth: usize,
    value: i32,
    bound: Bound,
    /// Whether the value rests on positions cut off by the depth limit.
    horizon: bool,
    /// Index of the best move in the (deterministic) move order, tried
    /// first when the position comes up again.
    best: usize,
}

/// The table's keys are already hashes ([`position_key`]): use them as
/// they are instead of hashing them again.
#[derive(Default)]
struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ b as u64;
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

type Table = HashMap<u64, Entry, std::hash::BuildHasherDefault<KeyHasher>>;

struct Search<'a> {
    dict: &'a Dictionary,
    table: Table,
    nodes: u64,
    deadline: Option<Instant>,
    cancel: Option<&'a AtomicBool>,
    out_of_time: bool,
}

/// Limits and starting state of a go-out endgame search.
#[derive(Debug, Clone, Copy)]
pub struct GoOutSettings {
    /// Scoreless turns already played in a row.
    pub scoreless: usize,
    /// Deepest search, in plies.
    pub max_depth: usize,
    /// Time allowed; `None` searches until solved or `max_depth`.
    pub budget: Option<Duration>,
}

impl Default for GoOutSettings {
    fn default() -> Self {
        GoOutSettings {
            scoreless: 0,
            max_depth: 30,
            budget: Some(Duration::from_secs(10)),
        }
    }
}

/// Solve the endgame with the bag empty, `ours` to move and `theirs` the
/// opponent's rack, deepening as `settings` allow. Returns `None` only if
/// `cancel` is set (or the budget runs out) before the first depth
/// finishes.
pub fn solve_go_out(
    board: &Board,
    dict: &Dictionary,
    ours: &[Tile],
    theirs: &[Tile],
    settings: &GoOutSettings,
    cancel: Option<&AtomicBool>,
) -> Option<GoOutResult> {
    let (scoreless, max_depth, budget) = (settings.scoreless, settings.max_depth, settings.budget);
    let mut search = Search {
        dict,
        table: Table::default(),
        nodes: 0,
        deadline: budget.map(|b| Instant::now() + b),
        cancel,
        out_of_time: false,
    };
    let mut best: Option<GoOutResult> = None;
    for depth in 1..=max_depth.max(1) {
        // Each depth searches the root moves in the order the previous one
        // ranked them, so the best line is found early and cuts the rest.
        let order = best
            .as_ref()
            .map(|b| b.lines.iter().map(|l| l.mv.clone()).collect());
        let Some((lines, horizon)) = search.root(board, ours, theirs, scoreless, depth, order)
        else {
            break;
        };
        let solved = !horizon;
        best = Some(GoOutResult {
            lines,
            solved,
            depth,
            nodes: search.nodes,
        });
        if solved {
            break;
        }
    }
    best
}

/// Total tile value of a rack.
fn rack_value(board: &Board, rack: &[Tile]) -> i32 {
    rack.iter()
        .map(|t| board.rules().tile_value(t) as i32)
        .sum()
}

/// `rack` less the tiles `mv` takes from it.
fn after_play(rack: &[Tile], mv: &Move) -> Vec<Tile> {
    let mut left = rack.to_vec();
    for t in mv.tiles_from_rack() {
        if let Some(i) = left
            .iter()
            .position(|x| x.is_blank == t.is_blank && (t.is_blank || x.letter == t.letter))
        {
            left.swap_remove(i);
        }
    }
    left
}

/// A key for the position: board tiles, both racks (order-free), the side
/// to move's rack first, and the scoreless count.
fn position_key(board: &Board, mover: &[Tile], other: &[Tile], scoreless: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            match board.get_tile(r, c) {
                Some(t) => mix(t.letter as u8 | if t.is_blank { 0x80 } else { 0 }),
                None => mix(0),
            }
        }
    }
    for rack in [mover, other] {
        let mut counts = [0u8; 27];
        for t in rack {
            let i = if t.is_blank {
                26
            } else {
                (t.letter as u8 - b'A') as usize
            };
            counts[i] += 1;
        }
        for n in counts {
            mix(n);
        }
        mix(0xFF);
    }
    mix(scoreless as u8);
    h
}

impl Search<'_> {
    fn stopped(&mut self) -> bool {
        if self.nodes.is_multiple_of(256)
            && (self.deadline.is_some_and(|d| Instant::now() >= d)
                || self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)))
        {
            self.out_of_time = true;
        }
        self.out_of_time
    }

    /// The side to move's moves, most promising first: plays that go out
    /// (by score plus the bonus), then by score, then passing.
    fn ordered_moves(&self, board: &Board, mover: &[Tile], other: &[Tile]) -> Vec<Move> {
        let mut moves = Solver::new(board, self.dict).find_moves_for(mover);
        let bonus = 2 * rack_value(board, other);
        moves.sort_by_cached_key(|m| {
            let out = m.tiles_used.len() == mover.len();
            std::cmp::Reverse(m.score as i32 + if out { bonus } else { 0 })
        });
        moves.push(Move::pass(mover));
        moves
    }

    /// Value and line of each root move at `depth`, best first, and whether
    /// any line reached the depth limit. `None` if time ran out.
    fn root(
        &mut self,
        board: &Board,
        ours: &[Tile],
        theirs: &[Tile],
        scoreless: usize,
        depth: usize,
        order: Option<Vec<Move>>,
    ) -> Option<(Vec<EndgameLine>, bool)> {
        let mut lines = Vec::new();
        let mut alpha = i32::MIN + 1;
        let mut horizon = false;
        let mut board = board.clone();
        let moves = order.unwrap_or_else(|| self.ordered_moves(&board, ours, theirs));
        for mv in moves {
            let (value, h) = self.child_value(
                &mut board,
                ours,
                theirs,
                scoreless,
                &mv,
                depth,
                alpha,
                i32::MAX,
            )?;
            // A move that does not beat the best so far was searched with
            // a window that only proves it is no better.
            let exact = value > alpha || lines.is_empty();
            // A cut-off move's bound is only proof if it did not rest on
            // the depth limit either, so any horizon makes the depth inexact.
            horizon |= h;
            if value > alpha {
                alpha = value;
            }
            lines.push(EndgameLine { mv, value, exact });
        }
        lines.sort_by(|a, b| b.value.cmp(&a.value).then(b.exact.cmp(&a.exact)));
        Some((lines, horizon))
    }

    /// The value of playing `mv` for the side holding `mover`, searched
    /// `depth` plies deep with the window (`alpha`, `beta`), and whether it
    /// rests on the depth limit. Leaves `board` as it found it.
    #[allow(clippy::too_many_arguments)]
    fn child_value(
        &mut self,
        board: &mut Board,
        mover: &[Tile],
        other: &[Tile],
        scoreless: usize,
        mv: &Move,
        depth: usize,
        alpha: i32,
        beta: i32,
    ) -> Option<(i32, bool)> {
        let score = mv.score as i32;
        let left = after_play(mover, mv);
        if mv.kind == MoveKind::Play && left.is_empty() {
            // Out: the game ends with the bonus.
            return Some((score + 2 * rack_value(board, other), false));
        }
        let next_scoreless = if score == 0 { scoreless + 1 } else { 0 };
        for (r, c, t) in &mv.tiles_used {
            board.place_tile(*r, *c, *t);
        }
        let reply = self.negamax(
            board,
            other,
            &left,
            next_scoreless,
            depth - 1,
            beta.saturating_neg().saturating_add(score),
            alpha.saturating_neg().saturating_add(score),
        );
        for (r, c, _) in &mv.tiles_used {
            board.remove_tile(*r, *c);
        }
        let (v, h) = reply?;
        Some((score - v, h))
    }

    /// The value for the side to move (holding `mover`), within the window
    /// (`alpha`, `beta`), and whether it rests on the depth limit.
    #[allow(clippy::too_many_arguments)] // the search state, recursively
    fn negamax(
        &mut self,
        board: &mut Board,
        mover: &[Tile],
        other: &[Tile],
        scoreless: usize,
        depth: usize,
        mut alpha: i32,
        beta: i32,
    ) -> Option<(i32, bool)> {
        self.nodes += 1;
        if self.stopped() {
            return None;
        }
        // Six scoreless turns: each side loses its own rack.
        let rack_diff = rack_value(board, other) - rack_value(board, mover);
        if scoreless >= Ending::SCORELESS_TURNS_TO_END {
            return Some((rack_diff, false));
        }
        if depth == 0 {
            return Some((rack_diff, true));
        }
        let key = position_key(board, mover, other, scoreless);
        if let Some(e) = self.table.get(&key).copied() {
            if e.depth >= depth || !e.horizon {
                match e.bound {
                    Bound::Exact => return Some((e.value, e.horizon)),
                    Bound::Lower if e.value >= beta => return Some((e.value, e.horizon)),
                    Bound::Upper if e.value <= alpha => return Some((e.value, e.horizon)),
                    _ => {}
                }
            }
        }
        let alpha_in = alpha;
        let mut best = i32::MIN + 1;
        let mut best_index = 0;
        let mut horizon = false;
        let moves = self.ordered_moves(board, mover, other);
        // The best move from an earlier visit first, then the rest in order.
        let first = self
            .table
            .get(&key)
            .map_or(0, |e| e.best.min(moves.len() - 1));
        let order = std::iter::once(first).chain((0..moves.len()).filter(|&i| i != first));
        for (n, i) in order.enumerate() {
            // Principal variation search: the first move gets the full
            // window; the rest are first asked only whether they beat it
            // (a null window), and searched fully only if they do.
            let mv = &moves[i];
            let (value, h) = if n == 0 {
                self.child_value(board, mover, other, scoreless, mv, depth, alpha, beta)?
            } else {
                let probe = self.child_value(
                    board,
                    mover,
                    other,
                    scoreless,
                    mv,
                    depth,
                    alpha,
                    alpha.saturating_add(1),
                )?;
                if probe.0 > alpha && probe.0 < beta {
                    self.child_value(board, mover, other, scoreless, mv, depth, alpha, beta)?
                } else {
                    probe
                }
            };
            if value > best {
                best = value;
                best_index = i;
            }
            horizon |= h;
            if value > alpha {
                alpha = value;
            }
            if alpha >= beta {
                break;
            }
        }
        let bound = if best <= alpha_in {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.table.insert(
            key,
            Entry {
                depth,
                value: best,
                bound,
                horizon,
                best: best_index,
            },
        );
        Some((best, horizon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::SCRABBLE;

    fn tiles(s: &str) -> Vec<Tile> {
        s.chars()
            .map(|c| {
                if c == '?' {
                    Tile::blank(' ')
                } else {
                    Tile::new(c)
                }
            })
            .collect()
    }

    fn cat_board() -> Board {
        let mut board = Board::with_rules(&SCRABBLE);
        for (c, l) in [(6, 'C'), (7, 'A'), (8, 'T')] {
            board.place_tile(7, c, Tile::new(l));
        }
        board
    }

    /// Going out at once is best when it is possible: S hooks for its
    /// score plus twice the opponent's VV.
    #[test]
    fn going_out_now_is_found_and_exact() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = cat_board();
        let r = solve_go_out(
            &board,
            &dict,
            &tiles("S"),
            &tiles("VV"),
            &GoOutSettings {
                scoreless: 0,
                max_depth: 10,
                budget: None,
            },
            None,
        )
        .unwrap();
        assert!(r.solved);
        let best = &r.lines[0];
        assert_eq!(best.mv.tiles_used.len(), 1);
        let s = Solver::new(&board, &dict).best_score(&tiles("S")) as i32;
        assert_eq!(best.value, s + 16);
    }

    /// With nothing playable on either side the game ends after six
    /// passes; the best "move" is passing, worth their rack minus ours.
    #[test]
    fn six_passes_end_a_stuck_game() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = Board::with_rules(&SCRABBLE);
        let r = solve_go_out(
            &board,
            &dict,
            &tiles("VV"),
            &tiles("QQ"),
            &GoOutSettings {
                scoreless: 0,
                max_depth: 10,
                budget: None,
            },
            None,
        )
        .unwrap();
        assert!(r.solved);
        assert_eq!(r.lines[0].value, 20 - 8);
    }

    /// The search agrees with a plain minimax without pruning or table on
    /// small endgames reached by self-play.
    #[test]
    fn matches_plain_minimax_on_small_endgames() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        fn minimax(
            dict: &Dictionary,
            board: &mut Board,
            mover: &[Tile],
            other: &[Tile],
            s: usize,
        ) -> i32 {
            if s >= 6 {
                return rack_value(board, other) - rack_value(board, mover);
            }
            let mut best = i32::MIN;
            let mut moves = Solver::new(board, dict).find_moves_for(mover);
            moves.push(Move::pass(mover));
            for mv in moves {
                let left = after_play(mover, &mv);
                let score = mv.score as i32;
                let v = if mv.kind == MoveKind::Play && left.is_empty() {
                    score + 2 * rack_value(board, other)
                } else {
                    for (r, c, t) in &mv.tiles_used {
                        board.place_tile(*r, *c, *t);
                    }
                    let v = score
                        - minimax(
                            dict,
                            board,
                            other,
                            &left,
                            if score == 0 { s + 1 } else { 0 },
                        );
                    for (r, c, _) in &mv.tiles_used {
                        board.remove_tile(*r, *c);
                    }
                    v
                };
                best = best.max(v);
            }
            best
        }
        let mut checked = 0;
        for (ours, theirs) in [("ST", "VE"), ("AT", "QE"), ("ER", "IN"), ("SE", "OX")] {
            let board = cat_board();
            let r = solve_go_out(
                &board,
                &dict,
                &tiles(ours),
                &tiles(theirs),
                &GoOutSettings {
                    scoreless: 0,
                    max_depth: 30,
                    budget: None,
                },
                None,
            )
            .unwrap();
            assert!(r.solved, "{ours} v {theirs}");
            let mut b = board.clone();
            let want = minimax(&dict, &mut b, &tiles(ours), &tiles(theirs), 0);
            assert_eq!(r.lines[0].value, want, "{ours} v {theirs}");
            checked += 1;
        }
        assert_eq!(checked, 4);
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use crate::rules::SCRABBLE;
    use crate::tile::RACK_SIZE;

    /// Scrabble endgames reached by greedy self-play: the board when the
    /// bag empties, the side to move's rack, the other's.
    pub(super) fn endgames(dict: &Dictionary, n: usize) -> Vec<(Board, Vec<Tile>, Vec<Tile>)> {
        (0..100u64)
            .filter_map(|seed| {
                let mut rng = fastrand::Rng::with_seed(seed);
                let mut bag = SCRABBLE.full_tile_set();
                rng.shuffle(&mut bag);
                let mut racks = [
                    bag.split_off(bag.len() - RACK_SIZE),
                    bag.split_off(bag.len() - RACK_SIZE),
                ];
                let mut board = Board::with_rules(&SCRABBLE);
                let mut to_move = 0;
                while !bag.is_empty() {
                    let mv = Solver::new(&board, dict).best_move_by_equity(&racks[to_move])?;
                    for (r, c, t) in &mv.tiles_used {
                        board.place_tile(*r, *c, *t);
                    }
                    racks[to_move] = after_play(&racks[to_move], &mv);
                    let draw = mv.tiles_used.len().min(bag.len());
                    racks[to_move].extend(bag.split_off(bag.len() - draw));
                    to_move = 1 - to_move;
                }
                Some((board, racks[to_move].clone(), racks[1 - to_move].clone()))
            })
            .take(n)
            .collect()
    }

    /// Run with `cargo test --release -p ravyn negamax_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn negamax_timing() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let dict = dict.for_rules(&SCRABBLE);
        for (board, ours, theirs) in endgames(&dict, 8) {
            let t = Instant::now();
            let r = solve_go_out(
                &board,
                &dict,
                &ours,
                &theirs,
                &GoOutSettings {
                    budget: Some(Duration::from_secs(20)),
                    ..GoOutSettings::default()
                },
                None,
            )
            .unwrap();
            let rack = |t: &[Tile]| {
                t.iter()
                    .map(|t| if t.is_blank { '?' } else { t.letter })
                    .collect::<String>()
            };
            println!(
                "{} v {}: best {} {} = {:+} {} depth {} nodes {} in {:?}",
                rack(&ours),
                rack(&theirs),
                r.lines[0].mv.coords(),
                r.lines[0].mv.word,
                r.lines[0].value,
                if r.solved { "SOLVED" } else { "estimate" },
                r.depth,
                r.nodes,
                t.elapsed()
            );
        }
    }
}
