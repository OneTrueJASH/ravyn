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
//! The root's moves after the first are searched in parallel, sharing the
//! table (see `root`).
//!
//! [`Ending::GoOut`]: crate::rules::Ending::GoOut

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;
use std::time::{Duration, Instant};

use crate::board::{Board, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::rules::Ending;
use crate::solver::{BoardAnalysis, Move, MoveKind, Solver};
use crate::tile::{Tile, RACK_SIZE};

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
    /// The best move, tried first (before any other move is generated)
    /// when the position comes up again.
    best: Option<Stored>,
}

/// A move as the table keeps it: the tiles it places (none for a pass)
/// and its score, without the allocations of a [`Move`].
#[derive(Clone, Copy, PartialEq)]
struct Stored {
    n: u8,
    tiles: [(u8, u8, Tile); RACK_SIZE],
    score: u16,
}

impl Stored {
    fn of(mv: &Move) -> Self {
        let mut tiles = [(0, 0, Tile::new('A')); RACK_SIZE];
        for (slot, &(r, c, t)) in tiles.iter_mut().zip(&mv.tiles_used) {
            *slot = (r as u8, c as u8, t);
        }
        Stored {
            n: mv.tiles_used.len() as u8,
            tiles,
            score: mv.score as u16,
        }
    }

    fn is(&self, mv: &Move) -> bool {
        mv.tiles_used.len() == self.n as usize
            && mv
                .tiles_used
                .iter()
                .zip(&self.tiles)
                .all(|(&(r, c, t), &(sr, sc, st))| r == sr as usize && c == sc as usize && t == st)
    }

    /// The move again, for a side holding `mover`; `None` if it no longer
    /// fits the board (only on a hash collision).
    fn rebuild(&self, board: &Board, mover: &[Tile]) -> Option<Move> {
        if self.n == 0 {
            return Some(Move::pass(mover));
        }
        let tiles: Vec<(usize, usize, Tile)> = self.tiles[..self.n as usize]
            .iter()
            .map(|&(r, c, t)| (r as usize, c as usize, t))
            .collect();
        if tiles
            .iter()
            .any(|&(r, c, _)| board.get_tile(r, c).is_some())
        {
            return None;
        }
        let (r0, c0, _) = tiles[0];
        Some(Move {
            kind: MoveKind::Play,
            word: String::new(),
            row: r0,
            col: c0,
            direction: crate::board::Direction::Horizontal,
            score: self.score as u32,
            is_bingo: tiles.len() == RACK_SIZE,
            tiles_used: tiles,
            leave: Vec::new(),
            equity: self.score as f32,
        })
    }
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

/// Shards of the shared transposition table: enough that the search
/// threads seldom wait for one another.
const SHARDS: usize = 64;

/// The transposition table the search threads share, in locked shards
/// chosen by the key's top bits.
struct SharedTable {
    shards: Vec<Mutex<Table>>,
}

impl SharedTable {
    fn new() -> Self {
        SharedTable {
            shards: (0..SHARDS).map(|_| Mutex::new(Table::default())).collect(),
        }
    }

    fn shard(&self, key: u64) -> &Mutex<Table> {
        &self.shards[(key >> 58) as usize % SHARDS]
    }

    fn get(&self, key: u64) -> Option<Entry> {
        self.shard(key).lock().ok()?.get(&key).copied()
    }

    fn insert(&self, key: u64, entry: Entry) {
        if let Ok(mut shard) = self.shard(key).lock() {
            shard.insert(key, entry);
        }
    }
}

/// What the search threads share: the table, the limits, and the count of
/// positions searched.
struct Shared<'a> {
    dict: &'a Dictionary,
    table: SharedTable,
    deadline: Option<Instant>,
    cancel: Option<&'a AtomicBool>,
    /// Set by the first thread to run out of time, stopping the others.
    stop: AtomicBool,
    nodes: AtomicU64,
}

/// One thread's search, adding its position count to the shared one when
/// done.
struct Search<'a> {
    shared: &'a Shared<'a>,
    nodes: u64,
    out_of_time: bool,
}

impl<'a> Search<'a> {
    fn new(shared: &'a Shared<'a>) -> Self {
        Search {
            shared,
            nodes: 0,
            out_of_time: false,
        }
    }
}

impl Drop for Search<'_> {
    fn drop(&mut self) {
        self.shared.nodes.fetch_add(self.nodes, Ordering::Relaxed);
    }
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
    let shared = Shared {
        dict,
        table: SharedTable::new(),
        deadline: budget.map(|b| Instant::now() + b),
        cancel,
        stop: AtomicBool::new(false),
        nodes: AtomicU64::new(0),
    };
    let mut best: Option<GoOutResult> = None;
    for depth in 1..=max_depth.max(1) {
        // Each depth searches the root moves in the order the previous one
        // ranked them, so the best line is found early and cuts the rest.
        let order = best
            .as_ref()
            .map(|b| b.lines.iter().map(|l| l.mv.clone()).collect());
        let Some((lines, horizon)) = root(&shared, board, ours, theirs, scoreless, depth, order)
        else {
            break;
        };
        let solved = !horizon;
        best = Some(GoOutResult {
            lines,
            solved,
            depth,
            nodes: shared.nodes.load(Ordering::Relaxed),
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

/// Value and line of each root move at `depth`, best first, and whether
/// any line reached the depth limit. `None` if time ran out.
///
/// The first move (the best of the previous depth) is searched alone, to
/// set the bar; the rest are searched in parallel, each with the best value
/// known when it starts (the window (`alpha`, `+inf`)). A move that comes
/// back above its `alpha` is exact; one that does not was only proved no
/// better, whatever the other threads found meanwhile. A solved result is
/// therefore the same however the threads run; an estimate (a depth that
/// reached its limit) can differ between runs, as it depends on which
/// table entries other threads stored first.
fn root(
    shared: &Shared,
    board: &Board,
    ours: &[Tile],
    theirs: &[Tile],
    scoreless: usize,
    depth: usize,
    order: Option<Vec<Move>>,
) -> Option<(Vec<EndgameLine>, bool)> {
    let analysis = Solver::new(board, shared.dict).analysis();
    let moves =
        order.unwrap_or_else(|| Search::new(shared).ordered_moves(board, &analysis, ours, theirs));
    let value_of = |mv: &Move, alpha: i32| {
        let mut search = Search::new(shared);
        let mut board = board.clone();
        search.child_value(
            &analysis,
            &mut board,
            ours,
            theirs,
            scoreless,
            mv,
            depth,
            alpha,
            i32::MAX,
        )
    };
    let (first, rest) = moves.split_first()?;
    let (v0, h0) = value_of(first, i32::MIN + 1)?;
    let alpha = AtomicI32::new(v0);
    let others = rest
        .par_iter()
        .map(|mv| {
            let a = alpha.load(Ordering::Relaxed);
            let (value, h) = value_of(mv, a)?;
            let exact = value > a;
            if exact {
                alpha.fetch_max(value, Ordering::Relaxed);
            }
            Some((value, h, exact))
        })
        .collect::<Option<Vec<_>>>()?;
    // A cut-off move's bound is only proof if it did not rest on the depth
    // limit either, so any horizon makes the depth inexact.
    let mut horizon = h0;
    let mut lines = vec![EndgameLine {
        mv: first.clone(),
        value: v0,
        exact: true,
    }];
    for (mv, (value, h, exact)) in rest.iter().zip(others) {
        horizon |= h;
        lines.push(EndgameLine {
            mv: mv.clone(),
            value,
            exact,
        });
    }
    // Stable: equal values keep the move order, whatever the threads did.
    lines.sort_by(|a, b| b.value.cmp(&a.value).then(b.exact.cmp(&a.exact)));
    Some((lines, horizon))
}

impl Search<'_> {
    fn stopped(&mut self) -> bool {
        if self.nodes.is_multiple_of(256) {
            let shared = self.shared;
            if shared.stop.load(Ordering::Relaxed)
                || shared.deadline.is_some_and(|d| Instant::now() >= d)
                || shared.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
            {
                shared.stop.store(true, Ordering::Relaxed);
                self.out_of_time = true;
            }
        }
        self.out_of_time
    }

    /// The side to move's moves, most promising first: plays that go out
    /// (by score plus the bonus), then by score, then passing.
    fn ordered_moves(
        &self,
        board: &Board,
        analysis: &BoardAnalysis,
        mover: &[Tile],
        other: &[Tile],
    ) -> Vec<Move> {
        let mut moves =
            Solver::with_analysis(board, self.shared.dict, analysis).find_moves_for(mover);
        let bonus = 2 * rack_value(board, other);
        moves.sort_by_cached_key(|m| {
            let out = m.tiles_used.len() == mover.len();
            std::cmp::Reverse(m.score as i32 + if out { bonus } else { 0 })
        });
        moves.push(Move::pass(mover));
        moves
    }

    /// The value of playing `mv` for the side holding `mover`, searched
    /// `depth` plies deep with the window (`alpha`, `beta`), and whether it
    /// rests on the depth limit. Leaves `board` as it found it.
    #[allow(clippy::too_many_arguments)]
    fn child_value(
        &mut self,
        analysis: &BoardAnalysis,
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
        let placed: Vec<(usize, usize)> = mv.tiles_used.iter().map(|&(r, c, _)| (r, c)).collect();
        // The child's analysis is made only if it searches its moves (not
        // at the depth limit, nor when the table answers).
        let reply = self.negamax(
            analysis,
            &placed,
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

    /// The value of a position one ply from the depth limit, and whether it
    /// rests on the limit. Each move leads to a position valued as the
    /// game ending there (each side losing its rack), so passing is worth
    /// `other - mover` (rack values) and a play `other + score - kept`,
    /// plus `other` again if it goes out: [`Solver::best_endgame_play`]
    /// finds the best play directly instead of searching every child.
    #[allow(clippy::too_many_arguments)]
    fn frontier(
        &mut self,
        analysis: &BoardAnalysis,
        board: &Board,
        mover: &[Tile],
        other: &[Tile],
        key: u64,
        beta: i32,
    ) -> (i32, bool) {
        let theirs = rack_value(board, other);
        let pass = theirs - rack_value(board, mover);
        let play = Solver::with_analysis(board, self.shared.dict, analysis)
            .best_endgame_play(mover, theirs as u32)
            .map(|m| {
                (
                    theirs + m.equity.round() as i32,
                    m.tiles_used.len() == mover.len(),
                )
            });
        let (value, out) = match play {
            Some((v, out)) if v > pass => (v, out),
            _ => (pass, false),
        };
        // Going out ends the game: no horizon. If that alone reaches beta
        // it proves the cut-off; otherwise the value is the best of
        // estimates at the limit.
        let proven = out && value >= beta;
        self.shared.table.insert(
            key,
            Entry {
                depth: 1,
                value,
                bound: if proven { Bound::Lower } else { Bound::Exact },
                horizon: !proven,
                best: None,
            },
        );
        (value, !proven)
    }

    /// The value for the side to move (holding `mover`), within the window
    /// (`alpha`, `beta`), and whether it rests on the depth limit. The
    /// board's analysis is `parent`'s with tiles just placed on `placed`.
    #[allow(clippy::too_many_arguments)] // the search state, recursively
    fn negamax(
        &mut self,
        parent: &BoardAnalysis,
        placed: &[(usize, usize)],
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
        if let Some(e) = self.shared.table.get(key) {
            if e.depth >= depth || !e.horizon {
                match e.bound {
                    Bound::Exact => return Some((e.value, e.horizon)),
                    Bound::Lower if e.value >= beta => return Some((e.value, e.horizon)),
                    Bound::Upper if e.value <= alpha => return Some((e.value, e.horizon)),
                    _ => {}
                }
            }
        }
        let owned;
        let analysis = if placed.is_empty() {
            parent
        } else {
            owned = parent.after_placing(board, self.shared.dict, placed);
            &owned
        };
        if depth == 1 {
            // Every reply ends at the depth limit, valued as each side
            // losing its rack: the best is one bounded search away.
            return Some(self.frontier(analysis, board, mover, other, key, beta));
        }
        let alpha_in = alpha;
        let mut best = i32::MIN + 1;
        let mut best_move: Option<Stored> = None;
        let mut horizon = false;
        // The best move from an earlier visit first, before generating any
        // other: if it cuts off, the rest are never generated.
        let remembered = self
            .shared
            .table
            .get(key)
            .and_then(|e| e.best)
            .and_then(|st| Some((st, st.rebuild(board, mover)?)));
        let mut tried = 0usize;
        if let Some((st, mv)) = &remembered {
            let (value, h) = self.child_value(
                analysis, board, mover, other, scoreless, mv, depth, alpha, beta,
            )?;
            tried = 1;
            best = value;
            best_move = Some(*st);
            horizon |= h;
            alpha = alpha.max(value);
        }
        if alpha < beta {
            let moves = self.ordered_moves(board, analysis, mover, other);
            for mv in &moves {
                if remembered.as_ref().is_some_and(|(st, _)| st.is(mv)) {
                    continue;
                }
                // Principal variation search: the first move gets the full
                // window; the rest are first asked only whether they beat
                // it (a null window), and searched fully only if they do.
                let (value, h) = if tried == 0 {
                    self.child_value(
                        analysis, board, mover, other, scoreless, mv, depth, alpha, beta,
                    )?
                } else {
                    let probe = self.child_value(
                        analysis,
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
                        self.child_value(
                            analysis, board, mover, other, scoreless, mv, depth, alpha, beta,
                        )?
                    } else {
                        probe
                    }
                };
                tried += 1;
                if value > best {
                    best = value;
                    best_move = Some(Stored::of(mv));
                }
                horizon |= h;
                if value > alpha {
                    alpha = value;
                }
                if alpha >= beta {
                    break;
                }
            }
        }
        let bound = if best <= alpha_in {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.shared.table.insert(
            key,
            Entry {
                depth,
                value: best,
                bound,
                horizon,
                best: best_move,
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
