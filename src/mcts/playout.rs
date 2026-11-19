//! One simulated game under Crossplay rules: move policies, the exact
//! two-ply ending, and the cut-off valuation.

use fastrand::Rng;

use crate::board::{Board, Direction, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::game::{best_exchange, top_exchanges, SolverState};
use crate::rules::Ending;
use crate::solver::{Move, MoveKind, Solver, EXCHANGE_MIN_BAG};
#[cfg(test)]
use crate::tile::Rack;
use crate::tile::{Tile, RACK_SIZE};

use super::eval::{MctsConfig, MctsEvaluator};

#[cfg(test)]
pub(super) fn rack_from(tiles: &[Tile]) -> Rack {
    let mut rack = Rack::new();
    for t in tiles {
        rack.add(*t);
    }
    rack
}

/// Moves the softmax opponent chooses among (the best by equity, within a
/// window of the best).
pub(super) const SOFTMAX_MOVES: usize = 16;

/// Which player is to move in a playout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Player {
    Us,
    Opp,
}

impl Player {
    pub(super) fn other(self) -> Player {
        match self {
            Player::Us => Player::Opp,
            Player::Opp => Player::Us,
        }
    }
}

/// One point of a playout's history: after `mover`'s move, the spread from
/// our point of view and the tiles left in the bag.
#[derive(Debug, Clone, Copy)]
pub struct TracePoint {
    pub mover_is_us: bool,
    pub spread: i32,
    pub bag: usize,
    pub was_bingo: bool,
    /// Phase-scaled leave value of our rack minus the opponent's, as the
    /// playout's cut-off valuation would add it.
    pub leave_adj: f32,
}

/// One simulated game from the position after our candidate move.
pub(super) struct Playout<'a> {
    pub(super) dict: &'a Dictionary,
    pub(super) board: Board,
    pub(super) racks: [Vec<Tile>; 2],
    pub(super) bag: Vec<Tile>,
    /// Signed spread from our point of view.
    pub(super) spread: i32,
    /// Player who drew the last tile, once the bag is empty (Crossplay's
    /// equal-turn ending only).
    pub(super) final_turn_owner: Option<Player>,
    /// Turns in a row that scored nothing (passes, exchanges, zero-point
    /// plays), for the go-out ending's six-scoreless-turns rule.
    pub(super) scoreless: usize,
    /// For returning exchanged tiles to random places in the bag.
    pub(super) rng: Rng,
    /// Whether each player's first move in this playout was a bingo.
    pub(super) first_move_bingo: [Option<bool>; 2],
    /// Softmax spread of the opponent's greedy-phase move choice, in
    /// equity points; 0 (the default) plays the best move every turn.
    pub(super) opp_temperature: f32,
    /// Move-by-move history, kept only when requested (self-play studies).
    pub(super) trace: Option<Vec<TracePoint>>,
    /// Dictionary (and so leave valuation) the opponent chooses moves
    /// with; the same as `dict` except in engine-versus-engine matches.
    pub(super) opp_dict: &'a Dictionary,
    /// Whether each player may pass in the greedy phase when exchanging is
    /// not allowed and keeping the rack beats every play.
    pub(super) allow_pass: [bool; 2],
    /// The previous move was a pass (two passes in a row are not allowed
    /// in the greedy phase, so playouts cannot stall).
    pub(super) last_was_pass: bool,
    /// Players that choose their greedy-phase moves by simulation instead
    /// of static equity (self-play matches only).
    pub(super) sim_choice: [Option<SimChoice>; 2],
}

/// Choosing a move by simulation in a self-play match: the top
/// `candidates` by static equity, `playouts` each.
#[derive(Debug, Clone, Copy)]
pub struct SimChoice {
    pub candidates: usize,
    pub playouts: usize,
}

/// Whether playout players consider passing when the bag is too small to
/// exchange (4.4 in the polish plan; validated by self-play).
pub(crate) const PASS_IN_PLAYOUTS: bool = true;

impl<'a> Playout<'a> {
    pub(super) fn new(
        dict: &'a Dictionary,
        board: Board,
        racks: [Vec<Tile>; 2],
        bag: Vec<Tile>,
        seed: u64,
    ) -> Self {
        Playout {
            dict,
            board,
            racks,
            bag,
            spread: 0,
            final_turn_owner: None,
            scoreless: 0,
            rng: Rng::with_seed(seed),
            first_move_bingo: [None, None],
            opp_temperature: 0.0,
            trace: None,
            opp_dict: dict,
            allow_pass: [PASS_IN_PLAYOUTS; 2],
            last_was_pass: false,
            sim_choice: [None, None],
        }
    }

    /// `p`'s move chosen by simulating the top candidates from its own
    /// point of view (the other rack unknown, dealt from the unseen tiles).
    pub(super) fn simulated_move(&self, p: Player, spec: SimChoice) -> Option<Move> {
        let dict = self.dict_for(p);
        let mut rack = crate::tile::Rack::new();
        for t in self.rack(p) {
            rack.add(*t);
        }
        let state = SolverState {
            board: self.board.clone(),
            rack,
        };
        let mut candidates = state.get_suggestions(dict);
        if candidates.len() <= 1 {
            return candidates.pop();
        }
        candidates.truncate(spec.candidates);
        let config = MctsConfig {
            num_candidates: spec.candidates,
            iterations_per_candidate: spec.playouts,
            max_plies: MctsConfig::for_bag(self.bag.len()).max_plies,
            parallel: true,
            score_diff: 0,
            time_budget: None,
            cancel: None,
        };
        MctsEvaluator::new(config)
            .evaluate(&state, &candidates, dict)
            .into_iter()
            .next()
            .map(|r| r.mv)
    }

    /// The dictionary `p` chooses moves with.
    pub(super) fn dict_for(&self, p: Player) -> &'a Dictionary {
        match p {
            Player::Us => self.dict,
            Player::Opp => self.opp_dict,
        }
    }

    /// Passing as a candidate for `p` in the greedy phase: only when the
    /// bag is too small to exchange (otherwise an exchange keeps the best
    /// tiles anyway), valued like any move at 0 points plus the leave of
    /// the whole rack.
    pub(super) fn pass_candidate(&self, p: Player) -> Option<Move> {
        if !self.allow_pass[p as usize]
            || self.last_was_pass
            || self.bag.is_empty()
            || self.bag.len() >= EXCHANGE_MIN_BAG
        {
            return None;
        }
        let rack = self.rack(p);
        let mut pass = Move::pass(rack);
        pass.equity = self
            .dict_for(p)
            .leave_value(rack, Some(self.unseen_for_phase()));
        Some(pass)
    }

    pub(super) fn rack(&self, p: Player) -> &[Tile] {
        &self.racks[p as usize]
    }

    pub(super) fn rack_mut(&mut self, p: Player) -> &mut Vec<Tile> {
        &mut self.racks[p as usize]
    }

    pub(super) fn unseen_for_phase(&self) -> usize {
        // From the mover's perspective the unseen count is bag + other rack,
        // which is the same for both players when both racks are full; use
        // bag + 7 as the phase indicator, floored at the real count.
        self.bag.len() + RACK_SIZE
    }

    /// Make `mv` for `p`: place or exchange its tiles, score it, draw, and
    /// for an exchange return the exchanged tiles to the bag afterwards (so
    /// the bag size is unchanged and an exchange can never empty it).
    pub(super) fn commit(&mut self, p: Player, mv: &Move) {
        for (r, c, tile) in &mv.tiles_used {
            self.board.place_tile(*r, *c, *tile);
        }
        for tile in mv.tiles_from_rack() {
            let rack = self.rack_mut(p);
            let idx = rack
                .iter()
                .position(|t| {
                    t.is_blank == tile.is_blank && (t.is_blank || t.letter == tile.letter)
                })
                .expect("move tiles come from the mover's rack");
            rack.swap_remove(idx);
        }
        self.last_was_pass = mv.kind == MoveKind::Pass;
        self.scoreless = if mv.score == 0 { self.scoreless + 1 } else { 0 };
        let signed = mv.score as i32;
        self.spread += if p == Player::Us { signed } else { -signed };
        if self.first_move_bingo[p as usize].is_none() {
            self.first_move_bingo[p as usize] = Some(mv.is_bingo);
        }

        let need = RACK_SIZE
            .saturating_sub(self.rack(p).len())
            .min(self.bag.len());
        if need > 0 {
            let drawn: Vec<Tile> = self.bag.drain(self.bag.len() - need..).collect();
            self.rack_mut(p).extend(drawn);
        }
        if let MoveKind::Exchange(returned) = &mv.kind {
            for &tile in returned {
                let at = self.rng.usize(..=self.bag.len());
                self.bag.insert(at, tile);
            }
        }
        // Only a play can draw the last tile: an exchange puts back as many
        // as it took, so the bag is checked after the return.
        if need > 0 && self.bag.is_empty() && self.board.rules().ending == Ending::EqualTurn {
            self.final_turn_owner = Some(p);
        }
        if self.trace.is_some() {
            let leave_adj = self.leave_adjustment();
            if let Some(trace) = &mut self.trace {
                trace.push(TracePoint {
                    mover_is_us: p == Player::Us,
                    spread: self.spread,
                    bag: self.bag.len(),
                    was_bingo: mv.is_bingo,
                    leave_adj,
                });
            }
        }
    }

    /// Our rack's leave value minus the opponent's, phase-scaled.
    /// The go-out ending, from `to_move`'s turn with the bag empty: both
    /// sides take their highest-scoring play (or pass) until one goes out,
    /// gaining twice the other's rack value in spread, or six turns in a
    /// row score nothing and each loses their own rack's value. Leave
    /// values no longer matter once nothing can be drawn, so moves are
    /// chosen by score; the exact endgame search is a separate solver.
    fn play_to_the_end(&mut self, mut to_move: Player) -> PlayoutOutcome {
        let rules = self.board.rules();
        let value = |rack: &[Tile]| rack.iter().map(|t| rules.tile_value(t) as i32).sum::<i32>();
        let sign = |p: Player| if p == Player::Us { 1 } else { -1 };
        loop {
            let last = to_move.other();
            if self.rack(last).is_empty() {
                self.spread += sign(last) * 2 * value(self.rack(to_move));
                break;
            }
            if self.scoreless >= Ending::SCORELESS_TURNS_TO_END {
                self.spread -= value(self.rack(Player::Us));
                self.spread += value(self.rack(Player::Opp));
                break;
            }
            let mv = self
                .best_by_score(to_move)
                .unwrap_or_else(|| Move::pass(self.rack(to_move)));
            self.commit(to_move, &mv);
            to_move = last;
        }
        PlayoutOutcome {
            value: self.spread as f32,
            finished: true,
            bag: 0,
        }
    }

    pub(super) fn leave_adjustment(&self) -> f32 {
        let unseen = self.unseen_for_phase();
        self.dict.leave_value(self.rack(Player::Us), Some(unseen))
            - self.dict.leave_value(self.rack(Player::Opp), Some(unseen))
    }

    /// Highest-scoring move for `p` (None means pass).
    pub(super) fn best_by_score(&self, p: Player) -> Option<Move> {
        Solver::new(&self.board, self.dict_for(p)).best_move_by_score(self.rack(p))
    }

    /// Best move by static equity for `p`, including exchanges when the bag
    /// allows them (None means pass).
    pub(super) fn best_by_equity(&self, p: Player) -> Option<Move> {
        let dict = self.dict_for(p);
        let solver = Solver::new(&self.board, dict);
        let best_play = solver.best_move_by_equity(self.rack(p));
        if self.bag.len() < EXCHANGE_MIN_BAG {
            return match (best_play, self.pass_candidate(p)) {
                (Some(play), Some(pass)) if pass.equity > play.equity => Some(pass),
                (play, _) => play,
            };
        }
        let unseen = self.unseen_for_phase();
        let best_exchange = best_exchange(self.rack(p), dict, Some(unseen));
        match (best_play, best_exchange) {
            (Some(play), Some(exch)) => Some(if exch.equity > play.equity {
                exch
            } else {
                play
            }),
            (play, exch) => play.or(exch),
        }
    }

    /// Move choice in the greedy phase: exact best for us, and for the
    /// opponent a softmax sample over near-best moves when the model says
    /// they err (`opp_temperature` > 0). The final two turns after the bag
    /// empties stay exact for both players regardless.
    pub(super) fn policy_move(&mut self, p: Player) -> Option<Move> {
        if let Some(spec) = self.sim_choice[p as usize] {
            if !self.bag.is_empty() {
                return self.simulated_move(p, spec);
            }
        }
        if p != Player::Opp || self.opp_temperature <= 0.0 {
            return self.best_by_equity(p);
        }
        let tau = self.opp_temperature;
        // Only the best SOFTMAX_MOVES of plays and of exchanges can make the
        // cut below, so only those are built.
        let mut moves = {
            let dict = self.dict_for(p);
            let solver = Solver::new(&self.board, dict);
            let mut moves = solver.top_moves_by_equity(self.rack(p), SOFTMAX_MOVES);
            if self.bag.len() >= EXCHANGE_MIN_BAG {
                let unseen = self.unseen_for_phase();
                moves.extend(top_exchanges(
                    self.rack(p),
                    dict,
                    Some(unseen),
                    SOFTMAX_MOVES,
                ));
            } else if let Some(pass) = self.pass_candidate(p) {
                moves.push(pass);
            }
            moves.sort_by(|a, b| {
                b.equity
                    .partial_cmp(&a.equity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            moves
        };
        let best_equity = moves.first()?.equity;
        // Errors far below the best are not plausible even for a weak
        // player; without the window the long tail of junk moves would
        // dominate the sample by sheer count.
        moves.retain(|m| m.equity >= best_equity - 3.0 * tau);
        moves.truncate(SOFTMAX_MOVES);
        let weights: Vec<f32> = moves
            .iter()
            .map(|m| ((m.equity - best_equity) / tau).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        let mut roll = self.rng.f32() * total;
        for (i, w) in weights.iter().enumerate() {
            roll -= w;
            if roll <= 0.0 {
                return Some(moves.swap_remove(i));
            }
        }
        moves.pop()
    }

    /// `p`'s last turn with `p.other()` holding the final move: choose the
    /// play maximising score minus the other's best reply, or pass.
    pub(super) fn best_last_turn(&self, p: Player) -> (Option<Move>, i32) {
        let solver = Solver::new(&self.board, self.dict_for(p));
        let other_rack = self.rack(p.other());

        // Only the reply's score matters, so no reply moves are built.
        let reply_score =
            |board: &Board| -> i32 { Solver::new(board, self.dict).best_score(other_rack) as i32 };

        // The other player's best replies on the current board. One still
        // legal after our move scores the same there, so it bounds their
        // reply from below: value(mv) <= mv.score - that score.
        let mut base_replies = Solver::new(&self.board, self.dict).find_moves_for(other_rack);
        base_replies.sort_by_key(|m| std::cmp::Reverse(m.score));
        base_replies.truncate(REPLY_BOUND_CANDIDATES);

        // Passing: the other player replies on the unchanged board.
        let mut best_value = -(base_replies.first().map_or(0, |m| m.score as i32));
        let mut best_move = None;

        let mut moves: Vec<(i32, Move)> = solver
            .find_moves_for(self.rack(p))
            .into_iter()
            .map(|mv| {
                let floor = base_replies
                    .iter()
                    .find(|r| reply_survives(&self.board, r, &mv.tiles_used))
                    .map_or(0, |r| r.score as i32);
                (mv.score as i32 - floor, mv)
            })
            .collect();
        // Highest bound first; ties by score keep the old order.
        moves.sort_by_key(|(bound, mv)| (std::cmp::Reverse(*bound), std::cmp::Reverse(mv.score)));

        for (bound, mv) in moves {
            // Nothing from here on can beat the best value found.
            if bound <= best_value {
                break;
            }
            let mut board = self.board.clone();
            for (r, c, tile) in &mv.tiles_used {
                board.place_tile(*r, *c, *tile);
            }
            let value = mv.score as i32 - reply_score(&board);
            if value > best_value {
                best_value = value;
                best_move = Some(mv);
            }
        }
        (best_move, best_value)
    }

    /// A fresh game: empty board, shuffled full bag, both racks drawn.
    pub(super) fn new_game(dict: &'a Dictionary, seed: u64) -> Self {
        let mut rng = Rng::with_seed(seed);
        let board = Board::new();
        let mut bag = board.rules().full_tile_set();
        rng.shuffle(&mut bag);
        let us: Vec<Tile> = bag.drain(bag.len() - RACK_SIZE..).collect();
        let opp: Vec<Tile> = bag.drain(bag.len() - RACK_SIZE..).collect();
        Self::new(dict, board, [us, opp], bag, rng.u64(..))
    }

    /// Play the game out from `to_move`'s turn. Returns the final spread
    /// (plus leave and tempo adjustments if the ply limit cut the game short).
    pub(super) fn run(&mut self, to_move: Player, max_plies: usize) -> f32 {
        self.run_with(to_move, max_plies, &mut |_, _| {}).value
    }

    /// As [`Self::run`], calling `observe(self, to_move)` before every move
    /// of the greedy phase (self-play studies sample positions this way).
    pub(super) fn run_with(
        &mut self,
        mut to_move: Player,
        max_plies: usize,
        observe: &mut dyn FnMut(&Playout<'a>, Player),
    ) -> PlayoutOutcome {
        let finished = |p: &Playout| PlayoutOutcome {
            value: p.spread as f32,
            finished: true,
            bag: p.bag.len(),
        };
        let mut plies = 0;
        loop {
            if self.board.rules().ending == Ending::GoOut && self.bag.is_empty() {
                return self.play_to_the_end(to_move);
            }
            match self.final_turn_owner {
                Some(owner) if owner == to_move => {
                    // Final move of the game: take the points.
                    if let Some(mv) = self.best_by_score(to_move) {
                        self.commit(to_move, &mv);
                    }
                    return finished(self);
                }
                Some(_) => {
                    // Last turn before the owner's final move: exact 2-ply.
                    let (mv, _) = self.best_last_turn(to_move);
                    if let Some(mv) = mv {
                        self.commit(to_move, &mv);
                    }
                    to_move = to_move.other();
                    continue;
                }
                None => {}
            }

            if plies >= max_plies {
                // Cut short: value the racks statically, plus the tempo of
                // whoever is to move.
                let tempo = if to_move == Player::Us { TEMPO } else { -TEMPO };
                return PlayoutOutcome {
                    value: self.spread as f32 + self.leave_adjustment() + tempo,
                    finished: false,
                    bag: self.bag.len(),
                };
            }

            observe(self, to_move);
            if let Some(mv) = self.policy_move(to_move) {
                self.commit(to_move, &mv);
            } else if self.bag.is_empty() {
                // No bag, no move: treat as the game ending here.
                return finished(self);
            } else if self.trace.is_some() {
                let leave_adj = self.leave_adjustment();
                if let Some(trace) = &mut self.trace {
                    trace.push(TracePoint {
                        mover_is_us: to_move == Player::Us,
                        spread: self.spread,
                        bag: self.bag.len(),
                        was_bingo: false,
                        leave_adj,
                    });
                }
            }
            plies += 1;
            to_move = to_move.other();
        }
    }
}

/// Worth of being the player to move when a playout is cut off mid-game,
/// in points of final spread. Measured by self-play (`study_spread_variance`,
/// 1000 games, 2026-08-20, with racks valued through the leave table): with
/// the bag holding 8+ tiles, the final spread minus (spread so far + leave
/// values) averaged +18.5 / +22.7 with us to move and -22.5 / -18.7 with the
/// opponent to move, whichever side had started the game (earlier runs of
/// 400-800 games gave 17.6-23.2). The same offset applies to every
/// candidate, so it changes win estimates, not rankings.
pub(super) const TEMPO: f32 = 21.0;

/// Standard deviation of the spread still to come from a mid-game
/// position, by tiles left in the bag. Same study: the variance of the
/// residual grows linearly with the bag, slope 79.4-79.8 per tile across
/// runs and an intercept between 92 and 213 (few samples at small bags,
/// where playouts usually finish exactly anyway): var = 150 + 79.8 * bag
/// (sd 12 with an empty bag, 64 with 50 tiles, 84 at the start).
pub(super) fn remaining_spread_sd(bag: usize) -> f32 {
    (150.0 + 79.8 * bag as f32).sqrt()
}

/// Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
pub(super) fn normal_cdf(x: f32) -> f32 {
    let z = (x as f64) / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    let erf = if z >= 0.0 { erf } else { -erf };
    (0.5 * (1.0 + erf)) as f32
}

/// What one playout reports back to its candidate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlayoutRecord {
    pub(crate) outcome: PlayoutOutcome,
    pub(crate) opp_bingo: bool,
    pub(crate) our_bingo: bool,
}

/// A playout's contribution to the win rate: 1, 0 or 1/2 for a finished
/// game (counting the score difference before the move), and for a game
/// cut off mid-way the probability that the remaining spread swing leaves
/// us ahead.
pub(crate) fn win_value(score_diff: i32, outcome: &PlayoutOutcome) -> f32 {
    let final_diff = score_diff as f32 + outcome.value;
    if outcome.finished {
        match final_diff.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => 1.0,
            Some(std::cmp::Ordering::Less) => 0.0,
            _ => 0.5,
        }
    } else {
        normal_cdf(final_diff / remaining_spread_sd(outcome.bag))
    }
}

/// Result of one playout.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlayoutOutcome {
    /// Spread change from our point of view (exact if `finished`, otherwise
    /// spread so far plus leave values plus the tempo of the side to move).
    pub(crate) value: f32,
    /// The game reached its end inside the ply limit.
    pub(crate) finished: bool,
    /// Tiles left in the bag where the playout stopped.
    pub(crate) bag: usize,
}

/// How many of the other player's best current-board replies
/// `best_last_turn` tries as a lower bound on their reply after our move.
const REPLY_BOUND_CANDIDATES: usize = 12;

/// Whether `reply`, legal on `board`, stays legal with the same score once
/// `ours` is placed: none of our tiles lands on its squares or just past
/// either end of its main word or of any cross-word it forms. Every other
/// square it reads is unchanged.
fn reply_survives(board: &Board, reply: &Move, ours: &[(usize, usize, Tile)]) -> bool {
    let n = BOARD_SIZE as i32;
    let is_ours = |r: i32, c: i32| {
        ours.iter()
            .any(|&(or, oc, _)| or as i32 == r && oc as i32 == c)
    };
    let placed = |r: i32, c: i32| {
        reply
            .tiles_used
            .iter()
            .any(|&(pr, pc, _)| pr as i32 == r && pc as i32 == c)
    };
    let filled = |r: i32, c: i32| {
        (0..n).contains(&r)
            && (0..n).contains(&c)
            && (!board.is_empty(r as usize, c as usize) || placed(r, c))
    };
    // The squares just past both ends of the run through (r, c) along (dr, dc).
    let run_ends_clear = |r: i32, c: i32, dr: i32, dc: i32| {
        let (mut a, mut b) = ((r, c), (r, c));
        while filled(a.0 - dr, a.1 - dc) {
            a = (a.0 - dr, a.1 - dc);
        }
        while filled(b.0 + dr, b.1 + dc) {
            b = (b.0 + dr, b.1 + dc);
        }
        !is_ours(a.0 - dr, a.1 - dc) && !is_ours(b.0 + dr, b.1 + dc)
    };
    let (dr, dc) = match reply.direction {
        Direction::Horizontal => (0, 1),
        Direction::Vertical => (1, 0),
    };
    let Some(&(r0, c0, _)) = reply.tiles_used.first() else {
        return true;
    };
    if reply
        .tiles_used
        .iter()
        .any(|&(r, c, _)| is_ours(r as i32, c as i32))
    {
        return false;
    }
    run_ends_clear(r0 as i32, c0 as i32, dr, dc)
        && reply
            .tiles_used
            .iter()
            .all(|&(r, c, _)| run_ends_clear(r as i32, c as i32, dc, dr))
}
