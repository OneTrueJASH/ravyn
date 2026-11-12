//! The fixed-budget evaluator: candidates played out in batches, with
//! paired-difference pruning between rounds.

use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastrand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::board::Board;
use crate::dictionary::Dictionary;
use crate::game::SolverState;
use crate::solver::{Move, MoveKind};
use crate::tile::{Tile, RACK_SIZE};

use super::opponent::OpponentModel;
use super::playout::{win_value, Player, Playout, PlayoutOutcome, PlayoutRecord};
use super::pool::{iteration_seed, position_fingerprint, TilePool, PLAYOUT_SEED};

/// Configuration for Monte Carlo evaluation
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Number of top candidates (by static equity) to evaluate
    pub num_candidates: usize,
    /// Playouts per candidate
    pub iterations_per_candidate: usize,
    /// Maximum plies per playout after the candidate move. The final two
    /// turns after the bag empties always run to completion; this bounds
    /// the greedy phase before that.
    pub max_plies: usize,
    /// Use parallel evaluation with Rayon
    pub parallel: bool,
    /// Current score difference (ours minus the opponent's) before the
    /// candidate move; playout win rates are measured against it.
    pub score_diff: i32,
    /// Wall-clock budget for the whole evaluation. Playouts stop when it
    /// runs out and every candidate reports the estimate it has; move
    /// generation with several unseen blanks can make playouts slow.
    pub time_budget: Option<Duration>,
    /// Set from another thread to stop the evaluation early (checked before
    /// every playout); the results then hold whatever was simulated.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            num_candidates: 20,
            iterations_per_candidate: 200,
            max_plies: 2,
            parallel: true,
            score_diff: 0,
            time_budget: Some(Duration::from_secs(8)),
            cancel: None,
        }
    }
}

impl MctsConfig {
    /// Settings for the pre-endgame (bag of 1–7 tiles): the game ends
    /// within a few plies, so play every playout to the end.
    pub fn pre_endgame() -> Self {
        MctsConfig {
            num_candidates: 25,
            iterations_per_candidate: 120,
            max_plies: 10,
            parallel: true,
            score_diff: 0,
            time_budget: Some(Duration::from_secs(8)),
            cancel: None,
        }
    }

    /// Settings by bag size. The ending is always played exactly whenever
    /// a playout reaches it; the depth decides how often it does:
    /// - 1-7 in the bag: to the end of the game (see `pre_endgame`)
    /// - 8-21: four plies, enough for either side to empty the bag
    /// - more: two plies (our move, the reply, our follow-up) plus leaves
    pub fn for_bag(bag: usize) -> Self {
        if bag <= RACK_SIZE {
            Self::pre_endgame()
        } else if bag <= 3 * RACK_SIZE {
            MctsConfig {
                num_candidates: 25,
                iterations_per_candidate: 150,
                max_plies: 4,
                ..Self::default()
            }
        } else {
            Self::default()
        }
    }
    pub fn with_score_diff(mut self, score_diff: i32) -> Self {
        self.score_diff = score_diff;
        self
    }

    /// Stop early once `cancel` is set.
    pub fn with_cancel(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|c| c.load(AtomicOrdering::Relaxed))
    }
}

/// Result of Monte Carlo evaluation for a single move
#[derive(Debug, Clone)]
pub struct MctsResult {
    /// The evaluated move
    pub mv: Move,
    /// Expected spread change from playing this move (our points minus the
    /// opponent's over the playout, plus leave values if it was cut short)
    pub mcts_equity: f32,
    /// Fraction of playouts we finished ahead in, counting the score
    /// difference before the move (ties count half). With the ply limit
    /// reached this is judged on spread plus leave values.
    pub win_rate: f32,
    /// Playouts completed
    pub iterations: usize,
    /// Share of playouts in which the opponent's reply was a bingo
    pub opp_bingo_rate: f32,
    /// Share of playouts in which our follow-up move was a bingo
    pub our_bingo_rate: f32,
}

/// Per-candidate simulation state, kept across pruning rounds.
pub(crate) struct CandidateSim<'a> {
    pub(super) mv: &'a Move,
    pub(super) board: Board,
    pub(super) our_leave: Vec<Tile>,
    pub(super) pool: TilePool,
    pub(super) bag_size: usize,
    pub(super) draw_count: usize,
    pub(super) score_diff: i32,
    pub(super) model: Option<&'a OpponentModel>,
    /// Base seed of this evaluation's playouts (see `run_batch`).
    pub(super) seed: u64,
    pub(super) total: f32,
    pub(super) wins: f32,
    pub(super) iterations: usize,
    /// Playouts in which the opponent's reply / our follow-up was a bingo.
    pub(super) opp_bingos: u32,
    pub(super) our_bingos: u32,
    /// Every playout value, in order; candidates share draw sequences, so
    /// element-wise differences between candidates are paired samples.
    pub(super) values: Vec<f32>,
}

impl<'a> CandidateSim<'a> {
    pub(crate) fn new(
        state: &SolverState,
        mv: &'a Move,
        score_diff: i32,
        model: Option<&'a OpponentModel>,
    ) -> Self {
        let mut base_state = state.clone();
        base_state.apply_move(mv);
        let our_leave: Vec<Tile> = base_state.rack.tiles();

        // Tiles we may draw from: the unseen tiles before our move (bag +
        // opponent's rack). A play moves tiles to the board, which leaves
        // this set unchanged; an exchange returns its tiles to the bag only
        // after the draw, so they must not be in the pool either. While the
        // bag has tiles the opponent holds a full rack.
        let pool = TilePool::from_remaining(&state.unseen_tiles());
        let bag_size = pool.len().saturating_sub(RACK_SIZE);
        let draw_count = mv.tiles_from_rack().len().min(bag_size);

        CandidateSim {
            mv,
            board: base_state.board,
            our_leave,
            pool,
            bag_size,
            draw_count,
            score_diff,
            model,
            seed: iteration_seed(PLAYOUT_SEED, position_fingerprint(state)),
            total: 0.0,
            wins: 0.0,
            iterations: 0,
            opp_bingos: 0,
            our_bingos: 0,
            values: Vec::new(),
        }
    }

    pub(super) fn mean(&self) -> f32 {
        if self.iterations == 0 {
            self.mv.score as f32
        } else {
            self.total / self.iterations as f32
        }
    }

    /// Add one playout. A finished game contributes a win, a loss or half
    /// a point for a tie; a cut-off game contributes the probability that
    /// the remaining spread swing (normal with the bag-dependent sd)
    /// leaves us ahead.
    pub(super) fn record(&mut self, outcome: PlayoutOutcome) {
        self.values.push(outcome.value);
        self.total += outcome.value;
        self.wins += win_value(self.score_diff, &outcome);
        self.iterations += 1;
    }

    /// Run playouts `start..start + count` (iteration numbers) and return
    /// what they found, without touching the accumulators, so batches of
    /// one candidate can run in parallel. Stops early at the deadline or on
    /// cancellation; the returned batch is then shorter.
    ///
    /// Playout `i` draws from its own generator seeded by `i` alone, the
    /// same for every candidate: common random numbers. Each iteration
    /// faces the same deal whichever candidate is played, which tightens
    /// comparisons between candidates, makes runs reproducible, and does
    /// not depend on how the work is split between threads.
    pub(crate) fn run_batch(
        &self,
        dict: &Dictionary,
        start: usize,
        count: usize,
        max_plies: usize,
        deadline: Option<Instant>,
        cancel: Option<&AtomicBool>,
    ) -> Vec<PlayoutRecord> {
        let mut out = Vec::with_capacity(count);
        if self.pool.is_empty() {
            // Opponent has nothing: our move ends the game.
            out.resize(
                count,
                PlayoutRecord {
                    outcome: PlayoutOutcome {
                        value: self.mv.score as f32,
                        finished: true,
                        bag: 0,
                    },
                    opp_bingo: false,
                    our_bingo: false,
                },
            );
            return out;
        }

        for i in start..start + count {
            if deadline.is_some_and(|d| Instant::now() >= d)
                || cancel.is_some_and(|c| c.load(AtomicOrdering::Relaxed))
            {
                break;
            }
            // Common random numbers: everything random in iteration i comes
            // from this generator, and it is consumed the same way for
            // every candidate (the deal, then two seeds), so all candidates
            // face the same opponent rack and the same bag order.
            let mut rng = Rng::with_seed(iteration_seed(self.seed, i as u64));
            let dealt = match self.model {
                Some(m) if m.has_marginals() => m.dealt(&self.pool, &mut rng),
                _ => self.pool.shuffled(&mut rng),
            };
            let playout_seed = rng.u64(..);
            let mut exchange_rng = Rng::with_seed(rng.u64(..));

            let split = dealt.len().min(RACK_SIZE);
            let opp_rack = dealt[..split].to_vec();
            let mut bag = dealt[split..].to_vec();
            // Our draw comes off the top of the bag, where every later
            // draw comes from too.
            let drawn = bag.split_off(bag.len() - self.draw_count.min(bag.len()));
            let mut our_rack = self.our_leave.clone();
            our_rack.extend_from_slice(&drawn);
            if let MoveKind::Exchange(returned) = &self.mv.kind {
                // Exchanged tiles go back after we draw and the bag is
                // reshuffled, with the iteration's own generator.
                bag.extend_from_slice(returned);
                exchange_rng.shuffle(&mut bag);
            }

            let seed = playout_seed;
            let mut playout =
                Playout::new(dict, self.board.clone(), [our_rack, opp_rack], bag, seed);
            playout.spread = self.mv.score as i32;
            playout.opp_temperature = self.model.map(|m| m.temperature).unwrap_or(0.0);
            // If our move emptied the bag, we own the final turn: the
            // opponent gets one last turn and then we play the final move.
            if playout.bag.is_empty() && self.bag_size > 0 {
                playout.final_turn_owner = Some(Player::Us);
            }

            let outcome = playout.run_with(Player::Opp, max_plies, &mut |_, _| {});
            out.push(PlayoutRecord {
                outcome,
                opp_bingo: playout.first_move_bingo[Player::Opp as usize] == Some(true),
                our_bingo: playout.first_move_bingo[Player::Us as usize] == Some(true),
            });
        }
        out
    }

    /// Add a batch's playouts to the accumulators, in iteration order.
    pub(super) fn absorb(&mut self, records: &[PlayoutRecord]) {
        for r in records {
            self.opp_bingos += u32::from(r.opp_bingo);
            self.our_bingos += u32::from(r.our_bingo);
            self.record(r.outcome);
        }
    }

    pub(super) fn result(&self) -> MctsResult {
        let n = self.iterations.max(1) as f32;
        MctsResult {
            mv: self.mv.clone(),
            mcts_equity: self.mean(),
            win_rate: if self.iterations == 0 {
                0.5
            } else {
                self.wins / n
            },
            iterations: self.iterations,
            opp_bingo_rate: self.opp_bingos as f32 / n,
            our_bingo_rate: self.our_bingos as f32 / n,
        }
    }
}

/// Playouts per scheduled job: small enough to spread a few candidates
/// over every core, large enough to amortise scheduling.
pub(super) const PLAYOUT_BATCH: usize = 8;

/// Candidates trailing the leader by more than this many points after a
/// pruning round — and by a statistically clear amount — are not simulated
/// further.
pub(super) const PRUNE_MARGIN: f32 = 4.0;

/// How many standard errors of the paired difference the deficit must
/// exceed before a candidate is pruned.
pub(super) const PRUNE_SIGMAS: f32 = 2.5;

/// Mean and standard error of the element-wise difference `a - b`.
pub(super) fn paired_difference(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, f32::INFINITY);
    }
    let diffs = a.iter().zip(b).map(|(x, y)| x - y);
    let mean = diffs.clone().sum::<f32>() / n as f32;
    if n < 2 {
        return (mean, f32::INFINITY);
    }
    let var = diffs.map(|d| (d - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    (mean, (var / n as f32).sqrt())
}

/// Monte Carlo evaluator for move selection
pub struct MctsEvaluator {
    pub(super) config: MctsConfig,
}

impl MctsEvaluator {
    pub fn new(config: MctsConfig) -> Self {
        MctsEvaluator { config }
    }

    /// Evaluate the top candidates and return results sorted by equity.
    ///
    /// Playouts run in rounds; after each round, candidates that trail the
    /// leader by more than [`PRUNE_MARGIN`] points, by at least
    /// [`PRUNE_SIGMAS`] standard errors of the paired difference, stop and
    /// keep their estimate so far. The budget concentrates on the moves that
    /// can still finish first. Results list the fully simulated candidates
    /// first (by expected spread), then the pruned ones, so a stale
    /// short-run estimate never outranks a complete one.
    pub fn evaluate(
        &self,
        state: &SolverState,
        candidates: &[Move],
        dictionary: &Dictionary,
    ) -> Vec<MctsResult> {
        self.evaluate_with_model(state, candidates, dictionary, None)
    }

    /// As [`Self::evaluate`], with an opponent model biasing how their
    /// rack is dealt and how strongly they play in the playouts.
    pub fn evaluate_with_model(
        &self,
        state: &SolverState,
        candidates: &[Move],
        dictionary: &Dictionary,
        model: Option<&OpponentModel>,
    ) -> Vec<MctsResult> {
        let num_to_evaluate = candidates.len().min(self.config.num_candidates);
        let mut sims: Vec<CandidateSim> = candidates[..num_to_evaluate]
            .iter()
            .map(|mv| CandidateSim::new(state, mv, self.config.score_diff, model))
            .collect();

        // Quarter, quarter, half: two pruning points.
        let total = self.config.iterations_per_candidate;
        let rounds = [total / 4, total / 4, total - 2 * (total / 4)];
        let mut active: Vec<usize> = (0..sims.len()).collect();
        let max_plies = self.config.max_plies;
        let deadline = self.config.time_budget.map(|b| Instant::now() + b);
        let cancel = self.config.cancel.as_deref();

        for (round, &playouts) in rounds.iter().enumerate() {
            if playouts == 0 {
                continue;
            }
            // One job per (candidate, batch of playouts), so every core
            // stays busy even when only a few candidates are left.
            let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
            for &ci in &active {
                let first = sims[ci].iterations;
                let mut start = first;
                while start < first + playouts {
                    let n = PLAYOUT_BATCH.min(first + playouts - start);
                    jobs.push((ci, start, n));
                    start += n;
                }
            }
            let run = |&(ci, start, n): &(usize, usize, usize)| {
                let records = sims[ci].run_batch(dictionary, start, n, max_plies, deadline, cancel);
                (ci, start, n, records)
            };
            let mut batches: Vec<(usize, usize, usize, Vec<PlayoutRecord>)> =
                if self.config.parallel {
                    jobs.par_iter().map(run).collect()
                } else {
                    jobs.iter().map(run).collect()
                };
            // Keep each candidate's playouts contiguous in iteration order:
            // after a batch cut short (deadline or cancel), later batches of
            // that candidate are dropped, so iteration i means the same
            // deal for every candidate in the paired comparisons.
            batches.sort_by_key(|&(ci, start, _, _)| (ci, start));
            let mut cut = vec![false; sims.len()];
            for (ci, _, n, records) in &batches {
                if cut[*ci] {
                    continue;
                }
                sims[*ci].absorb(records);
                if records.len() < *n {
                    cut[*ci] = true;
                }
            }
            if deadline.is_some_and(|d| Instant::now() >= d) || self.config.is_cancelled() {
                break;
            }

            if round + 1 < rounds.len() {
                active.sort_by(|&a, &b| {
                    sims[b]
                        .mean()
                        .partial_cmp(&sims[a].mean())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let leader = active[0];
                let contenders: Vec<usize> = active
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let (deficit, se) =
                            paired_difference(&sims[leader].values, &sims[i].values);
                        deficit <= PRUNE_MARGIN || deficit <= PRUNE_SIGMAS * se
                    })
                    .collect();
                if contenders.len() >= 4 {
                    active = contenders;
                } else {
                    active.truncate(4.min(active.len()));
                }
            }
        }

        let full = sims.iter().map(|s| s.iterations).max().unwrap_or(0);
        let mut results: Vec<MctsResult> = sims.iter().map(|s| s.result()).collect();
        // Fully simulated first, then candidates cut short, then any that
        // never got a playout (their numbers are placeholders).
        let tier = |r: &MctsResult| match r.iterations {
            0 => 0,
            n if n == full => 2,
            _ => 1,
        };
        results.sort_by(|a, b| {
            tier(b).cmp(&tier(a)).then(
                b.mcts_equity
                    .partial_cmp(&a.mcts_equity)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        results
    }
}
