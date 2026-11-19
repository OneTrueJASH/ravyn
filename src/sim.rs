//! Anytime simulation of every candidate move: plays, exchanges and pass.
//!
//! Every candidate ("arm") gets the same playouts the fixed-budget
//! evaluator in [`crate::mcts`] runs, scheduled so the user always has the
//! best available answer:
//!
//! 1. Static equity orders the arms (stage 0, done by the caller).
//! 2. Every arm gets a floor of `min_playouts` playouts.
//! 3. Racing: an arm keeps being sampled while its upper confidence bound
//!    on mean spread *or* on win rate reaches the leader's lower bound on
//!    that metric. Arms are never pruned on spread alone, so a play that
//!    is behind on spread but competitive on winning chances stays in.
//!    Because iteration `i` is the same deal for every arm, the bounds are
//!    taken on the paired difference to the leader, which is far tighter
//!    than comparing two independent intervals; and an arm within an
//!    indifference margin of the leader ([`SPREAD_EPSILON`],
//!    [`WIN_EPSILON`]) does not keep the race going on its own.
//! 4. Widening: with very many arms (racks with blanks) arms are admitted
//!    in batches by static equity, exchanges and pass first, and half of
//!    each round goes to racing the arms already floored.
//!
//! Once racing has settled on a leader, the leading few arms on each
//! metric keep being refined. The run stops when the user cancels it, the
//! budget runs out, or every contender reaches the per-arm playout cap. Results stream out as
//! [`SimSnapshot`]s; the per-arm statistics can be saved ([`SimState`]) and
//! a later run on the same position continues from them, because playout
//! `i` is the same deal in every run (see `CandidateSim::run_batch`).

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::dictionary::Dictionary;
use crate::game::SolverState;
use crate::mcts::{
    position_fingerprint, win_value, CandidateSim, Fnv, OpponentModel, PlayoutRecord,
};
use crate::solver::{Move, MoveKind};

/// Width of the confidence bounds used for racing, in standard errors.
const RACE_Z: f64 = 2.5;
/// Once racing has settled, this many leading arms on each metric keep
/// being refined until the budget runs out.
const REFINE_ARMS: usize = 5;
/// Differences in mean spread smaller than this are not worth more
/// playouts to resolve.
pub const SPREAD_EPSILON: f64 = 0.5;
/// Differences in win rate smaller than this are not worth more playouts.
pub const WIN_EPSILON: f64 = 0.005;
/// Playouts per scheduled job.
const BATCH: usize = 8;
/// Arms admitted at a time while widening.
const WIDEN_BATCH: usize = 256;
/// Jobs kept in flight per pool thread, so a thread that finishes always
/// finds another queued.
const JOBS_PER_THREAD: usize = 4;
/// How often the racing bounds are refreshed while results stream in.
const STATUS_EVERY: Duration = Duration::from_millis(2);

/// One finished job: the playouts of `arm` from iteration `start`
/// (`records` is shorter than `n` if it was stopped early).
struct JobResult {
    arm: usize,
    start: usize,
    n: usize,
    records: Vec<PlayoutRecord>,
}
/// How often a running simulation publishes a snapshot.
pub const PUBLISH_EVERY: Duration = Duration::from_millis(300);

/// What to simulate and for how long.
#[derive(Debug, Clone)]
pub struct SimSettings {
    /// Playouts every arm gets before racing decides (n0).
    pub min_playouts: usize,
    /// Wall-clock budget for one run; `None` runs until cancelled or
    /// settled.
    pub budget: Option<Duration>,
    /// Ply limit of the greedy phase of a playout (see `MctsConfig`).
    pub max_plies: usize,
    /// Score difference before the move (ours minus theirs), for win rates.
    pub score_diff: i32,
    /// No arm gets more playouts than this, so two arms too close to
    /// separate cannot race forever.
    pub max_playouts_per_arm: usize,
}

impl SimSettings {
    /// Settings for a bag of `bag` tiles, with the playout depth the
    /// fixed-budget evaluator uses (`MctsConfig::for_bag`).
    pub fn for_bag(bag: usize, min_playouts: usize, budget: Option<Duration>) -> Self {
        SimSettings {
            min_playouts: min_playouts.max(1),
            budget,
            max_plies: crate::mcts::MctsConfig::for_bag(bag).max_plies,
            score_diff: 0,
            max_playouts_per_arm: 5000,
        }
    }

    pub fn with_score_diff(mut self, score_diff: i32) -> Self {
        self.score_diff = score_diff;
        self
    }
}

/// Where an arm stands in the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmStatus {
    /// Below the playout floor (or not admitted yet).
    Pending,
    /// Still being sampled: it could be the best on spread or win rate.
    Racing,
    /// Clearly behind on both: no more playouts for now.
    Stopped,
    /// Reached the per-arm playout cap.
    Capped,
}

/// Running mean and variance (Welford).
#[derive(Debug, Clone, Copy, Default)]
struct Welford {
    n: u32,
    mean: f64,
    m2: f64,
}

impl Welford {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Standard error of the mean (infinite below two samples).
    fn se(&self) -> f64 {
        if self.n < 2 {
            return f64::INFINITY;
        }
        (self.m2 / (self.n - 1) as f64 / self.n as f64).sqrt()
    }
}

/// Accumulated playouts of one arm.
#[derive(Debug, Clone, Default)]
pub struct ArmStats {
    spread: Welford,
    win: Welford,
    our_bingos: u32,
    opp_bingos: u32,
    /// Spread of every playout in iteration order, for paired differences
    /// (iteration `i` is the same deal for every arm).
    values: Vec<f32>,
    /// Win value of every playout in iteration order.
    win_values: Vec<f32>,
}

impl ArmStats {
    /// Playouts so far.
    pub fn n(&self) -> usize {
        self.values.len()
    }

    fn push(&mut self, score_diff: i32, r: &PlayoutRecord) {
        let win = win_value(score_diff, &r.outcome);
        self.values.push(r.outcome.value);
        self.win_values.push(win);
        self.spread.push(r.outcome.value as f64);
        self.win.push(win as f64);
        self.our_bingos += u32::from(r.our_bingo);
        self.opp_bingos += u32::from(r.opp_bingo);
    }

    /// The per-iteration spread values (for tests and paired comparisons).
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// What the table shows for one arm.
#[derive(Debug, Clone, Copy)]
pub struct ArmSummary {
    pub n: usize,
    /// Mean final spread and its standard error.
    pub spread: f32,
    pub spread_se: f32,
    /// Mean win probability (0-1) and its standard error.
    pub win: f32,
    pub win_se: f32,
    /// Mean spread minus the best arm's over the iterations both have, and
    /// the standard error of that paired difference.
    pub delta: f32,
    pub delta_se: f32,
    /// Share of playouts in which our follow-up / the opponent's reply was
    /// a bingo.
    pub our_bingo: f32,
    pub opp_bingo: f32,
    pub status: ArmStatus,
}

/// The state of a run, as published while it goes and at the end.
#[derive(Debug, Clone)]
pub struct SimSnapshot {
    /// One summary per candidate, in the candidates' order.
    pub arms: Vec<ArmSummary>,
    /// Playouts run so far (including any resumed from a saved state).
    pub playouts: u64,
    /// Time spent so far (including resumed runs).
    pub elapsed: Duration,
    /// Arms with at least one playout.
    pub simulated: usize,
    /// Arms at or above the playout floor.
    pub floored: usize,
    /// Arms in total.
    pub total: usize,
    /// The run has ended (budget, settled or cancelled).
    pub done: bool,
    /// The budget of this run, for a progress gauge.
    pub budget: Option<Duration>,
    /// Elapsed time of this run alone (the gauge's numerator).
    pub run_elapsed: Duration,
}

/// Everything needed to continue a run later.
#[derive(Debug, Clone)]
pub struct SimState {
    /// Identifies position, candidates, opponent model and settings.
    pub key: u64,
    arms: Vec<ArmStats>,
    playouts: u64,
    elapsed: Duration,
}

impl SimState {
    /// Playouts saved in this state.
    pub fn playouts(&self) -> u64 {
        self.playouts
    }
}

/// Key under which a run's state is saved: a resumed run must see the
/// same position, the same candidates in the same order, the same
/// opponent model and the same playout rules.
pub fn sim_key(
    state: &SolverState,
    candidates: &[Move],
    model: Option<&OpponentModel>,
    settings: &SimSettings,
) -> u64 {
    let mut h = Fnv::new();
    h.write_u64(position_fingerprint(state));
    h.write_u64(candidates.len() as u64);
    for mv in candidates.iter().take(64) {
        h.write_u64(mv.score as u64);
        h.write_u64(mv.tiles_used.len() as u64);
    }
    h.write_u64(model.map_or(0, |m| m.fingerprint()));
    h.write_u64(settings.max_plies as u64);
    h.write_u64(settings.score_diff as u64);
    h.finish()
}

/// A simulation of every candidate in a position.
pub struct Simulation<'a> {
    settings: SimSettings,
    dict: &'a Dictionary,
    state: &'a SolverState,
    candidates: &'a [Move],
    model: Option<&'a OpponentModel>,
    /// Admission order (see the module docs).
    order: Vec<usize>,
    /// How many arms of `order` are admitted.
    admitted: usize,
    sims: Vec<Option<Arc<CandidateSim<'a>>>>,
    /// Playouts scheduled for each arm whose results have not been
    /// applied yet.
    in_flight: Vec<usize>,
    stats: Vec<ArmStats>,
    status: Vec<ArmStatus>,
    playouts: u64,
    prior_elapsed: Duration,
}

impl<'a> Simulation<'a> {
    /// Set up a run over `candidates` (in the order the caller lists them;
    /// summaries come back in that order), continuing from `resume` when
    /// it belongs to the same key.
    pub fn new(
        settings: SimSettings,
        dict: &'a Dictionary,
        state: &'a SolverState,
        candidates: &'a [Move],
        model: Option<&'a OpponentModel>,
        resume: Option<SimState>,
    ) -> Self {
        let n = candidates.len();
        // Exchanges and pass first (few and cheap, and often the question
        // with a bad rack), then plays by static equity.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let ka = candidates[a].kind != MoveKind::Play;
            let kb = candidates[b].kind != MoveKind::Play;
            kb.cmp(&ka).then(
                candidates[b]
                    .equity
                    .partial_cmp(&candidates[a].equity)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        let key = sim_key(state, candidates, model, &settings);
        let (stats, playouts, prior_elapsed) = match resume {
            Some(saved) if saved.key == key && saved.arms.len() == n => {
                (saved.arms, saved.playouts, saved.elapsed)
            }
            _ => (vec![ArmStats::default(); n], 0, Duration::ZERO),
        };
        // Arms with playouts were admitted before: resume widening after
        // the last of them.
        let admitted = order
            .iter()
            .rposition(|&a| stats[a].n() > 0)
            .map_or(0, |i| i + 1);

        let mut sim = Simulation {
            settings,
            dict,
            state,
            candidates,
            model,
            order,
            admitted: 0,
            sims: (0..n).map(|_| None).collect(),
            in_flight: vec![0; n],
            stats,
            status: vec![ArmStatus::Pending; n],
            playouts,
            prior_elapsed,
        };
        sim.admit(admitted);
        sim.update_status();
        sim
    }

    /// The key this run saves its state under.
    pub fn key(&self) -> u64 {
        sim_key(self.state, self.candidates, self.model, &self.settings)
    }

    /// Admit arms up to position `upto` of the admission order.
    fn admit(&mut self, upto: usize) {
        let upto = upto.min(self.order.len());
        let new: Vec<usize> = self.order[self.admitted..upto].to_vec();
        let (state, model, diff, candidates) = (
            self.state,
            self.model,
            self.settings.score_diff,
            self.candidates,
        );
        let built: Vec<(usize, CandidateSim<'a>)> = new
            .par_iter()
            .map(|&a| (a, CandidateSim::new(state, &candidates[a], diff, model)))
            .collect();
        for (a, sim) in built {
            self.sims[a] = Some(Arc::new(sim));
        }
        self.admitted = upto;
    }

    fn floor(&self) -> usize {
        self.settings.min_playouts
    }

    /// Pick up to `capacity` more jobs: (arm, first iteration, count).
    /// Their playouts count as in flight until their results are applied.
    fn next_jobs(&mut self, capacity: usize) -> Vec<(usize, usize, usize)> {
        let floor = self.floor();
        let mut jobs = Vec::new();

        let racing: Vec<usize> = (0..self.stats.len())
            .filter(|&a| self.status[a] == ArmStatus::Racing)
            .collect();
        let floor_work_left = self.admitted < self.order.len()
            || self.order[..self.admitted]
                .iter()
                .any(|&a| self.stats[a].n() < floor);
        // While some arms still lack their floor, at most half of a round
        // goes to racing (once there is anything to race).
        let racing_share = if !floor_work_left {
            capacity
        } else if racing.len() > 1 {
            capacity / 2
        } else {
            0
        };

        // Floor work, admitting more arms as the admitted ones fill up.
        let floor_capacity = capacity - racing_share;
        let mut pos = 0;
        while jobs.len() < floor_capacity {
            if pos == self.admitted {
                if self.admitted == self.order.len() {
                    break;
                }
                self.admit(self.admitted + WIDEN_BATCH);
            }
            let a = self.order[pos];
            let have = self.stats[a].n() + self.in_flight[a];
            if have < floor {
                let n = BATCH.min(floor - have);
                jobs.push((a, have, n));
                self.in_flight[a] += n;
            } else {
                pos += 1;
            }
        }

        // Racing: fewest playouts first, one batch per arm per pass.
        let mut racing = racing;
        racing.sort_by_key(|&a| self.stats[a].n());
        let cap = self.settings.max_playouts_per_arm;
        'fill: while jobs.len() < capacity && !racing.is_empty() {
            let mut added = false;
            for &a in &racing {
                if jobs.len() >= capacity {
                    break 'fill;
                }
                let have = self.stats[a].n() + self.in_flight[a];
                if have >= cap {
                    continue;
                }
                let n = BATCH.min(cap - have);
                jobs.push((a, have, n));
                self.in_flight[a] += n;
                added = true;
            }
            if !added {
                break;
            }
        }
        jobs
    }

    /// Recompute every arm's status from the racing bounds.
    fn update_status(&mut self) {
        let floor = self.floor();
        let floored: Vec<usize> = (0..self.stats.len())
            .filter(|&a| self.stats[a].n() >= floor)
            .collect();
        let best_by = |metric: &dyn Fn(&ArmStats) -> f64| {
            floored.iter().copied().max_by(|&a, &b| {
                metric(&self.stats[a])
                    .partial_cmp(&metric(&self.stats[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        };
        let lead_s = best_by(&|s| s.spread.mean);
        let lead_w = best_by(&|s| s.win.mean);
        let cap = self.settings.max_playouts_per_arm;

        // Upper bound on (arm - leader) over their common iterations; the
        // arm is still in the race while it could beat the leader by more
        // than epsilon (so exact ties do not race forever).
        let still_in = |a: usize, lead: Option<usize>, eps: f64, pick: fn(&ArmStats) -> &[f32]| {
            let Some(l) = lead else {
                return true;
            };
            if a == l {
                return true;
            }
            let (mean, se) = paired_difference(pick(&self.stats[a]), pick(&self.stats[l]));
            mean as f64 + RACE_Z * se as f64 >= eps
        };

        let mut status = vec![ArmStatus::Pending; self.stats.len()];
        for (a, st) in status.iter_mut().enumerate() {
            let s = &self.stats[a];
            *st = if s.n() < floor {
                ArmStatus::Pending
            } else if s.n() >= cap {
                ArmStatus::Capped
            } else if still_in(a, lead_s, SPREAD_EPSILON, |s| &s.values)
                || still_in(a, lead_w, WIN_EPSILON, |s| &s.win_values)
            {
                ArmStatus::Racing
            } else {
                ArmStatus::Stopped
            };
        }
        // Once the race has settled on a leader, keep refining the closest
        // contenders on both metrics with the rest of the budget: their
        // estimates tighten, and an arm stopped too early can come back.
        let all_floored =
            self.admitted == self.order.len() && self.stats.iter().all(|s| s.n() >= floor);
        let racing = status.iter().filter(|&&st| st == ArmStatus::Racing).count();
        if all_floored && racing <= 1 {
            for metric in [|s: &ArmStats| s.spread.mean, |s: &ArmStats| s.win.mean] {
                let mut top: Vec<usize> = (0..self.stats.len())
                    .filter(|&a| status[a] != ArmStatus::Capped)
                    .collect();
                top.sort_by(|&a, &b| {
                    metric(&self.stats[b])
                        .partial_cmp(&metric(&self.stats[a]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for &a in top.iter().take(REFINE_ARMS) {
                    status[a] = ArmStatus::Racing;
                }
            }
        }
        self.status = status;
    }

    /// True when nothing is left to do: every arm has its floor and none
    /// is racing (the contenders have all reached the playout cap).
    fn settled(&self) -> bool {
        let floor = self.floor();
        self.admitted == self.order.len()
            && self.stats.iter().all(|s| s.n() >= floor)
            && !self.status.contains(&ArmStatus::Racing)
    }

    /// Run until cancelled, out of budget, or settled, calling `publish`
    /// with a snapshot about every [`PUBLISH_EVERY`]. Returns the final
    /// snapshot (with `done` set).
    ///
    /// Jobs run on the rayon pool as a continuous pipeline: about
    /// four per thread are kept in flight, and each result
    /// is applied as it arrives and replaced by a new job, so no thread
    /// waits for the slowest job of a round. Results are applied in
    /// iteration order per arm (a job that finishes early waits for the
    /// ones before it), so every arm's playouts stay contiguous from
    /// iteration 0 and each playout is the same deal as in any other run.
    /// Which arms get extra playouts can vary slightly with thread timing.
    pub fn run(
        &mut self,
        cancel: &AtomicBool,
        publish: &mut dyn FnMut(SimSnapshot),
    ) -> SimSnapshot {
        let started = Instant::now();
        let deadline = self.settings.budget.map(|b| started + b);
        let capacity = (rayon::current_num_threads() * JOBS_PER_THREAD).max(8);
        let (dict, max_plies) = (self.dict, self.settings.max_plies);
        let mut last_publish = Instant::now();
        let mut last_status = Instant::now();

        let (tx, rx) = mpsc::channel::<std::thread::Result<JobResult>>();
        // The first job that panicked; re-raised once the others finish.
        let mut panicked: Option<Box<dyn std::any::Any + Send>> = None;
        // Results that arrived before an earlier job of the same arm.
        let mut waiting: Vec<JobResult> = Vec::new();
        // Arms whose playouts stopped short (cancel or deadline): later
        // results for them would leave a gap, so they are dropped.
        let mut cut = vec![false; self.stats.len()];
        let mut outstanding = 0usize;

        rayon::in_place_scope(|scope| {
            let mut stopping = false;
            loop {
                if !stopping
                    && (cancel.load(Ordering::Relaxed)
                        || deadline.is_some_and(|d| Instant::now() >= d)
                        || self.settled())
                {
                    stopping = true;
                }
                if !stopping && outstanding < capacity {
                    let mut jobs = self.next_jobs(capacity - outstanding);
                    if jobs.is_empty() && outstanding == 0 {
                        // Nothing to do on the bounds as last refreshed;
                        // fresh ones may start a race before giving up.
                        self.update_status();
                        last_status = Instant::now();
                        jobs = self.next_jobs(capacity);
                        stopping = jobs.is_empty();
                    }
                    for (a, start, n) in jobs {
                        let sim =
                            Arc::clone(self.sims[a].as_ref().expect("scheduled arms are admitted"));
                        let tx = tx.clone();
                        outstanding += 1;
                        scope.spawn(move |_| {
                            // A panic is reported like a result, or the
                            // coordinator would wait for it forever.
                            let records = std::panic::catch_unwind(AssertUnwindSafe(|| {
                                sim.run_batch(dict, start, n, max_plies, deadline, Some(cancel))
                            }));
                            // The receiver outlives every job in the scope.
                            let _ = tx.send(match records {
                                Ok(records) => Ok(JobResult {
                                    arm: a,
                                    start,
                                    n,
                                    records,
                                }),
                                Err(payload) => Err(payload),
                            });
                        });
                    }
                }
                if outstanding == 0 {
                    break;
                }

                // Wait for one result, then take whatever else is ready.
                let first = rx.recv().expect("jobs hold a sender until they finish");
                for r in std::iter::once(first).chain(std::iter::from_fn(|| rx.try_recv().ok())) {
                    outstanding -= 1;
                    match r {
                        Ok(r) => self.accept(r, &mut waiting, &mut cut),
                        Err(payload) => {
                            stopping = true;
                            panicked.get_or_insert(payload);
                        }
                    }
                }

                // Racing bounds cost a pass over every arm: refresh them
                // every few milliseconds rather than after every job.
                if last_status.elapsed() >= STATUS_EVERY || stopping {
                    self.update_status();
                    last_status = Instant::now();
                }
                if !stopping && last_publish.elapsed() >= PUBLISH_EVERY {
                    publish(self.snapshot(started.elapsed(), /* done= */ false));
                    last_publish = Instant::now();
                }
            }
        });
        if let Some(payload) = panicked {
            std::panic::resume_unwind(payload);
        }
        // Anything still waiting follows a short batch: its arm is cut.
        for r in waiting {
            self.in_flight[r.arm] -= r.n;
        }
        debug_assert!(
            self.in_flight.iter().all(|&n| n == 0),
            "every scheduled playout is applied or dropped"
        );
        self.update_status();
        self.snapshot(started.elapsed(), /* done= */ true)
    }

    /// Take one job's result: apply it if it is next in its arm's
    /// iteration order (then any waiting results it unblocks), hold it if
    /// an earlier job of the arm is still running, drop it if the arm was
    /// cut short.
    fn accept(&mut self, result: JobResult, waiting: &mut Vec<JobResult>, cut: &mut [bool]) {
        let mut next = Some(result);
        while let Some(r) = next.take() {
            let a = r.arm;
            if cut[a] {
                self.in_flight[a] -= r.n;
            } else if r.start == self.stats[a].n() {
                self.in_flight[a] -= r.n;
                for rec in &r.records {
                    self.stats[a].push(self.settings.score_diff, rec);
                }
                self.playouts += r.records.len() as u64;
                if r.records.len() < r.n {
                    cut[a] = true;
                }
                // The arm's following job may already be waiting.
                if let Some(i) = waiting
                    .iter()
                    .position(|w| w.arm == a && w.start == self.stats[a].n())
                {
                    next = Some(waiting.swap_remove(i));
                } else if cut[a] {
                    // Drop everything the cut arm still has waiting.
                    waiting.retain(|w| {
                        if w.arm == a {
                            self.in_flight[a] -= w.n;
                            false
                        } else {
                            true
                        }
                    });
                }
            } else {
                waiting.push(r);
            }
        }
    }

    /// Summaries of every arm, for the table.
    pub fn snapshot(&self, run_elapsed: Duration, done: bool) -> SimSnapshot {
        let floor = self.floor();
        // The best arm by mean spread, among those with their floor (or
        // among all with playouts before any has).
        let pool: Vec<usize> = {
            let floored: Vec<usize> = (0..self.stats.len())
                .filter(|&a| self.stats[a].n() >= floor)
                .collect();
            if floored.is_empty() {
                (0..self.stats.len())
                    .filter(|&a| self.stats[a].n() > 0)
                    .collect()
            } else {
                floored
            }
        };
        let best = pool.iter().copied().max_by(|&a, &b| {
            self.stats[a]
                .spread
                .mean
                .partial_cmp(&self.stats[b].spread.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let arms = (0..self.stats.len())
            .map(|a| {
                let s = &self.stats[a];
                let n = s.n();
                let (delta, delta_se) = match best {
                    Some(b) if n > 0 => paired_difference(&s.values, &self.stats[b].values),
                    _ => (0.0, f32::INFINITY),
                };
                let rate = |c: u32| if n == 0 { 0.0 } else { c as f32 / n as f32 };
                ArmSummary {
                    n,
                    spread: s.spread.mean as f32,
                    spread_se: s.spread.se() as f32,
                    win: s.win.mean as f32,
                    win_se: s.win.se() as f32,
                    delta,
                    delta_se,
                    our_bingo: rate(s.our_bingos),
                    opp_bingo: rate(s.opp_bingos),
                    status: self.status[a],
                }
            })
            .collect();
        SimSnapshot {
            arms,
            playouts: self.playouts,
            elapsed: self.prior_elapsed + run_elapsed,
            simulated: self.stats.iter().filter(|s| s.n() > 0).count(),
            floored: self.stats.iter().filter(|s| s.n() >= floor).count(),
            total: self.stats.len(),
            done,
            budget: self.settings.budget,
            run_elapsed,
        }
    }

    /// The state to resume from later.
    pub fn into_state(self, run_elapsed: Duration) -> SimState {
        let key = self.key();
        SimState {
            key,
            arms: self.stats,
            playouts: self.playouts,
            elapsed: self.prior_elapsed + run_elapsed,
        }
    }

    /// Per-arm statistics (tests).
    pub fn stats(&self) -> &[ArmStats] {
        &self.stats
    }
}

/// Mean and standard error of the element-wise difference `a - b` over
/// their common prefix.
fn paired_difference(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, f32::INFINITY);
    }
    let mut w = Welford::default();
    for i in 0..n {
        w.push((a[i] - b[i]) as f64);
    }
    (w.mean as f32, w.se() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data;

    const BEECH: &str = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15";
    const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";

    fn candidates(state: &SolverState, dict: &Dictionary) -> Vec<Move> {
        let mut c = state.get_suggestions(dict);
        c.push(Move::pass(&state.rack.tiles()));
        c
    }

    /// Common random numbers: two identical arms produce identical
    /// per-iteration results, whatever else is simulated alongside them.
    #[test]
    fn identical_arms_get_identical_playouts() {
        let Some(dict) = test_data::dictionary() else {
            return;
        };
        let state = SolverState::from_cgp(&format!("{} AEINRST", BEECH)).unwrap();
        let all = state.get_suggestions(&dict);
        let a = all[0].clone();
        let b = all[5].clone();
        let cands = vec![a.clone(), b, a];
        let mut settings = SimSettings::for_bag(state.bag_size(), 24, None);
        settings.max_playouts_per_arm = 24;
        let mut sim = Simulation::new(settings, &dict, &state, &cands, None, None);
        let cancel = AtomicBool::new(false);
        let snap = sim.run(&cancel, &mut |_| {});
        assert!(snap.done);
        let stats = sim.stats();
        assert_eq!(stats[0].n(), 24);
        assert_eq!(stats[0].values(), stats[2].values());
        assert_ne!(stats[0].values(), stats[1].values());
    }

    /// On the MID position every legal play, exchange and pass gets the
    /// playout floor well within the default budget.
    #[test]
    fn every_arm_reaches_the_floor_on_mid() {
        let Some(dict) = test_data::dictionary() else {
            return;
        };
        let state = SolverState::from_cgp(&format!("{} JSIRATU", MID)).unwrap();
        let cands = candidates(&state, &dict);
        assert!(cands.len() > 400, "{} candidates", cands.len());
        let settings = SimSettings::for_bag(state.bag_size(), 16, Some(Duration::from_secs(20)));
        let mut sim = Simulation::new(settings, &dict, &state, &cands, None, None);
        // Stop as soon as a snapshot shows every arm at the floor.
        let cancel = AtomicBool::new(false);
        let snap = sim.run(&cancel, &mut |s| {
            if s.floored == s.total {
                cancel.store(true, Ordering::Relaxed);
            }
        });
        assert_eq!(snap.total, cands.len());
        assert_eq!(snap.floored, cands.len(), "every arm reached the floor");
        assert!(snap.arms.iter().all(|a| a.n >= 16));
        assert!(snap.elapsed < Duration::from_secs(20), "{:?}", snap.elapsed);
        println!(
            "MID: {} arms floored after {} playouts in {:?}",
            snap.total, snap.playouts, snap.elapsed
        );
    }

    /// How a run progresses on bigger positions. Run with
    /// `cargo test --release sim_progress -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sim_progress() {
        let Some(dict) = test_data::dictionary() else {
            return;
        };
        for rack in ["AEINRST", "AEIO?ST", "DGIOUVW"] {
            let state = SolverState::from_cgp(&format!("{} {}", BEECH, rack)).unwrap();
            let cands = candidates(&state, &dict);
            let settings =
                SimSettings::for_bag(state.bag_size(), 16, Some(Duration::from_secs(20)));
            let mut sim = Simulation::new(settings, &dict, &state, &cands, None, None);
            let cancel = AtomicBool::new(false);
            let mut k = 0;
            let snap = sim.run(&cancel, &mut |s| {
                k += 1;
                if k % 10 == 0 {
                    let racing = s
                        .arms
                        .iter()
                        .filter(|a| a.status == ArmStatus::Racing)
                        .count();
                    println!(
                        "  {:>5.1}s simulated {}/{} floored {} racing {} playouts {}",
                        s.elapsed.as_secs_f32(),
                        s.simulated,
                        s.total,
                        s.floored,
                        racing,
                        s.playouts
                    );
                }
            });
            let racing = snap
                .arms
                .iter()
                .filter(|a| a.status == ArmStatus::Racing)
                .count();
            println!(
                "{rack}: {} arms, floored {}, racing {}, {} playouts in {:?}",
                snap.total, snap.floored, racing, snap.playouts, snap.elapsed
            );
        }
    }

    /// A cancelled run saves its state; resuming continues from it without
    /// repeating playouts, and the resumed values extend the saved ones.
    #[test]
    fn cancel_and_resume_continue_the_same_playouts() {
        let Some(dict) = test_data::dictionary() else {
            return;
        };
        let state = SolverState::from_cgp(&format!("{} AEINRST", BEECH)).unwrap();
        let mut cands = state.get_suggestions(&dict);
        cands.truncate(12);
        let mut settings = SimSettings::for_bag(state.bag_size(), 16, None);
        settings.max_playouts_per_arm = 64;

        // A full run for reference.
        let mut full = Simulation::new(settings.clone(), &dict, &state, &cands, None, None);
        let cancel = AtomicBool::new(false);
        full.run(&cancel, &mut |_| {});

        // A run cancelled from the start does nothing, then resumes.
        let mut first = Simulation::new(settings.clone(), &dict, &state, &cands, None, None);
        let stopped = AtomicBool::new(true);
        first.run(&stopped, &mut |_| {});
        let saved = first.into_state(Duration::ZERO);
        assert_eq!(saved.playouts(), 0);

        // Partial run: stop after the floor via a tiny cap, then resume
        // with the real cap.
        let mut partial_settings = settings.clone();
        partial_settings.max_playouts_per_arm = 16;
        let mut partial =
            Simulation::new(partial_settings, &dict, &state, &cands, None, Some(saved));
        partial.run(&cancel, &mut |_| {});
        let saved = partial.into_state(Duration::ZERO);
        assert!(saved.playouts() >= 16 * cands.len() as u64);

        let mut resumed = Simulation::new(settings, &dict, &state, &cands, None, Some(saved));
        resumed.run(&cancel, &mut |_| {});
        for (r, f) in resumed.stats().iter().zip(full.stats()) {
            let n = r.n().min(f.n());
            assert_eq!(
                r.values()[..n],
                f.values()[..n],
                "same deals in the same order"
            );
        }
    }
}
