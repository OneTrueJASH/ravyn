//! Engine-versus-engine self-play matches (`examples/selfplay.rs`).

use std::sync::Mutex;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::dictionary::Dictionary;

use super::playout::{Player, Playout, SimChoice};
use super::pool::iteration_seed;

/// One side of a self-play match: how it values leaves (through its
/// dictionary's leave table and phase scaling), whether it may pass when
/// the bag is too small to exchange, and whether it picks moves by static
/// equity or by simulating the top candidates. Both sides play the exact
/// two-ply ending.
#[derive(Clone, Copy)]
pub struct MatchEngine<'a> {
    pub name: &'a str,
    pub dict: &'a Dictionary,
    pub allow_pass: bool,
    /// Choose moves by simulation instead of static equity.
    pub sim: Option<SimChoice>,
}

/// Outcome of [`self_play_match`], from engine A's point of view.
#[derive(Debug, Clone, Copy)]
pub struct MatchResult {
    /// Games played (two per seed).
    pub games: usize,
    /// A's score per game: wins plus half the ties, divided by games.
    pub score: f64,
    /// Half-width of the 95% confidence interval of `score`.
    pub score_ci95: f64,
    /// A's mean final spread and its 95% half-width.
    pub spread: f64,
    pub spread_ci95: f64,
}

/// Play `pairs` pairs of games between engines A and B. Each pair uses one
/// seed twice, once with A holding the first rack and once with B holding
/// it, so both engines see the same tile stream; who moves first
/// alternates between pairs. Pairs are the unit of the confidence
/// intervals, which removes most of the luck of the draw.
pub fn self_play_match(a: &MatchEngine, b: &MatchEngine, pairs: usize, seed: u64) -> MatchResult {
    self_play_match_with_progress(a, b, pairs, seed, &|_| {})
}

/// [`self_play_match`], calling `progress` with the running result after
/// every finished pair (from whichever worker finished it, one call at a
/// time). The final result is the same as without it.
pub fn self_play_match_with_progress(
    a: &MatchEngine,
    b: &MatchEngine,
    pairs: usize,
    seed: u64,
    progress: &(dyn Fn(&MatchResult) + Sync),
) -> MatchResult {
    let running = Mutex::new(Running::default());
    let outcome = |spread: i32| match spread.cmp(&0) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => 0.0,
        std::cmp::Ordering::Equal => 0.5,
    };
    let per_pair: Vec<(f64, f64)> = (0..pairs as u64)
        .into_par_iter()
        .map(|k| {
            let game_seed = iteration_seed(seed, k);
            let first = if k % 2 == 0 { Player::Us } else { Player::Opp };
            let play = |us: &MatchEngine, them: &MatchEngine| -> i32 {
                let mut g = Playout::new_game(us.dict, game_seed);
                g.opp_dict = them.dict;
                g.allow_pass = [us.allow_pass, them.allow_pass];
                g.sim_choice = [us.sim, them.sim];
                g.run(first, 1000).round() as i32
            };
            let a_first_rack = play(a, b);
            let b_first_rack = -play(b, a);
            let pair = (
                (outcome(a_first_rack) + outcome(b_first_rack)) / 2.0,
                (a_first_rack + b_first_rack) as f64 / 2.0,
            );
            if let Ok(mut r) = running.lock() {
                r.add(pair);
                progress(&r.result());
            }
            pair
        })
        .collect();
    let mean_ci = |v: &mut dyn Iterator<Item = f64>| {
        let xs: Vec<f64> = v.collect();
        let n = xs.len().max(1) as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        (mean, 1.96 * (var / n).sqrt())
    };
    let (score, score_ci95) = mean_ci(&mut per_pair.iter().map(|p| p.0));
    let (spread, spread_ci95) = mean_ci(&mut per_pair.iter().map(|p| p.1));
    MatchResult {
        games: pairs * 2,
        score,
        score_ci95,
        spread,
        spread_ci95,
    }
}

/// Running sums of finished pairs, for progress reports.
#[derive(Default)]
struct Running {
    n: f64,
    score: (f64, f64),
    spread: (f64, f64),
}

impl Running {
    fn add(&mut self, (score, spread): (f64, f64)) {
        self.n += 1.0;
        self.score.0 += score;
        self.score.1 += score * score;
        self.spread.0 += spread;
        self.spread.1 += spread * spread;
    }

    fn result(&self) -> MatchResult {
        let n = self.n.max(1.0);
        let mean_ci = |(sum, sq): (f64, f64)| {
            let mean = sum / n;
            let var = ((sq - n * mean * mean) / (n - 1.0).max(1.0)).max(0.0);
            (mean, 1.96 * (var / n).sqrt())
        };
        let (score, score_ci95) = mean_ci(self.score);
        let (spread, spread_ci95) = mean_ci(self.spread);
        MatchResult {
            games: self.n as usize * 2,
            score,
            score_ci95,
            spread,
            spread_ci95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_sums_match_the_batch_statistics() {
        let pairs = [(1.0, 12.0), (0.5, -3.0), (0.0, -20.0), (1.0, 7.5)];
        let mut r = Running::default();
        pairs.iter().for_each(|&p| r.add(p));
        let got = r.result();
        let n = pairs.len() as f64;
        let mean = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let var = pairs.iter().map(|p| (p.1 - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert_eq!(got.games, 8);
        assert!((got.spread - mean).abs() < 1e-9);
        assert!((got.spread_ci95 - 1.96 * (var / n).sqrt()).abs() < 1e-9);
        assert!((got.score - 0.625).abs() < 1e-9);
    }
}
