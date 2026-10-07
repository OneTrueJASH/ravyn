//! Engine-versus-engine self-play matches (`examples/selfplay.rs`).

use super::*;

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
            (
                (outcome(a_first_rack) + outcome(b_first_rack)) / 2.0,
                (a_first_rack + b_first_rack) as f64 / 2.0,
            )
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
