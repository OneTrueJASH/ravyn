//! Self-play studies that produced the playout constants (tempo, the
//! spread variance, the bingo model, the loss calibration, the win
//! model). All are ignored tests; each says how to run it.

use super::playout::{normal_cdf, rack_from, Player, Playout, TEMPO};
use super::*;
use crate::rules::CROSSPLAY;
use crate::solver::{MoveKind, Solver};
use crate::tile::{Tile, RACK_SIZE};
use fastrand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::Instant;

/// Solve the normal equations `a x = b` (small dense system) by Gaussian
/// elimination with partial pivoting.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = a[col][col];
        if p.abs() < 1e-12 {
            continue;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row][col] / p;
            if f == 0.0 {
                continue;
            }
            let pivot_row = a[col].clone();
            for (x, pv) in a[row].iter_mut().zip(&pivot_row).skip(col) {
                *x -= f * pv;
            }
            b[row] -= f * b[col];
        }
    }
    (0..n)
        .map(|i| {
            if a[i][i].abs() < 1e-12 {
                0.0
            } else {
                b[i] / a[i][i]
            }
        })
        .collect()
}

/// Logistic regression by Newton's method with a small ridge term.
fn fit_logistic(xs: &[Vec<f64>], ys: &[f64], iters: usize) -> Vec<f64> {
    let d = xs[0].len();
    let mut w = vec![0.0; d];
    for _ in 0..iters {
        let mut grad = vec![0.0; d];
        let mut hess = vec![vec![0.0; d]; d];
        for (x, &y) in xs.iter().zip(ys) {
            let z: f64 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
            let p = 1.0 / (1.0 + (-z).exp());
            let r = p - y;
            for i in 0..d {
                grad[i] += r * x[i];
                let s = p * (1.0 - p) * x[i];
                for j in 0..d {
                    hess[i][j] += s * x[j];
                }
            }
        }
        for i in 0..d {
            hess[i][i] += 1e-3;
            grad[i] += 1e-3 * w[i];
        }
        let step = solve_linear(hess, grad);
        for i in 0..d {
            w[i] -= step[i];
        }
    }
    w
}

/// Self-play study of the spread still to come. For every position
/// with the opponent to move (where a playout cut off by the ply limit
/// stops), record final spread minus (spread + leave adjustment) and
/// report its mean and standard deviation by tiles left in the bag.
/// Run with `cargo test --release study_spread_variance -- --ignored --nocapture`.
#[test]
#[ignore]
fn study_spread_variance() {
    let Some(dict) = crate::test_data::dictionary() else {
        return;
    };
    let games: usize = std::env::var("GAMES")
        .ok()
        .and_then(|g| g.parse().ok())
        .unwrap_or(400);
    let started = Instant::now();
    // Half the games we start, half the opponent starts.
    let traces: Vec<(bool, Vec<TracePoint>, i32)> = (0..games as u64)
        .into_par_iter()
        .map(|seed| {
            let we_start = seed % 2 == 0;
            let mut g = Playout::new_game(&dict, 1000 + seed);
            g.trace = Some(Vec::new());
            let first = if we_start { Player::Us } else { Player::Opp };
            let final_spread = g.run(first, 1000) as i32;
            (we_start, g.trace.take().unwrap(), final_spread)
        })
        .collect();
    println!("{} games in {:?}", games, started.elapsed());

    // Residuals bucketed by bag size, opponent to move next; tempo
    // offsets by who started and who is to move.
    let mut by_bag: Vec<Vec<f64>> = vec![Vec::new(); 101];
    let mut by_bag_us: Vec<Vec<f64>> = vec![Vec::new(); 101];
    let mut tempo: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut plies = 0usize;
    let mut bingos = 0usize;
    let mut final_spreads = [Vec::new(), Vec::new()];
    for (we_start, trace, final_spread) in &traces {
        final_spreads[usize::from(*we_start)].push(*final_spread as f64);
        for t in trace {
            plies += 1;
            bingos += usize::from(t.was_bingo);
            let residual = *final_spread as f64 - (t.spread as f64 + t.leave_adj as f64);
            // Only mid-game points (bag >= 8) for the tempo estimate.
            if t.bag >= 8 {
                tempo[usize::from(*we_start) * 2 + usize::from(t.mover_is_us)].push(residual);
            }
            if t.mover_is_us {
                by_bag[t.bag].push(residual);
            } else {
                by_bag_us[t.bag].push(-residual);
            }
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "{} plies, {:.1}% bingos, mean final spread: we start {:+.1}, they start {:+.1}",
        plies,
        100.0 * bingos as f64 / plies as f64,
        mean(&final_spreads[1]),
        mean(&final_spreads[0])
    );
    println!("tempo (mean residual, bag >= 8):");
    println!(
        "  they started, we are to move:   {:+.1} (n={})",
        mean(&tempo[0]),
        tempo[0].len()
    );
    println!(
        "  they started, they are to move: {:+.1} (n={})",
        mean(&tempo[1]),
        tempo[1].len()
    );
    println!(
        "  we started, we are to move:     {:+.1} (n={})",
        mean(&tempo[2]),
        tempo[2].len()
    );
    println!(
        "  we started, they are to move:   {:+.1} (n={})",
        mean(&tempo[3]),
        tempo[3].len()
    );
    println!("bag  n     mean   sd   (opponent to move; residual = final - spread - leaves)");
    let mut fit_x = Vec::new();
    let mut fit_y = Vec::new();
    for (bag, v) in by_bag.iter().enumerate() {
        if v.len() < 30 {
            continue;
        }
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        println!("{:3} {:5} {:+6.1} {:5.1}", bag, v.len(), mean, var.sqrt());
        fit_x.push(bag as f64);
        fit_y.push(var);
    }
    // Least squares for var = alpha + beta * bag.
    let n = fit_x.len() as f64;
    let mx = fit_x.iter().sum::<f64>() / n;
    let my = fit_y.iter().sum::<f64>() / n;
    let sxy: f64 = fit_x
        .iter()
        .zip(&fit_y)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let sxx: f64 = fit_x.iter().map(|x| (x - mx).powi(2)).sum();
    let beta = sxy / sxx;
    let alpha = my - beta * mx;
    println!("variance fit: var(bag) = {:.1} + {:.2} * bag  (sd at 0: {:.1}, at 50: {:.1}, at 86: {:.1})",
        alpha, beta, alpha.max(0.0).sqrt(), (alpha + 50.0 * beta).sqrt(), (alpha + 86.0 * beta).sqrt());
    let mut all_us: Vec<f64> = by_bag_us.iter().flatten().copied().collect();
    all_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_us = all_us.iter().sum::<f64>() / all_us.len().max(1) as f64;
    println!(
        "mean residual with us to move (sign flipped): {:+.2}",
        mean_us
    );
}

/// Self-play study of how a cut-off position turns into a final result:
/// fits the tempo of the side to move per bag-size bucket, and a
/// logistic win model on the cut-off estimate (spread + leaves +
/// tempo), the bag, and the blanks and S's still unseen. Positions are
/// split by game seed into a fitting half and a held-out half; the
/// held-out log-loss and Brier score of the fitted model are printed
/// next to the normal-CDF model's. Run with
/// `GAMES=4000 cargo test --release study_win_model -- --ignored --nocapture`.
#[test]
#[ignore]
fn study_win_model() {
    let Some(dict) = crate::test_data::dictionary() else {
        return;
    };
    let games: u64 = std::env::var("GAMES")
        .ok()
        .and_then(|g| g.parse().ok())
        .unwrap_or(2000);
    let started = Instant::now();

    /// A position before a greedy-phase move, from our side.
    #[derive(Clone, Copy)]
    struct Sample {
        x: f64,
        bag: usize,
        us_to_move: bool,
        blanks: f64,
        s: f64,
        game: u64,
    }
    let unseen_counts = |p: &Playout| {
        let mut blanks = 0.0;
        let mut s = 0.0;
        for t in p.bag.iter().chain(p.rack(Player::Opp)) {
            if t.is_blank {
                blanks += 1.0;
            } else if t.letter == 'S' {
                s += 1.0;
            }
        }
        (blanks, s)
    };
    let data: Vec<(Vec<Sample>, f64)> = (0..games)
        .into_par_iter()
        .map(|seed| {
            let mut g = Playout::new_game(&dict, 40_000 + seed);
            let mut samples = Vec::new();
            let first = if seed % 2 == 0 {
                Player::Us
            } else {
                Player::Opp
            };
            let outcome = g.run_with(first, 1000, &mut |p, to_move| {
                if p.bag.is_empty() {
                    return;
                }
                let (blanks, s) = unseen_counts(p);
                samples.push(Sample {
                    x: p.spread as f64 + p.leave_adjustment() as f64,
                    bag: p.bag.len(),
                    us_to_move: to_move == Player::Us,
                    blanks,
                    s,
                    game: seed,
                });
            });
            (samples, outcome.value as f64)
        })
        .collect();
    println!("{} games in {:?}", games, started.elapsed());

    // Tempo per bag bucket: half the gap between the mean residual
    // (final - estimate) with us to move and with them to move.
    let buckets: [(usize, usize); 6] = [(1, 7), (8, 14), (15, 21), (22, 35), (36, 50), (51, 100)];
    let mut tempo = [0.0f64; 6];
    println!("bucket   n(us)  n(opp)  resid(us)  resid(opp)  tempo");
    for (bi, &(lo, hi)) in buckets.iter().enumerate() {
        let (mut su, mut nu, mut so, mut no) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (samples, fin) in &data {
            for smp in samples.iter().filter(|s| s.bag >= lo && s.bag <= hi) {
                if smp.us_to_move {
                    su += fin - smp.x;
                    nu += 1.0;
                } else {
                    so += fin - smp.x;
                    no += 1.0;
                }
            }
        }
        let (mu, mo) = (su / nu.max(1.0), so / no.max(1.0));
        tempo[bi] = (mu - mo) / 2.0;
        println!(
            "{:>3}-{:<3} {:>7} {:>7} {:>+10.2} {:>+11.2} {:>+6.2}",
            lo, hi, nu, no, mu, mo, tempo[bi]
        );
    }
    let tempo_of = |bag: usize| {
        buckets
            .iter()
            .position(|&(lo, hi)| bag >= lo && bag <= hi)
            .map_or(0.0, |i| tempo[i])
    };

    // Rows: features of the tempo-adjusted estimate; label = win.
    let row = |s: &Sample| {
        let t = if s.us_to_move {
            tempo_of(s.bag)
        } else {
            -tempo_of(s.bag)
        };
        let xt = s.x + t;
        let sd = (150.0 + 79.8 * s.bag as f64).sqrt();
        let z = xt / sd;
        ([z, z * s.bag as f64 / 50.0, z * s.blanks, z * s.s], xt, sd)
    };
    let label = |fin: f64| match fin.partial_cmp(&0.0) {
        Some(std::cmp::Ordering::Greater) => 1.0,
        Some(std::cmp::Ordering::Less) => 0.0,
        _ => 0.5,
    };
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (samples, fin) in &data {
        for s in samples {
            let (f, xt, sd) = row(s);
            let old = s.x + if s.us_to_move { 1.0 } else { -1.0 } * TEMPO as f64;
            let entry = (f, label(*fin), xt, sd, old);
            if s.game % 2 == 0 {
                train.push(entry);
            } else {
                test.push(entry);
            }
        }
    }

    // Logistic regression without intercept (the model is symmetric
    // in the sign of the estimate), fitted by Newton's method.
    let mut w = [0.0f64; 4];
    for _ in 0..25 {
        let mut grad = [0.0f64; 4];
        let mut hess = [[0.0f64; 4]; 4];
        for (f, y, ..) in &train {
            let z: f64 = (0..4).map(|i| w[i] * f[i]).sum();
            let p = 1.0 / (1.0 + (-z).exp());
            for i in 0..4 {
                grad[i] += (y - p) * f[i];
                for j in 0..4 {
                    hess[i][j] += p * (1.0 - p) * f[i] * f[j];
                }
            }
        }
        // Solve hess * step = grad (4x4 Gaussian elimination).
        let mut m = hess;
        let mut b = grad;
        for c in 0..4 {
            let piv = (c..4)
                .max_by(|&a, &bb| m[a][c].abs().partial_cmp(&m[bb][c].abs()).unwrap())
                .unwrap();
            m.swap(c, piv);
            b.swap(c, piv);
            for r in 0..4 {
                if r != c {
                    let k = m[r][c] / m[c][c];
                    let pivot_row = m[c];
                    for (x, p) in m[r].iter_mut().zip(pivot_row) {
                        *x -= k * p;
                    }
                    b[r] -= k * b[c];
                }
            }
        }
        for i in 0..4 {
            w[i] += b[i] / m[i][i];
        }
    }
    println!(
        "win model: logit = z * ({:.4} + {:.4} * bag/50 + {:.4} * blanks + {:.4} * S), z = estimate / sqrt(150 + 79.8 bag)",
        w[0], w[1], w[2], w[3]
    );

    type Entry = ([f64; 4], f64, f64, f64, f64);
    let eval = |name: &str, p_of: &dyn Fn(&Entry) -> f64| {
        let (mut ll, mut brier) = (0.0, 0.0);
        for e in &test {
            let p = p_of(e).clamp(1e-6, 1.0 - 1e-6);
            let y = e.1;
            ll -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
            brier += (p - y).powi(2);
        }
        let n = test.len() as f64;
        println!(
            "{:<34} held-out log-loss {:.4}  Brier {:.4}  (n={})",
            name,
            ll / n,
            brier / n,
            test.len()
        );
    };
    // The current model: constant tempo 21, normal CDF with the
    // variance fit.
    eval("normal CDF, constant tempo (old)", &|e| {
        normal_cdf((e.4 / e.3) as f32) as f64
    });
    eval("normal CDF, tempo per bucket", &|e| {
        normal_cdf((e.2 / e.3) as f32) as f64
    });
    eval("logistic fit, tempo per bucket", &|e| {
        let z: f64 = (0..4).map(|i| w[i] * e.0[i]).sum();
        1.0 / (1.0 + (-z).exp())
    });
    println!("total {:?}", started.elapsed());
}

/// Calibration of the observed-move-loss estimator: self-play games
/// where the opponent plays at a known softmax temperature, recording
/// their true equity loss per move (their rack is known here) next to
/// what `observed_move_loss` measures blind from sampled completions.
/// The linear fit measured = a + b*tau justifies (or corrects) how the
/// app maps a measured loss to a playout temperature.
/// Run with `cargo test --release study_loss_calibration -- --ignored --nocapture`.
#[test]
#[ignore]
fn study_loss_calibration() {
    use crate::game::observed_move_loss;

    let Some(dict) = crate::test_data::dictionary() else {
        return;
    };
    let games: usize = std::env::var("GAMES")
        .ok()
        .and_then(|g| g.parse().ok())
        .unwrap_or(24);
    let taus = [0.0f32, 4.0, 8.0, 12.0];
    let started = Instant::now();

    let full_counts = {
        let mut counts = std::collections::HashMap::new();
        for t in CROSSPLAY.full_tile_set() {
            let c = if t.is_blank { '?' } else { t.letter };
            *counts.entry(c).or_insert(0u32) += 1;
        }
        counts
    };
    // The unseen pool from our side of the table after the opponent's
    // move: everything minus the board and our own rack.
    let pool_after = |g: &Playout| {
        let mut counts = full_counts.clone();
        for r in 0..crate::board::BOARD_SIZE {
            for c in 0..crate::board::BOARD_SIZE {
                if let Some(t) = g.board.get_tile(r, c) {
                    let key = if t.is_blank { '?' } else { t.letter };
                    if let Some(n) = counts.get_mut(&key) {
                        *n = n.saturating_sub(1);
                    }
                }
            }
        }
        for t in g.rack(Player::Us) {
            let key = if t.is_blank { '?' } else { t.letter };
            if let Some(n) = counts.get_mut(&key) {
                *n = n.saturating_sub(1);
            }
        }
        counts
    };

    let mut fit_points = Vec::new();
    for &tau in &taus {
        let per_game: Vec<(Vec<f32>, Vec<f32>)> = (0..games as u64)
            .into_par_iter()
            .map(|seed| {
                let mut g = Playout::new_game(&dict, 9_000 + seed);
                g.opp_temperature = tau;
                let mut to_move = if seed % 2 == 0 {
                    Player::Us
                } else {
                    Player::Opp
                };
                let mut true_losses = Vec::new();
                let mut measured = Vec::new();
                for ply in 0..40u64 {
                    if g.final_turn_owner.is_some() || g.bag.is_empty() {
                        break;
                    }
                    if to_move == Player::Opp && g.rack(Player::Opp).len() == RACK_SIZE {
                        let best = g.best_by_equity(Player::Opp);
                        let prev_board = g.board.clone();
                        let chosen = g.policy_move(Player::Opp);
                        let (Some(best), Some(chosen)) = (best, chosen) else {
                            break;
                        };
                        true_losses.push((best.equity - chosen.equity).max(0.0));
                        let is_play = matches!(chosen.kind, MoveKind::Play);
                        g.commit(Player::Opp, &chosen);
                        if is_play {
                            if let Some(m) = observed_move_loss(
                                &prev_board,
                                &chosen.tiles_used,
                                chosen.score,
                                &pool_after(&g),
                                &dict,
                                10,
                                seed * 101 + ply,
                            ) {
                                measured.push(m);
                            }
                        }
                    } else if let Some(mv) = g.policy_move(to_move) {
                        g.commit(to_move, &mv);
                    } else {
                        break;
                    }
                    to_move = to_move.other();
                }
                (true_losses, measured)
            })
            .collect();

        let true_all: Vec<f32> = per_game.iter().flat_map(|(t, _)| t.clone()).collect();
        let meas_all: Vec<f32> = per_game.iter().flat_map(|(_, m)| m.clone()).collect();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
        println!(
            "tau {:>4.1}: {} opp moves, true loss {:>5.2}, measured {:>5.2} ({} measured)",
            tau,
            true_all.len(),
            mean(&true_all),
            mean(&meas_all),
            meas_all.len(),
        );
        fit_points.push((tau as f64, mean(&meas_all) as f64));
    }

    // Least-squares measured = a + b*tau.
    let n = fit_points.len() as f64;
    let sx: f64 = fit_points.iter().map(|(x, _)| x).sum();
    let sy: f64 = fit_points.iter().map(|(_, y)| y).sum();
    let sxx: f64 = fit_points.iter().map(|(x, _)| x * x).sum();
    let sxy: f64 = fit_points.iter().map(|(x, y)| x * y).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;
    println!(
        "fit: measured = {:.2} + {:.3}*tau  =>  temperature = (measured - {:.2}) / {:.3}",
        a, b, a, b
    );
    println!("total {:?}", started.elapsed());
}

/// Self-play study of bingo probability by leave: at positions from
/// self-play games, keep a random leave of 0-6 tiles from the mover's
/// rack plus bag, draw the rest, and record whether a bingo exists. A
/// logistic model on letter counts is fitted and printed; it drives the
/// bingo-bonus correction to the Scrabble-fitted leave table.
/// Run with `cargo test --release study_bingo_probability -- --ignored --nocapture`.
#[test]
#[ignore]
fn study_bingo_probability() {
    let Some(dict) = crate::test_data::dictionary() else {
        return;
    };
    let games: usize = std::env::var("GAMES")
        .ok()
        .and_then(|g| g.parse().ok())
        .unwrap_or(300);
    let started = Instant::now();
    // (letter counts incl. blank at 26, bingo?)
    let samples: Vec<([u8; 27], bool)> = (0..games as u64)
        .into_par_iter()
        .flat_map_iter(|seed| {
            let mut rng = Rng::with_seed(5000 + seed);
            let mut out = Vec::new();
            let mut g = Playout::new_game(&dict, 2000 + seed);
            g.run_with(Player::Us, 1000, &mut |p, to_move| {
                for _ in 0..3 {
                    let mut pool: Vec<Tile> = p.rack(to_move).to_vec();
                    pool.extend_from_slice(&p.bag);
                    if pool.len() < RACK_SIZE {
                        return;
                    }
                    rng.shuffle(&mut pool);
                    let k = rng.usize(0..=6);
                    let leave = &pool[..k];
                    let rack: Vec<Tile> = pool[..RACK_SIZE].to_vec();
                    let mut counts = [0u8; 27];
                    for t in leave {
                        let i = if t.is_blank {
                            26
                        } else {
                            (t.letter as u8 - b'A') as usize
                        };
                        counts[i] += 1;
                    }
                    let moves = Solver::new(&p.board, p.dict).find_moves(&rack_from(&rack));
                    let bingo = moves.iter().any(|m| m.is_bingo);
                    out.push((counts, bingo));
                }
            });
            out
        })
        .collect();
    println!(
        "{} samples from {} games in {:?}",
        samples.len(),
        games,
        started.elapsed()
    );

    let xs: Vec<Vec<f64>> = samples
        .iter()
        .map(|(c, _)| {
            let mut x = vec![1.0];
            x.extend(c.iter().map(|&n| n as f64));
            x
        })
        .collect();
    let ys: Vec<f64> = samples.iter().map(|(_, b)| f64::from(*b)).collect();
    let w = fit_logistic(&xs, &ys, 8);
    let p_of =
        |x: &[f64]| 1.0 / (1.0 + (-(x.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>())).exp());
    let p0 = p_of(
        &xs[0]
            .iter()
            .enumerate()
            .map(|(i, _)| if i == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
    );
    println!(
        "overall bingo rate {:.3}; p(empty leave) = {:.3}",
        ys.iter().sum::<f64>() / ys.len() as f64,
        p0
    );
    println!("bias {:+.3}", w[0]);
    println!("letter  weight  p(single)  correction = -10*(p-p0)");
    for i in 0..27 {
        let letter = if i == 26 {
            '?'
        } else {
            (b'A' + i as u8) as char
        };
        let mut x = vec![0.0; 28];
        x[0] = 1.0;
        x[i + 1] = 1.0;
        let p = p_of(&x);
        println!(
            "  {}   {:+.3}   {:.3}   {:+.2}",
            letter,
            w[i + 1],
            p,
            -10.0 * (p - p0)
        );
    }
    // Held-out check: log-loss of the model vs the constant model.
    let n = ys.len() as f64;
    let (mut ll_model, mut ll_const) = (0.0, 0.0);
    let pbar = ys.iter().sum::<f64>() / n;
    for (x, &y) in xs.iter().zip(&ys) {
        let p = p_of(x).clamp(1e-6, 1.0 - 1e-6);
        ll_model += y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        ll_const += y * pbar.ln() + (1.0 - y) * (1.0 - pbar).ln();
    }
    println!(
        "log-loss model {:.4} vs constant {:.4}",
        -ll_model / n,
        -ll_const / n
    );
    println!(
        "WEIGHTS {:?}",
        w.iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
}
