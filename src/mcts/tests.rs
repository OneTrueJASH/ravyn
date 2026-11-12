use super::eval::paired_difference;
use super::playout::{normal_cdf, remaining_spread_sd, Player, Playout};
use super::*;
use crate::board::Board;
use crate::game::SolverState;
use crate::solver::{Move, MoveKind, Solver};
use crate::tile::{Tile, RACK_SIZE};
use fastrand::Rng;
use std::time::{Duration, Instant};

#[test]
fn test_tile_pool_creation() {
    let mut remaining = std::collections::HashMap::new();
    remaining.insert('A', 3);
    remaining.insert('B', 2);
    remaining.insert('?', 1);

    let pool = TilePool::from_remaining(&remaining);
    assert_eq!(pool.len(), 6);
    assert!(!pool.is_empty());
    let mut rng = Rng::with_seed(1);
    assert_eq!(pool.shuffled(&mut rng).len(), 6);
}

/// A marginal saying the opponent kept Z more often than a uniform
/// deal would must raise Z's frequency in their dealt rack, without
/// changing which tiles are dealt overall.
#[test]
fn opponent_model_biases_the_deal() {
    let mut remaining = std::collections::HashMap::new();
    remaining.insert('A', 5);
    remaining.insert('E', 5);
    remaining.insert('B', 8);
    remaining.insert('Z', 2);
    let pool = TilePool::from_remaining(&remaining);
    assert_eq!(pool.len(), 20);

    let kept_size = 6;
    // Uniform expectation of Z among 6 kept tiles is 6 * 2/20 = 0.6;
    // the marginal says 1.5.
    let model = OpponentModel::new(0.0).with_marginals(&[('Z', 1.5)], kept_size, &pool);
    assert!(model.has_marginals());

    let mut rng = Rng::with_seed(7);
    let deals = 2000;
    let mut z_in_opp = 0usize;
    for i in 0..deals {
        let dealt = model.dealt(&pool, &mut rng);
        assert_eq!(dealt.len(), pool.len());
        if i == 0 {
            let mut sorted: Vec<char> = dealt.iter().map(|t| t.letter).collect();
            sorted.sort_unstable();
            let mut expected: Vec<char> = pool.tiles.iter().map(|t| t.letter).collect();
            expected.sort_unstable();
            assert_eq!(sorted, expected, "a deal must be a permutation of the pool");
        }
        z_in_opp += dealt[..RACK_SIZE]
            .iter()
            .filter(|t| t.letter == 'Z')
            .count();
    }
    // A uniform deal puts 7 * 2/20 = 0.7 Z's in their rack on average;
    // the tilt should push it well above that.
    let mean = z_in_opp as f32 / deals as f32;
    assert!(mean > 0.95, "expected biased Z mean above 0.95, got {mean}");
}

/// Passing is a playout candidate only while the bag holds 1-6 tiles
/// (no exchange possible), valued at the whole rack's leave, and never
/// twice in a row.
#[test]
fn pass_is_a_playout_candidate_without_exchanges() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let tiles = |s: &str| -> Vec<Tile> {
        s.chars()
            .map(|c| {
                if c == '?' {
                    Tile::blank(' ')
                } else {
                    Tile::new(c)
                }
            })
            .collect()
    };
    let with_bag = |bag: &str| {
        Playout::new(
            &dict,
            Board::new(),
            [tiles("??AEQVV"), tiles("EEIIOOU")],
            tiles(bag),
            1,
        )
    };
    let p = with_bag("ABC");
    let pass = p
        .pass_candidate(Player::Us)
        .expect("bag of 3: pass allowed");
    assert_eq!(pass.kind, MoveKind::Pass);
    let rack_value = dict.leave_value(&tiles("??AEQVV"), Some(3 + RACK_SIZE));
    assert_eq!(pass.equity, rack_value);

    assert!(with_bag("").pass_candidate(Player::Us).is_none());
    assert!(with_bag("ABCDEFG").pass_candidate(Player::Us).is_none());
    let mut p = with_bag("ABC");
    p.commit(Player::Us, &pass);
    assert!(p.pass_candidate(Player::Opp).is_none(), "no second pass");
    let mut off = with_bag("ABC");
    off.allow_pass = [false, false];
    assert!(off.pass_candidate(Player::Us).is_none());
}

/// With a temperature the simulated opponent errs sometimes but still
/// prefers better moves; our own side stays exact.
#[test]
fn opponent_policy_errs_at_temperature() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let mut playout = Playout::new_game(&dict, 42);

    let best = playout.best_by_equity(Player::Opp).unwrap();
    let pick = playout.policy_move(Player::Opp).unwrap();
    assert_eq!(
        (pick.word.clone(), pick.score),
        (best.word.clone(), best.score),
        "zero temperature must play the greedy move"
    );

    playout.opp_temperature = 8.0;
    let samples = 300;
    let mut erred = 0usize;
    let mut best_picked = 0usize;
    let mut total_gap = 0.0f32;
    for _ in 0..samples {
        let mv = playout.policy_move(Player::Opp).unwrap();
        let gap = best.equity - mv.equity;
        assert!(gap >= -1e-4, "sampled move cannot beat the best");
        assert!(gap <= 3.0 * 8.0 + 1e-4, "window bounds the error");
        total_gap += gap.max(0.0);
        if gap > 1e-6 {
            erred += 1;
        } else {
            best_picked += 1;
        }
    }
    assert!(erred > 0, "a warm opponent must err sometimes");
    assert!(
        best_picked > samples / 16,
        "the best move should stay the most likely pick"
    );
    assert!(
        total_gap / samples as f32 <= 2.0 * 8.0,
        "mean error should be on the order of the temperature"
    );

    let our_best = playout.best_by_equity(Player::Us).unwrap();
    let our_pick = playout.policy_move(Player::Us).unwrap();
    assert_eq!(
        (our_pick.word, our_pick.score),
        (our_best.word, our_best.score),
        "our own playout policy stays exact"
    );
}

/// `evaluate` and `evaluate_with_model(None)` are the same computation.
#[test]
fn evaluate_without_model_is_unchanged() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let mut state = SolverState::new();
    for ch in ['C', 'A', 'T', 'S', 'E', 'R', 'N'] {
        state.add_to_rack(ch);
    }
    let suggestions = state.get_suggestions(&dict);
    let config = MctsConfig {
        num_candidates: 3,
        iterations_per_candidate: 6,
        max_plies: 2,
        parallel: false,
        score_diff: 0,
        time_budget: None,
        cancel: None,
    };
    let evaluator = MctsEvaluator::new(config);
    let plain = evaluator.evaluate(&state, &suggestions, &dict);
    let with_none = evaluator.evaluate_with_model(&state, &suggestions, &dict, None);
    let key = |r: &MctsResult| (r.mv.word.clone(), r.mcts_equity, r.win_rate);
    assert_eq!(
        plain.iter().map(key).collect::<Vec<_>>(),
        with_none.iter().map(key).collect::<Vec<_>>()
    );
}

#[test]
fn test_mcts_evaluator_basic() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let state = SolverState::new();

    let mut test_state = state.clone();
    for ch in ['C', 'A', 'T', 'S', 'E', 'R', 'N'] {
        test_state.add_to_rack(ch);
    }

    let suggestions = test_state.get_suggestions(&dict);
    assert!(!suggestions.is_empty());

    let config = MctsConfig {
        num_candidates: 3,
        iterations_per_candidate: 5,
        max_plies: 2,
        parallel: false,
        score_diff: 0,
        time_budget: None,
        cancel: None,
    };

    let evaluator = MctsEvaluator::new(config);
    let results = evaluator.evaluate(&test_state, &suggestions, &dict);

    assert!(!results.is_empty());
    assert!(results.len() <= 3);
    for result in &results {
        assert!(result.mcts_equity.is_finite());
        assert!(result.iterations >= 1 && result.iterations <= 5);
        assert!((0.0..=1.0).contains(&result.win_rate));
    }
    // The leader always receives the full budget.
    assert_eq!(results[0].iterations, 5);
}

/// Two-tile bag. Playing out the ending must respect the equal-turn
/// rule: whoever draws the last tile gets the final move, nobody is
/// penalised for leftover tiles, and the playout ends after the two
/// final turns rather than when a rack empties.
#[test]
fn test_playout_follows_equal_turn_rule() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let mut board = Board::new();
    board.place_tile(7, 6, Tile::new('C'));
    board.place_tile(7, 7, Tile::new('A'));
    board.place_tile(7, 8, Tile::new('T'));

    // We move first and empty the bag (bag is empty from the start here,
    // with us as owner): the opponent gets a last turn, then we play.
    let mut playout = Playout::new(
        &dict,
        board.clone(),
        [
            vec![Tile::new('S'), Tile::new('E')],
            vec![Tile::new('Z'), Tile::new('A')],
        ],
        vec![],
        1,
    );
    playout.final_turn_owner = Some(Player::Us);
    let value = playout.run(Player::Opp, 10);
    // The opponent played (ZA somewhere, > 0 points) and we replied; the
    // playout stopped there even though both racks may still hold tiles.
    assert!(playout.racks[1].len() <= 2);
    assert_eq!(value, playout.spread as f32);

    // Reproduce by hand: the opponent's best last turn is the 2-ply
    // optimum, and our final move is our best score on the board after it.
    let mut check = Playout::new(
        &dict,
        board,
        [
            vec![Tile::new('S'), Tile::new('E')],
            vec![Tile::new('Z'), Tile::new('A')],
        ],
        vec![],
        1,
    );
    check.final_turn_owner = Some(Player::Us);
    let (opp_mv, opp_value) = check.best_last_turn(Player::Opp);
    let opp_score = opp_mv.as_ref().map(|m| m.score as i32).unwrap_or(0);
    if let Some(m) = &opp_mv {
        check.commit(Player::Opp, m);
    }
    let our_final = check
        .best_by_score(Player::Us)
        .map(|m| m.score as i32)
        .unwrap_or(0);
    assert_eq!(opp_value, opp_score - our_final);
    assert_eq!(value as i32, our_final - opp_score);
}

/// When our candidate draws the last tile we own the final turn; when it
/// does not, the opponent's draw decides. Either way the playout must
/// terminate with an empty bag and a finite value.
#[test]
fn test_pre_endgame_evaluation_terminates() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    // 87 tiles on the board and 4 in our rack: 9 unseen, bag of 2.
    const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";
    let state = SolverState::from_cgp(&format!("{} AEST", MID)).unwrap();
    assert_eq!(state.unseen_tile_count(), 9);

    let suggestions = state.get_suggestions(&dict);
    assert!(!suggestions.is_empty());
    let evaluator = MctsEvaluator::new(MctsConfig {
        num_candidates: 4,
        iterations_per_candidate: 10,
        max_plies: 10,
        parallel: false,
        score_diff: 0,
        time_budget: None,
        cancel: None,
    });
    let results = evaluator.evaluate(&state, &suggestions, &dict);
    assert_eq!(results.len(), 4.min(suggestions.len()));
    for r in &results {
        assert!(r.mcts_equity.is_finite());
        assert!(r.iterations >= 2);
    }
    assert_eq!(results[0].iterations, 10);
}

#[test]
fn normal_cdf_and_spread_sd() {
    assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
    assert!((normal_cdf(1.0) - 0.841_345).abs() < 1e-4);
    assert!((normal_cdf(-2.0) - 0.022_750).abs() < 1e-4);
    assert!(normal_cdf(10.0) > 0.999_99);
    assert!(remaining_spread_sd(0) > 11.0 && remaining_spread_sd(0) < 13.0);
    assert!(remaining_spread_sd(86) > 80.0 && remaining_spread_sd(86) < 90.0);
}

#[test]
fn paired_difference_statistics() {
    let (m, se) = paired_difference(&[], &[]);
    assert_eq!(m, 0.0);
    assert!(se.is_infinite());
    let (m, se) = paired_difference(&[5.0], &[2.0]);
    assert_eq!(m, 3.0);
    assert!(se.is_infinite());
    // Constant difference: zero standard error.
    let (m, se) = paired_difference(&[1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(m, 1.0);
    assert_eq!(se, 0.0);
    // Differences 2, -2, 2, -2: mean 0, sample sd = sqrt(16/3), se = sd/2.
    let (m, se) = paired_difference(&[2.0, 0.0, 2.0, 0.0], &[0.0, 2.0, 0.0, 2.0]);
    assert_eq!(m, 0.0);
    assert!((se - (16.0f32 / 3.0).sqrt() / 2.0).abs() < 1e-6);
}

/// Fully simulated candidates rank ahead of pruned ones regardless of
/// their short-run means.
#[test]
fn fully_simulated_candidates_rank_first() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";
    let state = SolverState::from_cgp(&format!("{} ES", MID)).unwrap();
    let suggestions = state.get_suggestions(&dict);
    let results = MctsEvaluator::new(MctsConfig {
        num_candidates: 12,
        iterations_per_candidate: 24,
        max_plies: 10,
        parallel: true,
        score_diff: 0,
        time_budget: None,
        cancel: None,
    })
    .evaluate(&state, &suggestions, &dict);
    let full = results.iter().map(|r| r.iterations).max().unwrap();
    assert_eq!(full, 24);
    let first_partial = results.iter().position(|r| r.iterations < full);
    if let Some(p) = first_partial {
        assert!(
            results[p..].iter().all(|r| r.iterations < full),
            "a fully simulated candidate is listed after a pruned one"
        );
        assert!(results[..p]
            .windows(2)
            .all(|w| w[0].mcts_equity >= w[1].mcts_equity));
    }
}

/// A tiny time budget stops playouts early and still returns a result
/// for every candidate.
#[test]
fn time_budget_is_honoured() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";
    let state = SolverState::from_cgp(&format!("{} AEST", MID)).unwrap();
    let suggestions = state.get_suggestions(&dict);
    let started = Instant::now();
    let results = MctsEvaluator::new(MctsConfig {
        num_candidates: 25,
        iterations_per_candidate: 400,
        max_plies: 10,
        parallel: true,
        score_diff: 0,
        time_budget: Some(Duration::from_millis(150)),
        cancel: None,
    })
    .evaluate(&state, &suggestions, &dict);
    // One playout may overrun the deadline per thread; allow slack.
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(results.len(), 25.min(suggestions.len()));
    assert!(results.iter().all(|r| r.mcts_equity.is_finite()));
    assert!(results.iter().any(|r| r.iterations < 400));
}

/// Timing harness for the pre-endgame playout. Run with
/// `cargo test --release bench_pre_endgame -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_pre_endgame() {
    use std::time::Instant;
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";
    // Same dense board with the three board blanks turned into real
    // tiles, so the three blanks are unseen and land in the opponent's
    // rack: the slowest realistic case for the 2-ply replies.
    const END_BLANKS_UNSEEN: &str = "3N3Q3M3/3OP2UNTEACH1/3BE2O3N3/3LA2T2MYOPE/2FER2H1LO4/2O6AB4/2R1F4U3DE/2TRIVET1D3AY/4LA1A1IODIZE/4KI1JINX2ES/5R1E1G4O/OUTIES1S3G2R/11L2E/8WAHINES/2EGALITE2A3";
    const EMPTY: &str = "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15";
    const BEECH: &str = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15";
    for (board, rack) in [
        (EMPTY, "AEINRST"),
        (BEECH, "AEINRS?"),
        (MID, "AEIST"),
        (MID, "AEST"),
        (MID, "ES"),
        (MID, "E"),
        (END_BLANKS_UNSEEN, "R"),
    ] {
        let state = SolverState::from_cgp(&format!("{} {}", board, rack)).unwrap();
        let bag = state.unseen_tile_count() - RACK_SIZE;
        let suggestions = state.get_suggestions(&dict);
        let config = MctsConfig::for_bag(bag);
        let t = Instant::now();
        let results = MctsEvaluator::new(config.clone()).evaluate(&state, &suggestions, &dict);
        let elapsed = t.elapsed();
        println!(
            "rack {:7} bag {:2} {} plies  {} candidates x {} playouts: {:?}",
            rack,
            bag,
            config.max_plies,
            results.len(),
            config.iterations_per_candidate,
            elapsed
        );
        for r in results.iter().take(5) {
            println!(
                "   {:10} {:>4} score {:>3} static {:>6.1} sim {:>6.1} win {:.0}% bingo {:.0}/{:.0} n={}",
                r.mv.word,
                r.mv.coords(),
                r.mv.score,
                r.mv.equity,
                r.mcts_equity,
                r.win_rate * 100.0,
                r.our_bingo_rate * 100.0,
                r.opp_bingo_rate * 100.0,
                r.iterations
            );
        }
    }
}

fn tiles_of(s: &str) -> Vec<Tile> {
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

fn multiset(v: &[Tile]) -> Vec<(bool, char)> {
    let mut k: Vec<(bool, char)> = v
        .iter()
        .map(|x| (x.is_blank, if x.is_blank { '?' } else { x.letter }))
        .collect();
    k.sort();
    k
}

/// An exchange in a playout: the exchanged tiles leave the rack, as many
/// are drawn, the bag keeps its size, nothing is lost, and the bag is
/// never "emptied" by it.
#[test]
fn playout_exchange_conserves_tiles_and_bag_size() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let bag = tiles_of("EEEEEEEEEE");
    let mut p = Playout::new(
        &dict,
        Board::new(),
        [tiles_of("QVWXZ?A"), tiles_of("AAAAAAA")],
        bag,
        7,
    );
    let before: Vec<Tile> = p.racks[0]
        .iter()
        .chain(&p.racks[1])
        .chain(&p.bag)
        .copied()
        .collect();
    let mv = Move::exchange(&p.racks[0], tiles_of("QV?"), 0.0);
    p.commit(Player::Us, &mv);
    assert_eq!(p.racks[0].len(), 7);
    assert_eq!(p.bag.len(), 10);
    assert!(p.final_turn_owner.is_none());
    assert_eq!(p.spread, 0);
    // Drew three E's (the bag was all E's), returned Q V ?.
    assert_eq!(multiset(&p.racks[0]), multiset(&tiles_of("WXZAEEE")));
    let after: Vec<Tile> = p.racks[0]
        .iter()
        .chain(&p.racks[1])
        .chain(&p.bag)
        .copied()
        .collect();
    assert_eq!(multiset(&before), multiset(&after));
    assert!(p.bag.iter().any(|t| t.is_blank) && p.bag.iter().any(|t| t.letter == 'Q'));
}

/// A seven-tile exchange with exactly seven tiles in the bag must not
/// count as drawing the last tile (regression: the emptiness check ran
/// before the exchanged tiles went back).
#[test]
fn full_exchange_with_bag_of_seven_does_not_claim_final_turn() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let mut p = Playout::new(
        &dict,
        Board::new(),
        [tiles_of("UUVVWWI"), tiles_of("AAAAAAA")],
        tiles_of("EEEEEEE"),
        9,
    );
    let mv = Move::exchange(&p.racks[0], tiles_of("UUVVWWI"), 0.0);
    p.commit(Player::Us, &mv);
    assert_eq!(p.bag.len(), 7);
    assert!(p.final_turn_owner.is_none());
    assert_eq!(multiset(&p.racks[0]), multiset(&tiles_of("EEEEEEE")));
}

/// The three ways the bag can empty, each followed by exactly one more
/// move for the other side and then the emptier's final move.
#[test]
fn playout_endings_follow_the_equal_turn_sequence() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let board = Board::new();
    let mut b = board.clone();
    b.place_tile(7, 6, Tile::new('C'));
    b.place_tile(7, 7, Tile::new('A'));
    b.place_tile(7, 8, Tile::new('T'));

    // (a) Our candidate emptied the bag: owner = Us from the start.
    let mut p = Playout::new(
        &dict,
        b.clone(),
        [tiles_of("RE"), tiles_of("SOD")],
        vec![],
        1,
    );
    p.final_turn_owner = Some(Player::Us);
    p.trace = Some(vec![]);
    let out = p.run_with(Player::Opp, 10, &mut |_, _| {});
    assert!(out.finished);
    let tr = p.trace.take().unwrap();
    assert_eq!(tr.len(), 2, "{:?}", tr);
    assert!(!tr[0].mover_is_us && tr[1].mover_is_us);

    // (b) The opponent's first move empties a one-tile bag: they own the
    // final turn; we get one last turn, then they play, then it ends.
    let mut p = Playout::new(
        &dict,
        b.clone(),
        [tiles_of("RE"), tiles_of("SOD")],
        tiles_of("E"),
        2,
    );
    p.trace = Some(vec![]);
    let out = p.run_with(Player::Opp, 20, &mut |_, _| {});
    assert!(out.finished);
    assert_eq!(p.final_turn_owner, Some(Player::Opp));
    let tr = p.trace.take().unwrap();
    assert_eq!(tr.len(), 3, "{:?}", tr);
    assert!(!tr[0].mover_is_us && tr[1].mover_is_us && !tr[2].mover_is_us);
    assert_eq!(tr[0].bag, 0);

    // (c) The opponent passes (no tiles), our move empties the bag: we
    // own it; the opponent (still no tiles) passes their last turn and
    // we play the final move.
    let mut p = Playout::new(
        &dict,
        b.clone(),
        [tiles_of("RES"), vec![]],
        tiles_of("D"),
        3,
    );
    p.trace = Some(vec![]);
    let out = p.run_with(Player::Opp, 20, &mut |_, _| {});
    assert!(out.finished);
    assert_eq!(p.final_turn_owner, Some(Player::Us));
    let tr = p.trace.take().unwrap();
    // Our emptying move, then our final move; the opponent's turns are
    // passes (recorded in the greedy phase, silent on their last turn)
    // that never change the spread.
    assert_eq!(tr.iter().filter(|t| t.mover_is_us).count(), 2, "{:?}", tr);
    for w in tr.windows(2) {
        if !w[1].mover_is_us {
            assert_eq!(w[0].spread, w[1].spread);
        }
    }
}

/// Exchange candidates draw from the pre-move pool and the exchanged
/// tiles end up back in the bag, so every playout conserves tiles; the
/// evaluator handles them alongside plays.
#[test]
fn exchange_candidates_evaluate() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let state =
        SolverState::from_cgp("15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15 QVWXZ?A")
            .unwrap();
    let candidates = state.get_suggestions(&dict);
    let results = MctsEvaluator::new(MctsConfig {
        num_candidates: 12,
        iterations_per_candidate: 8,
        max_plies: 2,
        parallel: false,
        score_diff: 0,
        time_budget: None,
        cancel: None,
    })
    .evaluate(&state, &candidates, &dict);
    assert!(results
        .iter()
        .any(|r| matches!(r.mv.kind, MoveKind::Exchange(_))));
    for r in &results {
        assert!(r.mcts_equity.is_finite());
        assert!((0.0..=1.0).contains(&r.win_rate));
        assert!((0.0..=1.0).contains(&r.opp_bingo_rate));
    }
    // Direct check of the dealing for an exchange candidate.
    let ex = candidates
        .iter()
        .find(|m| matches!(m.kind, MoveKind::Exchange(_)))
        .unwrap();
    let sim = CandidateSim::new(&state, ex, 0, None);
    let MoveKind::Exchange(exchanged) = &ex.kind else {
        unreachable!()
    };
    assert_eq!(sim.draw_count, exchanged.len());
    // Pool = unseen before the move = 100 - 9 board - 7 rack = 84.
    assert_eq!(sim.pool.len(), 84);
    assert_eq!(sim.bag_size, 77);
}

/// `best_last_turn` prunes with a lower bound on the reply (a current-board
/// reply our move leaves intact). It must return exactly the value of the
/// unpruned 2-ply search, on many positions, racks with blanks included.
#[test]
fn bounded_last_turn_matches_exhaustive_search() {
    let Some(dict) = crate::datafiles::test_data::dictionary() else {
        return;
    };
    let exhaustive = |p: &Playout, who: Player| -> i32 {
        let reply =
            |board: &Board| Solver::new(board, &dict).best_score(p.rack(who.other())) as i32;
        let mut best = -reply(&p.board);
        for mv in Solver::new(&p.board, &dict).find_moves_for(p.rack(who)) {
            let mut board = p.board.clone();
            for (r, c, t) in &mv.tiles_used {
                board.place_tile(*r, *c, *t);
            }
            best = best.max(mv.score as i32 - reply(&board));
        }
        best
    };
    let mut checked = 0;
    for seed in 0..8u64 {
        let mut game = Playout::new_game(&dict, seed);
        let mut who = Player::Us;
        for _ in 0..30 {
            if game.bag.len() < 30 {
                for side in [Player::Us, Player::Opp] {
                    let (_, value) = game.best_last_turn(side);
                    assert_eq!(value, exhaustive(&game, side), "seed {seed}");
                    checked += 1;
                }
            }
            let Some(mv) = game.best_by_score(who) else {
                break;
            };
            game.commit(who, &mv);
            if game.bag.is_empty() {
                break;
            }
            who = who.other();
        }
    }
    assert!(checked > 40, "only {checked} positions checked");
}
