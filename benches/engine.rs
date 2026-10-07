//! Engine benchmarks: move generation, playouts and the endgame solver.
//!
//! Run with `cargo bench --bench engine`. Positions are the ones the test
//! suite uses (EMPTY, MID, END, and BEECH for blank-heavy racks).

use std::path::Path;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use cpsolver2::dictionary::{load_dictionary_from, Dictionary};
use cpsolver2::endgame::EndgameSolver;
use cpsolver2::game::SolverState;
use cpsolver2::mcts::{MctsConfig, MctsEvaluator};
use cpsolver2::solver::{MoveKind, Solver};
use cpsolver2::tile::Tile;

const EMPTY: &str = "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15";
const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";
const BEECH: &str = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15";
const END: &str = "3N3Q3M3/3OP2UNTEACH1/3BE2O3N3/3LA2T2MYOPE/2FeR2H1LO4/2O6AB4/2R1F4U3De/2TRIVET1D3AY/4LA1A1IODIZE/4KI1JINX2ES/5R1E1G4O/OUTIES1S3G2R/11L2E/8WAHINES/2EGAlITE2A3";

fn dictionary() -> Dictionary {
    load_dictionary_from(Path::new(env!("CARGO_MANIFEST_DIR"))).expect("dictionary")
}

fn state(board: &str, rack: &str) -> SolverState {
    SolverState::from_cgp(&format!("{} {}", board, rack)).expect("valid CGP")
}

fn movegen(c: &mut Criterion) {
    let dict = dictionary();
    let mut group = c.benchmark_group("movegen");
    for (label, board, rack) in [
        ("empty_AEINRST", EMPTY, "AEINRST"),
        ("mid_JSIRATU", MID, "JSIRATU"),
        ("beech_AEIO?ST", BEECH, "AEIO?ST"),
        ("beech_??EINRS", BEECH, "??EINRS"),
        ("end_RCVSDW", END, "RCVSDW"),
    ] {
        let s = state(board, rack);
        let solver = Solver::new(&s.board, &dict);
        group.bench_function(BenchmarkId::new("find_moves", label), |b| {
            b.iter(|| solver.find_moves(&s.rack))
        });
    }
    group.finish();
}

/// Playouts per second for a fixed candidate set: the top candidates by
/// static equity, a fixed playout count and no time budget. The count of
/// playouts actually run (pruning stops some candidates early) is measured
/// once and used as the throughput unit, so criterion reports playouts/s.
fn playouts(c: &mut Criterion) {
    let dict = dictionary();
    let mut group = c.benchmark_group("playouts");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    for (label, board, rack, plies) in [
        ("beech_AEINRST_2ply", BEECH, "AEINRST", 2),
        ("beech_AEIO?ST_2ply", BEECH, "AEIO?ST", 2),
        ("mid_JSIRATU_4ply", MID, "JSIRATU", 4),
    ] {
        let s = state(board, rack);
        let mut candidates = s.get_suggestions(&dict);
        candidates.truncate(20);
        let config = MctsConfig {
            num_candidates: 20,
            iterations_per_candidate: 64,
            max_plies: plies,
            parallel: true,
            score_diff: 0,
            time_budget: None,
            cancel: None,
        };
        let evaluator = MctsEvaluator::new(config);
        let total: usize = evaluator
            .evaluate(&s, &candidates, &dict)
            .iter()
            .map(|r| r.iterations)
            .sum();
        group.throughput(Throughput::Elements(total as u64));
        group.bench_function(BenchmarkId::new("evaluate", label), |b| {
            b.iter(|| evaluator.evaluate(&s, &candidates, &dict))
        });
    }
    group.finish();
}

fn endgame(c: &mut Criterion) {
    let dict = dictionary();
    let s = state(END, "RCVSDW");
    let our: Vec<Tile> = s.rack.tiles();
    // An opponent rack for the 2-ply search (tile conservation does not
    // matter to the solver).
    let opp: Vec<Tile> = "AEEIST?"
        .chars()
        .map(|c| {
            if c == '?' {
                Tile::blank(' ')
            } else {
                Tile::new(c)
            }
        })
        .collect();
    let solver = EndgameSolver::new(&dict);
    let mut group = c.benchmark_group("endgame");
    group.sample_size(10);
    group.bench_function("solve_end_RCVSDW", |b| {
        b.iter(|| {
            let r = solver.solve(&s.board, &our, &opp, true);
            assert!(r.iter().any(|m| m.mv.kind == MoveKind::Pass));
            r
        })
    });
    group.finish();
}

criterion_group!(benches, movegen, playouts, endgame);
criterion_main!(benches);
