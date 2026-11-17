//! The `ravyn` command-line tool, run as a process.

use std::process::{Command, Output};

use ravyn::game::SolverState;
use ravyn::rules::CROSSPLAY;
use ravyn::solver::Solver;
use ravyn::tile::{Tile, RACK_SIZE};

mod common;

const BEECH: &str = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15";
const END: &str = "3N3Q3M3/3OP2UNTEACH1/3BE2O3N3/3LA2T2MYOPE/2FeR2H1LO4/2O6AB4/2R1F4U3De/2TRIVET1D3AY/4LA1A1IODIZE/4KI1JINX2ES/5R1E1G4O/OUTIES1S3G2R/11L2E/8WAHINES/2EGAlITE2A3";

fn ravyn(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ravyn"))
        .args(args)
        .arg("--data")
        .arg(common::root())
        .output()
        .expect("ravyn runs")
}

fn stdout(out: &Output) -> String {
    assert!(
        out.status.success(),
        "ravyn failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).expect("UTF-8")
}

#[test]
fn help_and_errors() {
    let help = Command::new(env!("CARGO_BIN_EXE_ravyn"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("preendgame"));

    let bad = ravyn(&["nonsense", "x"]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("unknown command"));

    let bad = ravyn(&["gen", "not a board"]);
    assert!(!bad.status.success());
}

#[test]
fn gen_lists_the_best_play_first() {
    if common::dictionary().is_none() {
        return;
    }
    let out = stdout(&ravyn(&[
        "gen",
        &format!("{BEECH} AEINRST/ 0/0 0"),
        "--top",
        "3",
    ]));
    assert!(out.starts_with("2633 candidates"), "{out}");
    let first = out.lines().nth(2).expect("a play");
    assert!(first.starts_with("D8 "), "{first}");
}

#[test]
fn sim_ranks_candidates() {
    if common::dictionary().is_none() {
        return;
    }
    let out = stdout(&ravyn(&[
        "sim",
        &format!("{BEECH} AEINRST/ 0/0 0"),
        "--candidates",
        "4",
        "--seconds",
        "1",
    ]));
    // Four candidates plus the pass, each with a row.
    assert!(out.starts_with("5 candidates"), "{out}");
    assert_eq!(out.lines().count(), 2 + 5, "{out}");
}

#[test]
fn endgame_uses_the_known_opponent_rack() {
    if common::dictionary().is_none() {
        return;
    }
    let out = stdout(&ravyn(&[
        "endgame",
        &format!("{END} RCVSDW/AEEIST? 300/310 0"),
        "--top",
        "1",
    ]));
    let best = out.lines().nth(2).expect("a move");
    // Their best reply is a bingo with the blank they were given.
    assert!(
        best.contains("-21"),
        "final margin from the CGP scores: {best}"
    );
}

#[test]
fn infer_reports_the_kept_letters() {
    // The rack count depends on the leave values: skip without the table.
    if common::dictionary_with_leaves().is_none() {
        return;
    }
    let out = stdout(&ravyn(&[
        "infer",
        &format!("{BEECH} AEILNOS/"),
        "--played",
        "QUOD",
        "--score",
        "43",
    ]));
    assert!(out.starts_with("77234 racks consistent"), "{out}");
}

/// A self-played position with one tile in the bag, solved by the tool.
#[test]
fn preendgame_solves_one_tile_in_the_bag() {
    let Some(dict) = common::dictionary() else {
        return;
    };
    let mut rng = fastrand::Rng::with_seed(3);
    let cgp = (0..200)
        .find_map(|_| {
            let mut bag = CROSSPLAY.full_tile_set();
            rng.shuffle(&mut bag);
            let mut racks = [
                bag.split_off(bag.len() - RACK_SIZE),
                bag.split_off(bag.len() - RACK_SIZE),
            ];
            let mut state = SolverState::new();
            let mut to_move = 0;
            while bag.len() > 1 {
                let mv = Solver::new(&state.board, &dict).best_move_by_equity(&racks[to_move])?;
                for (r, c, t) in &mv.tiles_used {
                    state.place_tile(*r, *c, *t);
                }
                for t in mv.tiles_from_rack() {
                    let i = racks[to_move].iter().position(|x: &Tile| {
                        x.is_blank == t.is_blank && (t.is_blank || x.letter == t.letter)
                    })?;
                    racks[to_move].swap_remove(i);
                }
                let draw = mv.tiles_used.len().min(bag.len());
                racks[to_move].extend(bag.split_off(bag.len() - draw));
                to_move = 1 - to_move;
            }
            (bag.len() == 1).then(|| {
                for t in &racks[to_move] {
                    state.add_to_rack(if t.is_blank { '?' } else { t.letter });
                }
                assert_eq!(state.unseen_tile_count(), RACK_SIZE + 1);
                state.to_cgp((0, 0))
            })
        })
        .expect("a position with one tile in the bag");
    let out = stdout(&ravyn(&["preendgame", &cgp, "--top", "3"]));
    assert!(
        out.contains("solved exactly over each possible last tile"),
        "{out}"
    );
    assert!(out.lines().count() >= 3, "{out}");
}

/// Under Scrabble rules, `endgame` runs the negamax search to the end of a
/// self-played game and reports whether it solved it.
#[test]
fn scrabble_endgame_is_searched_to_the_end() {
    use ravyn::rules::SCRABBLE;
    let Some(dict) = common::dictionary() else {
        return;
    };
    let dict = dict.for_rules(&SCRABBLE);
    // Greedy self-play until the bag empties, with small final racks.
    let cgp = (0..100u64)
        .find_map(|seed| {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut bag = SCRABBLE.full_tile_set();
            rng.shuffle(&mut bag);
            let mut racks = [
                bag.split_off(bag.len() - RACK_SIZE),
                bag.split_off(bag.len() - RACK_SIZE),
            ];
            let mut state = SolverState::new();
            state.board = ravyn::board::Board::with_rules(&SCRABBLE);
            let mut to_move = 0;
            while !bag.is_empty() {
                let mv = Solver::new(&state.board, &dict).best_move_by_equity(&racks[to_move])?;
                for (r, c, t) in &mv.tiles_used {
                    state.place_tile(*r, *c, *t);
                }
                for t in mv.tiles_from_rack() {
                    let i = racks[to_move].iter().position(|x: &Tile| {
                        x.is_blank == t.is_blank && (t.is_blank || x.letter == t.letter)
                    })?;
                    racks[to_move].swap_remove(i);
                }
                let draw = mv.tiles_used.len().min(bag.len());
                racks[to_move].extend(bag.split_off(bag.len() - draw));
                to_move = 1 - to_move;
            }
            (racks[1 - to_move].len() <= 3).then(|| {
                for t in &racks[to_move] {
                    state.add_to_rack(if t.is_blank { '?' } else { t.letter });
                }
                state.to_cgp((0, 0))
            })
        })
        .expect("a self-played endgame with a short opponent rack");
    let out = stdout(&ravyn(&[
        "endgame",
        &cgp,
        "--rules",
        "scrabble",
        "--seconds",
        "20",
        "--top",
        "3",
    ]));
    assert!(out.starts_with("solved exactly"), "{out}");
}
