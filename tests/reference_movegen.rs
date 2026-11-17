//! The engine against RefEngine's answers, recorded in
//! `fixtures/reference_movegen.txt` before RefEngine was removed from the
//! repository: move generation (the same plays with the same scores),
//! static equity (score plus leave table value) and rack inference.
//!
//! Needs the dictionary and leave table from the workspace root; skips
//! when they are missing or Git LFS pointers.

use std::collections::HashMap;
use std::path::Path;

use ravyn::board::Direction;
use ravyn::dictionary::Dictionary;
use ravyn::game::SolverState;
use ravyn::inference::{infer_leave, InferenceSettings};
use ravyn::leaves::bingo_bonus_correction;
use ravyn::rules::CROSSPLAY;
use ravyn::solver::{Move, MoveKind};
use ravyn::tile::Tile;

const BEECH: &str = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15";

mod common;

/// The dictionary with the leave table, or `None` (skip) without data.
fn dictionary() -> Option<Dictionary> {
    common::dictionary_with_leaves()
}

fn fixture() -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference_movegen.txt"),
    )
    .expect("fixture");
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// "8H" (row 8, column H, across) or "H8" (down) as (row, col, dir).
fn parse_coords(coords: &str) -> (usize, usize, Direction) {
    let (letters, digits): (String, String) = coords.chars().partition(|c| c.is_ascii_alphabetic());
    let row = digits.parse::<usize>().expect("row") - 1;
    let col = (letters.as_bytes()[0] - b'A') as usize;
    let dir = if coords.starts_with(|c: char| c.is_ascii_digit()) {
        Direction::Horizontal
    } else {
        Direction::Vertical
    };
    (row, col, dir)
}

/// The move as RefEngine writes it: coordinates and the word with letters
/// from blanks in lowercase.
fn notation(mv: &Move) -> (String, String) {
    let (dr, dc) = mv.direction.delta();
    let word = mv
        .word
        .chars()
        .enumerate()
        .map(|(i, ch)| {
            let r = (mv.row as i32 + dr * i as i32) as usize;
            let c = (mv.col as i32 + dc * i as i32) as usize;
            let blank = mv
                .tiles_used
                .iter()
                .any(|&(pr, pc, t)| pr == r && pc == c && t.is_blank);
            if blank {
                ch.to_ascii_lowercase()
            } else {
                ch
            }
        })
        .collect();
    (mv.coords(), word)
}

/// The play set normalized as the fixture's: best score per placement,
/// single-tile plays keyed by square and letter, sorted lines.
fn canonical_lines(state: &SolverState, plays: &[Move]) -> Vec<String> {
    let mut map: HashMap<(String, String), u32> = HashMap::new();
    for mv in plays {
        let (coords, word) = notation(mv);
        let (row, col, dir) = parse_coords(&coords);
        let (dr, dc) = dir.delta();
        let placed: Vec<(usize, usize, char)> = word
            .chars()
            .enumerate()
            .filter_map(|(i, ch)| {
                let r = (row as i32 + dr * i as i32) as usize;
                let c = (col as i32 + dc * i as i32) as usize;
                state
                    .board
                    .get_tile(r, c)
                    .is_none()
                    .then(|| (r, c, ch.to_ascii_uppercase()))
            })
            .collect();
        let key = if placed.len() == 1 {
            let (r, c, ch) = placed[0];
            (format!("single@{r},{c}"), ch.to_string())
        } else {
            (coords, word.to_uppercase())
        };
        let e = map.entry(key).or_insert(0);
        *e = (*e).max(mv.score);
    }
    let mut lines: Vec<String> = map
        .into_iter()
        .map(|((a, b), v)| format!("{a}\t{b}\t{v}"))
        .collect();
    lines.sort();
    lines
}

/// The engine generates exactly RefEngine's plays, with the same scores, on
/// 32 positions (named and random; at most two blanks, where RefEngine is a
/// reference).
#[test]
fn move_generation_matches_refengine() {
    let Some(dict) = dictionary() else {
        return;
    };
    let mut checked = 0;
    for row in fixture().iter().filter(|r| r[0] == "movegen") {
        let (label, cgp, count, hash) = (&row[1], &row[2], &row[3], &row[4]);
        let state = SolverState::from_cgp(cgp).expect("valid CGP");
        let plays: Vec<Move> = state
            .get_suggestions(&dict)
            .into_iter()
            .filter(|m| m.kind == MoveKind::Play)
            .collect();
        let lines = canonical_lines(&state, &plays);
        assert_eq!(lines.len().to_string(), *count, "{label}: number of plays");
        assert_eq!(
            format!("{:016x}", fnv(&lines.join("\n"))),
            *hash,
            "{label}: plays or scores differ"
        );
        checked += 1;
    }
    assert_eq!(checked, 32);
}

/// Static equity (score plus raw leave table value, the engine's Crossplay
/// bingo correction removed) matches RefEngine's on three opening racks.
#[test]
fn static_equity_matches_refengine() {
    let Some(dict) = dictionary() else {
        return;
    };
    let mut expected: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    for row in fixture().iter().filter(|r| r[0] == "equity") {
        expected
            .entry(row[1].clone())
            .or_default()
            .push((row[2].clone(), row[3].parse().expect("equity")));
    }
    assert_eq!(expected.len(), 3);
    for (rack, values) in &expected {
        let state = SolverState::from_cgp(&format!(
            "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 {rack}"
        ))
        .unwrap();
        let ours: HashMap<String, f32> = state
            .get_suggestions(&dict)
            .iter()
            .filter(|m| m.kind == MoveKind::Play)
            .map(|m| {
                let (coords, word) = notation(m);
                let correction = bingo_bonus_correction(&m.leave, CROSSPLAY.bingo_bonus);
                (
                    format!("{:016x}", fnv(&format!("{coords}\t{word}"))),
                    m.equity - correction,
                )
            })
            .collect();
        let mut compared = 0;
        for (key, eq) in values {
            if let Some(ours) = ours.get(key) {
                assert!(
                    (ours - eq).abs() < 0.02,
                    "{rack}: {ours:.2} vs RefEngine {eq:.2}"
                );
                compared += 1;
            }
        }
        assert!(compared >= 50, "{rack}: only {compared} plays compared");
    }
}

/// Rack inference agrees with RefEngine's to within 0.1 tiles per letter
/// (they differ only in leave values: corrected to Crossplay's 40-point
/// bingo and phase-weighted here, raw in RefEngine).
#[test]
fn inference_matches_refengine() {
    let Some(dict) = dictionary() else {
        return;
    };
    let mut checked = 0;
    for row in fixture().iter().filter(|r| r[0] == "inference") {
        let (opp, ours, coords, word, marginals) = (&row[1], &row[2], &row[3], &row[4], &row[5]);
        let pre = SolverState::from_cgp(&format!("{BEECH} {opp}")).unwrap();
        let mv = pre
            .get_suggestions(&dict)
            .into_iter()
            .find(|m| m.coords() == *coords && m.word == *word)
            .expect("the recorded play");
        let ours: Vec<Tile> = ours.chars().map(Tile::new).collect();
        let engine = infer_leave(
            &pre.board,
            &dict,
            &ours,
            &mv.tiles_from_rack(),
            mv.score,
            &InferenceSettings::default(),
            None,
        )
        .expect("the engine finds consistent racks");
        for item in marginals.split(',') {
            let (letter, expected) = item.split_once(':').expect("letter:expected");
            let letter = letter.chars().next().unwrap();
            let expected: f32 = expected.parse().unwrap();
            let got = engine
                .letter_marginals
                .iter()
                .find(|m| m.letter == letter)
                .map_or(0.0, |m| m.expected);
            assert!(
                (got - expected).abs() < 0.1,
                "{opp} plays {word}: {letter} {got:.3} vs RefEngine {expected:.3}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 3);
}
