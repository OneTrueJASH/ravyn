//! The engine against RefEngine's answers, recorded in
//! `fixtures/reference_movegen.txt` before RefEngine was removed from the
//! repository: move generation, the same plays with the same scores. (The
//! fixture's equity and inference records depended on RefEngine's leave table,
//! which the engine no longer uses.)
//!
//! Needs the dictionary and leave table from the workspace root; skips
//! when they are missing or Git LFS pointers.

use std::collections::HashMap;
use std::path::Path;

use ravyn::board::Direction;
use ravyn::dictionary::Dictionary;
use ravyn::game::SolverState;
use ravyn::solver::{Move, MoveKind};

mod common;

/// The dictionary, or `None` (skip) without data.
fn dictionary() -> Option<Dictionary> {
    common::dictionary()
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
