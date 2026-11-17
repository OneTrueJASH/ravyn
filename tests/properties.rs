//! Property tests: CGP round trips, tile conservation, and the place/undo
//! invariants of applying moves, over random positions.

use std::path::Path;
use std::sync::OnceLock;

use proptest::prelude::*;

use cpsolver2::board::BOARD_SIZE;
use cpsolver2::datafiles::{file_status, FileStatus, WORDS_FST};
use cpsolver2::dictionary::{load_dictionary_from, Dictionary};
use cpsolver2::game::SolverState;
use cpsolver2::rules::CROSSPLAY;
use cpsolver2::solver::MoveKind;
use cpsolver2::tile::{Tile, RACK_SIZE};

/// A position drawn without replacement from the real tile set: `board`
/// tiles on random squares (blanks given random letters) and up to seven
/// in the rack. The board need not spell words.
fn position() -> impl Strategy<Value = SolverState> {
    (
        any::<u64>(),
        0usize..=90,
        0usize..=RACK_SIZE,
        prop::collection::vec(0usize..BOARD_SIZE * BOARD_SIZE, 90),
    )
        .prop_map(|(seed, on_board, in_rack, squares)| {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut bag = CROSSPLAY.full_tile_set();
            rng.shuffle(&mut bag);
            let mut state = SolverState::new();
            let mut placed = 0;
            for sq in squares {
                if placed == on_board {
                    break;
                }
                let (r, c) = (sq / BOARD_SIZE, sq % BOARD_SIZE);
                if state.get_tile(r, c).is_some() {
                    continue;
                }
                let t = bag.pop().expect("100 tiles");
                let tile = if t.is_blank {
                    Tile::blank((b'A' + rng.u8(0..26)) as char)
                } else {
                    t
                };
                state.place_tile(r, c, tile);
                placed += 1;
            }
            for _ in 0..in_rack {
                let t = bag.pop().expect("tiles left");
                state.add_to_rack(if t.is_blank { '?' } else { t.letter });
            }
            state
        })
}

fn board_tiles(state: &SolverState) -> Vec<(usize, usize, Tile)> {
    let mut v = Vec::new();
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if let Some(t) = state.get_tile(r, c) {
                v.push((r, c, *t));
            }
        }
    }
    v
}

fn rack_key(state: &SolverState) -> Vec<(bool, char)> {
    let mut k: Vec<(bool, char)> = state
        .rack
        .tiles()
        .iter()
        .map(|t| (t.is_blank, if t.is_blank { '?' } else { t.letter }))
        .collect();
    k.sort_unstable();
    k
}

fn dictionary() -> Option<&'static Dictionary> {
    static DICT: OnceLock<Option<Dictionary>> = OnceLock::new();
    DICT.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        (file_status(&root.join(WORDS_FST)) == FileStatus::Present)
            .then(|| load_dictionary_from(root).expect("dictionary"))
    })
    .as_ref()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Writing a position as CGP and reading it back gives the same board
    /// (letters and blank flags) and rack, and the scores survive.
    #[test]
    fn cgp_round_trip(state in position(), ours in 0u32..700, theirs in 0u32..700) {
        let cgp = state.to_cgp((ours, theirs));
        let back = SolverState::from_cgp(&cgp).expect("own CGP parses");
        prop_assert_eq!(board_tiles(&back), board_tiles(&state));
        prop_assert_eq!(rack_key(&back), rack_key(&state));
        prop_assert_eq!(back.to_cgp((ours, theirs)), cgp);
    }

    /// Board + rack + unseen always accounts for the 100 tiles, letter by
    /// letter.
    #[test]
    fn tiles_are_conserved(state in position()) {
        let unseen = state.unseen_tile_count();
        let board = board_tiles(&state).len();
        let rack = state.rack.tiles().len();
        prop_assert_eq!(board + rack + unseen, 100);
        let mut counts = state.unseen_tiles();
        for (_, _, t) in board_tiles(&state) {
            *counts.entry(if t.is_blank { '?' } else { t.letter }).or_default() += 1;
        }
        for t in state.rack.tiles() {
            *counts.entry(if t.is_blank { '?' } else { t.letter }).or_default() += 1;
        }
        let mut full = std::collections::HashMap::new();
        for t in CROSSPLAY.full_tile_set() {
            *full.entry(if t.is_blank { '?' } else { t.letter }).or_insert(0u32) += 1;
        }
        prop_assert_eq!(counts, full);
    }

    /// Placing a tile and removing it again leaves the board as it was.
    #[test]
    fn place_then_remove_restores_the_board(
        state in position(),
        sq in 0usize..BOARD_SIZE * BOARD_SIZE,
        letter in 0u8..26,
        blank in any::<bool>(),
    ) {
        let (r, c) = (sq / BOARD_SIZE, sq % BOARD_SIZE);
        prop_assume!(state.get_tile(r, c).is_none());
        let mut s = state.clone();
        let letter = (b'A' + letter) as char;
        s.place_tile(r, c, if blank { Tile::blank(letter) } else { Tile::new(letter) });
        let t = s.get_tile(r, c).copied().expect("placed");
        prop_assert_eq!(t.is_blank, blank);
        s.remove_tile(r, c);
        prop_assert_eq!(board_tiles(&s), board_tiles(&state));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Applying any generated play moves exactly its tiles from the rack to
    /// the board; taking them back restores the position.
    #[test]
    fn apply_move_moves_exactly_its_tiles(seed in any::<u64>(), rack_size in 1usize..=RACK_SIZE) {
        let Some(dict) = dictionary() else {
            return Ok(());
        };
        // A real position: a few greedy plays from a fresh bag.
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut bag = CROSSPLAY.full_tile_set();
        rng.shuffle(&mut bag);
        let mut state = SolverState::new();
        for _ in 0..rng.usize(0..6) {
            state.clear_rack();
            for t in bag.iter().rev().take(RACK_SIZE) {
                state.add_to_rack(if t.is_blank { '?' } else { t.letter });
            }
            let Some(mv) = state
                .get_suggestions(dict)
                .into_iter()
                .find(|m| m.kind == MoveKind::Play)
            else {
                break;
            };
            for (_, _, t) in &mv.tiles_used {
                let i = bag
                    .iter()
                    .rposition(|b| b.is_blank == t.is_blank && (t.is_blank || b.letter == t.letter))
                    .expect("tile came from the bag");
                bag.remove(i);
            }
            state.apply_move(&mv);
        }
        state.clear_rack();
        for t in bag.iter().take(rack_size) {
            state.add_to_rack(if t.is_blank { '?' } else { t.letter });
        }

        let before_board = board_tiles(&state);
        let before_rack = rack_key(&state);
        for mv in state.get_suggestions(dict).iter().take(40) {
            let mut s = state.clone();
            s.apply_move(mv);
            let added = board_tiles(&s).len() - before_board.len();
            let used = before_rack.len() - s.rack.tiles().len();
            prop_assert_eq!(added, mv.tiles_used.len());
            prop_assert_eq!(used, mv.tiles_from_rack().len());
            // Undo: take the placed tiles back into the rack.
            for (r, c, t) in &mv.tiles_used {
                s.remove_tile(*r, *c);
                s.add_to_rack(if t.is_blank { '?' } else { t.letter });
            }
            if let MoveKind::Exchange(tiles) = &mv.kind {
                for t in tiles {
                    s.add_to_rack(if t.is_blank { '?' } else { t.letter });
                }
            }
            prop_assert_eq!(board_tiles(&s), before_board.clone());
            prop_assert_eq!(rack_key(&s), before_rack.clone());
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// The parsers the fuzz targets cover never panic on arbitrary input
    /// (a stable-Rust stand-in for `cargo fuzz`, which needs nightly).
    #[test]
    fn parsers_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..256), text in "\\PC{0,120}") {
        let _ = cpsolver2::leaves::LeaveValues::parse(&bytes);
        if let Ok(state) = SolverState::from_cgp(&text) {
            let cgp = state.to_cgp((0, 0));
            let again = SolverState::from_cgp(&cgp).expect("own CGP parses");
            prop_assert_eq!(again.to_cgp((0, 0)), cgp);
        }
    }

    /// CGP-shaped input (rows of digits and letters) exercises the board
    /// reader past its first checks.
    #[test]
    fn cgp_shaped_input_never_panics(
        rows in prop::collection::vec("[0-9A-Za-z]{0,17}", 15),
        rack in "[A-Z?]{0,9}",
    ) {
        let text = format!("{} {}/ 0/0 0", rows.join("/"), rack);
        if let Ok(state) = SolverState::from_cgp(&text) {
            prop_assert!(state.rack.tiles().len() <= RACK_SIZE);
            let cgp = state.to_cgp((0, 0));
            prop_assert_eq!(SolverState::from_cgp(&cgp).unwrap().to_cgp((0, 0)), cgp);
        }
    }
}
