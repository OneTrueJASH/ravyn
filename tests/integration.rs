//! Integration tests for the Crossplay solver.
//!
//! These tests exercise end-to-end flows that cross module boundaries,
//! verifying that the solver, board, rack, endgame, and dictionary
//! components work together correctly.

use cpsolver2::board::Board;
use cpsolver2::dictionary::{load_default_dictionary, Dictionary};
use cpsolver2::endgame::EndgameSolver;
use cpsolver2::game::SolverState;
use cpsolver2::tile::Tile;

/// Load the dictionary once for tests that need it.
fn dict() -> Dictionary {
    load_default_dictionary().expect("dictionary must be loadable for integration tests")
}

// ---------------------------------------------------------------------------
// 1. Full solve flow
// ---------------------------------------------------------------------------

#[test]
fn test_full_solve_flow() {
    let dictionary = dict();

    // Set up a board with "CAT" placed horizontally at row 7.
    let mut state = SolverState::new();
    state.place_tile(7, 6, 'C', false);
    state.place_tile(7, 7, 'A', false);
    state.place_tile(7, 8, 'T', false);

    // Give ourselves a rack of tiles.
    for ch in ['S', 'E', 'R', 'I', 'N', 'G', 'D'] {
        state.add_to_rack(ch);
    }

    let suggestions = state.get_suggestions(&dictionary);

    // We should get a non-empty list of legal moves.
    assert!(
        !suggestions.is_empty(),
        "Solver should return at least one valid move"
    );

    // Every returned move should have a positive score and non-empty tiles_used.
    for mv in &suggestions {
        assert!(
            mv.score > 0,
            "Move '{}' should have a positive score",
            mv.word
        );
        assert!(
            !mv.tiles_used.is_empty(),
            "Move '{}' should place at least one tile",
            mv.word
        );
    }

    // Moves should be sorted by equity (descending).
    for pair in suggestions.windows(2) {
        assert!(
            pair[0].equity >= pair[1].equity,
            "Moves should be sorted by equity: {} ({}) >= {} ({})",
            pair[0].word,
            pair[0].equity,
            pair[1].word,
            pair[1].equity,
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Endgame solve flow
// ---------------------------------------------------------------------------

#[test]
fn test_endgame_solve_flow() {
    let dictionary = dict();

    // Create a board with "CAT" placed so there are anchor points.
    let mut board = Board::new();
    board.place_tile(7, 7, Tile::new('C'));
    board.place_tile(7, 8, Tile::new('A'));
    board.place_tile(7, 9, Tile::new('T'));

    let our_rack = vec![Tile::new('S'), Tile::new('E')];
    let opp_rack = vec![Tile::new('R'), Tile::new('D')];

    let solver = EndgameSolver::new(&dictionary);
    let results = solver.solve(&board, &our_rack, &opp_rack, true);

    assert!(
        !results.is_empty(),
        "Endgame solver should return at least one move"
    );

    // Each result's endgame_value must equal our_score - opponent_best_score.
    for r in &results {
        let expected = r.mv.score as i32 - r.opponent_best_score as i32;
        assert_eq!(
            r.endgame_value, expected,
            "Endgame value for '{}' should be our_score({}) - opp_best({})",
            r.mv.word, r.mv.score, r.opponent_best_score,
        );
    }

    // Results should be sorted by endgame_value descending.
    for pair in results.windows(2) {
        assert!(
            pair[0].endgame_value >= pair[1].endgame_value,
            "Endgame results should be sorted by value: {} >= {}",
            pair[0].endgame_value,
            pair[1].endgame_value,
        );
    }

    // The principal variation for a 2-ply solve should contain our move (player 0).
    let best = &results[0];
    assert!(!best.pv.is_empty(), "PV should contain at least our move");
    assert_eq!(best.pv[0].player, 0, "First PV entry should be our move");
}

// ---------------------------------------------------------------------------
// 3. Move apply and resolve
// ---------------------------------------------------------------------------

#[test]
fn test_move_apply_and_resolve() {
    let dictionary = dict();

    // Place "CAT" on the board, give ourselves an "S" to extend to "CATS".
    let mut state = SolverState::new();
    state.place_tile(7, 7, 'C', false);
    state.place_tile(7, 8, 'A', false);
    state.place_tile(7, 9, 'T', false);

    state.add_to_rack('S');
    state.add_to_rack('E');
    state.add_to_rack('R');

    let moves = state.get_suggestions(&dictionary);
    assert!(!moves.is_empty(), "Should find moves extending CAT");

    // Pick the top move by equity and apply it.
    let top_move = &moves[0];
    let tiles_placed_count = top_move.tiles_used.len();
    let rack_before = state.rack.len();

    state.apply_move(top_move);

    // Board should have the new tiles.
    for &(row, col, ref tile) in &top_move.tiles_used {
        let board_tile = state
            .get_tile(row, col)
            .expect("Tile should be on board after apply");
        assert_eq!(
            board_tile.letter, tile.letter,
            "Board tile at ({},{}) should match the move",
            row, col,
        );
    }

    // Rack should have fewer tiles.
    assert_eq!(
        state.rack.len(),
        rack_before - tiles_placed_count,
        "Rack should shrink by the number of tiles placed"
    );

    // Generate moves again from the new position — the solver should still work.
    let new_moves = state.get_suggestions(&dictionary);
    // With remaining rack tiles and more tiles on the board, we may or may not
    // find moves, but the solver must not panic.
    // If we still have rack tiles, we almost certainly find something.
    if !state.rack.is_empty() {
        assert!(
            !new_moves.is_empty(),
            "With remaining rack tiles on a non-trivial board, should find moves"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Unseen tile tracking
// ---------------------------------------------------------------------------

#[test]
fn test_unseen_tile_tracking() {
    // A fresh state has 100 total tiles, all unseen.
    let mut state = SolverState::new();
    let total_tiles = state.unseen_tile_count(); // Should be 100.
    assert_eq!(total_tiles, 100, "Fresh board should have 100 unseen tiles");

    // Place 3 tiles on the board.
    state.place_tile(7, 7, 'C', false);
    state.place_tile(7, 8, 'A', false);
    state.place_tile(7, 9, 'T', false);

    // Add 2 tiles to the rack.
    state.add_to_rack('S');
    state.add_to_rack('E');

    // Unseen = total - board(3) - rack(2) = 95
    assert_eq!(
        state.unseen_tile_count(),
        95,
        "Unseen should be 100 - 3 board - 2 rack = 95"
    );

    // Apply a move that places the S on the board.
    let dictionary = dict();
    let moves = state.get_suggestions(&dictionary);

    // Find a move that uses at least 1 rack tile.
    let mv = moves
        .iter()
        .find(|m| !m.tiles_used.is_empty())
        .expect("Should find a move that places tiles");

    let unseen_before = state.unseen_tile_count();
    state.apply_move(mv);

    // Tiles moved from rack to board: unseen count should stay the same
    // because unseen = total - board - rack, and board goes up by N while
    // rack goes down by N.
    assert_eq!(
        state.unseen_tile_count(),
        unseen_before,
        "Moving tiles from rack to board should not change unseen count"
    );

    // But total board tiles increased and rack tiles decreased.
    let board_count: u32 = state.count_board_tiles().values().sum();
    let rack_count: u32 = state.count_rack_tiles().values().sum();
    assert_eq!(
        board_count as usize + rack_count as usize + state.unseen_tile_count(),
        100,
        "board + rack + unseen should always equal 100"
    );
}

// ---------------------------------------------------------------------------
// 5. Endgame detection
// ---------------------------------------------------------------------------

#[test]
fn test_endgame_detection() {
    let mut state = SolverState::new();

    // Not endgame at the start.
    assert!(!state.is_endgame(), "Fresh board should not be endgame");

    // Progressively place tiles. We need to get unseen <= 7 (RACK_SIZE).
    // We'll place tiles from the standard distribution until we cross the
    // threshold. Use a simple strategy: place common letters across the board.
    let letters_to_place: Vec<char> = {
        let mut v = Vec::new();
        // Approximation of distribution (excluding blanks counted as '?'):
        // We need to place enough that only 7 unseen remain.
        // Total tiles = 100, so we need board + rack >= 93.
        // We'll place 93 tiles on the board with no rack tiles.
        let dist = [
            ('A', 9),
            ('B', 2),
            ('C', 2),
            ('D', 4),
            ('E', 12),
            ('F', 2),
            ('G', 3),
            ('H', 3),
            ('I', 8),
            ('J', 1),
            ('K', 1),
            ('L', 4),
            ('M', 2),
            ('N', 5),
            ('O', 8),
            ('P', 2),
            ('Q', 1),
            ('R', 6),
            ('S', 5),
            ('T', 6),
            ('U', 3),
            ('V', 2),
            ('W', 2),
            ('X', 1),
            ('Y', 2),
            ('Z', 1),
        ];
        for (ch, count) in dist {
            for _ in 0..count {
                v.push(ch);
            }
        }
        v
    };

    let mut placed = 0;
    let target_placed = 93; // leaves 7 unseen (the 3 blanks + 4 letters)

    let mut became_endgame = false;
    for ch in &letters_to_place {
        if placed >= target_placed {
            break;
        }
        let row = placed / 15;
        let col = placed % 15;
        if row < 15 {
            state.place_tile(row, col, *ch, false);
            placed += 1;
        }

        if state.is_endgame() && !became_endgame {
            became_endgame = true;
            let unseen = state.unseen_tile_count();
            assert!(
                unseen <= 7,
                "Endgame should trigger when unseen <= 7 (RACK_SIZE), got {}",
                unseen,
            );
        }
    }

    assert!(
        state.is_endgame(),
        "After placing {} tiles, should be in endgame (unseen={})",
        placed,
        state.unseen_tile_count(),
    );
    assert!(
        state.unseen_tile_count() <= 7,
        "Unseen tile count ({}) should be <= 7 in endgame",
        state.unseen_tile_count(),
    );
}

// ---------------------------------------------------------------------------
// 6. Blank tile move generation
// ---------------------------------------------------------------------------

#[test]
fn test_blank_tile_move_generation() {
    let dictionary = dict();

    // Place "AT" on the board to give the solver something to hook onto.
    let mut state = SolverState::new();
    state.place_tile(7, 7, 'A', false);
    state.place_tile(7, 8, 'T', false);

    // Give ourselves a blank plus some regular tiles.
    state.add_to_rack('?'); // blank
    state.add_to_rack('E');
    state.add_to_rack('R');
    state.add_to_rack('S');
    state.add_to_rack('I');
    state.add_to_rack('N');
    state.add_to_rack('G');

    let moves = state.get_suggestions(&dictionary);
    assert!(
        !moves.is_empty(),
        "Should find moves with a blank tile in the rack"
    );

    // At least some moves should use the blank tile.
    let blank_moves: Vec<_> = moves
        .iter()
        .filter(|m| m.tiles_used.iter().any(|(_, _, t)| t.is_blank))
        .collect();

    assert!(
        !blank_moves.is_empty(),
        "Some moves should use the blank tile (found {} total moves but none use blank)",
        moves.len(),
    );

    // Blank tiles placed on the board should have value 0 and is_blank = true.
    for mv in &blank_moves {
        for (_, _, tile) in &mv.tiles_used {
            if tile.is_blank {
                assert_eq!(tile.value(), 0, "Blank tile should have 0 point value");
                assert!(
                    tile.letter.is_ascii_uppercase(),
                    "Blank tile should be designated as an uppercase letter, got '{}'",
                    tile.letter,
                );
            }
        }
    }
}
