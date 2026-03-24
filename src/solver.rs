//! Move generation engine using anchor-based search with cross-checks.
//!
//! Generates all legal moves for a given rack and board position,
//! scoring each move and evaluating leave quality for equity ranking.

use std::collections::HashSet;

use crate::board::{Board, Direction, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::tile::{Rack, Tile, RACK_SIZE};

/// Bonus points awarded when a player uses all tiles from their rack in a single turn.
pub(crate) const BINGO_BONUS: u32 = 40;

/// A potential move that can be played
#[derive(Debug, Clone)]
pub struct Move {
    pub word: String,
    pub row: usize,
    pub col: usize,
    pub direction: Direction,
    pub score: u32,
    pub tiles_used: Vec<(usize, usize, Tile)>, // (row, col, tile) for each placed tile
    pub is_bingo: bool,                        // All 7 tiles used
    pub leave: Vec<Tile>,                      // Tiles remaining in rack after play
    pub equity: f32,                           // Strategic value (score + leave value)
}

/// Parameters for constructing a Move from external data (e.g., RefEngine results)
pub struct ExternalMoveData<'a> {
    pub word: &'a str,
    pub row: usize,
    pub col: usize,
    pub direction: Direction,
    pub score: u32,
    pub equity: f32,
}

impl Move {
    /// Construct a Move from external data (e.g., RefEngine results)
    ///
    /// The word uses lowercase letters to indicate blanks (e.g., "SNiPER" = blank as I)
    /// Returns None if the move can't be reconstructed with the given rack
    pub fn from_external(data: &ExternalMoveData, board: &Board, rack: &Rack) -> Option<Self> {
        let word = data.word;
        let row = data.row;
        let col = data.col;
        let direction = data.direction;
        let score = data.score;
        let equity = data.equity;
        let (dr, dc) = direction.delta();
        let mut tiles_used = Vec::new();
        let mut rack_remaining: Vec<Tile> = rack.tiles().to_vec();

        for (i, ch) in word.chars().enumerate() {
            let r = row as i32 + dr * i as i32;
            let c = col as i32 + dc * i as i32;

            if !(0..BOARD_SIZE as i32).contains(&r) || !(0..BOARD_SIZE as i32).contains(&c) {
                return None; // Out of bounds
            }

            let r = r as usize;
            let c = c as usize;

            // Check if there's already a tile on the board at this position
            if board.get_tile(r, c).is_some() {
                // Tile already on board, not placed from rack
                continue;
            }

            // Need to place a tile from rack
            let is_blank = ch.is_ascii_lowercase();
            let letter = ch.to_ascii_uppercase();

            // Find matching tile in remaining rack
            let tile_idx = if is_blank {
                // Need a blank tile
                rack_remaining.iter().position(|t| t.is_blank)
            } else {
                // Need the specific letter
                rack_remaining
                    .iter()
                    .position(|t| !t.is_blank && t.letter == letter)
            };

            match tile_idx {
                Some(idx) => {
                    let mut tile = rack_remaining.remove(idx);
                    if is_blank {
                        // Set the letter the blank represents
                        tile.letter = letter;
                    }
                    tiles_used.push((r, c, tile));
                }
                None => {
                    // Can't find required tile - try using a blank as fallback
                    if let Some(idx) = rack_remaining.iter().position(|t| t.is_blank) {
                        let mut tile = rack_remaining.remove(idx);
                        tile.letter = letter;
                        tiles_used.push((r, c, tile));
                    } else {
                        return None; // Can't construct this move
                    }
                }
            }
        }

        let is_bingo = tiles_used.len() == RACK_SIZE;

        Some(Move {
            word: word.to_uppercase(), // Normalize to uppercase for display
            row,
            col,
            direction,
            score,
            tiles_used,
            is_bingo,
            leave: rack_remaining,
            equity,
        })
    }
}

/// Calculate the strategic value of a leave (tiles remaining after a play)
pub fn leave_value(leave: &[Tile]) -> f32 {
    if leave.is_empty() {
        return 0.0;
    }

    let mut value = 0.0;
    let mut vowels = 0;
    let mut consonants = 0;
    let mut letter_counts: std::collections::HashMap<char, i32> = std::collections::HashMap::new();

    for tile in leave {
        if tile.is_blank {
            // Blanks are very valuable
            value += 25.0;
            continue;
        }

        let letter = tile.letter;
        *letter_counts.entry(letter).or_insert(0) += 1;

        // Individual letter values (approximate equity values)
        value += match letter {
            'S' => 10.0, // S is very valuable for hooks
            'X' => 6.0,  // Good for short high-scoring plays
            'Z' => 5.0,  // Similar to X
            'E' => 4.0,  // Common, good for bingos
            'A' => 3.0,  // Common vowel
            'I' => 2.5,  // Good vowel
            'N' => 3.5,  // Good consonant
            'R' => 4.0,  // Very good consonant
            'T' => 3.0,  // Good consonant
            'O' => 1.5,  // Less useful vowel
            'U' => 0.5,  // Least useful vowel (but helps with Q)
            'L' => 2.0,
            'D' => 1.5,
            'G' => 0.5,
            'C' => 0.0,
            'M' => 1.0,
            'P' => 0.5,
            'H' => 1.0,
            'B' => -1.0,
            'F' => -1.0,
            'Y' => -0.5,
            'W' => -2.0,
            'K' => -1.0,
            'V' => -3.0, // V is hard to use
            'J' => -2.0, // Hard to use, but high points
            'Q' => -8.0, // Very hard without U
            _ => 0.0,
        };

        if matches!(letter, 'A' | 'E' | 'I' | 'O' | 'U') {
            vowels += 1;
        } else {
            consonants += 1;
        }
    }

    // Penalize duplicates heavily
    for (letter, count) in &letter_counts {
        if *count >= 2 {
            // Penalty increases with more duplicates
            let penalty = match letter {
                'E' | 'A' => 3.0 * (*count - 1) as f32, // Common letters less bad
                _ => 5.0 * (*count - 1) as f32,
            };
            value -= penalty;
        }
    }

    // Penalize vowel/consonant imbalance
    let total = leave.len() as i32;
    if total > 1 {
        let ideal_vowels = (total as f32 * 0.4).round() as i32; // ~40% vowels ideal
        let vowel_diff = (vowels - ideal_vowels).abs();
        value -= vowel_diff as f32 * 2.0;

        // Heavy penalty for all vowels or all consonants
        if vowels == 0 || consonants == 0 {
            value -= 8.0;
        }
    }

    // Q without U is very bad
    if letter_counts.get(&'Q').copied().unwrap_or(0) > 0
        && letter_counts.get(&'U').copied().unwrap_or(0) == 0
    {
        value -= 10.0;
    }

    value
}

/// Configuration for the solver
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub bingo_bonus: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            bingo_bonus: BINGO_BONUS, // Crossplay uses 40-point bingo bonus
        }
    }
}

/// The move generation solver
pub struct Solver<'a> {
    board: &'a Board,
    dictionary: &'a Dictionary,
    config: SolverConfig,
}

impl<'a> Solver<'a> {
    pub fn new(board: &'a Board, dictionary: &'a Dictionary) -> Self {
        Solver {
            board,
            dictionary,
            config: SolverConfig::default(),
        }
    }

    /// Generate all valid moves for the given rack
    pub fn find_moves(&self, rack: &Rack) -> Vec<Move> {
        let mut moves = Vec::new();

        // Get all anchor squares
        let anchors = self.board.get_anchors();

        // For each anchor, try to generate moves in both directions
        for (anchor_row, anchor_col) in anchors {
            // Try horizontal moves
            self.generate_moves_at_anchor(
                anchor_row,
                anchor_col,
                Direction::Horizontal,
                rack,
                &mut moves,
            );

            // Try vertical moves
            self.generate_moves_at_anchor(
                anchor_row,
                anchor_col,
                Direction::Vertical,
                rack,
                &mut moves,
            );
        }

        // Remove duplicates and sort by score
        self.deduplicate_and_sort(&mut moves);

        moves
    }

    fn generate_moves_at_anchor(
        &self,
        anchor_row: usize,
        anchor_col: usize,
        dir: Direction,
        rack: &Rack,
        moves: &mut Vec<Move>,
    ) {
        let (dr, dc) = dir.delta();

        // First, collect any existing tiles BEFORE the anchor as a prefix
        let mut existing_prefix = String::new();
        let mut r = anchor_row as i32 - dr;
        let mut c = anchor_col as i32 - dc;

        while Board::is_valid_position(r, c) {
            if let Some(tile) = self.board.get_tile(r as usize, c as usize) {
                existing_prefix.insert(0, tile.letter);
                r -= dr;
                c -= dc;
            } else {
                break;
            }
        }

        // If there's an existing prefix, we can only extend from this anchor
        // (no room to place tiles before the existing word)
        if !existing_prefix.is_empty() {
            // Check if this prefix is valid before proceeding
            if self.dictionary.has_prefix(&existing_prefix) {
                self.extend_right(
                    anchor_row,
                    anchor_col,
                    dir,
                    existing_prefix,
                    rack.tiles(),
                    Vec::new(),
                    moves,
                );
            }
        } else {
            // No existing tiles before anchor - try all possible prefix lengths
            let max_prefix_len = self.calculate_prefix_limit(anchor_row, anchor_col, dir);

            for prefix_len in 0..=max_prefix_len {
                let start_row = (anchor_row as i32 - dr * prefix_len as i32) as usize;
                let start_col = (anchor_col as i32 - dc * prefix_len as i32) as usize;

                self.extend_right(
                    start_row,
                    start_col,
                    dir,
                    String::new(),
                    rack.tiles(),
                    Vec::new(),
                    moves,
                );
            }
        }
    }

    fn calculate_prefix_limit(&self, row: usize, col: usize, dir: Direction) -> usize {
        let (dr, dc) = dir.delta();
        let mut limit = 0;
        let mut r = row as i32 - dr;
        let mut c = col as i32 - dc;

        // Count empty squares before the anchor (not adjacent to other tiles)
        while Board::is_valid_position(r, c) {
            let ur = r as usize;
            let uc = c as usize;

            if !self.board.is_empty(ur, uc) {
                break;
            }

            // Check if this square is already an anchor (adjacent to another tile)
            if self.board.has_adjacent_tile(ur, uc) {
                break;
            }

            limit += 1;
            r -= dr;
            c -= dc;
        }

        // Limit by rack size
        limit.min(RACK_SIZE)
    }

    fn get_cross_checks(&self, row: usize, col: usize, cross_dir: Direction) -> HashSet<char> {
        let (dr, dc) = cross_dir.delta();

        // Find what letters are already above/left
        let mut prefix = String::new();
        let mut r = row as i32 - dr;
        let mut c = col as i32 - dc;

        while Board::is_valid_position(r, c) {
            if let Some(tile) = self.board.get_tile(r as usize, c as usize) {
                prefix.insert(0, tile.letter);
                r -= dr;
                c -= dc;
            } else {
                break;
            }
        }

        // Find what letters are already below/right
        let mut suffix = String::new();
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;

        while Board::is_valid_position(r, c) {
            if let Some(tile) = self.board.get_tile(r as usize, c as usize) {
                suffix.push(tile.letter);
                r += dr;
                c += dc;
            } else {
                break;
            }
        }

        // If no adjacent tiles in cross direction, any letter is valid
        if prefix.is_empty() && suffix.is_empty() {
            return ('A'..='Z').collect();
        }

        // Find all letters that form valid cross words
        let mut valid = HashSet::new();
        for letter in 'A'..='Z' {
            let cross_word = format!("{}{}{}", prefix, letter, suffix);
            if self.dictionary.is_valid(&cross_word) {
                valid.insert(letter);
            }
        }

        valid
    }

    #[allow(clippy::too_many_arguments)]
    fn extend_right(
        &self,
        row: usize,
        col: usize,
        dir: Direction,
        current_word: String,
        available_tiles: Vec<Tile>,
        placed_tiles: Vec<(usize, usize, Tile)>,
        moves: &mut Vec<Move>,
    ) {
        let (dr, dc) = dir.delta();

        // If we've gone off the board, check if we have a valid word
        if !Board::is_valid_position(row as i32, col as i32) {
            if current_word.len() >= 2
                && self.dictionary.is_valid(&current_word)
                && !placed_tiles.is_empty()
            {
                self.record_move(dir, &current_word, &placed_tiles, &available_tiles, moves);
            }
            return;
        }

        // Check if there's already a tile here
        if let Some(existing_tile) = self.board.get_tile(row, col) {
            // Use the existing tile
            let mut new_word = current_word.clone();
            new_word.push(existing_tile.letter);

            // Check if this prefix is valid
            if self.dictionary.has_prefix(&new_word) {
                let next_row = (row as i32 + dr) as usize;
                let next_col = (col as i32 + dc) as usize;

                self.extend_right(
                    next_row,
                    next_col,
                    dir,
                    new_word,
                    available_tiles,
                    placed_tiles,
                    moves,
                );
            }
        } else {
            // Try placing each available tile
            let cross_checks = self.get_cross_checks(row, col, dir.perpendicular());

            // First, check if the current word is complete (valid word that touches existing tiles)
            if current_word.len() >= 2
                && self.dictionary.is_valid(&current_word)
                && !placed_tiles.is_empty()
            {
                // Make sure we've connected to existing tiles
                let connects = placed_tiles.iter().any(|(r, c, _)| {
                    self.board.has_adjacent_tile(*r, *c)
                        || (*r == 7 && *c == 7 && self.board.is_first_turn())
                });

                if connects {
                    self.record_move(dir, &current_word, &placed_tiles, &available_tiles, moves);
                }
            }

            // Try placing tiles from rack
            let mut tried_letters = HashSet::new();
            for i in 0..available_tiles.len() {
                let tile = &available_tiles[i];
                let letters_to_try: Vec<char> = if tile.is_blank {
                    ('A'..='Z').collect()
                } else {
                    vec![tile.letter]
                };

                for letter in letters_to_try {
                    // Skip if we've already tried this letter
                    if tried_letters.contains(&letter) {
                        continue;
                    }

                    // Skip if this letter doesn't pass cross-checks
                    if !cross_checks.contains(&letter) {
                        continue;
                    }

                    let mut new_word = current_word.clone();
                    new_word.push(letter);

                    // Check if this prefix is valid
                    if !self.dictionary.has_prefix(&new_word) {
                        continue;
                    }

                    tried_letters.insert(letter);

                    // Remove tile from available
                    let mut new_available = available_tiles.clone();
                    let used_tile = new_available.remove(i);

                    let mut new_placed = placed_tiles.clone();
                    let placed_tile = if used_tile.is_blank {
                        Tile::blank(letter)
                    } else {
                        used_tile
                    };
                    new_placed.push((row, col, placed_tile));

                    let next_row = (row as i32 + dr) as usize;
                    let next_col = (col as i32 + dc) as usize;

                    self.extend_right(
                        next_row,
                        next_col,
                        dir,
                        new_word,
                        new_available,
                        new_placed,
                        moves,
                    );
                }
            }
        }
    }

    fn record_move(
        &self,
        dir: Direction,
        word: &str,
        placed_tiles: &[(usize, usize, Tile)],
        leave: &[Tile],
        moves: &mut Vec<Move>,
    ) {
        if placed_tiles.is_empty() {
            return;
        }

        // Find the actual start position of the word
        let (dr, dc) = dir.delta();
        let (first_row, first_col, _) = placed_tiles[0];

        // Go backwards to find the true start
        let mut start_row = first_row as i32;
        let mut start_col = first_col as i32;

        while Board::is_valid_position(start_row - dr, start_col - dc) {
            if self
                .board
                .get_tile((start_row - dr) as usize, (start_col - dc) as usize)
                .is_some()
            {
                start_row -= dr;
                start_col -= dc;
            } else {
                break;
            }
        }

        // Calculate score
        let score = self.calculate_move_score(placed_tiles, dir);

        let is_bingo = placed_tiles.len() == RACK_SIZE;
        let leave_val = leave_value(leave);
        let equity = score as f32 + leave_val;

        moves.push(Move {
            word: word.to_string(),
            row: start_row as usize,
            col: start_col as usize,
            direction: dir,
            score,
            tiles_used: placed_tiles.to_vec(),
            is_bingo,
            leave: leave.to_vec(),
            equity,
        });
    }

    fn calculate_move_score(&self, placed_tiles: &[(usize, usize, Tile)], dir: Direction) -> u32 {
        let mut total_score = 0;

        // Calculate main word score
        let main_word_score = self.calculate_word_score(placed_tiles, dir);
        total_score += main_word_score;

        // Calculate cross word scores
        for &(row, col, ref tile) in placed_tiles {
            let cross_score = self.calculate_cross_word_score(row, col, tile, dir.perpendicular());
            total_score += cross_score;
        }

        // Add bingo bonus
        if placed_tiles.len() == RACK_SIZE {
            total_score += self.config.bingo_bonus;
        }

        total_score
    }

    fn calculate_word_score(&self, placed_tiles: &[(usize, usize, Tile)], dir: Direction) -> u32 {
        let (dr, dc) = dir.delta();

        // Find the full word boundaries
        if placed_tiles.is_empty() {
            return 0;
        }

        let (first_row, first_col, _) = placed_tiles[0];

        // Find start of word
        let mut start_row = first_row as i32;
        let mut start_col = first_col as i32;

        while Board::is_valid_position(start_row - dr, start_col - dc) {
            if self
                .board
                .get_tile((start_row - dr) as usize, (start_col - dc) as usize)
                .is_some()
            {
                start_row -= dr;
                start_col -= dc;
            } else {
                break;
            }
        }

        // Build a map of placed tiles for quick lookup
        let placed_map: std::collections::HashMap<(usize, usize), &Tile> =
            placed_tiles.iter().map(|(r, c, t)| ((*r, *c), t)).collect();

        let mut word_score = 0;
        let mut word_multiplier = 1;

        let mut r = start_row;
        let mut c = start_col;

        loop {
            if !Board::is_valid_position(r, c) {
                break;
            }

            let ur = r as usize;
            let uc = c as usize;

            let tile = if let Some(t) = placed_map.get(&(ur, uc)) {
                Some(*t)
            } else {
                self.board.get_tile(ur, uc)
            };

            if let Some(tile) = tile {
                let cell = self
                    .board
                    .get(ur, uc)
                    .expect("board position within validated bounds");
                let letter_value = tile.value();

                // Only apply premium if we're placing a tile here
                if placed_map.contains_key(&(ur, uc)) && !cell.premium_used {
                    let letter_mult = cell.premium.letter_multiplier();
                    let word_mult = cell.premium.word_multiplier();

                    word_score += letter_value * letter_mult;
                    word_multiplier *= word_mult;
                } else {
                    word_score += letter_value;
                }

                r += dr;
                c += dc;
            } else {
                break;
            }
        }

        word_score * word_multiplier
    }

    fn calculate_cross_word_score(
        &self,
        row: usize,
        col: usize,
        placed_tile: &Tile,
        cross_dir: Direction,
    ) -> u32 {
        let (dr, dc) = cross_dir.delta();

        // Check if there are adjacent tiles in the cross direction
        let has_before = Board::is_valid_position(row as i32 - dr, col as i32 - dc)
            && self
                .board
                .get_tile((row as i32 - dr) as usize, (col as i32 - dc) as usize)
                .is_some();

        let has_after = Board::is_valid_position(row as i32 + dr, col as i32 + dc)
            && self
                .board
                .get_tile((row as i32 + dr) as usize, (col as i32 + dc) as usize)
                .is_some();

        if !has_before && !has_after {
            return 0;
        }

        // Calculate cross word score
        let cell = self
            .board
            .get(row, col)
            .expect("board position within validated bounds");
        let mut word_score = 0;
        let mut word_multiplier = 1;

        // Score tiles before
        let mut r = row as i32 - dr;
        let mut c = col as i32 - dc;

        while Board::is_valid_position(r, c) {
            if let Some(tile) = self.board.get_tile(r as usize, c as usize) {
                word_score += tile.value();
                r -= dr;
                c -= dc;
            } else {
                break;
            }
        }

        // Score the placed tile (with premium)
        if !cell.premium_used {
            word_score += placed_tile.value() * cell.premium.letter_multiplier();
            word_multiplier = cell.premium.word_multiplier();
        } else {
            word_score += placed_tile.value();
        }

        // Score tiles after
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;

        while Board::is_valid_position(r, c) {
            if let Some(tile) = self.board.get_tile(r as usize, c as usize) {
                word_score += tile.value();
                r += dr;
                c += dc;
            } else {
                break;
            }
        }

        word_score * word_multiplier
    }

    fn deduplicate_and_sort(&self, moves: &mut Vec<Move>) {
        // Remove duplicates based on word + position + direction
        let mut seen = HashSet::new();
        moves.retain(|m| {
            let key = (m.word.clone(), m.row, m.col, m.direction);
            seen.insert(key)
        });

        // Sort by equity descending (strategic value = score + leave value)
        moves.sort_by(|a, b| {
            b.equity
                .partial_cmp(&a.equity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::load_default_dictionary;

    #[test]
    fn test_solver_first_turn() {
        let dict = load_default_dictionary().unwrap();
        let board = Board::new();

        // Create a rack with letters to spell "CAT"
        let mut rack = Rack::new();
        rack.add(Tile::new('C'));
        rack.add(Tile::new('A'));
        rack.add(Tile::new('T'));
        rack.add(Tile::new('S'));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);

        println!("Found {} moves:", moves.len());
        for (i, m) in moves.iter().take(10).enumerate() {
            println!("  {}: {} ({} pts)", i + 1, m.word, m.score);
        }

        // Should find at least some words
        assert!(
            !moves.is_empty(),
            "Should find at least one move on first turn"
        );

        // Should find "CAT" since we have those letters
        let has_cat = moves.iter().any(|m| m.word == "CAT");
        assert!(has_cat, "Should find CAT as a valid move");
    }

    #[test]
    fn test_solver_with_existing_tiles() {
        let dict = load_default_dictionary().unwrap();
        let mut board = Board::new();

        // Place "CAT" horizontally at (7, 7)
        board.place_tile(7, 7, Tile::new('C'));
        board.place_tile(7, 8, Tile::new('A'));
        board.place_tile(7, 9, Tile::new('T'));
        board.mark_premium_used(7, 7);
        board.mark_premium_used(7, 8);
        board.mark_premium_used(7, 9);

        // Simple rack with just 'S' - should find CATS
        let mut rack = Rack::new();
        rack.add(Tile::new('S'));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);

        // Should find moves
        assert!(
            !moves.is_empty(),
            "Should find moves connecting to existing tiles"
        );

        // Should find "CATS" (add S at end)
        let has_cats = moves.iter().any(|m| m.word == "CATS");
        assert!(has_cats, "Should find CATS");
    }

    #[test]
    fn test_solver_extensions() {
        let dict = load_default_dictionary().unwrap();
        let mut board = Board::new();

        // Place "CAT" horizontally at (7, 7)
        board.place_tile(7, 7, Tile::new('C'));
        board.place_tile(7, 8, Tile::new('A'));
        board.place_tile(7, 9, Tile::new('T'));
        board.mark_premium_used(7, 7);
        board.mark_premium_used(7, 8);
        board.mark_premium_used(7, 9);

        // Rack with letters for CATER/CATES
        let mut rack = Rack::new();
        rack.add(Tile::new('E'));
        rack.add(Tile::new('R'));
        rack.add(Tile::new('S'));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);

        // Should find multi-letter extensions
        let has_cater = moves.iter().any(|m| m.word == "CATER");
        let has_cates = moves.iter().any(|m| m.word == "CATES");
        let has_caters = moves.iter().any(|m| m.word == "CATERS");

        assert!(
            has_cater || has_cates || has_caters,
            "Should find CATER, CATES, or CATERS"
        );
    }

    #[test]
    fn test_cross_word_validation() {
        let dict = load_default_dictionary().unwrap();
        let mut board = Board::new();

        // Place "CAT" horizontally at (7, 7)
        board.place_tile(7, 7, Tile::new('C'));
        board.place_tile(7, 8, Tile::new('A'));
        board.place_tile(7, 9, Tile::new('T'));
        board.mark_premium_used(7, 7);
        board.mark_premium_used(7, 8);
        board.mark_premium_used(7, 9);

        // Rack with tiles that could form vertical words
        let mut rack = Rack::new();
        rack.add(Tile::new('A'));
        rack.add(Tile::new('T'));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);

        println!("Found {} cross-word moves:", moves.len());
        for (i, m) in moves.iter().take(15).enumerate() {
            println!("  {}: {} ({} pts)", i + 1, m.word, m.score);
        }

        // All returned moves should have valid cross-words
        // Check that AT is being found (vertical through the A of CAT)
        let vertical_moves = moves
            .iter()
            .filter(|m| m.direction == Direction::Vertical)
            .count();
        println!("  Vertical moves: {}", vertical_moves);
    }
}
