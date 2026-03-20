use crate::board::Board;
use crate::dictionary::Dictionary;
use crate::solver::{Move, Solver};
use crate::tile::{Rack, Tile};
use std::collections::HashMap;

/// Tile distribution for Crossplay (100 tiles)
fn standard_distribution() -> HashMap<char, u32> {
    [
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
        ('?', 3), // ? = blank
    ]
    .into_iter()
    .collect()
}

/// The solver state - tracks board and rack for finding optimal plays
#[derive(Clone)]
pub struct SolverState {
    pub board: Board,
    pub rack: Rack,
}

impl SolverState {
    pub fn new() -> Self {
        SolverState {
            board: Board::new(),
            rack: Rack::new(),
        }
    }

    /// Place a tile on the board (for recreating game state)
    pub fn place_tile(&mut self, row: usize, col: usize, letter: char, is_blank: bool) {
        if row >= 15 || col >= 15 {
            return;
        }

        let tile = if is_blank {
            Tile::blank(letter)
        } else {
            Tile::new(letter)
        };

        self.board.place_tile(row, col, tile);
        self.board.mark_premium_used(row, col);
    }

    /// Remove a tile from the board
    pub fn remove_tile(&mut self, row: usize, col: usize) -> Option<Tile> {
        self.board.remove_tile(row, col)
    }

    /// Get tile at position
    pub fn get_tile(&self, row: usize, col: usize) -> Option<&Tile> {
        self.board.get_tile(row, col)
    }

    /// Get mutable tile at position
    pub fn get_tile_mut(&mut self, row: usize, col: usize) -> Option<&mut Tile> {
        self.board.get_tile_mut(row, col)
    }

    /// Add a letter to the rack (if space available)
    pub fn add_to_rack(&mut self, letter: char) -> bool {
        let upper = letter.to_ascii_uppercase();
        if upper == '?' {
            self.rack.add(Tile::blank(' '))
        } else if upper.is_ascii_alphabetic() {
            self.rack.add(Tile::new(upper))
        } else {
            false
        }
    }

    /// Remove last tile from rack (finds rightmost occupied slot)
    pub fn pop_from_rack(&mut self) -> bool {
        // Find the last occupied slot
        if let Some(idx) = self.rack.slots.iter().rposition(|s| s.is_some()) {
            self.rack.clear_slot(idx);
            true
        } else {
            false
        }
    }

    /// Clear the rack
    pub fn clear_rack(&mut self) {
        self.rack = Rack::new();
    }

    /// Count tiles placed on the board by letter
    pub fn count_board_tiles(&self) -> HashMap<char, u32> {
        let mut counts: HashMap<char, u32> = HashMap::new();
        for row in 0..15 {
            for col in 0..15 {
                if let Some(tile) = self.board.get_tile(row, col) {
                    let key = if tile.is_blank { '?' } else { tile.letter };
                    *counts.entry(key).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    /// Count tiles in rack by letter
    pub fn count_rack_tiles(&self) -> HashMap<char, u32> {
        let mut counts: HashMap<char, u32> = HashMap::new();
        for tile in self.rack.tiles() {
            let key = if tile.is_blank { '?' } else { tile.letter };
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    /// Calculate unseen tiles (tiles not on board or in our rack)
    /// This includes both the bag AND opponent's rack
    pub fn unseen_tiles(&self) -> HashMap<char, u32> {
        let total = standard_distribution();
        let board = self.count_board_tiles();
        let rack = self.count_rack_tiles();

        let mut remaining = total;
        for (letter, count) in board {
            if let Some(r) = remaining.get_mut(&letter) {
                *r = r.saturating_sub(count);
            }
        }
        for (letter, count) in rack {
            if let Some(r) = remaining.get_mut(&letter) {
                *r = r.saturating_sub(count);
            }
        }
        remaining
    }

    /// Count total unseen tiles (bag + opponent's rack)
    pub fn unseen_tile_count(&self) -> usize {
        self.unseen_tiles().values().sum::<u32>() as usize
    }

    /// Estimate tiles remaining in bag (unseen minus opponent's max rack of 7)
    /// Returns 0 if opponent likely has all remaining tiles
    pub fn estimated_bag_count(&self) -> usize {
        let unseen = self.unseen_tile_count();
        unseen.saturating_sub(7)
    }

    /// Check if we're in endgame (bag is empty, opponent has all remaining tiles)
    /// This is true when unseen tiles <= 7
    pub fn is_endgame(&self) -> bool {
        self.unseen_tile_count() <= 7
    }

    /// Get suggested moves using the solver
    pub fn get_suggestions(&self, dictionary: &Dictionary) -> Vec<Move> {
        let solver = Solver::new(&self.board, dictionary);
        solver.find_moves(&self.rack)
    }

    /// Apply a move to the board (clear used tiles from rack, keeping positions)
    pub fn apply_move(&mut self, mv: &Move) {
        for (row, col, tile) in &mv.tiles_used {
            self.board.place_tile(*row, *col, *tile);
            self.board.mark_premium_used(*row, *col);

            // Clear the matching tile from the rack (leaves slot empty)
            // For blanks, we match by is_blank (the letter is what it became)
            // For regular tiles, we match by letter
            if tile.is_blank {
                if let Some(idx) = self.rack.find_index(|t| t.is_blank) {
                    self.rack.clear_slot(idx);
                }
            } else if let Some(idx) = self.rack.find_index(|t| !t.is_blank && t.letter == tile.letter) {
                self.rack.clear_slot(idx);
            }
        }
    }

    /// Reset everything
    pub fn reset(&mut self) {
        self.board = Board::new();
        self.rack = Rack::new();
    }
}

impl Default for SolverState {
    fn default() -> Self {
        Self::new()
    }
}
