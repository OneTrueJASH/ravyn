//! The 15x15 board: cells, premium squares and directions. The premium
//! layout comes from the board's [`Rules`].
//!
//! Provides the [`Board`] struct, anchor detection, and tile placement logic.

use crate::rules::{Rules, CROSSPLAY};
use crate::tile::Tile;

/// Side length of the square game board.
pub const BOARD_SIZE: usize = 15;

/// Premium square types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Premium {
    Normal,
    DoubleLetter,
    TripleLetter,
    DoubleWord,
    TripleWord,
    /// Crossplay's centre star: the first play covers it; it scores as a
    /// plain square.
    Center,
}

impl Premium {
    pub fn letter_multiplier(&self) -> u32 {
        match self {
            Premium::DoubleLetter => 2,
            Premium::TripleLetter => 3,
            _ => 1,
        }
    }

    pub fn word_multiplier(&self) -> u32 {
        match self {
            Premium::DoubleWord => 2,
            Premium::TripleWord => 3,
            _ => 1,
        }
    }
}

/// Direction of word placement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Horizontal,
    Vertical,
}

impl Direction {
    pub fn delta(&self) -> (i32, i32) {
        match self {
            Direction::Horizontal => (0, 1),
            Direction::Vertical => (1, 0),
        }
    }

    pub fn perpendicular(&self) -> Direction {
        match self {
            Direction::Horizontal => Direction::Vertical,
            Direction::Vertical => Direction::Horizontal,
        }
    }
}

/// A cell on the board, which may contain a placed tile
#[derive(Debug, Clone)]
pub struct Cell {
    pub tile: Option<Tile>,
    pub premium: Premium,
}

impl Cell {
    pub fn new(premium: Premium) -> Self {
        Cell {
            tile: None,
            premium,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tile.is_none()
    }
}

/// A 15x15 board: its squares and the rules it is played under.
#[derive(Debug, Clone)]
pub struct Board {
    pub cells: [[Cell; BOARD_SIZE]; BOARD_SIZE],
    rules: &'static Rules,
}

impl Board {
    /// An empty board under [`CROSSPLAY`] rules.
    pub fn new() -> Self {
        Self::with_rules(&CROSSPLAY)
    }

    /// An empty board under `rules`, with its premium squares.
    pub fn with_rules(rules: &'static Rules) -> Self {
        Board {
            cells: std::array::from_fn(|r| {
                std::array::from_fn(|c| Cell::new(rules.premiums[r][c]))
            }),
            rules,
        }
    }

    /// The rules this board is played under.
    #[inline]
    pub fn rules(&self) -> &'static Rules {
        self.rules
    }

    /// Get the cell at a position (0-indexed)
    pub fn get(&self, row: usize, col: usize) -> Option<&Cell> {
        self.cells.get(row).and_then(|r| r.get(col))
    }

    /// Get a mutable reference to a cell
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Cell> {
        self.cells.get_mut(row).and_then(|r| r.get_mut(col))
    }

    /// Get the tile at a position
    pub fn get_tile(&self, row: usize, col: usize) -> Option<&Tile> {
        self.get(row, col).and_then(|c| c.tile.as_ref())
    }

    /// Get a mutable reference to the tile at a position
    pub fn get_tile_mut(&mut self, row: usize, col: usize) -> Option<&mut Tile> {
        self.get_mut(row, col).and_then(|c| c.tile.as_mut())
    }

    /// Place a tile at a position
    pub fn place_tile(&mut self, row: usize, col: usize, tile: Tile) -> bool {
        if let Some(cell) = self.get_mut(row, col) {
            if cell.tile.is_none() {
                cell.tile = Some(tile);
                return true;
            }
        }
        false
    }

    /// Remove a tile from a position (returns the tile if present)
    pub fn remove_tile(&mut self, row: usize, col: usize) -> Option<Tile> {
        if let Some(cell) = self.get_mut(row, col) {
            cell.tile.take()
        } else {
            None
        }
    }

    /// Check if a position is valid
    pub fn is_valid_position(row: i32, col: i32) -> bool {
        (0..BOARD_SIZE as i32).contains(&row) && (0..BOARD_SIZE as i32).contains(&col)
    }

    /// Check if a position is empty
    pub fn is_empty(&self, row: usize, col: usize) -> bool {
        self.get(row, col).map(|c| c.is_empty()).unwrap_or(true)
    }

    /// Check if a position has an adjacent tile
    pub fn has_adjacent_tile(&self, row: usize, col: usize) -> bool {
        let checks = [
            (row.wrapping_sub(1), col),
            (row + 1, col),
            (row, col.wrapping_sub(1)),
            (row, col + 1),
        ];

        for (r, c) in checks {
            if r < BOARD_SIZE && c < BOARD_SIZE && !self.is_empty(r, c) {
                return true;
            }
        }
        false
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_premium_squares() {
        let board = Board::new();

        // Check triple word squares (Crossplay layout)
        assert_eq!(board.cells[0][3].premium, Premium::TripleWord);
        assert_eq!(board.cells[14][11].premium, Premium::TripleWord);

        // Check triple letter corners
        assert_eq!(board.cells[0][0].premium, Premium::TripleLetter);
        assert_eq!(board.cells[14][14].premium, Premium::TripleLetter);

        // Check center
        assert_eq!(board.cells[7][7].premium, Premium::Center);

        // Check a double letter
        assert_eq!(board.cells[0][7].premium, Premium::DoubleLetter);
    }

    #[test]
    fn test_tile_placement() {
        let mut board = Board::new();
        assert!(board.is_empty(7, 7));

        board.place_tile(7, 7, Tile::new('A'));
        assert!(!board.is_empty(7, 7));
        assert_eq!(board.get_tile(7, 7).unwrap().letter, 'A');
    }
}
