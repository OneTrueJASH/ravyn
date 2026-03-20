//! 15x15 Crossplay game board with cells, premium squares, and direction types.
//!
//! Provides the [`Board`] struct, anchor detection, and tile placement logic.

use crate::tile::Tile;

/// Side length of the square game board.
pub(crate) const BOARD_SIZE: usize = 15;

/// Premium square types on the Crossplay board
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Premium {
    Normal,
    DoubleLetter,
    TripleLetter,
    DoubleWord,
    TripleWord,
    Center, // Center star (must be played on first move, scores as normal)
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
    /// True if the premium was used (tile was placed here this turn or previously)
    pub premium_used: bool,
}

impl Cell {
    pub fn new(premium: Premium) -> Self {
        Cell {
            tile: None,
            premium,
            premium_used: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tile.is_none()
    }
}

/// The 15x15 Crossplay game board
#[derive(Debug, Clone)]
pub struct Board {
    pub cells: [[Cell; BOARD_SIZE]; BOARD_SIZE],
}

impl Board {
    /// Create a new board with Crossplay premium square layout
    pub fn new() -> Self {
        // Initialize all cells as Normal
        let mut cells: [[Cell; BOARD_SIZE]; BOARD_SIZE] =
            std::array::from_fn(|_| std::array::from_fn(|_| Cell::new(Premium::Normal)));

        // Set up premium squares based on actual NYT Crossplay layout
        // Triple Word squares (purple)
        let triple_word = [
            (0, 3),
            (0, 11),
            (3, 0),
            (3, 14),
            (11, 0),
            (11, 14),
            (14, 3),
            (14, 11),
        ];
        for (r, c) in triple_word {
            cells[r][c].premium = Premium::TripleWord;
        }

        // Double Word squares (light blue/lavender)
        let double_word = [
            (1, 1),
            (1, 13),
            (3, 7),
            (7, 3),
            (7, 11),
            (11, 7),
            (13, 1),
            (13, 13),
        ];
        for (r, c) in double_word {
            cells[r][c].premium = Premium::DoubleWord;
        }

        // Center star
        cells[7][7].premium = Premium::Center;

        // Triple Letter squares (green)
        let triple_letter = [
            (0, 0),
            (0, 14),
            (1, 6),
            (1, 8),
            (4, 5),
            (4, 9),
            (5, 4),
            (5, 10),
            (6, 1),
            (6, 13),
            (8, 1),
            (8, 13),
            (9, 4),
            (9, 10),
            (10, 5),
            (10, 9),
            (13, 6),
            (13, 8),
            (14, 0),
            (14, 14),
        ];
        for (r, c) in triple_letter {
            cells[r][c].premium = Premium::TripleLetter;
        }

        // Double Letter squares (tan/beige)
        let double_letter = [
            (0, 7),   // H1
            (2, 4),   // E3
            (2, 10),  // K3
            (3, 3),   // D4
            (3, 11),  // L4
            (4, 2),   // C5
            (4, 12),  // M5
            (5, 7),   // H6
            (7, 0),   // A8
            (7, 5),   // F8
            (7, 9),   // J8
            (7, 14),  // O8
            (9, 7),   // H10
            (10, 2),  // C11
            (10, 12), // M11
            (11, 3),  // D12
            (11, 11), // L12
            (12, 4),  // E13
            (12, 10), // K13
            (14, 7),  // H15
        ];
        for (r, c) in double_letter {
            cells[r][c].premium = Premium::DoubleLetter;
        }

        Board { cells }
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

    /// Mark the premium at a position as used
    pub fn mark_premium_used(&mut self, row: usize, col: usize) {
        if let Some(cell) = self.get_mut(row, col) {
            cell.premium_used = true;
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

    /// Check if the board is completely empty (first turn)
    pub fn is_first_turn(&self) -> bool {
        self.cells
            .iter()
            .all(|row| row.iter().all(|c| c.is_empty()))
    }

    /// Get all anchor squares (empty squares adjacent to placed tiles)
    pub fn get_anchors(&self) -> Vec<(usize, usize)> {
        let mut anchors = Vec::new();

        // On first turn, only the center is an anchor
        if self.is_first_turn() {
            anchors.push((7, 7));
            return anchors;
        }

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.is_empty(row, col) && self.has_adjacent_tile(row, col) {
                    anchors.push((row, col));
                }
            }
        }
        anchors
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

    #[test]
    fn test_anchors_first_turn() {
        let board = Board::new();
        let anchors = board.get_anchors();
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0], (7, 7));
    }
}
