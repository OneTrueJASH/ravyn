//! Tile representation, letter point values (Crossplay scoring), and 7-slot rack.
//!
//! Crossplay uses different point values than standard Scrabble.

/// Number of tile slots in a player's rack.
pub(crate) const RACK_SIZE: usize = 7;

/// Represents a single tile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub letter: char,
    pub is_blank: bool,
}

impl Tile {
    pub fn new(letter: char) -> Self {
        Tile {
            letter: letter.to_ascii_uppercase(),
            is_blank: false,
        }
    }

    pub fn blank(designated_letter: char) -> Self {
        Tile {
            letter: designated_letter.to_ascii_uppercase(),
            is_blank: true,
        }
    }

    /// Get the point value of this tile (blanks are always 0)
    pub fn value(&self) -> u32 {
        if self.is_blank {
            return 0;
        }
        letter_value(self.letter)
    }
}

/// Get the point value of a letter (Crossplay-specific values)
pub fn letter_value(letter: char) -> u32 {
    match letter.to_ascii_uppercase() {
        'A' | 'E' | 'I' | 'N' | 'O' | 'R' | 'S' | 'T' => 1,
        'D' | 'L' | 'U' => 2,
        'C' | 'H' | 'M' | 'P' => 3,
        'B' | 'F' | 'G' | 'Y' => 4,
        'W' => 5,
        'K' | 'V' => 6,
        'X' => 8,
        'J' | 'Q' | 'Z' => 10,
        _ => 0,
    }
}

/// Player's rack of tiles (7 slots, some may be empty)
#[derive(Debug, Clone)]
pub struct Rack {
    pub slots: [Option<Tile>; RACK_SIZE],
}

impl Rack {
    pub fn new() -> Self {
        Rack {
            slots: [None; RACK_SIZE],
        }
    }

    /// Add a tile to the first empty slot
    pub fn add(&mut self, tile: Tile) -> bool {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.is_none()) {
            *slot = Some(tile);
            true
        } else {
            false
        }
    }

    /// Clear a slot by setting it to None
    pub fn clear_slot(&mut self, index: usize) {
        if index < RACK_SIZE {
            self.slots[index] = None;
        }
    }

    /// Get tiles as a vector (for solver compatibility)
    pub fn tiles(&self) -> Vec<Tile> {
        self.slots.iter().filter_map(|s| *s).collect()
    }

    /// Count of actual tiles (not empty slots)
    #[cfg_attr(not(test), allow(dead_code))] // used by tests and library consumers
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_none())
    }

    /// Find index of a tile matching the predicate
    pub fn find_index<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&Tile) -> bool,
    {
        self.slots
            .iter()
            .position(|s| s.as_ref().map(&predicate).unwrap_or(false))
    }
}

impl Default for Rack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letter_values() {
        // Crossplay values
        assert_eq!(letter_value('E'), 1);
        assert_eq!(letter_value('A'), 1);
        assert_eq!(letter_value('D'), 2);
        assert_eq!(letter_value('C'), 3);
        assert_eq!(letter_value('B'), 4);
        assert_eq!(letter_value('W'), 5);
        assert_eq!(letter_value('K'), 6);
        assert_eq!(letter_value('V'), 6);
        assert_eq!(letter_value('X'), 8);
        assert_eq!(letter_value('Q'), 10);
        assert_eq!(letter_value('Z'), 10);
        assert_eq!(letter_value('J'), 10);
    }

    #[test]
    fn test_rack_operations() {
        let mut rack = Rack::new();
        rack.add(Tile::new('A'));
        rack.add(Tile::new('B'));
        assert_eq!(rack.len(), 2);

        rack.clear_slot(0);
        assert_eq!(rack.len(), 1);
    }
}
