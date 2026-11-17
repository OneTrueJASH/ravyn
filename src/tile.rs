//! Tiles and the 7-slot rack. Point values belong to the rules
//! ([`crate::rules::Rules::tile_value`]).

/// Number of tile slots in a player's rack.
pub const RACK_SIZE: usize = 7;

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
    fn test_rack_operations() {
        let mut rack = Rack::new();
        rack.add(Tile::new('A'));
        rack.add(Tile::new('B'));
        assert_eq!(rack.len(), 2);

        rack.clear_slot(0);
        assert_eq!(rack.len(), 1);
    }
}
