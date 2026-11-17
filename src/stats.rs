//! Game statistics: phase detection, tile tracking, and rack analysis.
//!
//! Computes real-time stats from the current game state including bag estimates,
//! power tile locations, and rack composition metrics.

use std::collections::HashMap;

use crate::game::SolverState;
use crate::tile::RACK_SIZE;

/// Game phase based on unseen tiles (bag + opponent's rack)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    Early,   // 77+ unseen (70+ in bag)
    Mid,     // 37-76 unseen (30-69 in bag)
    Late,    // 8-36 unseen (1-29 in bag)
    Endgame, // ≤RACK_SIZE unseen (bag empty, opponent has all)
}

const RACK_SIZE_U32: u32 = RACK_SIZE as u32;

impl GamePhase {
    /// Determine game phase from unseen tile count (bag + opponent's rack)
    pub fn from_unseen_count(count: u32) -> Self {
        match count {
            0..=RACK_SIZE_U32 => GamePhase::Endgame, // Bag empty, opponent has ≤RACK_SIZE
            8..=36 => GamePhase::Late,               // 1-29 in bag + 7 opponent
            37..=76 => GamePhase::Mid,               // 30-69 in bag + 7 opponent
            _ => GamePhase::Early,                   // 70+ in bag + 7 opponent
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            GamePhase::Early => "Early",
            GamePhase::Mid => "Mid",
            GamePhase::Late => "Late",
            GamePhase::Endgame => "Endgame",
        }
    }
}

/// Location of a power tile (Q, Z, J, X)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLocation {
    InBag(u32),   // Count remaining in bag
    OnBoard,      // All copies on board
    InRack,       // All copies in rack
    Split,        // Some on board, some in rack/bag
    BoardAndRack, // Some on board, some in rack
}

/// Computed game statistics
#[derive(Debug, Clone)]
pub struct GameStats {
    // Rack analysis
    pub rack_vowels: usize,
    pub rack_consonants: usize,
    pub rack_blanks: usize,
    pub rack_value: u32,
    pub rack_duplicates: Vec<char>,

    // Bag/game state (unseen = bag + opponent's rack)
    pub estimated_bag: u32,
    pub blanks_unseen: u32,
    pub s_unseen: u32,
    pub power_tiles: HashMap<char, TileLocation>,
    pub game_phase: GamePhase,
}

impl GameStats {
    /// Compute all statistics from the current game state
    pub fn compute(state: &SolverState) -> Self {
        let unseen = state.unseen_tiles();
        let rack_counts = state.count_rack_tiles();
        let board_counts = state.count_board_tiles();

        // Rack analysis
        let (rack_vowels, rack_consonants, rack_blanks, rack_value, rack_duplicates) =
            Self::analyze_rack(state);

        // Unseen tiles (bag + opponent's rack)
        let unseen_tiles: u32 = unseen.values().sum();
        let estimated_bag = unseen_tiles.saturating_sub(RACK_SIZE as u32); // Opponent has up to RACK_SIZE
        let blanks_unseen = unseen.get(&'?').copied().unwrap_or(0);
        let s_unseen = unseen.get(&'S').copied().unwrap_or(0);

        // Power tiles
        let power_tiles = Self::compute_power_tile_locations(&unseen, &board_counts, &rack_counts);

        let game_phase = GamePhase::from_unseen_count(unseen_tiles);

        GameStats {
            rack_vowels,
            rack_consonants,
            rack_blanks,
            rack_value,
            rack_duplicates,
            estimated_bag,
            blanks_unseen,
            s_unseen,
            power_tiles,
            game_phase,
        }
    }

    fn analyze_rack(state: &SolverState) -> (usize, usize, usize, u32, Vec<char>) {
        const VOWELS: &[char] = &['A', 'E', 'I', 'O', 'U'];

        let mut vowels = 0usize;
        let mut consonants = 0usize;
        let mut blanks = 0usize;
        let mut value = 0u32;
        let mut letter_counts: HashMap<char, usize> = HashMap::new();

        for tile in state.rack.tiles() {
            if tile.is_blank {
                blanks += 1;
            } else {
                let letter = tile.letter;
                value += state.board.rules().letter_value(letter);

                if VOWELS.contains(&letter) {
                    vowels += 1;
                } else {
                    consonants += 1;
                }

                *letter_counts.entry(letter).or_insert(0) += 1;
            }
        }

        // Find duplicates
        let mut duplicates: Vec<char> = letter_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .flat_map(|(letter, count)| std::iter::repeat_n(letter, count))
            .collect();
        duplicates.sort();

        (vowels, consonants, blanks, value, duplicates)
    }

    fn compute_power_tile_locations(
        remaining: &HashMap<char, u32>,
        board_counts: &HashMap<char, u32>,
        rack_counts: &HashMap<char, u32>,
    ) -> HashMap<char, TileLocation> {
        let mut locations = HashMap::new();

        for &letter in &['Q', 'Z', 'J', 'X'] {
            let in_bag = remaining.get(&letter).copied().unwrap_or(0);
            let on_board = board_counts.get(&letter).copied().unwrap_or(0);
            let in_rack = rack_counts.get(&letter).copied().unwrap_or(0);

            let location = match (in_bag > 0, on_board > 0, in_rack > 0) {
                (true, false, false) => TileLocation::InBag(in_bag),
                (false, true, false) => TileLocation::OnBoard,
                (false, false, true) => TileLocation::InRack,
                (false, true, true) => TileLocation::BoardAndRack,
                (true, true, _) | (true, _, true) => TileLocation::Split,
                (false, false, false) => TileLocation::InBag(0), // Shouldn't happen
            };

            locations.insert(letter, location);
        }

        locations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::SolverState;

    #[test]
    fn test_game_phase() {
        // Test with unseen count (bag + opponent's rack of up to 7)
        assert_eq!(GamePhase::from_unseen_count(100), GamePhase::Early); // 93+ in bag
        assert_eq!(GamePhase::from_unseen_count(77), GamePhase::Early); // 70 in bag + 7 opp
        assert_eq!(GamePhase::from_unseen_count(76), GamePhase::Mid); // 69 in bag + 7 opp
        assert_eq!(GamePhase::from_unseen_count(37), GamePhase::Mid); // 30 in bag + 7 opp
        assert_eq!(GamePhase::from_unseen_count(36), GamePhase::Late); // 29 in bag + 7 opp
        assert_eq!(GamePhase::from_unseen_count(8), GamePhase::Late); // 1 in bag + 7 opp
        assert_eq!(GamePhase::from_unseen_count(7), GamePhase::Endgame); // 0 in bag, opp has 7
        assert_eq!(GamePhase::from_unseen_count(3), GamePhase::Endgame); // 0 in bag, opp has 3
    }

    #[test]
    fn test_empty_state_stats() {
        let state = SolverState::new();
        let stats = GameStats::compute(&state);

        assert_eq!(stats.rack_vowels, 0);
        assert_eq!(stats.rack_consonants, 0);
        assert_eq!(stats.rack_blanks, 0);
        assert_eq!(stats.rack_value, 0);
        assert!(stats.rack_duplicates.is_empty());

        // Full unseen tiles (100 total in Crossplay - empty rack)
        assert_eq!(state.unseen_tile_count(), 100);
        assert_eq!(stats.estimated_bag, 93); // 100 - 7 (opponent's rack)
        assert_eq!(stats.blanks_unseen, 3);
        assert_eq!(stats.s_unseen, 5);
        assert_eq!(stats.game_phase, GamePhase::Early);
    }

    #[test]
    fn test_rack_analysis() {
        let mut state = SolverState::new();
        state.add_to_rack('A'); // vowel, 1pt
        state.add_to_rack('E'); // vowel, 1pt
        state.add_to_rack('E'); // vowel, 1pt (duplicate)
        state.add_to_rack('S'); // consonant, 1pt
        state.add_to_rack('T'); // consonant, 1pt
        state.add_to_rack('?'); // blank

        let stats = GameStats::compute(&state);

        assert_eq!(stats.rack_vowels, 3);
        assert_eq!(stats.rack_consonants, 2);
        assert_eq!(stats.rack_blanks, 1);
        assert_eq!(stats.rack_value, 5); // A(1) + E(1) + E(1) + S(1) + T(1)
        assert_eq!(stats.rack_duplicates, vec!['E', 'E']);
    }
}
