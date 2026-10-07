//! Game state tracking: board + rack, tile distribution, and unseen tile calculation.
//!
//! The [`SolverState`] struct combines a board and rack, providing methods for
//! move application, tile counting, and endgame detection.

use std::collections::HashMap;

use crate::board::{Board, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::solver::{rack_counts, tiles_from_counts, LeaveTable, Move, Solver, EXCHANGE_MIN_BAG};
use crate::tile::{Rack, Tile, RACK_SIZE};

/// Total number of tiles in a standard Crossplay game.
pub(crate) const TOTAL_TILES: usize = 100;

/// Number of blank (wildcard) tiles in the distribution.
pub(crate) const BLANK_COUNT: usize = 3;

/// Tile distribution for Crossplay (100 tiles)
fn standard_distribution() -> HashMap<char, u32> {
    let dist: HashMap<char, u32> = [
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
        ('?', BLANK_COUNT as u32), // ? = blank
    ]
    .into_iter()
    .collect();
    debug_assert_eq!(
        dist.values().sum::<u32>() as usize,
        TOTAL_TILES,
        "Tile distribution must sum to TOTAL_TILES"
    );
    dist
}

/// Every tile in a fresh bag, in distribution order.
#[cfg_attr(not(test), allow(dead_code))] // self-play studies and library consumers
pub fn full_tile_set() -> Vec<Tile> {
    let mut dist: Vec<(char, u32)> = standard_distribution().into_iter().collect();
    dist.sort();
    let mut tiles = Vec::with_capacity(TOTAL_TILES);
    for (letter, count) in dist {
        for _ in 0..count {
            tiles.push(if letter == '?' {
                Tile::blank(' ')
            } else {
                Tile::new(letter)
            });
        }
    }
    tiles
}

/// Every distinct exchange from `rack` (one per multiset of exchanged
/// tiles), valued by the phase-scaled leave value of the tiles kept. The
/// caller checks that the bag permits exchanging.
pub fn exchange_moves(rack: &[Tile], dictionary: &Dictionary, unseen: Option<usize>) -> Vec<Move> {
    let n = rack.len();
    let mut seen: std::collections::HashSet<Vec<(bool, char)>> = std::collections::HashSet::new();
    let mut moves = Vec::new();
    for mask in 1u32..(1u32 << n) {
        let exchanged: Vec<Tile> = (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| rack[i])
            .collect();
        let mut key: Vec<(bool, char)> = exchanged.iter().map(|t| (t.is_blank, t.letter)).collect();
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        let mut kept = rack.to_vec();
        for t in &exchanged {
            if let Some(i) = kept
                .iter()
                .position(|k| k.is_blank == t.is_blank && (k.is_blank || k.letter == t.letter))
            {
                kept.swap_remove(i);
            }
        }
        let equity = dictionary.leave_value(&kept, unseen);
        moves.push(Move::exchange(rack, exchanged, equity));
    }
    moves
}

/// Every distinct exchange from `rack` as (kept-subset index, equity),
/// enumerated through the rack's subset leave table without building
/// moves. The full rack (nothing exchanged) is excluded.
fn exchange_values<'d>(
    rack: &[Tile],
    dictionary: &'d Dictionary,
    unseen: Option<usize>,
) -> (LeaveTable<'d>, Vec<(usize, f32)>) {
    let mut table = LeaveTable::new(rack_counts(rack), dictionary, unseen);
    let full = table.full_index();
    let values = (0..full).map(|i| (i, table.value(i))).collect();
    (table, values)
}

/// The exchange that keeps the subset at `kept_index`.
fn exchange_from_subset(rack: &[Tile], table: &LeaveTable, kept_index: usize, equity: f32) -> Move {
    let kept = table.subset_counts(kept_index);
    let mut exchanged = rack_counts(rack);
    for (e, k) in exchanged.iter_mut().zip(kept) {
        *e -= k;
    }
    Move::exchange(rack, tiles_from_counts(&exchanged), equity)
}

/// The best exchange by the leave value of the tiles kept (the same
/// valuation as [`exchange_moves`]), building only that one move.
pub fn best_exchange(
    rack: &[Tile],
    dictionary: &Dictionary,
    unseen: Option<usize>,
) -> Option<Move> {
    let (table, values) = exchange_values(rack, dictionary, unseen);
    let &(index, equity) = values
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    Some(exchange_from_subset(rack, &table, index, equity))
}

/// The `k` best exchanges, best first, building only those moves.
pub fn top_exchanges(
    rack: &[Tile],
    dictionary: &Dictionary,
    unseen: Option<usize>,
    k: usize,
) -> Vec<Move> {
    let (table, mut values) = exchange_values(rack, dictionary, unseen);
    values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    values
        .iter()
        .take(k)
        .map(|&(index, equity)| exchange_from_subset(rack, &table, index, equity))
        .collect()
}

/// Estimate how much equity the opponent gave up with an observed play:
/// the gap between the best move available and the move they made,
/// averaged over sampled completions of their hidden rack.
///
/// `prev_board` is the board before their move, `placed` the tiles it put
/// down, `score` its score, and `pool` the unseen tiles now (their rack
/// plus the bag), which is exactly the set their kept tiles came from.
/// Their rack is completed by joining the played tiles with `7 - played`
/// tiles sampled uniformly from the pool; for each completion the best
/// play (and exchange, when the bag allowed one) is generated on the
/// pre-move board and compared with the observed move's equity. Returns
/// `None` when no full rack can be reconstructed.
pub fn observed_move_loss(
    prev_board: &Board,
    placed: &[(usize, usize, Tile)],
    score: u32,
    pool: &HashMap<char, u32>,
    dictionary: &Dictionary,
    samples: usize,
    seed: u64,
) -> Option<f32> {
    if placed.is_empty() || placed.len() > RACK_SIZE {
        return None;
    }
    let kept_size = RACK_SIZE - placed.len();
    let mut pool_tiles: Vec<Tile> = Vec::new();
    for (&letter, &count) in pool {
        for _ in 0..count {
            pool_tiles.push(if letter == '?' {
                Tile::blank(' ')
            } else {
                Tile::new(letter)
            });
        }
    }
    if pool_tiles.len() < kept_size {
        return None;
    }
    // Canonical order so the same seed samples the same completions.
    pool_tiles.sort_by_key(|t| (t.is_blank, t.letter));

    let played: Vec<Tile> = placed
        .iter()
        .map(|(_, _, t)| if t.is_blank { Tile::blank(' ') } else { *t })
        .collect();
    let prev_count = (0..BOARD_SIZE)
        .flat_map(|r| (0..BOARD_SIZE).map(move |c| (r, c)))
        .filter(|&(r, c)| prev_board.get_tile(r, c).is_some())
        .count();
    // The unseen count from their side of the table when they moved, and
    // the bag size then (they had not yet redrawn).
    let their_unseen = TOTAL_TILES.saturating_sub(prev_count + RACK_SIZE);
    let bag_then = (pool_tiles.len() + placed.len()).saturating_sub(RACK_SIZE);

    let solver = Solver::new(prev_board, dictionary);
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut total_loss = 0.0f32;
    let mut measured = 0usize;
    for _ in 0..samples {
        rng.shuffle(&mut pool_tiles);
        let kept = &pool_tiles[..kept_size];
        let mut rack_tiles = played.clone();
        rack_tiles.extend_from_slice(kept);

        let best_play = solver.best_move_by_equity(&rack_tiles).map(|m| m.equity);
        let best_exchange = if bag_then >= EXCHANGE_MIN_BAG {
            best_exchange(&rack_tiles, dictionary, Some(their_unseen)).map(|m| m.equity)
        } else {
            None
        };
        let best = match (best_play, best_exchange) {
            (Some(p), Some(e)) => p.max(e),
            (Some(p), None) => p,
            (None, Some(e)) => e,
            (None, None) => continue,
        };
        let observed = score as f32 + dictionary.leave_value(kept, Some(their_unseen));
        total_loss += (best - observed).max(0.0);
        measured += 1;
    }
    if measured == 0 {
        return None;
    }
    Some(total_loss / measured as f32)
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
        if row >= BOARD_SIZE || col >= BOARD_SIZE {
            return;
        }

        let tile = if is_blank {
            Tile::blank(letter)
        } else {
            Tile::new(letter)
        };

        self.board.place_tile(row, col, tile);
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
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
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

    /// Check if we're in endgame (bag is empty, opponent has all remaining tiles)
    /// This is true when unseen tiles <= 7
    pub fn is_endgame(&self) -> bool {
        self.unseen_tile_count() <= RACK_SIZE
    }

    /// Tiles in the bag: the unseen tiles minus the opponent's full rack.
    /// (While the bag has tiles the opponent always holds seven.)
    pub fn bag_size(&self) -> usize {
        self.unseen_tile_count().saturating_sub(RACK_SIZE)
    }

    /// All candidate moves for our rack, best static equity first: every
    /// legal play, plus every distinct exchange when the bag allows one.
    pub fn get_suggestions(&self, dictionary: &Dictionary) -> Vec<Move> {
        let solver = Solver::new(&self.board, dictionary);
        let mut moves = solver.find_moves(&self.rack);
        let unseen = self.unseen_tile_count();
        if unseen.saturating_sub(RACK_SIZE) >= EXCHANGE_MIN_BAG {
            moves.extend(exchange_moves(&self.rack.tiles(), dictionary, Some(unseen)));
            moves.sort_by(|a, b| {
                b.equity
                    .partial_cmp(&a.equity)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.score.cmp(&a.score))
            });
        }
        moves
    }

    /// Apply a move: place its tiles on the board (for a play) and clear the
    /// tiles it consumes from the rack, keeping slot positions. Exchanged
    /// tiles leave the rack too; the replacements are typed in afterwards.
    pub fn apply_move(&mut self, mv: &Move) {
        for (row, col, tile) in &mv.tiles_used {
            self.board.place_tile(*row, *col, *tile);
        }
        for tile in mv.tiles_from_rack() {
            // For blanks, match by is_blank (the letter is what it became);
            // for regular tiles, match by letter.
            let idx = if tile.is_blank {
                self.rack.find_index(|t| t.is_blank)
            } else {
                self.rack
                    .find_index(|t| !t.is_blank && t.letter == tile.letter)
            };
            if let Some(idx) = idx {
                self.rack.clear_slot(idx);
            }
        }
    }

    /// Build a state from the board and rack sections of a CGP string
    /// (`<board> <rack>[/<opp_rack>] ...`). Rows are separated by `/`, digits
    /// are runs of empty squares, uppercase letters are tiles and lowercase
    /// letters are blanks designated as that letter. Only the board and our
    /// rack are read; anything after them is ignored.
    #[cfg_attr(not(test), allow(dead_code))] // used by tests and library consumers
    pub fn from_cgp(cgp: &str) -> Result<Self, String> {
        let mut sections = cgp.split_whitespace();
        let board_str = sections.next().ok_or("empty CGP")?;
        let rack_str = sections.next().unwrap_or("");

        let mut state = SolverState::new();
        let rows: Vec<&str> = board_str.split('/').collect();
        if rows.len() != BOARD_SIZE {
            return Err(format!("expected {} rows, got {}", BOARD_SIZE, rows.len()));
        }
        for (row, row_str) in rows.iter().enumerate() {
            let mut col = 0;
            let mut run = 0;
            for ch in row_str.chars() {
                if let Some(d) = ch.to_digit(10) {
                    run = run * 10 + d as usize;
                    continue;
                }
                col += run;
                run = 0;
                if !ch.is_ascii_alphabetic() {
                    return Err(format!("bad board character {:?} in row {}", ch, row + 1));
                }
                if col >= BOARD_SIZE {
                    return Err(format!("row {} is too long", row + 1));
                }
                state.place_tile(row, col, ch.to_ascii_uppercase(), ch.is_ascii_lowercase());
                col += 1;
            }
            col += run;
            if col != BOARD_SIZE {
                return Err(format!(
                    "row {} has {} squares, expected {}",
                    row + 1,
                    col,
                    BOARD_SIZE
                ));
            }
        }

        for ch in rack_str.split('/').next().unwrap_or("").chars() {
            if !state.add_to_rack(ch) {
                return Err(format!("bad or excess rack tile {:?}", ch));
            }
        }
        Ok(state)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Direction;
    use crate::solver::{Move, MoveKind};
    use crate::tile::Tile;

    fn t(s: &str) -> Vec<Tile> {
        s.chars()
            .map(|c| {
                if c == '?' {
                    Tile::blank(' ')
                } else {
                    Tile::new(c)
                }
            })
            .collect()
    }
    fn key(v: &[Tile]) -> Vec<(bool, char)> {
        let mut k: Vec<(bool, char)> = v
            .iter()
            .map(|x| (x.is_blank, if x.is_blank { '?' } else { x.letter }))
            .collect();
        k.sort();
        k
    }

    #[test]
    fn test_new_state_is_empty() {
        let state = SolverState::new();
        for row in 0..15 {
            for col in 0..15 {
                assert!(state.get_tile(row, col).is_none());
            }
        }
        assert!(state.rack.is_empty());
        assert_eq!(state.rack.len(), 0);
    }

    #[test]
    fn test_place_and_get_tile() {
        let mut state = SolverState::new();
        state.place_tile(7, 7, 'A', false);
        let tile = state.get_tile(7, 7).expect("tile should be present");
        assert_eq!(tile.letter, 'A');
        assert!(!tile.is_blank);
    }

    #[test]
    fn test_place_blank_tile() {
        let mut state = SolverState::new();
        state.place_tile(3, 4, 'S', true);
        let tile = state.get_tile(3, 4).expect("tile should be present");
        assert!(tile.is_blank);
        assert_eq!(tile.letter, 'S');
    }

    #[test]
    fn test_remove_tile() {
        let mut state = SolverState::new();
        state.place_tile(0, 0, 'Z', false);
        assert!(state.get_tile(0, 0).is_some());
        let removed = state.remove_tile(0, 0);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().letter, 'Z');
        assert!(state.get_tile(0, 0).is_none());
    }

    #[test]
    fn test_add_to_rack() {
        let mut state = SolverState::new();
        assert!(state.add_to_rack('A'));
        assert!(state.add_to_rack('B'));
        assert!(state.add_to_rack('C'));
        assert_eq!(state.rack.len(), 3);
        let tiles = state.rack.tiles();
        assert_eq!(tiles[0].letter, 'A');
        assert_eq!(tiles[1].letter, 'B');
        assert_eq!(tiles[2].letter, 'C');
    }

    #[test]
    fn test_add_to_rack_full() {
        let mut state = SolverState::new();
        for ch in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] {
            assert!(state.add_to_rack(ch), "should succeed adding '{}'", ch);
        }
        assert_eq!(state.rack.len(), 7);
        assert!(!state.add_to_rack('H'));
        assert_eq!(state.rack.len(), 7);
    }

    #[test]
    fn test_pop_from_rack() {
        let mut state = SolverState::new();
        state.add_to_rack('X');
        state.add_to_rack('Y');
        assert_eq!(state.rack.len(), 2);
        assert!(state.pop_from_rack());
        assert_eq!(state.rack.len(), 1);
        let tiles = state.rack.tiles();
        assert_eq!(tiles[0].letter, 'X');
    }

    #[test]
    fn test_clear_rack() {
        let mut state = SolverState::new();
        state.add_to_rack('A');
        state.add_to_rack('B');
        state.add_to_rack('C');
        assert_eq!(state.rack.len(), 3);
        state.clear_rack();
        assert!(state.rack.is_empty());
        assert_eq!(state.rack.len(), 0);
    }

    #[test]
    fn test_unseen_tiles_empty_board() {
        let state = SolverState::new();
        let unseen = state.unseen_tiles();
        let total: u32 = unseen.values().sum();
        assert_eq!(total as usize, TOTAL_TILES);
    }

    #[test]
    fn test_unseen_tiles_with_board_and_rack() {
        let mut state = SolverState::new();
        state.place_tile(7, 7, 'C', false);
        state.place_tile(7, 8, 'A', false);
        state.place_tile(7, 9, 'T', false);
        state.add_to_rack('D');
        state.add_to_rack('O');

        let unseen = state.unseen_tiles();
        let total: u32 = unseen.values().sum();
        assert_eq!(total, 95);
    }

    #[test]
    fn test_is_endgame() {
        let mut state = SolverState::new();
        assert!(!state.is_endgame());

        // Place enough tiles to leave exactly 7 unseen
        let dist = standard_distribution();
        let mut placed = 0;
        let target = 93;
        'outer: for (&letter, &count) in &dist {
            if letter == '?' {
                continue;
            }
            for _ in 0..count {
                if placed >= target {
                    break 'outer;
                }
                let row = placed / 15;
                let col = placed % 15;
                if row < 15 {
                    state.place_tile(row, col, letter, false);
                    placed += 1;
                }
            }
        }
        assert!(state.unseen_tile_count() <= 7);
        assert!(state.is_endgame());
    }

    #[test]
    fn test_apply_move() {
        let mut state = SolverState::new();
        state.add_to_rack('C');
        state.add_to_rack('A');
        state.add_to_rack('T');

        let mv = Move {
            kind: MoveKind::Play,
            word: "CAT".to_string(),
            row: 7,
            col: 7,
            direction: Direction::Horizontal,
            score: 10,
            tiles_used: vec![
                (7, 7, Tile::new('C')),
                (7, 8, Tile::new('A')),
                (7, 9, Tile::new('T')),
            ],
            is_bingo: false,
            leave: vec![],
            equity: 10.0,
        };

        state.apply_move(&mv);

        assert_eq!(state.get_tile(7, 7).unwrap().letter, 'C');
        assert_eq!(state.get_tile(7, 8).unwrap().letter, 'A');
        assert_eq!(state.get_tile(7, 9).unwrap().letter, 'T');
        assert!(state.rack.is_empty());
    }

    #[test]
    fn test_from_cgp_round_trip() {
        let cgp = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15 AEI?/ 0/0 0";
        let state = SolverState::from_cgp(cgp).expect("valid CGP");
        assert_eq!(state.get_tile(7, 3).unwrap().letter, 'B');
        assert_eq!(state.get_tile(7, 7).unwrap().letter, 'H');
        let r = state.get_tile(11, 7).unwrap();
        assert!(r.is_blank && r.letter == 'R');
        assert_eq!(state.rack.len(), 4);
        assert!(state.rack.tiles()[3].is_blank);
        assert_eq!(state.unseen_tile_count(), 100 - 9 - 4);

        assert!(SolverState::from_cgp("15/15 A").is_err());
        assert!(SolverState::from_cgp("16/15/15/15/15/15/15/15/15/15/15/15/15/15/15 A").is_err());
        assert!(
            SolverState::from_cgp("15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 ABCDEFGH").is_err()
        );
    }

    #[test]
    fn test_get_suggestions() {
        let Some(dict) = crate::datafiles::test_data::dictionary() else {
            return;
        };

        let mut state = SolverState::new();
        state.place_tile(7, 6, 'C', false);
        state.place_tile(7, 7, 'A', false);
        state.place_tile(7, 8, 'T', false);

        state.add_to_rack('S');
        state.add_to_rack('E');
        state.add_to_rack('R');
        state.add_to_rack('I');
        state.add_to_rack('N');

        let suggestions = state.get_suggestions(&dict);
        assert!(
            !suggestions.is_empty(),
            "should find at least one valid move"
        );
    }

    #[test]
    fn exchange_moves_complete_and_distinct() {
        let Some(dict) = crate::datafiles::test_data::dictionary() else {
            return;
        };
        let rack = t("AABC??D");
        let moves = exchange_moves(&rack, &dict, Some(80));
        // distinct multisets: A:0-2 (3) * B (2) * C (2) * ?:0-2 (3) * D (2) - 1 = 71
        assert_eq!(moves.len(), 71);
        let mut seen = std::collections::HashSet::new();
        for m in &moves {
            let MoveKind::Exchange(ex) = &m.kind else {
                panic!()
            };
            assert!(seen.insert(key(ex)));
            // leave + exchanged == rack as multisets
            let mut all = m.leave.clone();
            all.extend(ex.iter().copied());
            assert_eq!(key(&all), key(&rack));
            assert_eq!(m.coords(), "-");
            assert_eq!(m.score, 0);
            assert!(m.tiles_used.is_empty());
            assert_eq!(m.tiles_from_rack().len(), ex.len());
        }
    }

    #[test]
    fn get_suggestions_exchanges_only_with_bag_of_seven() {
        let Some(dict) = crate::datafiles::test_data::dictionary() else {
            return;
        };
        // MID board: 87 tiles; rack AEST -> unseen 9 -> bag 2: no exchanges.
        const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";
        let s = SolverState::from_cgp(&format!("{} AEST", MID)).unwrap();
        assert_eq!(s.bag_size(), 2);
        assert!(s
            .get_suggestions(&dict)
            .iter()
            .all(|m| m.kind == MoveKind::Play));
        // BEECH board: bag 77: exchanges present, sorted by equity with plays.
        let s =
            SolverState::from_cgp("15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15 QVWXZ?A")
                .unwrap();
        let sugg = s.get_suggestions(&dict);
        let n_ex = sugg
            .iter()
            .filter(|m| matches!(m.kind, MoveKind::Exchange(_)))
            .count();
        assert!(n_ex > 0);
        for w in sugg.windows(2) {
            assert!(w[0].equity >= w[1].equity - 1e-4);
        }
        // With a rack this bad, some exchange should beat the median play.
        let best_ex = sugg
            .iter()
            .position(|m| matches!(m.kind, MoveKind::Exchange(_)))
            .unwrap();
        assert!(
            best_ex < sugg.len() / 2,
            "best exchange at rank {} of {}",
            best_ex,
            sugg.len()
        );
    }

    /// The measured loss is deterministic for a seed, never negative, and
    /// bigger for a worse observed move: the same placement reported with
    /// a lower score loses exactly the score difference more (the sampled
    /// completions and thus the best alternatives are identical).
    #[test]
    fn observed_move_loss_ranks_worse_moves_higher() {
        let Some(dict) = crate::datafiles::test_data::dictionary() else {
            return;
        };
        let state =
            SolverState::from_cgp("15/15/15/15/15/15/15/3BEECH7/15/15/15/15/15/15/15 AAA").unwrap();
        // The opponent hooked two tiles onto BEECH.
        let placed = vec![
            (6usize, 3usize, Tile::new('T')),
            (8usize, 3usize, Tile::new('I')),
        ];
        let mut pool = HashMap::new();
        pool.insert('A', 3u32);
        pool.insert('E', 3);
        pool.insert('R', 2);
        pool.insert('S', 2);
        pool.insert('T', 2);
        pool.insert('N', 2);
        pool.insert('O', 2);

        let good = observed_move_loss(&state.board, &placed, 30, &pool, &dict, 8, 11).unwrap();
        let bad = observed_move_loss(&state.board, &placed, 2, &pool, &dict, 8, 11).unwrap();
        assert!(good >= 0.0);
        assert!(bad > good, "lower score must measure a bigger loss");
        assert!(
            bad - good <= 28.0 + 1e-3,
            "the gap is bounded by the score difference"
        );

        let again = observed_move_loss(&state.board, &placed, 30, &pool, &dict, 8, 11).unwrap();
        assert_eq!(good, again, "same seed, same measurement");

        // A rack that cannot be completed measures nothing.
        let empty_pool = HashMap::new();
        assert!(observed_move_loss(&state.board, &placed, 30, &empty_pool, &dict, 8, 11).is_none());
    }

    #[test]
    fn apply_exchange_clears_exactly_those_tiles() {
        let Some(dict) = crate::datafiles::test_data::dictionary() else {
            return;
        };
        let mut s =
            SolverState::from_cgp("15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15 QVWXZ?A")
                .unwrap();
        let rack = s.rack.tiles();
        let mv = Move::exchange(&rack, t("QV?"), dict.leave_value(&t("WXZA"), Some(84)));
        let board_before = s.board.clone();
        s.apply_move(&mv);
        assert_eq!(key(&s.rack.tiles()), key(&t("WXZA")));
        for r in 0..15 {
            for c in 0..15 {
                assert_eq!(
                    s.board.get_tile(r, c).map(|x| x.letter),
                    board_before.get_tile(r, c).map(|x| x.letter)
                );
            }
        }
    }
}
