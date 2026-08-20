//! Move generation engine using anchor-based search with cross-checks.
//!
//! Generates all legal moves for a given rack and board position,
//! scoring each move and evaluating leave quality for equity ranking.
//!
//! The search walks the dictionary FST one node at a time: every placed
//! letter is a single transition lookup, and blanks enumerate the node's
//! outgoing transitions rather than all 26 letters. Cross-checks are
//! precomputed bitmasks, and scores accumulate incrementally during the
//! search so recording a move is allocation-light.

use std::collections::HashMap;

use fst::raw::Node;

use crate::board::{Board, Direction, Premium, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::game::TOTAL_TILES;
use crate::tile::{letter_value, Rack, Tile, RACK_SIZE};

/// Bonus points awarded when a player uses all tiles from their rack in a single turn.
pub(crate) const BINGO_BONUS: u32 = 40;

/// Bitmask with every letter A–Z allowed.
const ALL_LETTERS: u32 = (1 << 26) - 1;

/// Index of the blank in the rack count array.
const BLANK_IDX: usize = 26;

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
                    let idx = rack_remaining.iter().position(|t| t.is_blank)?;
                    let mut tile = rack_remaining.remove(idx);
                    tile.letter = letter;
                    tiles_used.push((r, c, tile));
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

    /// Board coordinates in the conventional notation: row number first for
    /// horizontal plays (`8D`), column letter first for vertical plays (`D8`).
    pub fn coords(&self) -> String {
        if self.tiles_used.is_empty() {
            return "-".to_string();
        }
        let col_letter = (b'A' + self.col as u8) as char;
        match self.direction {
            Direction::Horizontal => format!("{}{}", self.row + 1, col_letter),
            Direction::Vertical => format!("{}{}", col_letter, self.row + 1),
        }
    }
}

/// Per-letter heuristic leave value with game-phase scaling.
///
/// This is the fallback when no fitted leave table is loaded; prefer
/// [`Dictionary::leave_value`], which uses the table when available.
pub fn leave_value_with_phase(leave: &[Tile], unseen_count: Option<usize>) -> f32 {
    if leave.is_empty() {
        return 0.0;
    }

    let phase_scale = crate::dictionary::phase_scale(unseen_count);

    let mut value = 0.0;
    let mut vowels: i32 = 0;
    let mut consonants: i32 = 0;
    let mut counts = [0i32; 26];

    for tile in leave {
        if tile.is_blank {
            // Blanks are extremely valuable: they enable bingos and flexible plays.
            // Value scales with game phase (less valuable in endgame with no redraws).
            value += 25.0;
            continue;
        }

        let letter = tile.letter;
        if let Some(idx) = letter_index(letter) {
            counts[idx] += 1;
        }

        // Individual letter values calibrated for Crossplay.
        // These approximate the expected equity contribution of having each
        // letter on your rack, based on hookability, bingo potential, and
        // scoring power in Crossplay's tile distribution and point values.
        value += match letter {
            'S' => 8.0,  // Excellent: hooks (plurals, verb forms), bingo-friendly
            'R' => 4.5,  // Top consonant for bingos (-ER, -ING+R, RE-)
            'E' => 3.5,  // Most common, essential for bingos (-ED, -ER, -ES)
            'N' => 3.5,  // Strong bingo consonant (-ING, -TION, UN-)
            'A' => 2.5,  // Good vowel, many 2-letter words (AA, AB, AD...)
            'I' => 2.0,  // Good vowel, -ING, -TION
            'T' => 2.5,  // Good consonant, -TION, -EST
            'L' => 2.0,  // Decent consonant, -LY, -LE
            'D' => 1.5,  // Good for -ED past tenses
            'O' => 1.0,  // Less versatile vowel
            'H' => 1.0,  // CH, SH, TH combos
            'G' => 0.5,  // -ING (good with I,N), otherwise middling
            'U' => 0.5,  // Weak vowel but essential for Q plays
            'M' => 0.5,  // Average consonant
            'P' => 0.0,  // Average consonant
            'C' => -0.5, // Limited hooks, awkward combos
            'Y' => -0.5, // Semi-vowel, hard to combine
            'B' => -1.5, // Few useful combos
            'F' => -1.5, // Few useful combos
            'K' => -1.5, // High point value but hard to use in bingos
            'W' => -2.0, // Very limited combos
            'J' => -2.5, // High points but almost no bingo potential
            'X' => 3.5,  // High scoring in short plays (AX, EX, XI, XU)
            'Z' => 2.5,  // High scoring (ZA, ZO, ZAP, ZIP)
            'V' => -3.5, // Hardest letter: few words, no 2-letter words
            'Q' => -6.0, // Very hard without U, limited word options
            _ => 0.0,
        };

        if matches!(letter, 'A' | 'E' | 'I' | 'O' | 'U') {
            vowels += 1;
        } else {
            consonants += 1;
        }
    }

    // Penalize duplicate tiles (diminishing returns on same letter)
    for (idx, &count) in counts.iter().enumerate() {
        if count >= 2 {
            let extra = (count - 1) as f32;
            let penalty = match (b'A' + idx as u8) as char {
                // Common tiles: duplicates are less painful
                'E' => 2.5 * extra,
                'A' | 'I' | 'O' => 3.0 * extra,
                // S is special: first S is great, second is still good for hooks
                'S' => 2.0 * extra,
                // Uncommon tiles: duplicates are very bad
                'V' | 'W' | 'Q' | 'J' | 'K' | 'X' | 'Z' => 6.0 * extra,
                _ => 4.0 * extra,
            };
            value -= penalty;
        }
    }

    // Penalize vowel/consonant imbalance
    let total_letters = vowels + consonants;
    if total_letters > 1 {
        // Ideal ratio is ~40% vowels for bingo potential
        let ideal_vowels = (total_letters as f32 * 0.4).round() as i32;
        let vowel_diff = (vowels - ideal_vowels).abs();
        value -= vowel_diff as f32 * 2.5;

        // Heavy penalty for all vowels or all consonants
        if vowels == 0 || consonants == 0 {
            value -= 10.0;
        }
    }

    let has = |c: char| counts[(c as u8 - b'A') as usize] > 0;

    // Q without U is very bad
    if has('Q') && !has('U') {
        value -= 12.0;
    }

    // Synergy bonuses for bingo-friendly combinations
    if has('I') && has('N') && has('G') {
        value += 3.0; // -ING is a powerful bingo ending
    }
    if has('E') && has('R') {
        value += 1.5; // -ER is very common
    }
    if has('E') && has('D') {
        value += 1.0; // -ED is common
    }

    value * phase_scale
}

/// Index 0–25 for an uppercase ASCII letter, `None` otherwise.
#[inline]
fn letter_index(letter: char) -> Option<usize> {
    let b = letter as u32;
    if (b'A' as u32..=b'Z' as u32).contains(&b) {
        Some((b - b'A' as u32) as usize)
    } else {
        None
    }
}

#[inline]
fn index_letter(idx: usize) -> char {
    (b'A' + idx as u8) as char
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

/// Board-derived tables that depend only on the position, not the rack.
///
/// Indexed `[direction][row][col]` where direction 0 = horizontal word,
/// 1 = vertical word (the cross-check constrains the perpendicular word).
struct BoardInfo {
    /// Allowed letters (bit `i` = letter `A + i`) for a tile placed here as
    /// part of a word in the given direction.
    cross_mask: [[[u32; BOARD_SIZE]; BOARD_SIZE]; 2],
    /// Sum of existing tile values in the perpendicular word through this
    /// square (0 when there is no perpendicular word).
    cross_score: [[[u32; BOARD_SIZE]; BOARD_SIZE]; 2],
    /// Whether placing here forms a perpendicular word at all.
    cross_has: [[[bool; BOARD_SIZE]; BOARD_SIZE]; 2],
    /// Empty squares adjacent to a tile (or the centre on the first turn).
    anchor: [[bool; BOARD_SIZE]; BOARD_SIZE],
    /// Number of tiles on the board (for game-phase estimation).
    tile_count: usize,
}

#[inline]
fn dir_index(dir: Direction) -> usize {
    match dir {
        Direction::Horizontal => 0,
        Direction::Vertical => 1,
    }
}

/// A tile placed during generation: position along the lane, letter index,
/// and whether it came from a blank.
#[derive(Clone, Copy)]
struct Placed {
    pos: usize,
    letter: usize,
    is_blank: bool,
}

/// Mutable search state for one `find_moves` call.
struct Gen<'s, 'a> {
    solver: &'s Solver<'a>,
    info: &'s BoardInfo,
    dir: Direction,
    di: usize,
    /// Row (horizontal) or column (vertical) being searched.
    lane: usize,
    /// Position of the anchor along the lane.
    anchor_pos: usize,
    /// Position of the first letter of the main word (set once the left
    /// part is attached to the board).
    word_start: usize,
    /// Remaining rack: counts per letter, blanks at index 26.
    rack: [u8; 27],
    rack_len: usize,
    /// Letters of the main word so far (indices 0–25).
    word: [u8; BOARD_SIZE],
    word_len: usize,
    placed: [Placed; RACK_SIZE],
    placed_len: usize,
    /// Unseen tile count (bag + opponent rack) for leave phase scaling.
    unseen: usize,
    /// Leave value per remaining-rack signature (at most 2^7 distinct).
    leave_cache: HashMap<[u8; 27], f32>,
    moves: &'s mut Vec<Move>,
}

impl<'a> Solver<'a> {
    pub fn new(board: &'a Board, dictionary: &'a Dictionary) -> Self {
        Solver {
            board,
            dictionary,
            config: SolverConfig::default(),
        }
    }

    /// Generate all valid moves for the given rack, sorted by equity.
    ///
    /// Every distinct tile placement appears exactly once. Single-tile plays
    /// that form words in both directions are reported once, labelled with
    /// the longer word.
    pub fn find_moves(&self, rack: &Rack) -> Vec<Move> {
        let tiles = rack.tiles();
        if tiles.is_empty() {
            return Vec::new();
        }

        let info = self.analyze_board();
        let unseen = TOTAL_TILES.saturating_sub(info.tile_count + tiles.len());

        let mut counts = [0u8; 27];
        for t in &tiles {
            if t.is_blank {
                counts[BLANK_IDX] += 1;
            } else if let Some(idx) = letter_index(t.letter) {
                counts[idx] += 1;
            }
        }

        let mut moves = Vec::new();
        let mut gen = Gen {
            solver: self,
            info: &info,
            dir: Direction::Horizontal,
            di: 0,
            lane: 0,
            anchor_pos: 0,
            word_start: 0,
            rack: counts,
            rack_len: tiles.len(),
            word: [0; BOARD_SIZE],
            word_len: 0,
            placed: [Placed {
                pos: 0,
                letter: 0,
                is_blank: false,
            }; RACK_SIZE],
            placed_len: 0,
            unseen,
            leave_cache: HashMap::new(),
            moves: &mut moves,
        };

        // On the first turn the board is symmetric, so every vertical play
        // mirrors a horizontal one; list only horizontal plays.
        let directions: &[Direction] = if info.tile_count == 0 {
            &[Direction::Horizontal]
        } else {
            &[Direction::Horizontal, Direction::Vertical]
        };

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if !info.anchor[row][col] {
                    continue;
                }
                for &dir in directions {
                    gen.dir = dir;
                    gen.di = dir_index(dir);
                    let (lane, pos) = match dir {
                        Direction::Horizontal => (row, col),
                        Direction::Vertical => (col, row),
                    };
                    gen.lane = lane;
                    gen.anchor_pos = pos;
                    gen.generate_at_anchor();
                }
            }
        }

        Self::dedup_single_tile_plays(&mut moves);

        // Sort by equity descending (strategic value = score + leave value),
        // breaking ties by raw score so the order is stable across runs.
        moves.sort_by(|a, b| {
            b.equity
                .partial_cmp(&a.equity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.score.cmp(&a.score))
        });

        moves
    }

    /// Score a set of placed tiles against the current board.
    ///
    /// This is the public entry point for scoring externally-detected moves
    /// (e.g., opponent tiles placed by the user). The board should be the
    /// state BEFORE the tiles were placed (so premiums are still available).
    /// `placed_tiles` must be in word order along `dir`.
    pub fn score_placed_tiles(&self, placed_tiles: &[(usize, usize, Tile)], dir: Direction) -> u32 {
        if placed_tiles.is_empty() {
            return 0;
        }
        let (dr, dc) = dir.delta();

        // Walk back from the first placed tile over existing tiles to the
        // true start of the main word.
        let (mut r, mut c, _) = placed_tiles[0];
        loop {
            let pr = r as i32 - dr;
            let pc = c as i32 - dc;
            if Board::is_valid_position(pr, pc)
                && self.board.get_tile(pr as usize, pc as usize).is_some()
            {
                r = pr as usize;
                c = pc as usize;
            } else {
                break;
            }
        }

        let mut main_sum = 0;
        let mut word_mult = 1;
        let mut cross_total = 0;
        let mut ri = r as i32;
        let mut ci = c as i32;
        while Board::is_valid_position(ri, ci) {
            let (ur, uc) = (ri as usize, ci as usize);
            if let Some(&(_, _, tile)) = placed_tiles
                .iter()
                .find(|(pr, pc, _)| *pr == ur && *pc == uc)
            {
                let premium = self.premium_at(ur, uc);
                let lm = premium.letter_multiplier();
                let wm = premium.word_multiplier();
                main_sum += tile.value() * lm;
                word_mult *= wm;
                cross_total +=
                    self.cross_word_score(ur, uc, tile.value() * lm, wm, dir.perpendicular());
            } else if let Some(tile) = self.board.get_tile(ur, uc) {
                main_sum += tile.value();
            } else {
                break;
            }
            ri += dr;
            ci += dc;
        }

        let mut score = main_sum * word_mult + cross_total;
        if placed_tiles.len() == RACK_SIZE {
            score += self.config.bingo_bonus;
        }
        score
    }

    /// Premium of a square as it applies to a tile placed there now.
    #[inline]
    fn premium_at(&self, row: usize, col: usize) -> Premium {
        self.board
            .get(row, col)
            .map(|c| c.premium)
            .unwrap_or(Premium::Normal)
    }

    /// Score of the perpendicular word formed by placing a tile worth
    /// `placed_value` (already letter-multiplied) at (row, col), or 0 if
    /// there is no perpendicular word.
    fn cross_word_score(
        &self,
        row: usize,
        col: usize,
        placed_value: u32,
        word_mult: u32,
        cross_dir: Direction,
    ) -> u32 {
        let (dr, dc) = cross_dir.delta();
        let mut sum = 0;
        let mut found = false;

        let mut r = row as i32 - dr;
        let mut c = col as i32 - dc;
        while Board::is_valid_position(r, c) {
            match self.board.get_tile(r as usize, c as usize) {
                Some(t) => {
                    sum += t.value();
                    found = true;
                    r -= dr;
                    c -= dc;
                }
                None => break,
            }
        }

        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;
        while Board::is_valid_position(r, c) {
            match self.board.get_tile(r as usize, c as usize) {
                Some(t) => {
                    sum += t.value();
                    found = true;
                    r += dr;
                    c += dc;
                }
                None => break,
            }
        }

        if found {
            (sum + placed_value) * word_mult
        } else {
            0
        }
    }

    /// Build the rack-independent tables: anchors and per-square cross-checks.
    fn analyze_board(&self) -> BoardInfo {
        let mut info = BoardInfo {
            cross_mask: [[[ALL_LETTERS; BOARD_SIZE]; BOARD_SIZE]; 2],
            cross_score: [[[0; BOARD_SIZE]; BOARD_SIZE]; 2],
            cross_has: [[[false; BOARD_SIZE]; BOARD_SIZE]; 2],
            anchor: [[false; BOARD_SIZE]; BOARD_SIZE],
            tile_count: 0,
        };

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.board.get_tile(row, col).is_some() {
                    info.tile_count += 1;
                }
            }
        }

        if info.tile_count == 0 {
            info.anchor[BOARD_SIZE / 2][BOARD_SIZE / 2] = true;
            return info;
        }

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.board.get_tile(row, col).is_some() {
                    continue;
                }
                if !self.board.has_adjacent_tile(row, col) {
                    continue;
                }
                info.anchor[row][col] = true;
                for dir in [Direction::Horizontal, Direction::Vertical] {
                    let di = dir_index(dir);
                    let (mask, score, has) = self.cross_check(row, col, dir.perpendicular());
                    info.cross_mask[di][row][col] = mask;
                    info.cross_score[di][row][col] = score;
                    info.cross_has[di][row][col] = has;
                }
            }
        }

        info
    }

    /// Compute which letters can be placed at (row, col) given the existing
    /// tiles before and after it along `cross_dir`. Returns the allowed
    /// letter mask, the sum of the neighbouring tile values, and whether a
    /// perpendicular word exists at all.
    fn cross_check(&self, row: usize, col: usize, cross_dir: Direction) -> (u32, u32, bool) {
        let (dr, dc) = cross_dir.delta();

        let mut prefix = [0u8; BOARD_SIZE];
        let mut prefix_len = 0;
        let mut score = 0;
        let mut r = row as i32 - dr;
        let mut c = col as i32 - dc;
        while Board::is_valid_position(r, c) {
            match self.board.get_tile(r as usize, c as usize) {
                Some(t) => {
                    prefix[prefix_len] = t.letter as u8;
                    prefix_len += 1;
                    score += t.value();
                    r -= dr;
                    c -= dc;
                }
                None => break,
            }
        }

        let mut suffix = [0u8; BOARD_SIZE];
        let mut suffix_len = 0;
        let mut r = row as i32 + dr;
        let mut c = col as i32 + dc;
        while Board::is_valid_position(r, c) {
            match self.board.get_tile(r as usize, c as usize) {
                Some(t) => {
                    suffix[suffix_len] = t.letter as u8;
                    suffix_len += 1;
                    score += t.value();
                    r += dr;
                    c += dc;
                }
                None => break,
            }
        }

        if prefix_len == 0 && suffix_len == 0 {
            return (ALL_LETTERS, 0, false);
        }

        // Walk the prefix (collected backwards) to its node.
        let dict = self.dictionary;
        let mut node = dict.root();
        for i in (0..prefix_len).rev() {
            match dict.child(node, prefix[i]) {
                Some(n) => node = n,
                None => return (0, score, true),
            }
        }

        let mut mask = 0u32;
        for t in node.transitions() {
            let Some(idx) = letter_index(t.inp as char) else {
                continue;
            };
            let mut n = dict.node_at(t.addr);
            let mut ok = true;
            for &b in &suffix[..suffix_len] {
                match dict.child(n, b) {
                    Some(next) => n = next,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && n.is_final() {
                mask |= 1 << idx;
            }
        }

        (mask, score, true)
    }

    /// Single-tile plays are generated once per direction when they form
    /// both a horizontal and a vertical word. Keep one copy, preferring the
    /// longer main word for display.
    fn dedup_single_tile_plays(moves: &mut Vec<Move>) {
        let mut best: HashMap<(usize, usize, char, bool), usize> = HashMap::new();
        let mut keep = vec![true; moves.len()];
        for (i, mv) in moves.iter().enumerate() {
            if mv.tiles_used.len() != 1 {
                continue;
            }
            let (r, c, t) = mv.tiles_used[0];
            let key = (r, c, t.letter, t.is_blank);
            match best.get(&key) {
                Some(&j) => {
                    if moves[j].word.len() >= mv.word.len() {
                        keep[i] = false;
                    } else {
                        keep[j] = false;
                        best.insert(key, i);
                    }
                }
                None => {
                    best.insert(key, i);
                }
            }
        }
        if keep.iter().any(|k| !k) {
            let mut idx = 0;
            moves.retain(|_| {
                let k = keep[idx];
                idx += 1;
                k
            });
        }
    }
}

impl<'s, 'a> Gen<'s, 'a> {
    #[inline]
    fn rc(&self, pos: usize) -> (usize, usize) {
        match self.dir {
            Direction::Horizontal => (self.lane, pos),
            Direction::Vertical => (pos, self.lane),
        }
    }

    #[inline]
    fn tile_at(&self, pos: usize) -> Option<&'a Tile> {
        let (r, c) = self.rc(pos);
        self.solver.board.get_tile(r, c)
    }

    fn generate_at_anchor(&mut self) {
        let dict = self.solver.dictionary;
        let anchor = self.anchor_pos;

        if anchor > 0 && self.tile_at(anchor - 1).is_some() {
            // Tiles immediately before the anchor form a fixed prefix: the
            // word must start at the beginning of that run.
            let mut start = anchor - 1;
            while start > 0 && self.tile_at(start - 1).is_some() {
                start -= 1;
            }
            let mut node = dict.root();
            let mut sum = 0;
            self.word_len = 0;
            for pos in start..anchor {
                let tile = self.tile_at(pos).expect("run of existing tiles");
                match dict.child(node, tile.letter as u8) {
                    Some(n) => node = n,
                    None => {
                        // Existing tiles are not a prefix of any word; no
                        // extension is possible from this anchor.
                        self.word_len = 0;
                        return;
                    }
                }
                self.word[self.word_len] = letter_index(tile.letter).unwrap_or(0) as u8;
                self.word_len += 1;
                sum += tile.value();
            }
            self.word_start = start;
            self.extend_right(node, anchor, sum, 1, 0);
            self.word_len = 0;
            return;
        }

        // No tile before the anchor: build left parts over the run of empty,
        // non-anchor squares to the left. Squares further left belong to
        // another anchor and will be covered from there.
        let mut limit = 0;
        let mut pos = anchor;
        while pos > 0 {
            pos -= 1;
            let (r, c) = self.rc(pos);
            if self.tile_at(pos).is_some() || self.info.anchor[r][c] {
                break;
            }
            limit += 1;
        }
        let limit = limit.min(self.rack_len.saturating_sub(1));

        self.word_len = 0;
        self.placed_len = 0;
        self.left_part(dict.root(), limit);
    }

    /// Recursively build left parts (tiles placed before the anchor), and for
    /// each one — including the empty left part — extend rightwards from
    /// the anchor.
    fn left_part(&mut self, node: Node<'a>, limit: usize) {
        self.attach_left_part_and_extend(node);

        if limit == 0 {
            return;
        }

        let dict = self.solver.dictionary;
        for t in node.transitions() {
            let Some(idx) = letter_index(t.inp as char) else {
                continue;
            };
            let is_blank = if self.rack[idx] > 0 {
                false
            } else if self.rack[BLANK_IDX] > 0 {
                true
            } else {
                continue;
            };
            let take = if is_blank { BLANK_IDX } else { idx };
            self.rack[take] -= 1;
            self.word[self.word_len] = idx as u8;
            self.word_len += 1;
            self.placed[self.placed_len] = Placed {
                pos: 0, // assigned when attached
                letter: idx,
                is_blank,
            };
            self.placed_len += 1;

            self.left_part(dict.node_at(t.addr), limit - 1);

            self.placed_len -= 1;
            self.word_len -= 1;
            self.rack[take] += 1;
        }
    }

    /// Assign board squares to the current left part (it ends just before
    /// the anchor), accumulate its score, and extend right from the anchor.
    fn attach_left_part_and_extend(&mut self, node: Node<'a>) {
        let len = self.placed_len;
        let start = self.anchor_pos - len;
        self.word_start = start;

        let mut main_sum = 0;
        let mut word_mult = 1;
        for i in 0..len {
            let pos = start + i;
            self.placed[i].pos = pos;
            let (r, c) = self.rc(pos);
            let premium = self.solver.premium_at(r, c);
            let value = if self.placed[i].is_blank {
                0
            } else {
                letter_value(index_letter(self.placed[i].letter))
            };
            main_sum += value * premium.letter_multiplier();
            word_mult *= premium.word_multiplier();
            // Left-part squares are never anchors, so they have no
            // perpendicular neighbours and contribute no cross-word score.
        }

        self.extend_right(node, self.anchor_pos, main_sum, word_mult, 0);
    }

    fn extend_right(
        &mut self,
        node: Node<'a>,
        pos: usize,
        main_sum: u32,
        word_mult: u32,
        cross_total: u32,
    ) {
        let dict = self.solver.dictionary;

        if pos == BOARD_SIZE {
            if node.is_final() && self.placed_len > 0 && self.word_len >= 2 {
                self.record(main_sum * word_mult + cross_total);
            }
            return;
        }

        if let Some(tile) = self.tile_at(pos) {
            if let Some(child) = dict.child(node, tile.letter as u8) {
                self.word[self.word_len] = letter_index(tile.letter).unwrap_or(0) as u8;
                self.word_len += 1;
                self.extend_right(
                    child,
                    pos + 1,
                    main_sum + tile.value(),
                    word_mult,
                    cross_total,
                );
                self.word_len -= 1;
            }
            return;
        }

        // Empty square: the word may end here (only once it covers the anchor).
        if node.is_final() && self.placed_len > 0 && self.word_len >= 2 && pos > self.anchor_pos {
            self.record(main_sum * word_mult + cross_total);
        }

        if self.placed_len == self.rack_len {
            return;
        }

        let (r, c) = self.rc(pos);
        let mask = self.info.cross_mask[self.di][r][c];
        if mask == 0 {
            return;
        }
        let has_cross = self.info.cross_has[self.di][r][c];
        let cross_score = self.info.cross_score[self.di][r][c];
        let premium = self.solver.premium_at(r, c);
        let lm = premium.letter_multiplier();
        let wm = premium.word_multiplier();

        for t in node.transitions() {
            let Some(idx) = letter_index(t.inp as char) else {
                continue;
            };
            if mask & (1 << idx) == 0 {
                continue;
            }
            // Prefer the real tile; fall back to a blank. Using the real tile
            // first keeps the blank available for later squares, which
            // explores a superset of plays; `record` fixes up which square
            // gets the blank for the best score.
            let is_blank = if self.rack[idx] > 0 {
                false
            } else if self.rack[BLANK_IDX] > 0 {
                true
            } else {
                continue;
            };
            let take = if is_blank { BLANK_IDX } else { idx };
            let value = if is_blank {
                0
            } else {
                letter_value(index_letter(idx))
            };
            let placed_value = value * lm;
            let new_cross = if has_cross {
                cross_total + (cross_score + placed_value) * wm
            } else {
                cross_total
            };

            self.rack[take] -= 1;
            self.word[self.word_len] = idx as u8;
            self.word_len += 1;
            self.placed[self.placed_len] = Placed {
                pos,
                letter: idx,
                is_blank,
            };
            self.placed_len += 1;

            self.extend_right(
                dict.node_at(t.addr),
                pos + 1,
                main_sum + placed_value,
                word_mult * wm,
                new_cross,
            );

            self.placed_len -= 1;
            self.word_len -= 1;
            self.rack[take] += 1;
        }
    }

    /// Record the current word as a move. `raw_score` excludes the bingo bonus.
    fn record(&mut self, raw_score: u32) {
        let n = self.placed_len;
        let mut score = raw_score;
        if n == RACK_SIZE {
            score += self.solver.config.bingo_bonus;
        }

        // Work on a copy: the blank fix-up below may move blank flags between
        // squares, and the live `self.placed` stack must stay in sync with
        // the rack counts for the rest of the search.
        let mut placed = self.placed;
        if placed[..n].iter().any(|p| p.is_blank) {
            let delta = self.optimize_blank_squares(&mut placed[..n]);
            score = (score as i64 + delta).max(0) as u32;
        }

        let word: String = self.word[..self.word_len]
            .iter()
            .map(|&i| index_letter(i as usize))
            .collect();

        let tiles_used: Vec<(usize, usize, Tile)> = placed[..n]
            .iter()
            .map(|p| {
                let (r, c) = self.rc(p.pos);
                let letter = index_letter(p.letter);
                let tile = if p.is_blank {
                    Tile::blank(letter)
                } else {
                    Tile::new(letter)
                };
                (r, c, tile)
            })
            .collect();

        let mut leave = Vec::with_capacity(self.rack_len - n);
        for (idx, &count) in self.rack.iter().enumerate() {
            for _ in 0..count {
                if idx == BLANK_IDX {
                    leave.push(Tile::blank(' '));
                } else {
                    leave.push(Tile::new(index_letter(idx)));
                }
            }
        }

        let leave_val = match self.leave_cache.get(&self.rack) {
            Some(&v) => v,
            None => {
                let v = self
                    .solver
                    .dictionary
                    .leave_value(&leave, Some(self.unseen));
                self.leave_cache.insert(self.rack, v);
                v
            }
        };
        let equity = score as f32 + leave_val;
        let (row, col) = self.rc(self.word_start);

        self.moves.push(Move {
            word,
            row,
            col,
            direction: self.dir,
            score,
            tiles_used,
            is_bingo: n == RACK_SIZE,
            leave,
            equity,
        });
    }

    /// When a letter appears among the placed tiles both as a real tile and
    /// as a blank, the search always committed the real tile to the earlier
    /// square. Move the real tiles onto the squares where they score most
    /// (updating `placed` in place) and return the score adjustment.
    fn optimize_blank_squares(&self, placed: &mut [Placed]) -> i64 {
        let n = placed.len();
        let mut word_mult = 1;
        for p in placed.iter() {
            let (r, c) = self.rc(p.pos);
            word_mult *= self.solver.premium_at(r, c).word_multiplier();
        }

        // Points gained per unit of letter value at each placed square.
        let coef = |p: &Placed| -> u32 {
            let (r, c) = self.rc(p.pos);
            let premium = self.solver.premium_at(r, c);
            let lm = premium.letter_multiplier();
            let mut v = lm * word_mult;
            if self.info.cross_has[self.di][r][c] {
                v += lm * premium.word_multiplier();
            }
            v
        };

        let mut delta = 0i64;
        let mut handled = 0u32;
        for i in 0..n {
            let letter = placed[i].letter;
            if handled & (1 << letter) != 0 {
                continue;
            }
            handled |= 1 << letter;

            let has_blank = placed.iter().any(|p| p.letter == letter && p.is_blank);
            let has_real = placed.iter().any(|p| p.letter == letter && !p.is_blank);
            if !(has_blank && has_real) {
                continue;
            }

            // Indices of all squares holding this letter, best coefficient
            // first (stable, so ties keep the search's original assignment).
            let mut idxs: Vec<usize> = (0..n).filter(|&j| placed[j].letter == letter).collect();
            idxs.sort_by_key(|&j| std::cmp::Reverse(coef(&placed[j])));
            let real_count = idxs.iter().filter(|&&j| !placed[j].is_blank).count();
            let value = letter_value(index_letter(letter)) as i64;

            let before: i64 = idxs
                .iter()
                .filter(|&&j| !placed[j].is_blank)
                .map(|&j| coef(&placed[j]) as i64)
                .sum();
            let after: i64 = idxs[..real_count]
                .iter()
                .map(|&j| coef(&placed[j]) as i64)
                .sum();
            for (k, &j) in idxs.iter().enumerate() {
                placed[j].is_blank = k >= real_count;
            }
            delta += value * (after - before);
        }

        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::load_default_dictionary;
    use std::collections::HashSet;

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

        // Every first-turn move must cover the centre square.
        for m in &moves {
            assert!(
                m.tiles_used.iter().any(|(r, c, _)| *r == 7 && *c == 7),
                "{} at ({},{}) does not cover the centre",
                m.word,
                m.row,
                m.col
            );
        }
    }

    #[test]
    fn test_solver_with_existing_tiles() {
        let dict = load_default_dictionary().unwrap();
        let mut board = Board::new();

        // Place "CAT" horizontally at (7, 7)
        board.place_tile(7, 7, Tile::new('C'));
        board.place_tile(7, 8, Tile::new('A'));
        board.place_tile(7, 9, Tile::new('T'));

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
        // Front hooks must also appear.
        assert!(
            moves.iter().any(|m| m.word == "SCAT"),
            "Should find the front hook SCAT"
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

        // Rack with tiles that could form vertical words
        let mut rack = Rack::new();
        rack.add(Tile::new('A'));
        rack.add(Tile::new('T'));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);

        // Every word formed (main and perpendicular) must be valid.
        for m in &moves {
            let mut board2 = board.clone();
            for (r, c, t) in &m.tiles_used {
                board2.place_tile(*r, *c, *t);
            }
            for (r, c, _) in &m.tiles_used {
                for dir in [Direction::Horizontal, Direction::Vertical] {
                    let (dr, dc) = dir.delta();
                    let mut sr = *r as i32;
                    let mut sc = *c as i32;
                    while Board::is_valid_position(sr - dr, sc - dc)
                        && board2
                            .get_tile((sr - dr) as usize, (sc - dc) as usize)
                            .is_some()
                    {
                        sr -= dr;
                        sc -= dc;
                    }
                    let mut word = String::new();
                    while Board::is_valid_position(sr, sc) {
                        match board2.get_tile(sr as usize, sc as usize) {
                            Some(t) => word.push(t.letter),
                            None => break,
                        }
                        sr += dr;
                        sc += dc;
                    }
                    if word.len() >= 2 {
                        assert!(
                            dict.is_valid(&word),
                            "{} forms invalid word {}",
                            m.word,
                            word
                        );
                    }
                }
            }
        }
        assert!(moves.iter().any(|m| m.direction == Direction::Vertical));
    }

    /// Proves that rack tile order does NOT affect which plays are found.
    /// This is the regression test for the blank-tile ordering bug:
    /// before the fix, having blank before a matching regular tile caused
    /// the solver to miss plays that needed the blank at a later position.
    #[test]
    fn test_blank_ordering_does_not_affect_play_set() {
        let dict = load_default_dictionary().unwrap();
        let board = Board::new(); // first turn

        // Rack 1: blank FIRST, then regular tiles
        let mut rack_blank_first = Rack::new();
        rack_blank_first.add(Tile::blank(' '));
        rack_blank_first.add(Tile::new('S'));
        rack_blank_first.add(Tile::new('E'));
        rack_blank_first.add(Tile::new('T'));
        rack_blank_first.add(Tile::new('R'));
        rack_blank_first.add(Tile::new('I'));
        rack_blank_first.add(Tile::new('N'));

        // Rack 2: blank LAST, same letters
        let mut rack_blank_last = Rack::new();
        rack_blank_last.add(Tile::new('S'));
        rack_blank_last.add(Tile::new('E'));
        rack_blank_last.add(Tile::new('T'));
        rack_blank_last.add(Tile::new('R'));
        rack_blank_last.add(Tile::new('I'));
        rack_blank_last.add(Tile::new('N'));
        rack_blank_last.add(Tile::blank(' '));

        let solver = Solver::new(&board, &dict);
        let moves1 = solver.find_moves(&rack_blank_first);
        let moves2 = solver.find_moves(&rack_blank_last);

        let words1: HashSet<_> = moves1
            .iter()
            .map(|m| (m.word.clone(), m.row, m.col, m.direction, m.score))
            .collect();
        let words2: HashSet<_> = moves2
            .iter()
            .map(|m| (m.word.clone(), m.row, m.col, m.direction, m.score))
            .collect();

        assert_eq!(
            words1, words2,
            "Rack tile order should not affect which plays are found or their scores"
        );
        assert_eq!(moves1.len(), moves2.len());
    }

    /// Test that blank tiles enable plays with letters not in the rack.
    #[test]
    fn test_blank_enables_missing_letters() {
        let dict = load_default_dictionary().unwrap();
        let board = Board::new();

        // Rack with blank and common letters but no Z, X, J, Q
        let mut rack = Rack::new();
        rack.add(Tile::blank(' '));
        rack.add(Tile::new('E'));
        rack.add(Tile::new('S'));
        rack.add(Tile::new('T'));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);

        let blank_words: HashSet<_> = moves
            .iter()
            .filter(|m| m.tiles_used.iter().any(|(_, _, t)| t.is_blank))
            .map(|m| m.word.clone())
            .collect();

        assert!(
            blank_words.len() > 20,
            "Should find many words using blank as various letters, found {}",
            blank_words.len()
        );

        // Specific check: ZEST should be found (Z from blank)
        assert!(
            moves.iter().any(|m| m.word == "ZEST"),
            "Should find ZEST (Z from blank)"
        );
    }

    /// Test blank plays on a populated board with cross-check constraints.
    #[test]
    fn test_blank_with_cross_checks() {
        let dict = load_default_dictionary().unwrap();
        let mut board = Board::new();

        // Place "WORD" horizontally at (7, 6)-(7, 9)
        board.place_tile(7, 6, Tile::new('W'));
        board.place_tile(7, 7, Tile::new('O'));
        board.place_tile(7, 8, Tile::new('R'));
        board.place_tile(7, 9, Tile::new('D'));

        let mut rack = Rack::new();
        rack.add(Tile::blank(' '));
        rack.add(Tile::new('E'));
        rack.add(Tile::new('A'));
        rack.add(Tile::new('T'));
        rack.add(Tile::new('S'));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);

        let blank_plays: Vec<_> = moves
            .iter()
            .filter(|m| m.tiles_used.iter().any(|(_, _, t)| t.is_blank))
            .collect();
        assert!(
            !blank_plays.is_empty(),
            "Should find plays using blank on populated board"
        );

        // Verify both rack orders produce the same plays
        let mut rack_reversed = Rack::new();
        rack_reversed.add(Tile::new('S'));
        rack_reversed.add(Tile::new('T'));
        rack_reversed.add(Tile::new('A'));
        rack_reversed.add(Tile::new('E'));
        rack_reversed.add(Tile::blank(' '));

        let moves_reversed = solver.find_moves(&rack_reversed);
        let words1: HashSet<_> = moves
            .iter()
            .map(|m| (m.word.clone(), m.row, m.col, m.direction, m.score))
            .collect();
        let words2: HashSet<_> = moves_reversed
            .iter()
            .map(|m| (m.word.clone(), m.row, m.col, m.direction, m.score))
            .collect();

        assert_eq!(
            words1, words2,
            "Rack order should not affect plays on populated board"
        );
    }

    /// When a word needs the same letter twice and the rack has one real
    /// copy plus a blank, the real tile must land on the square where it
    /// scores most (the blank is worth 0 regardless of premium).
    #[test]
    fn test_blank_goes_on_cheaper_square() {
        let dict = load_default_dictionary().unwrap();
        let board = Board::new();

        // Property check: for every move, swapping the blank flag between two
        // placed squares holding the same letter must not increase the score.
        let mut rack = Rack::new();
        for ch in ['E', 'E', 'T', 'S', 'A'] {
            rack.add(Tile::new(ch));
        }
        rack.add(Tile::blank(' '));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);
        for m in &moves {
            let n = m.tiles_used.len();
            for i in 0..n {
                for j in 0..n {
                    let (_, _, ti) = m.tiles_used[i];
                    let (_, _, tj) = m.tiles_used[j];
                    if i == j || ti.letter != tj.letter || ti.is_blank == tj.is_blank {
                        continue;
                    }
                    let mut swapped = m.tiles_used.clone();
                    swapped[i].2.is_blank = tj.is_blank;
                    swapped[j].2.is_blank = ti.is_blank;
                    let alt = solver.score_placed_tiles(&swapped, m.direction);
                    assert!(
                        alt <= m.score,
                        "{} at {}: blank placement scores {} but swapping gives {}",
                        m.word,
                        m.coords(),
                        m.score,
                        alt
                    );
                }
            }
        }

        // Concrete case: rack has one A and a blank; "ALA" style words put
        // the two A's on different premiums. Check a known word explicitly.
        let mut rack = Rack::new();
        rack.add(Tile::new('A'));
        rack.add(Tile::blank(' '));
        rack.add(Tile::new('L'));
        let moves = solver.find_moves(&rack);
        let ala: Vec<_> = moves.iter().filter(|m| m.word == "ALA").collect();
        assert!(!ala.is_empty(), "ALA should be generated");
        for m in ala {
            let real_a = m
                .tiles_used
                .iter()
                .find(|(_, _, t)| t.letter == 'A' && !t.is_blank)
                .expect("one real A");
            let blank_a = m
                .tiles_used
                .iter()
                .find(|(_, _, t)| t.letter == 'A' && t.is_blank)
                .expect("one blank A");
            let prem = |rc: (usize, usize)| board.get(rc.0, rc.1).unwrap().premium;
            let real_lm = prem((real_a.0, real_a.1)).letter_multiplier();
            let blank_lm = prem((blank_a.0, blank_a.1)).letter_multiplier();
            assert!(
                real_lm >= blank_lm,
                "ALA at {}: real A on x{} square, blank on x{}",
                m.coords(),
                real_lm,
                blank_lm
            );
        }
    }

    /// Incremental scoring during search must agree with the standalone
    /// scorer used for opponent moves.
    #[test]
    fn test_incremental_score_matches_standalone() {
        let dict = load_default_dictionary().unwrap();
        let mut board = Board::new();
        board.place_tile(7, 3, Tile::new('B'));
        board.place_tile(7, 4, Tile::new('E'));
        board.place_tile(7, 5, Tile::new('E'));
        board.place_tile(7, 6, Tile::new('C'));
        board.place_tile(7, 7, Tile::new('H'));
        board.place_tile(8, 7, Tile::new('I'));
        board.place_tile(9, 7, Tile::new('K'));
        board.place_tile(10, 7, Tile::new('E'));
        board.place_tile(11, 7, Tile::blank('R'));

        let mut rack = Rack::new();
        for ch in ['A', 'E', 'I', 'S', 'T', 'R'] {
            rack.add(Tile::new(ch));
        }
        rack.add(Tile::blank(' '));

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);
        assert!(moves.len() > 100);
        for m in &moves {
            let standalone = solver.score_placed_tiles(&m.tiles_used, m.direction);
            assert_eq!(
                m.score,
                standalone,
                "{} at {}: incremental {} vs standalone {}",
                m.word,
                m.coords(),
                m.score,
                standalone
            );
        }
    }

    /// Each distinct tile placement must appear exactly once.
    #[test]
    fn test_no_duplicate_placements() {
        let dict = load_default_dictionary().unwrap();
        let mut board = Board::new();
        board.place_tile(7, 6, Tile::new('C'));
        board.place_tile(7, 7, Tile::new('A'));
        board.place_tile(7, 8, Tile::new('T'));
        board.place_tile(6, 8, Tile::new('O'));

        let mut rack = Rack::new();
        for ch in ['S', 'E', 'R', 'A', 'T'] {
            rack.add(Tile::new(ch));
        }

        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);
        let mut seen = HashSet::new();
        for m in &moves {
            let mut key: Vec<(usize, usize, char, bool)> = m
                .tiles_used
                .iter()
                .map(|(r, c, t)| (*r, *c, t.letter, t.is_blank))
                .collect();
            key.sort();
            assert!(
                seen.insert(key.clone()),
                "duplicate placement for {} at {}",
                m.word,
                m.coords()
            );
        }
        // The single S hooked after CAT is a one-tile play and must be listed
        // exactly once even though it is reachable from both directions.
        assert_eq!(
            moves
                .iter()
                .filter(|m| {
                    m.tiles_used.len() == 1
                        && m.tiles_used[0].0 == 7
                        && m.tiles_used[0].1 == 9
                        && m.tiles_used[0].2.letter == 'S'
                })
                .count(),
            1
        );
    }

    #[test]
    fn test_bingo_bonus_and_leave() {
        let dict = load_default_dictionary().unwrap();
        let board = Board::new();
        let mut rack = Rack::new();
        for ch in ['R', 'E', 'T', 'A', 'I', 'N', 'S'] {
            rack.add(Tile::new(ch));
        }
        let solver = Solver::new(&board, &dict);
        let moves = solver.find_moves(&rack);
        let retains = moves
            .iter()
            .find(|m| m.word == "RETAINS" && m.direction == Direction::Horizontal && m.col == 3)
            .expect("RETAINS at 8D");
        assert!(retains.is_bingo);
        assert!(retains.leave.is_empty());
        // 7 one-point tiles on row 7 starting at col 3 (DW) with DL at 5 and 9:
        // (1+1+2+1+1+1+2) * 2 = 18, plus 40 bingo.
        assert_eq!(retains.score, 18 + BINGO_BONUS);
        assert_eq!(retains.coords(), "8D");
    }

    /// Timing harness for move generation. Run with
    /// `cargo test --release bench_movegen -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_movegen() {
        use crate::game::SolverState;
        use std::time::Instant;

        let dict = load_default_dictionary().unwrap();
        const MID: &str = "14U/5HELM3ZIN/4VIG7T/6AYES3BI/6DOWER2AL/9GOOFS1/1BRA4GOWN1I1/POETIC1QI2L1N1/OX3H1U3Y3/R3FINAlS5/T4V1K1P5/A4e1EME5/L6DON5/EtHER3AD5/D2RECOAT6";
        const BEECH: &str = "15/15/15/15/15/15/15/3BEECH7/7I7/7K7/7E7/7r7/15/15/15";
        const END: &str = "3N3Q3M3/3OP2UNTEACH1/3BE2O3N3/3LA2T2MYOPE/2FeR2H1LO4/2O6AB4/2R1F4U3De/2TRIVET1D3AY/4LA1A1IODIZE/4KI1JINX2ES/5R1E1G4O/OUTIES1S3G2R/11L2E/8WAHINES/2EGAlITE2A3";
        let positions = [
            (
                "empty AEINRST",
                "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AEINRST".to_string(),
            ),
            ("midgame JSIRATU", format!("{} JSIRATU", MID)),
            ("endgame RCVSDW", format!("{} RCVSDW", END)),
            ("beech AEIO?ST", format!("{} AEIO?ST", BEECH)),
            ("beech ??EINRS", format!("{} ??EINRS", BEECH)),
        ];
        for (name, cgp) in &positions {
            let state = SolverState::from_cgp(cgp).unwrap();
            let solver = Solver::new(&state.board, &dict);
            let _ = solver.find_moves(&state.rack);
            let iters = 20;
            let t = Instant::now();
            let mut n = 0;
            for _ in 0..iters {
                n = solver.find_moves(&state.rack).len();
            }
            println!("{:18} {:6} moves  {:?}/call", name, n, t.elapsed() / iters);
        }
    }
}
