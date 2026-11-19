//! The rules of a game: board layout, tile values and distribution, the
//! bingo bonus and how the game ends.
//!
//! Every board carries its rules (see [`crate::board::Board::with_rules`]),
//! so scoring, move generation, tile counting and the simulation read them
//! from the position they are given. [`CROSSPLAY`] and [`SCRABBLE`] are the
//! presets; board and rack size are fixed at compile time
//! ([`BOARD_SIZE`], [`RACK_SIZE`]).
//!
//! Rules live for the whole program (`&'static`): use a preset, or leak a
//! custom set once with `Box::leak`.

use crate::board::{Premium, BOARD_SIZE};
use crate::tile::Tile;

#[cfg(doc)]
use crate::tile::RACK_SIZE;

/// How the game ends once the bag is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Ending {
    /// Crossplay: when a player empties the bag, the opponent takes one
    /// more turn and the player who emptied it plays the last move.
    /// Unplayed tiles score nothing and going out earns no bonus.
    EqualTurn,
    /// Scrabble: play continues after the bag empties until a player uses
    /// their last tile, who then gains the value of the opponent's rack
    /// while the opponent loses it; or until six turns in a row score
    /// nothing, when each player loses the value of their own rack.
    GoOut,
}

impl Ending {
    /// Consecutive scoreless turns that end a [`Ending::GoOut`] game.
    pub const SCORELESS_TURNS_TO_END: usize = 6;
}

/// The rules of a game.
#[derive(Debug, PartialEq, Eq)]
pub struct Rules {
    /// Short name, for logs and display.
    pub name: &'static str,
    /// Premium square at each position, `[row][col]`.
    pub premiums: [[Premium; BOARD_SIZE]; BOARD_SIZE],
    /// The square the first play must cover, `(row, col)`.
    pub start: (usize, usize),
    /// Points for each letter, `A` to `Z`. Blanks score 0.
    pub letter_values: [u8; 26],
    /// Tiles of each letter in a fresh bag, `A` to `Z`, then blanks.
    pub distribution: [u8; 27],
    /// Points for playing every tile on the rack in one turn.
    pub bingo_bonus: u32,
    /// How the game ends.
    pub ending: Ending,
}

impl Rules {
    /// Points for `letter` (either case); 0 for anything but `A` to `Z`.
    #[inline]
    pub fn letter_value(&self, letter: char) -> u32 {
        match letter.to_ascii_uppercase() {
            c @ 'A'..='Z' => self.letter_values[(c as u8 - b'A') as usize] as u32,
            _ => 0,
        }
    }

    /// Points for letter index `idx` (0 = `A`).
    #[inline]
    pub(crate) fn index_value(&self, idx: usize) -> u32 {
        self.letter_values[idx] as u32
    }

    /// Points for `tile`: its letter's value, or 0 for a blank.
    #[inline]
    pub fn tile_value(&self, tile: &Tile) -> u32 {
        if tile.is_blank {
            0
        } else {
            self.letter_value(tile.letter)
        }
    }

    /// Tiles of `letter` in a fresh bag (`'?'` for blanks).
    pub fn count(&self, letter: char) -> u32 {
        match letter.to_ascii_uppercase() {
            '?' => self.distribution[26] as u32,
            c @ 'A'..='Z' => self.distribution[(c as u8 - b'A') as usize] as u32,
            _ => 0,
        }
    }

    /// Blank tiles in a fresh bag.
    pub fn blank_count(&self) -> usize {
        self.distribution[26] as usize
    }

    /// Tiles in a fresh bag.
    pub fn total_tiles(&self) -> usize {
        self.distribution.iter().map(|&n| n as usize).sum()
    }

    /// Every tile in a fresh bag: blanks first, then `A` to `Z`.
    pub fn full_tile_set(&self) -> Vec<Tile> {
        let mut tiles = Vec::with_capacity(self.total_tiles());
        tiles.extend((0..self.blank_count()).map(|_| Tile::blank(' ')));
        for (i, &n) in self.distribution[..26].iter().enumerate() {
            let letter = (b'A' + i as u8) as char;
            tiles.extend((0..n).map(|_| Tile::new(letter)));
        }
        tiles
    }
}

/// Set `squares` to `premium` in `layout`.
const fn mark(
    mut layout: [[Premium; BOARD_SIZE]; BOARD_SIZE],
    squares: &[(usize, usize)],
    premium: Premium,
) -> [[Premium; BOARD_SIZE]; BOARD_SIZE] {
    let mut i = 0;
    while i < squares.len() {
        layout[squares[i].0][squares[i].1] = premium;
        i += 1;
    }
    layout
}

/// Crossplay's premium squares. The centre star scores as a plain
/// square.
const fn crossplay_layout() -> [[Premium; BOARD_SIZE]; BOARD_SIZE] {
    let layout = [[Premium::Normal; BOARD_SIZE]; BOARD_SIZE];
    let layout = mark(
        layout,
        &[
            (0, 3),
            (0, 11),
            (3, 0),
            (3, 14),
            (11, 0),
            (11, 14),
            (14, 3),
            (14, 11),
        ],
        Premium::TripleWord,
    );
    let layout = mark(
        layout,
        &[
            (1, 1),
            (1, 13),
            (3, 7),
            (7, 3),
            (7, 11),
            (11, 7),
            (13, 1),
            (13, 13),
        ],
        Premium::DoubleWord,
    );
    let layout = mark(layout, &[(7, 7)], Premium::Center);
    let layout = mark(
        layout,
        &[
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
        ],
        Premium::TripleLetter,
    );
    mark(
        layout,
        &[
            (0, 7),
            (2, 4),
            (2, 10),
            (3, 3),
            (3, 11),
            (4, 2),
            (4, 12),
            (5, 7),
            (7, 0),
            (7, 5),
            (7, 9),
            (7, 14),
            (9, 7),
            (10, 2),
            (10, 12),
            (11, 3),
            (11, 11),
            (12, 4),
            (12, 10),
            (14, 7),
        ],
        Premium::DoubleLetter,
    )
}

/// Crossplay.
#[rustfmt::skip] // keep the letter rulers lined up with the values
pub static CROSSPLAY: Rules = Rules {
    name: "Crossplay",
    premiums: crossplay_layout(),
    start: (BOARD_SIZE / 2, BOARD_SIZE / 2),
    //              A  B  C  D  E  F  G  H  I  J   K  L  M  N  O  P  Q   R  S  T  U  V  W  X  Y  Z
    letter_values: [1, 4, 3, 2, 1, 4, 4, 3, 1, 10, 6, 2, 3, 1, 1, 3, 10, 1, 1, 1, 2, 6, 5, 8, 4, 10],
    //             A  B  C  D  E   F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z  ?
    distribution: [9, 2, 2, 4, 12, 2, 3, 3, 8, 1, 1, 4, 2, 5, 8, 2, 1, 6, 5, 6, 3, 2, 2, 1, 2, 1, 3],
    bingo_bonus: 40,
    ending: Ending::EqualTurn,
};

/// Standard (English-language) Scrabble's premium squares. The centre is a
/// double word square.
const fn scrabble_layout() -> [[Premium; BOARD_SIZE]; BOARD_SIZE] {
    let layout = [[Premium::Normal; BOARD_SIZE]; BOARD_SIZE];
    let layout = mark(
        layout,
        &[
            (0, 0),
            (0, 7),
            (0, 14),
            (7, 0),
            (7, 14),
            (14, 0),
            (14, 7),
            (14, 14),
        ],
        Premium::TripleWord,
    );
    let layout = mark(
        layout,
        &[
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (10, 10),
            (11, 11),
            (12, 12),
            (13, 13),
            (1, 13),
            (2, 12),
            (3, 11),
            (4, 10),
            (10, 4),
            (11, 3),
            (12, 2),
            (13, 1),
            (7, 7),
        ],
        Premium::DoubleWord,
    );
    let layout = mark(
        layout,
        &[
            (1, 5),
            (1, 9),
            (5, 1),
            (5, 5),
            (5, 9),
            (5, 13),
            (9, 1),
            (9, 5),
            (9, 9),
            (9, 13),
            (13, 5),
            (13, 9),
        ],
        Premium::TripleLetter,
    );
    mark(
        layout,
        &[
            (0, 3),
            (0, 11),
            (2, 6),
            (2, 8),
            (3, 0),
            (3, 7),
            (3, 14),
            (6, 2),
            (6, 6),
            (6, 8),
            (6, 12),
            (7, 3),
            (7, 11),
            (8, 2),
            (8, 6),
            (8, 8),
            (8, 12),
            (11, 0),
            (11, 7),
            (11, 14),
            (12, 6),
            (12, 8),
            (14, 3),
            (14, 11),
        ],
        Premium::DoubleLetter,
    )
}

/// Standard English-language Scrabble: its board, values and 100 tiles, a
/// 50-point bingo, and the go-out ending.
#[rustfmt::skip] // keep the letter rulers lined up with the values
pub static SCRABBLE: Rules = Rules {
    name: "Scrabble",
    premiums: scrabble_layout(),
    start: (BOARD_SIZE / 2, BOARD_SIZE / 2),
    //              A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q   R  S  T  U  V  W  X  Y  Z
    letter_values: [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10],
    //             A  B  C  D  E   F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z  ?
    distribution: [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2],
    bingo_bonus: 50,
    ending: Ending::GoOut,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossplay_values() {
        let r = &CROSSPLAY;
        for (letters, value) in [
            ("AEINORST", 1),
            ("DLU", 2),
            ("CHMP", 3),
            ("BFGY", 4),
            ("W", 5),
            ("KV", 6),
            ("X", 8),
            ("JQZ", 10),
        ] {
            for c in letters.chars() {
                assert_eq!(r.letter_value(c), value, "{c}");
                assert_eq!(r.letter_value(c.to_ascii_lowercase()), value, "{c}");
            }
        }
        assert_eq!(r.letter_value('?'), 0);
        assert_eq!(r.tile_value(&Tile::blank('Q')), 0);
        assert_eq!(r.tile_value(&Tile::new('Q')), 10);
    }

    #[test]
    fn crossplay_bag() {
        let r = &CROSSPLAY;
        assert_eq!(r.total_tiles(), 100);
        assert_eq!(r.blank_count(), 3);
        assert_eq!(r.count('E'), 12);
        assert_eq!(r.count('?'), 3);
        let set = r.full_tile_set();
        assert_eq!(set.len(), 100);
        assert!(set[..3].iter().all(|t| t.is_blank));
        assert_eq!(set[3], Tile::new('A'));
        assert_eq!(set[99], Tile::new('Z'));
    }

    /// Unlike Crossplay in every setting: no premiums, an off-centre start,
    /// every letter worth 1, a 50-point bingo and 108 tiles.
    static PLAIN: Rules = Rules {
        name: "Plain",
        premiums: [[Premium::Normal; BOARD_SIZE]; BOARD_SIZE],
        start: (3, 4),
        letter_values: [1; 26],
        distribution: [4; 27],
        bingo_bonus: 50,
        ending: Ending::EqualTurn,
    };

    fn tiles(letters: &str) -> Vec<Tile> {
        letters.chars().map(Tile::new).collect()
    }

    #[test]
    fn move_generation_follows_the_boards_rules() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = crate::board::Board::with_rules(&PLAIN);
        let moves = crate::solver::Solver::new(&board, &dict).find_moves_for(&tiles("AEINRST"));
        let plays: Vec<_> = moves
            .iter()
            .filter(|m| m.kind == crate::solver::MoveKind::Play)
            .collect();
        assert!(!plays.is_empty());
        for m in &plays {
            assert!(
                m.tiles_used.iter().any(|&(r, c, _)| (r, c) == PLAIN.start),
                "{} misses the start square",
                m.word
            );
            // No premiums and 1 point a letter: the score is the length,
            // plus PLAIN's bingo bonus for seven tiles.
            let bonus = if m.is_bingo { 50 } else { 0 };
            assert_eq!(m.score as usize, m.word.len() + bonus, "{}", m.word);
        }
        assert!(plays.iter().any(|m| m.is_bingo));
    }

    #[test]
    fn tile_counts_follow_the_boards_rules() {
        let mut state = crate::game::SolverState::new();
        state.board = crate::board::Board::with_rules(&PLAIN);
        assert_eq!(state.unseen_tile_count(), 108);
        assert!(state.add_to_rack('Q'));
        assert_eq!(state.unseen_tile_count(), 107);
        assert_eq!(state.unseen_tiles()[&'?'], 4);
    }

    #[test]
    fn leave_values_use_the_rules_bingo_bonus() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        if !dict.has_leave_table() {
            return;
        }
        let blank = [Tile::blank(' ')];
        let crossplay = dict.leave_value(&blank, None);
        let dict = dict.for_rules(&PLAIN);
        assert_eq!(dict.bingo_bonus(), 50);
        // The table was fitted under a 50-point bonus, so PLAIN needs no
        // correction; Crossplay's 40 makes a bingo-prone leave worth less.
        assert!(dict.leave_value(&blank, None) > crossplay);
    }

    #[test]
    fn scrabble_preset() {
        let r = &SCRABBLE;
        assert_eq!(r.total_tiles(), 100);
        assert_eq!(r.blank_count(), 2);
        assert_eq!(r.letter_value('K'), 5);
        assert_eq!(r.letter_value('J'), 8);
        let p = &r.premiums;
        let count = |want: Premium| p.iter().flatten().filter(|&&q| q == want).count();
        assert_eq!(count(Premium::TripleWord), 8);
        assert_eq!(count(Premium::DoubleWord), 17);
        assert_eq!(count(Premium::TripleLetter), 12);
        assert_eq!(count(Premium::DoubleLetter), 24);
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                assert_eq!(p[r][c], p[c][r]);
                assert_eq!(p[r][c], p[BOARD_SIZE - 1 - r][c]);
            }
        }
        assert_eq!(p[7][7], Premium::DoubleWord);
    }

    /// The best opening bingo with seven 1-point tiles scores 66 in
    /// Scrabble: doubled by the centre square, one letter on a double
    /// letter square, plus 50.
    #[test]
    fn scrabble_scores_an_opening_bingo() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = crate::board::Board::with_rules(&SCRABBLE);
        let best = crate::solver::Solver::new(&board, &dict)
            .best_move_by_score(&tiles("AEINRST"))
            .unwrap();
        assert!(best.is_bingo);
        assert_eq!(best.score, 66, "{}", best.word);
    }

    #[test]
    fn crossplay_layout_is_symmetric() {
        let p = &CROSSPLAY.premiums;
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                assert_eq!(p[r][c], p[c][r], "diagonal ({r},{c})");
                assert_eq!(p[r][c], p[BOARD_SIZE - 1 - r][c], "vertical ({r},{c})");
                assert_eq!(p[r][c], p[r][BOARD_SIZE - 1 - c], "horizontal ({r},{c})");
            }
        }
        assert_eq!(p[7][7], Premium::Center);
    }
}
