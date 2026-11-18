//! GCG, the plain-text record of a crossword game: reading, writing,
//! replaying and reviewing.
//!
//! A GCG file is a list of pragmas (`#player1 nick Full Name`, `#note …`,
//! `#lexicon …`) and events, one per line:
//!
//! ```text
//! #player1 alice Alice
//! #player2 sam Sam
//! >alice: AEINRST 8D STAINER +72 72
//! >sam: ?EGLNOT D4 tOG.LE +20 20
//! >alice: ADEEIMU -AEU +0 72
//! >sam: ABEIORT - +0 20
//! ```
//!
//! An event is `>nick: RACK MOVE SCORE TOTAL`; the rack (before the move)
//! may be left out. Plays give the square and direction (`8D`, row first,
//! goes across; `D8`, column first, goes down) and the word, with `.` (or
//! a parenthesised letter) for a tile already on the board and lowercase
//! for a blank. The other moves are `-` (pass), `-TILES` or `-N` (an
//! exchange, shown or not), `--` (the previous play withdrawn after a
//! challenge), `(challenge)` (a challenge bonus), `(time)` (a time
//! penalty) and `(TILES)` (points for tiles left on a rack at the end).
//!
//! [`Game::replay`] rebuilds the board before every event, and [`review`]
//! compares each move whose rack is known with the engine's best by
//! static equity.

use std::fmt;

use crate::board::{Board, Direction, BOARD_SIZE};
use crate::dictionary::Dictionary;
use crate::game::SolverState;
use crate::rules::Rules;
use crate::solver::{Move, MoveKind, Solver};
use crate::tile::{Tile, RACK_SIZE};

/// A problem in a GCG file, at a line (1-based; 0 for the file as a whole).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcgError {
    pub line: usize,
    pub message: String,
}

impl GcgError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        GcgError {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for GcgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "{}", self.message)
        } else {
            write!(f, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for GcgError {}

/// A player: the nickname events use, and a full name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Player {
    pub nick: String,
    pub name: String,
}

/// Where a play goes and what it spells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// First square of the word, 0-based.
    pub row: usize,
    pub col: usize,
    pub direction: Direction,
    /// The word, one character per square: `.` for a tile already on the
    /// board, an uppercase letter for a tile placed, lowercase for a blank.
    pub word: String,
}

impl Placement {
    /// Parse a position (`8D` across, `D8` down) and a word (`.` or
    /// `(LETTERS)` for tiles already on the board, lowercase for blanks).
    pub fn parse(position: &str, word: &str) -> Result<Self, String> {
        let (row, col, direction) = parse_position(position)
            .ok_or_else(|| format!("bad position {position:?} (8D across, D8 down)"))?;
        let mut squares = String::new();
        let mut through = false;
        for ch in word.chars() {
            match ch {
                '(' if !through => through = true,
                ')' if through => through = false,
                '.' if !through => squares.push('.'),
                c if c.is_ascii_alphabetic() => squares.push(if through { '.' } else { c }),
                _ => return Err(format!("bad word {word:?}")),
            }
        }
        if through || squares.is_empty() || !squares.chars().any(|c| c != '.') {
            return Err(format!("bad word {word:?}"));
        }
        let (dr, dc) = direction.delta();
        let len = squares.chars().count() as i32;
        if !Board::is_valid_position(row as i32 + dr * (len - 1), col as i32 + dc * (len - 1)) {
            return Err(format!("{word} at {position} runs off the board"));
        }
        Ok(Placement {
            row,
            col,
            direction,
            word: squares,
        })
    }

    /// The placement of a play ([`MoveKind::Play`]); `None` for anything
    /// else. Squares of the word the move did not place become `.`.
    pub fn of_move(mv: &Move) -> Option<Self> {
        if mv.kind != MoveKind::Play {
            return None;
        }
        let (dr, dc) = mv.direction.delta();
        let word = (0..mv.word.chars().count())
            .map(|i| {
                let r = (mv.row as i32 + dr * i as i32) as usize;
                let c = (mv.col as i32 + dc * i as i32) as usize;
                match mv
                    .tiles_used
                    .iter()
                    .find(|&&(pr, pc, _)| pr == r && pc == c)
                {
                    Some((_, _, t)) if t.is_blank => t.letter.to_ascii_lowercase(),
                    Some((_, _, t)) => t.letter,
                    None => '.',
                }
            })
            .collect::<String>();
        Some(Placement {
            row: mv.row,
            col: mv.col,
            direction: mv.direction,
            word,
        })
    }

    /// The position in GCG notation: `8D` across, `D8` down.
    pub fn position(&self) -> String {
        let letter = (b'A' + self.col as u8) as char;
        match self.direction {
            Direction::Horizontal => format!("{}{}", self.row + 1, letter),
            Direction::Vertical => format!("{}{}", letter, self.row + 1),
        }
    }

    /// The squares of the word, in order.
    fn squares(&self) -> impl Iterator<Item = (usize, usize, char)> + '_ {
        let (dr, dc) = self.direction.delta();
        self.word.chars().enumerate().map(move |(i, ch)| {
            (
                (self.row as i32 + dr * i as i32) as usize,
                (self.col as i32 + dc * i as i32) as usize,
                ch,
            )
        })
    }

    /// The tiles this play puts on `board`, in word order. Fails if a `.`
    /// square is empty or a placed square is taken.
    pub fn tiles(&self, board: &Board) -> Result<Vec<(usize, usize, Tile)>, String> {
        let mut placed = Vec::new();
        for (r, c, ch) in self.squares() {
            let label = square_name(r, c);
            match (ch, board.get_tile(r, c)) {
                ('.', Some(_)) => {}
                ('.', None) => return Err(format!("{label} should hold a tile but is empty")),
                (_, Some(t)) => {
                    return Err(format!("{label} already holds {}", t.letter));
                }
                (ch, None) => placed.push((
                    r,
                    c,
                    if ch.is_ascii_lowercase() {
                        Tile::blank(ch)
                    } else {
                        Tile::new(ch)
                    },
                )),
            }
        }
        Ok(placed)
    }

    /// The placement of tiles just put on `board` (in any order): the
    /// direction they line up in (a single tile goes across when it touches
    /// a tile left or right of it, else down) and the whole word they make
    /// with the tiles already there. `None` if they are not in one line.
    pub fn from_tiles(board: &Board, placed: &[(usize, usize, Tile)]) -> Option<Self> {
        let &(r0, c0, _) = placed.first()?;
        let direction = if placed.len() == 1 {
            let beside = |c: Option<usize>| c.is_some_and(|c| board.get_tile(r0, c).is_some());
            if beside(c0.checked_sub(1)) || beside(Some(c0 + 1).filter(|&c| c < BOARD_SIZE)) {
                Direction::Horizontal
            } else {
                Direction::Vertical
            }
        } else if placed.iter().all(|&(r, _, _)| r == r0) {
            Direction::Horizontal
        } else if placed.iter().all(|&(_, c, _)| c == c0) {
            Direction::Vertical
        } else {
            return None;
        };
        let (dr, dc) = direction.delta();
        let along = |&(r, c, _): &(usize, usize, Tile)| if dc == 1 { c } else { r };
        let first = placed.iter().min_by_key(|t| along(t))?;
        let last = placed.iter().map(along).max()?;
        let at = |i: i32| (first.0 as i32 + dr * i, first.1 as i32 + dc * i);
        // Back over tiles already there to the start of the word.
        let mut start = 0;
        loop {
            let (r, c) = at(start - 1);
            if !Board::is_valid_position(r, c) || board.get_tile(r as usize, c as usize).is_none() {
                break;
            }
            start -= 1;
        }
        let mut word = String::new();
        let mut i = start;
        loop {
            let (r, c) = at(i);
            if !Board::is_valid_position(r, c) {
                break;
            }
            let (r, c) = (r as usize, c as usize);
            let ch = match placed.iter().find(|&&(pr, pc, _)| pr == r && pc == c) {
                Some((_, _, t)) if t.is_blank => t.letter.to_ascii_lowercase(),
                Some((_, _, t)) => t.letter,
                None if board.get_tile(r, c).is_some() => '.',
                // A gap inside the play: not one word.
                None if (if dc == 1 { c } else { r }) < last => return None,
                None => break,
            };
            word.push(ch);
            i += 1;
        }
        let (row, col) = at(start);
        Some(Placement {
            row: row as usize,
            col: col as usize,
            direction,
            word,
        })
    }

    /// The whole word on `board`, uppercase, existing tiles filled in.
    pub fn word_on(&self, board: &Board) -> String {
        self.squares()
            .map(|(r, c, ch)| match ch {
                '.' => board.get_tile(r, c).map_or('?', |t| t.letter),
                ch => ch.to_ascii_uppercase(),
            })
            .collect()
    }
}

/// `H8`-style name of a square, for messages.
fn square_name(row: usize, col: usize) -> String {
    format!("{}{}", (b'A' + col as u8) as char, row + 1)
}

/// `8D` → (7, 3, across); `D8` → (7, 3, down).
fn parse_position(s: &str) -> Option<(usize, usize, Direction)> {
    let s = s.to_ascii_uppercase();
    let digits_first = s.starts_with(|c: char| c.is_ascii_digit());
    let split = s.find(|c: char| c.is_ascii_digit() != digits_first)?;
    let (a, b) = s.split_at(split);
    let (number, letter, direction) = if digits_first {
        (a, b, Direction::Horizontal)
    } else {
        (b, a, Direction::Vertical)
    };
    let row = number.parse::<usize>().ok()?.checked_sub(1)?;
    let mut letters = letter.chars();
    let col = letters.next()? as usize;
    if letters.next().is_some() || !(b'A' as usize..b'A' as usize + BOARD_SIZE).contains(&col) {
        return None;
    }
    (row < BOARD_SIZE).then_some((row, col - b'A' as usize, direction))
}

/// What an event does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Play(Placement),
    /// These tiles (`?` for a blank) went back in the bag.
    Exchange(String),
    /// An exchange of this many tiles, which are not shown.
    ExchangeCount(usize),
    Pass,
    /// The player's previous play came off the board after a challenge.
    Withdrawn,
    /// Points for an unsuccessful challenge of the player's play.
    ChallengeBonus,
    /// A time penalty.
    TimePenalty,
    /// Points for tiles left on a rack when the game ended: gained from the
    /// opponent's rack, or lost from one's own.
    EndRack(String),
}

/// One line of play.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// 0 for the first player, 1 for the second.
    pub player: usize,
    /// The rack before the move (`?` for a blank); empty when unknown.
    pub rack: String,
    pub action: Action,
    /// Points this event added (negative for a withdrawal or penalty).
    pub score: i32,
    /// The player's total after it.
    pub total: i32,
    /// `#note` lines that follow the event.
    pub notes: Vec<String>,
    /// Line in the file it came from (0 when built in code).
    pub line: usize,
}

/// A whole game record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Game {
    pub players: [Player; 2],
    /// Every other pragma, in file order, as (name, value): `lexicon`,
    /// `title`, `id`, …
    pub pragmas: Vec<(String, String)>,
    /// Notes before the first event.
    pub notes: Vec<String>,
    pub events: Vec<Event>,
}

/// Letters of a rack as GCG writes them: uppercase, `?` for a blank.
pub fn rack_string(tiles: &[Tile]) -> String {
    tiles
        .iter()
        .map(|t| if t.is_blank { '?' } else { t.letter })
        .collect()
}

/// Tiles from a GCG rack string (`?` for a blank).
pub fn parse_rack(rack: &str) -> Vec<Tile> {
    rack.chars()
        .map(|c| {
            if c == '?' {
                Tile::blank(' ')
            } else {
                Tile::new(c)
            }
        })
        .collect()
}

fn is_rack(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic() || c == '?')
}

fn parse_score(s: &str) -> Option<i32> {
    s.strip_prefix('+').unwrap_or(s).parse().ok()
}

impl Game {
    /// A record with no events between these two players.
    pub fn new(first: Player, second: Player) -> Self {
        Game {
            players: [first, second],
            ..Game::default()
        }
    }

    /// Read a GCG file.
    pub fn parse(text: &str) -> Result<Self, GcgError> {
        let mut game = Game::default();
        let mut named = [false; 2];
        // Where a continuation line of a note goes.
        let mut in_note = false;
        for (index, raw) in text.lines().enumerate() {
            let line_no = index + 1;
            let line = raw.trim_end_matches('\r');
            // Names, notes and pragmas end up on screen: a control
            // character (an escape sequence, say) could drive the terminal
            // that shows them, so a file holding one is refused.
            if let Some(c) = line.chars().find(|&c| c.is_control() && c != '\t') {
                return Err(GcgError::new(
                    line_no,
                    format!("control character {:?} in the file", c),
                ));
            }
            if let Some(pragma) = line.strip_prefix('#') {
                let (name, value) = pragma
                    .split_once(char::is_whitespace)
                    .unwrap_or((pragma, ""));
                let value = value.trim();
                in_note = false;
                match name {
                    "player1" | "player2" => {
                        let slot = usize::from(name == "player2");
                        let (nick, full) =
                            value.split_once(char::is_whitespace).unwrap_or((value, ""));
                        if nick.is_empty() || nick.contains(':') {
                            return Err(GcgError::new(
                                line_no,
                                format!("#{name} needs a nickname (without a colon)"),
                            ));
                        }
                        if named[slot] {
                            return Err(GcgError::new(
                                line_no,
                                format!(
                                    "player {} is already {}",
                                    slot + 1,
                                    game.players[slot].nick
                                ),
                            ));
                        }
                        if named[1 - slot] && game.players[1 - slot].nick == nick {
                            return Err(GcgError::new(line_no, format!("both players are {nick}")));
                        }
                        game.players[slot] = Player {
                            nick: nick.to_string(),
                            name: full.trim().to_string(),
                        };
                        named[slot] = true;
                    }
                    "note" => {
                        match game.events.last_mut() {
                            Some(event) => event.notes.push(value.to_string()),
                            None => game.notes.push(value.to_string()),
                        }
                        in_note = true;
                    }
                    "character-encoding" => {}
                    _ => game.pragmas.push((name.to_string(), value.to_string())),
                }
            } else if let Some(body) = line.strip_prefix('>') {
                in_note = false;
                let event = parse_event(body, line_no, &mut game.players, &mut named)?;
                game.events.push(event);
            } else if line.trim().is_empty() {
                in_note = false;
            } else if in_note {
                let note = match game.events.last_mut() {
                    Some(event) => event.notes.last_mut(),
                    None => game.notes.last_mut(),
                };
                if let Some(note) = note {
                    note.push('\n');
                    note.push_str(line);
                }
            } else {
                return Err(GcgError::new(
                    line_no,
                    format!("not a pragma or an event: {line:?}"),
                ));
            }
        }
        if game.events.is_empty() && !named.iter().any(|&n| n) {
            return Err(GcgError::new(0, "no players and no events: not a GCG file"));
        }
        Ok(game)
    }

    /// The record as GCG text.
    pub fn to_gcg(&self) -> String {
        let mut out = String::from("#character-encoding UTF-8\n");
        for (i, p) in self.players.iter().enumerate() {
            if p.nick.is_empty() {
                continue;
            }
            let line = format!("#player{} {} {}", i + 1, p.nick, p.name);
            out.push_str(line.trim_end());
            out.push('\n');
        }
        for (name, value) in &self.pragmas {
            let line = format!("#{name} {value}");
            out.push_str(line.trim_end());
            out.push('\n');
        }
        let notes = |out: &mut String, notes: &[String]| {
            for note in notes {
                out.push_str(&format!("#note {note}\n"));
            }
        };
        notes(&mut out, &self.notes);
        for event in &self.events {
            let mut fields = vec![format!(">{}:", self.players[event.player].nick)];
            if !event.rack.is_empty() {
                fields.push(event.rack.clone());
            }
            match &event.action {
                Action::Play(p) => {
                    fields.push(p.position());
                    fields.push(p.word.clone());
                }
                Action::Exchange(tiles) => fields.push(format!("-{tiles}")),
                Action::ExchangeCount(n) => fields.push(format!("-{n}")),
                Action::Pass => fields.push("-".into()),
                Action::Withdrawn => fields.push("--".into()),
                Action::ChallengeBonus => fields.push("(challenge)".into()),
                Action::TimePenalty => fields.push("(time)".into()),
                Action::EndRack(tiles) => fields.push(format!("({tiles})")),
            }
            fields.push(format!("{:+}", event.score));
            fields.push(event.total.to_string());
            out.push_str(&fields.join(" "));
            out.push('\n');
            notes(&mut out, &event.notes);
        }
        out
    }

    /// Each player's total before event `index` (after the last event when
    /// `index` is past the end).
    pub fn scores_before(&self, index: usize) -> [i32; 2] {
        let mut scores = [0; 2];
        for event in self.events.iter().take(index) {
            scores[event.player] = event.total;
        }
        scores
    }

    /// Append an event for `player`, totalling its score onto theirs.
    pub fn push(&mut self, player: usize, rack: &str, action: Action, score: i32) -> &mut Event {
        let total = self.scores_before(self.events.len())[player] + score;
        self.events.push(Event {
            player,
            rack: rack.to_string(),
            action,
            score,
            total,
            notes: Vec::new(),
            line: 0,
        });
        self.events.last_mut().expect("just pushed")
    }

    /// Append a move the engine knows (play, exchange or pass) made by
    /// `player` from `rack`.
    pub fn push_move(&mut self, player: usize, rack: &[Tile], mv: &Move) -> &mut Event {
        let action = match &mv.kind {
            MoveKind::Play => Action::Play(Placement::of_move(mv).expect("a play")),
            MoveKind::Exchange(tiles) => Action::Exchange(rack_string(tiles)),
            MoveKind::Pass => Action::Pass,
        };
        self.push(player, &rack_string(rack), action, mv.score as i32)
    }

    /// Rebuild the board before every event under `rules`.
    pub fn replay(&self, rules: &'static Rules) -> Result<Replay, GcgError> {
        let mut board = Board::with_rules(rules);
        let mut boards = Vec::with_capacity(self.events.len() + 1);
        let mut placed: Vec<Vec<(usize, usize, Tile)>> = Vec::with_capacity(self.events.len());
        let mut warnings = Vec::new();
        let mut totals = [0i32; 2];
        for (i, event) in self.events.iter().enumerate() {
            boards.push(board.clone());
            let mut tiles = Vec::new();
            match &event.action {
                Action::Play(p) => {
                    tiles = p.tiles(&board).map_err(|m| GcgError::new(event.line, m))?;
                    for &(r, c, t) in &tiles {
                        board.place_tile(r, c, t);
                    }
                }
                Action::Withdrawn => {
                    let previous = (0..i).rev().find(|&j| {
                        self.events[j].player == event.player
                            && matches!(self.events[j].action, Action::Play(_))
                    });
                    let Some(j) = previous else {
                        return Err(GcgError::new(
                            event.line,
                            "withdrawn, but no play to withdraw",
                        ));
                    };
                    for &(r, c, _) in &placed[j] {
                        board.remove_tile(r, c);
                    }
                }
                _ => {}
            }
            if totals[event.player] + event.score != event.total {
                warnings.push(GcgError::new(
                    event.line,
                    format!(
                        "{} {:+} makes {}, not {}",
                        totals[event.player],
                        event.score,
                        totals[event.player] + event.score,
                        event.total
                    ),
                ));
            }
            totals[event.player] = event.total;
            placed.push(tiles);
        }
        boards.push(board);
        Ok(Replay {
            boards,
            placed,
            warnings,
        })
    }
}

fn parse_event(
    body: &str,
    line: usize,
    players: &mut [Player; 2],
    named: &mut [bool; 2],
) -> Result<Event, GcgError> {
    let err = |m: String| GcgError::new(line, m);
    let (nick, rest) = body
        .split_once(':')
        .ok_or_else(|| err("an event is >nick: MOVE SCORE TOTAL".into()))?;
    let nick = nick.trim();
    if nick.is_empty() || nick.contains(char::is_whitespace) {
        return Err(err(format!("bad player nickname {nick:?}")));
    }
    let player = match (0..2).position(|i| named[i] && players[i].nick == nick) {
        Some(p) => p,
        None => {
            // Without #player pragmas, players are named as they appear.
            let slot = named
                .iter()
                .position(|&n| !n)
                .ok_or_else(|| err(format!("unknown player {nick:?}")))?;
            players[slot] = Player {
                nick: nick.to_string(),
                name: String::new(),
            };
            named[slot] = true;
            slot
        }
    };
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() < 3 {
        return Err(err(format!(
            "expected MOVE SCORE TOTAL, got {:?}",
            rest.trim()
        )));
    }
    let n = tokens.len();
    let total = tokens[n - 1]
        .parse::<i32>()
        .map_err(|_| err(format!("bad total {:?}", tokens[n - 1])))?;
    let score =
        parse_score(tokens[n - 2]).ok_or_else(|| err(format!("bad score {:?}", tokens[n - 2])))?;
    let mut mid = &tokens[..n - 2];
    let mut rack = String::new();
    if mid.len() >= 2 && is_rack(mid[0]) {
        rack = mid[0].to_ascii_uppercase();
        mid = &mid[1..];
    }
    if rack.chars().count() > RACK_SIZE {
        return Err(err(format!("rack {rack} has more than {RACK_SIZE} tiles")));
    }
    let action = match mid {
        [position, word] => Action::Play(Placement::parse(position, word).map_err(err)?),
        ["-"] => Action::Pass,
        ["--"] => Action::Withdrawn,
        ["(challenge)"] => Action::ChallengeBonus,
        ["(time)"] => Action::TimePenalty,
        [s] if s.starts_with('(') && s.ends_with(')') && is_rack(&s[1..s.len() - 1]) => {
            Action::EndRack(s[1..s.len() - 1].to_ascii_uppercase())
        }
        [s] if s.starts_with('-') => {
            let tiles = &s[1..];
            if let Ok(count) = tiles.parse::<usize>() {
                Action::ExchangeCount(count)
            } else if is_rack(tiles) {
                Action::Exchange(tiles.to_ascii_uppercase())
            } else {
                return Err(err(format!("bad exchange {s:?}")));
            }
        }
        _ => return Err(err(format!("can't read the move in {:?}", rest.trim()))),
    };
    Ok(Event {
        player,
        rack,
        action,
        score,
        total,
        notes: Vec::new(),
        line,
    })
}

/// The boards of a replayed game.
#[derive(Debug, Clone)]
pub struct Replay {
    /// The board before each event, then the final board (one more than
    /// there are events).
    pub boards: Vec<Board>,
    /// Tiles each event placed (empty for anything but a play).
    pub placed: Vec<Vec<(usize, usize, Tile)>>,
    /// Totals that don't add up; the file's totals are kept.
    pub warnings: Vec<GcgError>,
}

/// How a move whose rack is known compares with the engine's best.
#[derive(Debug, Clone)]
pub struct MoveReview {
    /// Index of the event in [`Game::events`].
    pub event: usize,
    pub player: usize,
    /// The move made, valued as the engine values it.
    pub played: Move,
    /// The engine's best move by static equity.
    pub best: Move,
    /// Equity given up: best minus played (negative when the move played
    /// beats every move the engine knows, e.g. a phony).
    pub loss: f32,
    /// The play is not among the engine's legal moves (a phony, or a word
    /// missing from the word list).
    pub unknown_word: bool,
    /// The score the engine gives the play, when it differs from the record.
    pub rescored: Option<u32>,
    /// Tiles in the bag when the move was made.
    pub bag: usize,
}

/// Review every play, exchange and pass whose rack is recorded: the move
/// played against the engine's best by static equity (score plus leave
/// value). Moves later withdrawn are reviewed as played. A move that
/// can't be valued (its tiles are not on the recorded rack) is skipped and
/// reported in the second list.
pub fn review(
    game: &Game,
    replay: &Replay,
    dictionary: &Dictionary,
) -> (Vec<MoveReview>, Vec<GcgError>) {
    let mut reviews = Vec::new();
    let mut skipped = Vec::new();
    for (i, event) in game.events.iter().enumerate() {
        let reviewable = matches!(
            event.action,
            Action::Play(_) | Action::Exchange(_) | Action::Pass
        );
        if !reviewable || event.rack.is_empty() {
            continue;
        }
        let board = &replay.boards[i];
        let rack = parse_rack(&event.rack);
        let state = SolverState {
            board: board.clone(),
            rack: {
                let mut r = crate::tile::Rack::new();
                for t in &rack {
                    r.add(*t);
                }
                r
            },
        };
        let unseen = state.unseen_tile_count();
        let bag = state.bag_size();
        let mut moves = state.get_suggestions(dictionary);
        let pass = {
            let mut p = Move::pass(&rack);
            p.equity = dictionary.leave_value(&rack, Some(unseen));
            p
        };
        let (played, unknown_word, rescored) = match &event.action {
            Action::Play(_) => {
                let placed = &replay.placed[i];
                let same = |m: &Move| {
                    m.kind == MoveKind::Play
                        && m.tiles_used.len() == placed.len()
                        && placed.iter().all(|p| m.tiles_used.contains(p))
                };
                match moves.iter().find(|m| same(m)) {
                    Some(m) => {
                        let rescored = (m.score as i32 != event.score).then_some(m.score);
                        (m.clone(), false, rescored)
                    }
                    None => match unlisted_play(event, placed, &rack, board, dictionary, unseen) {
                        Ok(mv) => (mv, true, None),
                        Err(m) => {
                            skipped.push(GcgError::new(event.line, m));
                            continue;
                        }
                    },
                }
            }
            Action::Exchange(tiles) => {
                let out = parse_rack(tiles);
                let key = |ts: &[Tile]| {
                    let mut k: Vec<(bool, char)> = ts
                        .iter()
                        .map(|t| (t.is_blank, if t.is_blank { '?' } else { t.letter }))
                        .collect();
                    k.sort();
                    k
                };
                let found = moves.iter().find(|m| match &m.kind {
                    MoveKind::Exchange(x) => key(x) == key(&out),
                    _ => false,
                });
                let mv = match found {
                    Some(m) => m.clone(),
                    None => {
                        // Exchanged with too few in the bag, or tiles the
                        // rack didn't hold: value it anyway.
                        let mut m = Move::exchange(&rack, out, 0.0);
                        m.equity = dictionary.leave_value(&m.leave, Some(unseen));
                        m
                    }
                };
                (mv, false, None)
            }
            _ => (pass.clone(), false, None),
        };
        moves.push(pass);
        let best = moves
            .iter()
            .max_by(|a, b| {
                a.equity
                    .total_cmp(&b.equity)
                    .then(b.score.cmp(&a.score).reverse())
            })
            .cloned()
            .unwrap_or_else(|| played.clone());
        let best = if best.equity < played.equity {
            played.clone()
        } else {
            best
        };
        reviews.push(MoveReview {
            event: i,
            player: event.player,
            loss: best.equity - played.equity,
            played,
            best,
            unknown_word,
            rescored,
            bag,
        });
    }
    (reviews, skipped)
}

/// A play the move generator doesn't list, built and valued by hand.
fn unlisted_play(
    event: &Event,
    placed: &[(usize, usize, Tile)],
    rack: &[Tile],
    board: &Board,
    dictionary: &Dictionary,
    unseen: usize,
) -> Result<Move, String> {
    let Action::Play(p) = &event.action else {
        unreachable!("only plays")
    };
    let mut leave = rack.to_vec();
    for &(_, _, t) in placed {
        let i = leave
            .iter()
            .position(|k| {
                if t.is_blank {
                    k.is_blank
                } else {
                    !k.is_blank && k.letter == t.letter
                }
            })
            .ok_or_else(|| format!("{} is not on the rack {}", t.letter, event.rack))?;
        leave.swap_remove(i);
    }
    let score = Solver::new(board, dictionary).score_placed_tiles(placed, p.direction);
    Ok(Move {
        kind: MoveKind::Play,
        word: p.word_on(board),
        row: p.row,
        col: p.col,
        direction: p.direction,
        score,
        tiles_used: placed.to_vec(),
        is_bingo: placed.len() == RACK_SIZE,
        equity: score as f32 + dictionary.leave_value(&leave, Some(unseen)),
        leave,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::CROSSPLAY;

    const SAMPLE: &str = "\
#character-encoding UTF-8
#player1 alice Alice B
#player2 sam Sam
#lexicon NWL2023
>alice: AEINRST 8D STAINER +72 72
#note A bingo to open.
>sam: AGHLNOT D7 A.H +12 12
>alice: ADEEIMU -AEU +0 72
>sam: ABEIORT - +0 12
>alice: ADEIMOR 9E dA +7 79
>alice: -- -7 72
>sam: (challenge) +5 17
";

    #[test]
    fn positions_parse_both_ways() {
        assert_eq!(parse_position("8D"), Some((7, 3, Direction::Horizontal)));
        assert_eq!(parse_position("D8"), Some((7, 3, Direction::Vertical)));
        assert_eq!(parse_position("15o"), Some((14, 14, Direction::Horizontal)));
        assert_eq!(parse_position("P1"), None);
        assert_eq!(parse_position("16A"), None);
        assert_eq!(parse_position("8"), None);
        assert_eq!(parse_position("0A"), None);
    }

    #[test]
    fn events_parse() {
        let game = Game::parse(SAMPLE).unwrap();
        assert_eq!(game.players[0].nick, "alice");
        assert_eq!(game.players[0].name, "Alice B");
        assert_eq!(game.pragmas, vec![("lexicon".into(), "NWL2023".into())]);
        let actions: Vec<&Action> = game.events.iter().map(|e| &e.action).collect();
        assert!(
            matches!(actions[0], Action::Play(p) if p.word == "STAINER" && p.row == 7 && p.col == 3)
        );
        assert!(
            matches!(actions[1], Action::Play(p) if p.word == "A.H" && p.direction == Direction::Vertical)
        );
        assert_eq!(actions[2], &Action::Exchange("AEU".into()));
        assert_eq!(actions[3], &Action::Pass);
        assert!(matches!(actions[4], Action::Play(p) if p.word == "dA"));
        assert_eq!(actions[5], &Action::Withdrawn);
        assert_eq!(actions[6], &Action::ChallengeBonus);
        assert_eq!(game.events[0].notes, vec!["A bingo to open.".to_string()]);
        assert_eq!(game.events[6].rack, "");
        assert_eq!(game.scores_before(7), [72, 17]);
    }

    #[test]
    fn writing_reads_back_the_same_game() {
        let game = Game::parse(SAMPLE).unwrap();
        let again = Game::parse(&game.to_gcg()).unwrap();
        let strip = |g: &Game| {
            let mut g = g.clone();
            g.events.iter_mut().for_each(|e| e.line = 0);
            g
        };
        assert_eq!(strip(&game), strip(&again));
    }

    #[test]
    fn replay_places_and_withdraws() {
        let game = Game::parse(SAMPLE).unwrap();
        let replay = game.replay(&CROSSPLAY).unwrap();
        assert_eq!(replay.boards.len(), game.events.len() + 1);
        assert!(replay.warnings.is_empty(), "{:?}", replay.warnings);
        let after_ash = &replay.boards[2];
        assert_eq!(after_ash.get_tile(6, 3).unwrap().letter, 'A');
        // A.H runs through the S of STAINER at D8.
        assert_eq!(replay.placed[1].len(), 2);
        // The withdrawn dA leaves the final board as it was before it.
        let before_da = &replay.boards[4];
        let last = replay.boards.last().unwrap();
        assert!(last.get_tile(8, 4).is_none());
        assert_eq!(
            (0..BOARD_SIZE * BOARD_SIZE)
                .filter(|i| last.get_tile(i / 15, i % 15).is_some())
                .count(),
            (0..BOARD_SIZE * BOARD_SIZE)
                .filter(|i| before_da.get_tile(i / 15, i % 15).is_some())
                .count()
        );
        assert!(replay.boards[5].get_tile(8, 4).unwrap().is_blank);
    }

    #[test]
    fn replay_reports_bad_squares_and_totals() {
        let bad = "#player1 a A\n#player2 b B\n>a: ABC 8D A.C +5 5\n";
        let err = Game::parse(bad).unwrap().replay(&CROSSPLAY).unwrap_err();
        assert_eq!(err.line, 3);
        assert!(err.message.contains("empty"), "{err}");

        let sums = "#player1 a A\n#player2 b B\n>a: CAT 8H CAT +5 6\n";
        let replay = Game::parse(sums).unwrap().replay(&CROSSPLAY).unwrap();
        assert_eq!(replay.warnings.len(), 1);
    }

    #[test]
    fn parenthesised_through_tiles_and_end_rack() {
        let text = ">a: QUIZ 8D QU(I)Z +30 30\n>b: (EE) +4 4\n>a: (time) -10 20\n>b: -2 +0 4\n";
        let game = Game::parse(text).unwrap();
        assert!(matches!(&game.events[0].action, Action::Play(p) if p.word == "QU.Z"));
        assert_eq!(game.events[1].action, Action::EndRack("EE".into()));
        assert_eq!(game.events[2].action, Action::TimePenalty);
        assert_eq!(game.events[3].action, Action::ExchangeCount(2));
        assert_eq!(game.players[1].nick, "b");
    }

    /// A file carrying terminal control sequences (here an OSC 52 clipboard
    /// write in a player's name, and an escape in a note) is refused, so
    /// they never reach the screen.
    #[test]
    fn control_characters_are_refused() {
        for text in [
            "#player1 a\u{1b}]52;c;ZWNobw==\u{7} A\n>a: AB 8D AB +4 4\n",
            "#player1 a A\n>a: AB 8D AB +4 4\n#note \u{1b}[2J\n",
            "#player1 a A\n>a: AB 8D AB +4 4\n#note plain\n\u{9b}31m\n",
        ] {
            let err = Game::parse(text).unwrap_err();
            assert!(err.message.contains("control character"), "{err}");
        }
        // Tabs and Windows line ends are fine.
        assert!(Game::parse("#player1\ta A\r\n>a: AB 8D AB +4 4\r\n").is_ok());
    }

    #[test]
    fn junk_is_rejected_with_its_line() {
        for (text, line) in [
            ("hello\n", 1),
            (">a: AB 8D +5\n", 1),
            ("#player1 a A\n>a: AB 8Z AB +4 4\n", 2),
            ("#player1 a A\n#player2 b B\n>c: - +0 0\n", 3),
            (">a: ABCDEFGH - +0 0\n", 1),
        ] {
            let err = Game::parse(text).unwrap_err();
            assert_eq!(err.line, line, "{text:?}: {err}");
        }
    }

    #[test]
    fn placements_from_placed_tiles() {
        let game = Game::parse(SAMPLE).unwrap();
        let replay = game.replay(&CROSSPLAY).unwrap();
        // A.H, rebuilt from the tiles it placed on the board before it.
        let p = Placement::from_tiles(&replay.boards[1], &replay.placed[1]).unwrap();
        assert_eq!((p.position().as_str(), p.word.as_str()), ("D7", "A.H"));
        // A single tile beside NASTIER goes across and takes in the word.
        let one = [(7, 10, Tile::new('S'))];
        let p = Placement::from_tiles(&replay.boards[1], &one).unwrap();
        assert_eq!((p.position().as_str(), p.word.as_str()), ("8D", ".......S"));
        // Tiles not in one line, or with a gap, are not a play.
        let scattered = [(0, 0, Tile::new('A')), (1, 1, Tile::new('B'))];
        assert!(Placement::from_tiles(&replay.boards[1], &scattered).is_none());
        let gap = [(0, 0, Tile::new('A')), (0, 2, Tile::new('B'))];
        assert!(Placement::from_tiles(&replay.boards[1], &gap).is_none());
    }

    #[test]
    fn moves_become_events() {
        let mv = Move {
            kind: MoveKind::Play,
            word: "CATS".into(),
            row: 6,
            col: 7,
            direction: Direction::Vertical,
            score: 9,
            tiles_used: vec![
                (6, 7, Tile::new('C')),
                (8, 7, Tile::new('T')),
                (9, 7, Tile::blank('S')),
            ],
            is_bingo: false,
            leave: vec![],
            equity: 9.0,
        };
        let mut game = Game::new(
            Player {
                nick: "me".into(),
                name: String::new(),
            },
            Player {
                nick: "you".into(),
                name: String::new(),
            },
        );
        game.push_move(0, &parse_rack("CT?"), &mv);
        game.push_move(1, &parse_rack("AB"), &Move::pass(&parse_rack("AB")));
        let text = game.to_gcg();
        assert!(text.contains(">me: CT? H7 C.Ts +9 9\n"), "{text}");
        assert!(text.contains(">you: AB - +0 0\n"), "{text}");
    }
}
