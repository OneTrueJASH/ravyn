//! Crossplay-native endgame solver
//!
//! Crossplay endgame is fundamentally different from Scrabble:
//!
//! **Equal Turn Rule:** When the bag hits 0 tiles, the opponent gets one last
//! turn, then the player who emptied the bag gets one final turn. That's it —
//! exactly 2 remaining turns total once the bag is empty.
//!
//! **No leftover tile penalty:** Unplayed tiles in your rack don't reduce your
//! score and aren't added to the opponent's score.
//!
//! **No go-out bonus:** Emptying your rack gives no bonus points.
//!
//! So once the bag is empty there are only two situations, and the caller
//! (which knows who emptied the bag) picks one as a [`FinalTurn`]:
//!
//! - [`FinalTurn::Opponent`]: the opponent emptied the bag. We play our
//!   last turn and they play the final move. This is an exact 2-ply search:
//!   for each of our moves (and for passing), apply it and find the
//!   opponent's highest-scoring reply; rank by `our_score - their_best`.
//!   The opponent's rack is fully known (it is every unseen tile).
//! - [`FinalTurn::Us`]: we emptied the bag and the opponent has
//!   already taken their last turn. Our move ends the game: rank by score.
//!
//! Deeper search is deliberately absent — there are no further turns to
//! search. See PRIMER.md and the project memory note on endgame rules.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::board::Board;
use crate::dictionary::Dictionary;
use crate::solver::{BoardAnalysis, Move, Solver};
use crate::tile::{Rack, Tile};

/// Who plays the very last move of the game.
///
/// Crossplay's equal-turn rule: when a player draws the last tile from the
/// bag, the other player gets one more turn and then the player who emptied
/// the bag plays the final move. So when the bag is empty on our turn, the
/// opponent only gets to reply if *they* emptied the bag. The app infers
/// this from moves applied through the suggestions list and the user can
/// override it with Ctrl+E.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FinalTurn {
    /// The opponent emptied the bag: we play our last turn, they reply.
    #[default]
    Opponent,
    /// We emptied the bag: the opponent has already played their last
    /// turn, so our next move ends the game.
    Us,
}

impl FinalTurn {
    pub fn toggle(self) -> Self {
        match self {
            FinalTurn::Opponent => FinalTurn::Us,
            FinalTurn::Us => FinalTurn::Opponent,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            FinalTurn::Opponent => "OPP",
            FinalTurn::Us => "YOU",
        }
    }
}

/// A move in the principal variation (the optimal 2-move sequence)
#[derive(Debug, Clone)]
pub struct PvMove {
    pub word: String,
    pub score: u32,
    pub player: usize, // 0 = us, 1 = opponent
}

/// Result of endgame evaluation for a single candidate move
#[derive(Debug, Clone)]
pub struct EndgameMove {
    /// The move itself (a pass has `tiles_used` empty and word [`PASS_WORD`])
    pub mv: Move,
    /// Net scoring advantage: our_score - opponent_best_response_score.
    /// For last-move (no opponent response), this equals the move score.
    pub endgame_value: i32,
    /// Principal variation: our move, then opponent's best response (if any)
    pub pv: Vec<PvMove>,
    /// Opponent's best response score (0 if no opponent turn remains)
    pub opponent_best_score: u32,
    /// Opponent's best response word (empty if no opponent turn remains)
    pub opponent_best_word: String,
}

/// Exhaustive endgame solver for Crossplay
pub struct EndgameSolver<'a> {
    dictionary: &'a Dictionary,
}

impl<'a> EndgameSolver<'a> {
    pub fn new(dictionary: &'a Dictionary) -> Self {
        EndgameSolver { dictionary }
    }

    /// Evaluate every move available to us with the bag empty.
    ///
    /// `opponent_rack` must be the opponent's complete rack (all unseen
    /// tiles). With [`FinalTurn::Opponent`] the result includes a pass candidate,
    /// since every play may open more for the opponent than it scores.
    /// Results are sorted best first: by endgame value, then raw score.
    pub fn solve(
        &self,
        board: &Board,
        our_rack: &[Tile],
        opponent_rack: &[Tile],
        final_turn: FinalTurn,
    ) -> Vec<EndgameMove> {
        self.solve_cancellable(board, our_rack, opponent_rack, final_turn, None)
            .unwrap_or_default()
    }

    /// What the side to move can get when the other side plays the last
    /// move of the game: the most of (its play's score minus the other
    /// side's best final score), over every play and passing. The value of
    /// [`Self::solve`]'s best move with [`FinalTurn::Opponent`], without
    /// ranking every move.
    ///
    /// Plays are tried from the highest score down. A final reply scores
    /// at least 0, so a play can be worth no more than its own score, and
    /// the search stops at the first play that cannot beat the best value
    /// so far.
    pub fn best_value(&self, board: &Board, mover: &[Tile], last: &[Tile]) -> i32 {
        let analysis = Solver::new(board, self.dictionary).analysis();
        self.best_value_with(board, &analysis, mover, last)
    }

    /// [`Self::best_value`] with `board`'s analysis already made. The board
    /// after each play is analyzed incrementally from it, which is most of
    /// the saving when one board is solved for many pairs of racks.
    pub fn best_value_with(
        &self,
        board: &Board,
        analysis: &BoardAnalysis,
        mover: &[Tile],
        last: &[Tile],
    ) -> i32 {
        // Passing: the other side then takes its best final play here.
        let mut best =
            -(Solver::with_analysis(board, self.dictionary, analysis).best_score(last) as i32);
        // Plays from the highest score down, fetched a few at a time: the
        // search usually stops after the first handful, and building every
        // play (tens of thousands with a blank) would cost the most.
        let solver = Solver::with_analysis(board, self.dictionary, analysis);
        let mut after = board.clone();
        let mut squares = Vec::with_capacity(crate::tile::RACK_SIZE);
        let mut fetched = 16;
        let mut done = 0;
        'search: loop {
            let moves = solver.top_moves_by_score(mover, fetched);
            for mv in &moves[done..] {
                if mv.score as i32 <= best {
                    break 'search;
                }
                squares.clear();
                for (r, c, t) in &mv.tiles_used {
                    after.place_tile(*r, *c, *t);
                    squares.push((*r, *c));
                }
                let next = analysis.after_placing(&after, self.dictionary, &squares);
                let final_score =
                    Solver::with_analysis(&after, self.dictionary, &next).best_score(last) as i32;
                best = best.max(mv.score as i32 - final_score);
                for (r, c, _) in &mv.tiles_used {
                    after.remove_tile(*r, *c);
                }
            }
            if moves.len() < fetched {
                break;
            }
            done = moves.len();
            fetched *= 4;
        }
        best
    }

    /// As [`Self::solve`], checking `cancel` before each of our moves is
    /// searched. Returns `None` if the search was cancelled.
    pub fn solve_cancellable(
        &self,
        board: &Board,
        our_rack: &[Tile],
        opponent_rack: &[Tile],
        final_turn: FinalTurn,
        cancel: Option<&AtomicBool>,
    ) -> Option<Vec<EndgameMove>> {
        let opponent_has_turn = final_turn == FinalTurn::Opponent;
        let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
        let solver = Solver::new(board, self.dictionary);
        let mut rack = Rack::new();
        for t in our_rack {
            rack.add(*t);
        }
        let mut our_moves = solver.find_moves(&rack);

        log::info!(
            "Endgame solver: {} candidate moves, {} our tiles, {} opp tiles, final turn: {:?}",
            our_moves.len(),
            our_rack.len(),
            opponent_rack.len(),
            final_turn
        );

        let mut results: Vec<EndgameMove> = if !opponent_has_turn {
            // Our move ends the game: pure score maximization.
            our_moves
                .drain(..)
                .map(|mv| {
                    let pv = vec![PvMove {
                        word: mv.word.clone(),
                        score: mv.score,
                        player: 0,
                    }];
                    EndgameMove {
                        endgame_value: mv.score as i32,
                        pv,
                        opponent_best_score: 0,
                        opponent_best_word: String::new(),
                        mv,
                    }
                })
                .collect()
        } else {
            our_moves.push(Move::pass(our_rack));

            // 2-ply: apply our move, then find the opponent's best reply.
            // Candidates are independent, so evaluate them in parallel.
            let evaluated: Option<Vec<EndgameMove>> = our_moves
                .par_iter()
                .map(|mv| {
                    if cancelled() {
                        return None;
                    }
                    let mut new_board = board.clone();
                    for (r, c, tile) in &mv.tiles_used {
                        new_board.place_tile(*r, *c, *tile);
                    }

                    // The opponent's move is the last of the game: they simply
                    // take the most points (passing scores 0). Only that one
                    // reply is built.
                    let opp_solver = Solver::new(&new_board, self.dictionary);
                    let best_opp = opp_solver.best_move_by_score(opponent_rack);
                    let (opp_best_score, opp_best_word) = match &best_opp {
                        Some(opp_mv) => (opp_mv.score, opp_mv.word.clone()),
                        None => (0, String::new()),
                    };

                    let mut pv = vec![PvMove {
                        word: mv.word.clone(),
                        score: mv.score,
                        player: 0,
                    }];
                    if let Some(opp_mv) = &best_opp {
                        pv.push(PvMove {
                            word: opp_mv.word.clone(),
                            score: opp_mv.score,
                            player: 1,
                        });
                    }

                    Some(EndgameMove {
                        mv: mv.clone(),
                        endgame_value: mv.score as i32 - opp_best_score as i32,
                        pv,
                        opponent_best_score: opp_best_score,
                        opponent_best_word: opp_best_word,
                    })
                })
                .collect();
            evaluated?
        };
        if cancelled() {
            return None;
        }

        // Sort by endgame value descending, break ties by raw score
        results.sort_by(|a, b| {
            b.endgame_value
                .cmp(&a.endgame_value)
                .then(b.mv.score.cmp(&a.mv.score))
        });

        log::info!("Endgame solver complete: {} moves evaluated", results.len());
        if let Some(best) = results.first() {
            log::info!(
                "Best: {} at {} score={} value={} (opp responds: {} for {})",
                best.mv.word,
                best.mv.coords(),
                best.mv.score,
                best.endgame_value,
                best.opponent_best_word,
                best.opponent_best_score
            );
        }

        Some(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::PASS_WORD;

    fn make_board_with_cat() -> Board {
        let mut board = Board::new();
        board.place_tile(7, 7, Tile::new('C'));
        board.place_tile(7, 8, Tile::new('A'));
        board.place_tile(7, 9, Tile::new('T'));
        board
    }

    #[test]
    fn test_last_move_pure_score() {
        // When opponent has no turn remaining, just maximize score
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = make_board_with_cat();

        let our_rack = vec![Tile::new('S'), Tile::new('X')];
        let opp_rack = vec![]; // No opponent tiles

        let solver = EndgameSolver::new(&dict);
        let results = solver.solve(&board, &our_rack, &opp_rack, FinalTurn::Us);

        assert!(!results.is_empty());

        // endgame_value should equal raw score (no opponent response)
        for r in &results {
            assert_eq!(
                r.endgame_value, r.mv.score as i32,
                "Last move: endgame value should equal score for {}",
                r.mv.word
            );
            assert_eq!(r.opponent_best_score, 0);
        }

        // Should be sorted by score descending
        for pair in results.windows(2) {
            assert!(pair[0].mv.score >= pair[1].mv.score);
        }

        println!("Last move test ({} moves):", results.len());
        for (i, r) in results.iter().take(5).enumerate() {
            println!("  {}: {} score={}", i + 1, r.mv.word, r.mv.score);
        }
    }

    #[test]
    fn test_two_ply_opponent_response() {
        // With opponent having a turn, we should consider their response
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = make_board_with_cat();

        let our_rack = vec![Tile::new('S')];
        let opp_rack = vec![Tile::new('E')];

        let solver = EndgameSolver::new(&dict);
        let results = solver.solve(&board, &our_rack, &opp_rack, FinalTurn::Opponent);

        assert!(!results.is_empty());

        // Every result should have opponent response info
        for r in &results {
            // PV should have our move + opponent response (if opponent can play)
            assert!(!r.pv.is_empty());
            assert_eq!(r.pv[0].player, 0, "First PV move should be ours");
            assert_eq!(
                r.endgame_value,
                r.mv.score as i32 - r.opponent_best_score as i32,
                "Value should be our_score - opp_best for {}",
                r.mv.word
            );
        }

        println!("Two-ply test ({} moves):", results.len());
        for (i, r) in results.iter().take(5).enumerate() {
            println!(
                "  {}: {} score={} value={} (opp: {} for {})",
                i + 1,
                r.mv.word,
                r.mv.score,
                r.endgame_value,
                r.opponent_best_word,
                r.opponent_best_score
            );
        }
    }

    #[test]
    fn test_blocking_matters() {
        // A lower-scoring move that blocks the opponent's best play
        // should rank higher than a higher-scoring move that leaves
        // the opponent a big scoring opportunity
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = make_board_with_cat();

        // Give us S and opponent X — opponent wants premium squares
        let our_rack = vec![Tile::new('S')];
        let opp_rack = vec![Tile::new('X')];

        let solver = EndgameSolver::new(&dict);
        let results = solver.solve(&board, &our_rack, &opp_rack, FinalTurn::Opponent);

        assert!(!results.is_empty());

        println!("Blocking test ({} moves):", results.len());
        for (i, r) in results.iter().take(10).enumerate() {
            println!(
                "  {}: {} at ({},{}) score={} value={} (opp: {} for {})",
                i + 1,
                r.mv.word,
                r.mv.row,
                r.mv.col,
                r.mv.score,
                r.endgame_value,
                r.opponent_best_word,
                r.opponent_best_score
            );
        }

        // The best move by endgame_value may not be the highest raw score
        // (if blocking opponent's scoring opportunity is worth more)
        let best_value = results[0].endgame_value;
        let best_score_move = results.iter().max_by_key(|r| r.mv.score).unwrap();
        if best_value > best_score_move.endgame_value {
            println!(
                "Blocking confirmed: best value move ({}, val={}) beats highest scorer ({}, val={})",
                results[0].mv.word, best_value,
                best_score_move.mv.word, best_score_move.endgame_value
            );
        }
    }

    /// The pass candidate is worth minus the opponent's best play on the
    /// unchanged board, and every other candidate's value is its score
    /// minus the opponent's best reply — recomputed here independently.
    #[test]
    fn test_pass_candidate_and_values_are_consistent() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        let board = make_board_with_cat();

        let our_rack = vec![Tile::new('Q'), Tile::new('I')];
        let opp_rack = vec![Tile::new('Z'), Tile::new('A'), Tile::new('S')];

        let solver = EndgameSolver::new(&dict);
        let results = solver.solve(&board, &our_rack, &opp_rack, FinalTurn::Opponent);

        let mut opp = Rack::new();
        for t in &opp_rack {
            opp.add(*t);
        }
        let best_reply = |b: &Board| -> u32 {
            Solver::new(b, &dict)
                .find_moves(&opp)
                .iter()
                .map(|m| m.score)
                .max()
                .unwrap_or(0)
        };

        let pass = results
            .iter()
            .find(|r| r.mv.word == PASS_WORD)
            .expect("pass candidate present when the opponent replies");
        assert!(pass.mv.tiles_used.is_empty());
        assert_eq!(pass.mv.score, 0);
        assert_eq!(pass.mv.coords(), "-");
        assert_eq!(pass.endgame_value, -(best_reply(&board) as i32));
        assert_eq!(pass.mv.leave.len(), our_rack.len());

        for r in &results {
            let mut b = board.clone();
            for (row, col, t) in &r.mv.tiles_used {
                b.place_tile(*row, *col, *t);
            }
            assert_eq!(
                r.endgame_value,
                r.mv.score as i32 - best_reply(&b) as i32,
                "{} at {}",
                r.mv.word,
                r.mv.coords()
            );
        }

        // Sorted best first.
        for pair in results.windows(2) {
            assert!(pair[0].endgame_value >= pair[1].endgame_value);
        }

        // Without an opponent turn there is no pass candidate.
        let last = solver.solve(&board, &our_rack, &opp_rack, FinalTurn::Us);
        assert!(last.iter().all(|r| r.mv.word != PASS_WORD));
    }

    /// `best_value` is the value of the best move `solve` ranks, on the
    /// endgame test position from both sides.
    #[test]
    fn best_value_matches_the_full_search() {
        let Some(dict) = crate::test_data::dictionary() else {
            return;
        };
        const END: &str = "3N3Q3M3/3OP2UNTEACH1/3BE2O3N3/3LA2T2MYOPE/2FeR2H1LO4/2O6AB4/2R1F4U3De/2TRIVET1D3AY/4LA1A1IODIZE/4KI1JINX2ES/5R1E1G4O/OUTIES1S3G2R/11L2E/8WAHINES/2EGAlITE2A3";
        let state = crate::game::SolverState::from_cgp(&format!("{} RCVSDW", END)).unwrap();
        let ours = state.rack.tiles();
        let solver = EndgameSolver::new(&dict);
        for opp in ["AEEIST?", "DGILNOU", "QZ", ""] {
            let opp: Vec<Tile> = opp
                .chars()
                .map(|c| {
                    if c == '?' {
                        Tile::blank(' ')
                    } else {
                        Tile::new(c)
                    }
                })
                .collect();
            for (mover, last) in [(&ours, &opp), (&opp, &ours)] {
                let full = solver.solve(&state.board, mover, last, FinalTurn::Opponent);
                assert_eq!(
                    solver.best_value(&state.board, mover, last),
                    full[0].endgame_value
                );
            }
        }
    }
}
