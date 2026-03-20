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
//! This means endgame strategy is a simple 2-ply problem:
//! - For each of your candidate moves, find the opponent's best response
//! - Rank by: your_score - opponent_best_response_score
//!
//! For the absolute last move (opponent has 0 tiles), it's even simpler:
//! just pick the highest-scoring play.

use crate::board::Board;
use crate::dictionary::Dictionary;
use crate::logging::log;
use crate::solver::{Move, Solver};
use crate::tile::{Rack, Tile};

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
    /// The move itself
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

    /// Solve an endgame position where the opponent still has one turn.
    ///
    /// This is the "opponent's last turn, then our last turn" case:
    /// we are the player who emptied the bag, so opponent plays next,
    /// then we get the final move.
    ///
    /// For each of our candidate moves (played on our final turn), we need
    /// to know what the board looks like after the opponent's move. But we
    /// don't know the opponent's move yet — they play BEFORE us.
    ///
    /// So the correct framing is:
    /// - We're about to see the opponent play their last turn
    /// - Then we play our last turn
    /// - We want to evaluate: for every possible opponent move, what's our
    ///   best response? But we can't control the opponent's move.
    ///
    /// **Actually, the caller decides context.** This solver handles two cases:
    ///
    /// ## Case 1: `opponent_has_turn = true` (2 ply)
    /// We play first, opponent responds. For each of our moves:
    /// - Apply our move to the board
    /// - Find opponent's best scoring move on the resulting board
    /// - endgame_value = our_score - opponent_best_score
    ///
    /// ## Case 2: `opponent_has_turn = false` (1 ply / last move)
    /// We play last, no opponent response. Just rank by score.
    pub fn solve(
        &self,
        board: &Board,
        our_rack: &[Tile],
        opponent_rack: &[Tile],
        opponent_has_turn: bool,
    ) -> Vec<EndgameMove> {
        let solver = Solver::new(board, self.dictionary);
        let mut rack = Rack::new();
        for t in our_rack {
            rack.add(*t);
        }
        let our_moves = solver.find_moves(&rack);
        let total = our_moves.len();

        log(&format!(
            "Endgame solver: {} candidate moves, {} our tiles, {} opp tiles, opponent_has_turn={}",
            total, our_rack.len(), opponent_rack.len(), opponent_has_turn
        ));

        let mut results: Vec<EndgameMove> = Vec::with_capacity(total);

        for mv in &our_moves {
            if !opponent_has_turn {
                // Last move — no opponent response. Pure score maximization.
                let pv = vec![PvMove {
                    word: mv.word.clone(),
                    score: mv.score,
                    player: 0,
                }];
                results.push(EndgameMove {
                    mv: mv.clone(),
                    endgame_value: mv.score as i32,
                    pv,
                    opponent_best_score: 0,
                    opponent_best_word: String::new(),
                });
                continue;
            }

            // 2-ply: apply our move, then find opponent's best response
            let mut new_board = board.clone();
            for (r, c, tile) in &mv.tiles_used {
                new_board.place_tile(*r, *c, *tile);
                new_board.mark_premium_used(*r, *c);
            }

            let opp_solver = Solver::new(&new_board, self.dictionary);
            let mut opp_rack = Rack::new();
            for t in opponent_rack {
                opp_rack.add(*t);
            }
            let opp_moves = opp_solver.find_moves(&opp_rack);

            // Opponent picks their highest-scoring move (no further turns to
            // plan for — this is their last turn, no tile penalties)
            let best_opp = opp_moves.iter().max_by_key(|m| m.score);

            let (opp_best_score, opp_best_word) = match best_opp {
                Some(opp_mv) => (opp_mv.score, opp_mv.word.clone()),
                None => (0, String::new()), // Opponent can't play — passes
            };

            let endgame_value = mv.score as i32 - opp_best_score as i32;

            let mut pv = vec![PvMove {
                word: mv.word.clone(),
                score: mv.score,
                player: 0,
            }];
            if let Some(opp_mv) = best_opp {
                pv.push(PvMove {
                    word: opp_mv.word.clone(),
                    score: opp_mv.score,
                    player: 1,
                });
            }

            results.push(EndgameMove {
                mv: mv.clone(),
                endgame_value,
                pv,
                opponent_best_score: opp_best_score,
                opponent_best_word: opp_best_word,
            });
        }

        // Sort by endgame value descending, break ties by raw score
        results.sort_by(|a, b| {
            b.endgame_value
                .cmp(&a.endgame_value)
                .then(b.mv.score.cmp(&a.mv.score))
        });

        // Log summary
        log(&format!("Endgame solver complete: {} moves evaluated", results.len()));
        if let Some(best) = results.first() {
            log(&format!(
                "Best: {} at ({},{}) {:?} score={} value={} (opp responds: {} for {})",
                best.mv.word, best.mv.row, best.mv.col, best.mv.direction,
                best.mv.score, best.endgame_value,
                best.opponent_best_word, best.opponent_best_score
            ));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::load_default_dictionary;

    fn make_board_with_cat() -> Board {
        let mut board = Board::new();
        board.place_tile(7, 7, Tile::new('C'));
        board.place_tile(7, 8, Tile::new('A'));
        board.place_tile(7, 9, Tile::new('T'));
        board.mark_premium_used(7, 7);
        board.mark_premium_used(7, 8);
        board.mark_premium_used(7, 9);
        board
    }

    #[test]
    fn test_last_move_pure_score() {
        // When opponent has no turn remaining, just maximize score
        let dict = load_default_dictionary().unwrap();
        let board = make_board_with_cat();

        let our_rack = vec![Tile::new('S'), Tile::new('X')];
        let opp_rack = vec![]; // No opponent tiles

        let solver = EndgameSolver::new(&dict);
        let results = solver.solve(&board, &our_rack, &opp_rack, false);

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
        let dict = load_default_dictionary().unwrap();
        let board = make_board_with_cat();

        let our_rack = vec![Tile::new('S')];
        let opp_rack = vec![Tile::new('E')];

        let solver = EndgameSolver::new(&dict);
        let results = solver.solve(&board, &our_rack, &opp_rack, true);

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
                i + 1, r.mv.word, r.mv.score, r.endgame_value,
                r.opponent_best_word, r.opponent_best_score
            );
        }
    }

    #[test]
    fn test_blocking_matters() {
        // A lower-scoring move that blocks the opponent's best play
        // should rank higher than a higher-scoring move that leaves
        // the opponent a big scoring opportunity
        let dict = load_default_dictionary().unwrap();
        let board = make_board_with_cat();

        // Give us S and opponent X — opponent wants premium squares
        let our_rack = vec![Tile::new('S')];
        let opp_rack = vec![Tile::new('X')];

        let solver = EndgameSolver::new(&dict);
        let results = solver.solve(&board, &our_rack, &opp_rack, true);

        assert!(!results.is_empty());

        println!("Blocking test ({} moves):", results.len());
        for (i, r) in results.iter().take(10).enumerate() {
            println!(
                "  {}: {} at ({},{}) score={} value={} (opp: {} for {})",
                i + 1, r.mv.word, r.mv.row, r.mv.col, r.mv.score,
                r.endgame_value, r.opponent_best_word, r.opponent_best_score
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
}
