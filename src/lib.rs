//! RAVYN (Racing Arms by Value: Ȳ over n): a crossword-game analysis
//! engine.
//!
//! It generates every legal play ([`solver`]), values each by score and
//! leave ([`leaves`]), and ranks the candidates by simulation ([`sim`]):
//! each candidate is a bandit arm, each playout samples an outcome *Y*, and
//! arms race until their mean Ȳₙ separates them. The bag-empty ending is
//! solved exactly ([`endgame`]).
//!
//! The rules (board layout, tile values and distribution, bingo bonus,
//! ending) are data ([`rules`]): every board carries its own, and
//! [`rules::CROSSPLAY`] is the default.
//!
//! Diagnostics go through the [`log`](https://docs.rs/log) facade; an
//! application installs whichever logger it likes.

pub mod board;
pub mod dictionary;
pub mod endgame;
pub mod fetch;
pub mod game;
pub mod gcg;
pub mod inference;
pub mod leavegen;
pub mod leaves;
pub mod mcts;
pub mod negamax;
pub mod preendgame;
pub mod rules;
pub mod sim;
pub mod solver;
pub mod stats;
pub mod tile;

#[cfg(test)]
mod test_data;
