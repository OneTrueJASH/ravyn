//! Monte Carlo move evaluation under Crossplay rules.
//!
//! Each candidate move is played out many times against random draws from
//! the unseen tiles. Playouts follow Crossplay's ending exactly:
//!
//! - Players draw after playing; the player who draws the last tile from the
//!   bag "owns" the final turn. The other player then gets one last turn and
//!   the owner plays the final move. Nobody goes out for a bonus and leftover
//!   tiles are never scored.
//! - On a player's last turn (the reply after the bag empties) they pick the
//!   move that maximises their score minus the owner's best final reply,
//!   passing if that is better. On the final turn the owner simply takes the
//!   most points. Both players know each other's racks at that point (the
//!   unseen pool is the other rack), so this is exact given the playout
//!   reached that position.
//! - While the bag still has tiles, players play the best move by static
//!   equity (score plus phase-scaled leave value).
//!
//! If a playout hits the ply limit before the game ends, the position is
//! scored by the spread so far plus the phase-scaled leave values. This is
//! the pre-endgame engine (bag of 1–7 tiles), where RefEngine's simulation
//! would apply Scrabble's go-out rules, and the general fallback when
//! RefEngine is unavailable. It is not meant for positions where the bag is
//! already empty: who owns the final turn is then outside its knowledge,
//! and [`crate::endgame::EndgameSolver`] handles those exactly.

mod eval;
mod opponent;
mod playout;
mod pool;
mod selfplay;
#[cfg(test)]
mod studies;
#[cfg(test)]
mod tests;

pub(crate) use eval::CandidateSim;
pub use eval::{MctsConfig, MctsEvaluator, MctsResult};
pub use opponent::OpponentModel;
pub(crate) use playout::{win_value, PlayoutOutcome, PlayoutRecord};
pub use playout::{SimChoice, TracePoint};
pub(crate) use pool::Fnv;
pub use pool::{position_fingerprint, TilePool};
pub use selfplay::{self_play_match, MatchEngine, MatchResult};

#[allow(unused_imports)]
use self::{eval::*, opponent::*, playout::*, pool::*, selfplay::*};

use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastrand::Rng;
use rayon::prelude::*;

use crate::board::Board;
use crate::dictionary::Dictionary;
use crate::game::full_tile_set;
use crate::game::SolverState;
use crate::game::{best_exchange, top_exchanges};
use crate::solver::{Move, MoveKind, Solver, EXCHANGE_MIN_BAG};
#[cfg(test)]
use crate::tile::Rack;
use crate::tile::{Tile, RACK_SIZE};
