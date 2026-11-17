//! `ravyn`: the RAVYN engine from the command line.
//!
//! Positions are CGP strings: `BOARD RACK[/OPP] [OURS/THEIRS ...]`, the
//! board as 15 rows separated by `/`, digits for runs of empty squares and
//! lowercase letters for blanks; the rack with `?` for a blank, optionally
//! followed by the opponent's rack when it is known; then the scores.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ravyn::dictionary::Dictionary;
use ravyn::endgame::{EndgameSolver, FinalTurn};
use ravyn::game::SolverState;
use ravyn::inference::{infer_leave, InferenceSettings};
use ravyn::leaves::LeaveValues;
use ravyn::negamax::{solve_go_out, GoOutSettings};
use ravyn::preendgame::solve_one_in_bag;
use ravyn::rules::{Ending, Rules, CROSSPLAY, SCRABBLE};
use ravyn::sim::{ArmStatus, SimSettings, Simulation};
use ravyn::solver::{Move, MoveKind};
use ravyn::tile::{Tile, RACK_SIZE};

const USAGE: &str = "\
ravyn: the RAVYN crossword-game engine

usage: ravyn <command> '<CGP>' [options]

commands:
  gen         every play, exchange and pass, best static equity first
  sim         simulate the best candidates; rank by mean final spread
  endgame     bag empty: exact search (Crossplay: our play, their reply,
              our last; Scrabble: negamax to the end, --seconds budget)
  preendgame  one tile in the bag: exact over every possible last tile
              (Crossplay)
  infer       what the opponent kept: CGP is the board BEFORE their play,
              with OUR rack; needs --played and --score

options:
  --rules NAME    `crossplay` (default) or `scrabble`
  --top N         rows to print (default 20)
  --candidates N  sim: candidates to simulate (default 30)
  --seconds S     sim and Scrabble endgame: time budget (default 10)
  --last WHO      endgame: who plays the final move, `them` (default:
                  they emptied the bag) or `us`
  --played TILES  infer: the tiles they played, `?` for a blank
  --score N       infer: what their play scored
  --data DIR      directory holding data/words.fst and the leave table
                  external/refengine/data/lexica/AMERICAN.klv2 (default:
                  $RAVYN_DATA, then the current directory)

CGP: BOARD RACK[/OPP] [OURS/THEIRS ...], e.g.
  ravyn gen '15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AEINRST/ 0/0 0'";

const COMMANDS: [&str; 5] = ["gen", "sim", "endgame", "preendgame", "infer"];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ravyn: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Parsed command line.
struct Options {
    command: String,
    cgp: String,
    top: usize,
    candidates: usize,
    seconds: u64,
    last: FinalTurn,
    played: Option<String>,
    score: Option<u32>,
    data: Option<PathBuf>,
    rules: &'static Rules,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut positional = Vec::new();
    let mut o = Options {
        command: String::new(),
        cgp: String::new(),
        top: 20,
        candidates: 30,
        seconds: 10,
        last: FinalTurn::Opponent,
        played: None,
        score: None,
        data: None,
        rules: &CROSSPLAY,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        let number = |name: &str, v: String| {
            v.parse::<u64>()
                .map_err(|_| format!("{name}: not a number: {v}"))
        };
        match arg.as_str() {
            "--top" => o.top = number("--top", value("--top")?)? as usize,
            "--candidates" => {
                o.candidates = number("--candidates", value("--candidates")?)? as usize
            }
            "--seconds" => o.seconds = number("--seconds", value("--seconds")?)?,
            "--score" => o.score = Some(number("--score", value("--score")?)? as u32),
            "--played" => o.played = Some(value("--played")?),
            "--data" => o.data = Some(PathBuf::from(value("--data")?)),
            "--rules" => {
                o.rules = match value("--rules")?.to_ascii_lowercase().as_str() {
                    "crossplay" => &CROSSPLAY,
                    "scrabble" => &SCRABBLE,
                    other => {
                        return Err(format!("--rules: `crossplay` or `scrabble`, not {other}"))
                    }
                }
            }
            "--last" => {
                o.last = match value("--last")?.as_str() {
                    "them" => FinalTurn::Opponent,
                    "us" => FinalTurn::Us,
                    other => return Err(format!("--last: `us` or `them`, not {other}")),
                }
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => positional.push(arg.clone()),
        }
    }
    let mut positional = positional.into_iter();
    o.command = positional.next().ok_or(USAGE)?;
    if !COMMANDS.contains(&o.command.as_str()) {
        return Err(format!(
            "unknown command {} (one of: {}; --help for more)",
            o.command,
            COMMANDS.join(", ")
        ));
    }
    o.cgp = positional.next().ok_or("missing the position (CGP)")?;
    if let Some(extra) = positional.next() {
        return Err(format!("unexpected argument {extra} (quote the CGP)"));
    }
    Ok(o)
}

/// Our score minus theirs, from the CGP's score field (0 if absent).
fn score_diff(cgp: &str) -> i32 {
    cgp.split_whitespace()
        .nth(2)
        .and_then(|s| s.split_once('/'))
        .and_then(|(a, b)| Some(a.parse::<i32>().ok()? - b.parse::<i32>().ok()?))
        .unwrap_or(0)
}

/// The opponent's rack, if the CGP gives it after the slash.
fn opponent_rack(cgp: &str) -> Option<Vec<Tile>> {
    let field = cgp.split_whitespace().nth(1)?;
    let (_, opp) = field.split_once('/')?;
    (!opp.is_empty()).then(|| tiles(opp))
}

fn tiles(letters: &str) -> Vec<Tile> {
    letters
        .chars()
        .map(|c| {
            if c == '?' {
                Tile::blank(' ')
            } else {
                Tile::new(c)
            }
        })
        .collect()
}

/// Every unseen tile from our side: the bag plus the opponent's rack.
fn unseen(state: &SolverState) -> Vec<Tile> {
    let mut counts: Vec<(char, u32)> = state.unseen_tiles().into_iter().collect();
    counts.sort();
    counts
        .into_iter()
        .flat_map(|(c, n)| {
            (0..n).map(move |_| {
                if c == '?' {
                    Tile::blank(' ')
                } else {
                    Tile::new(c)
                }
            })
        })
        .collect()
}

fn load_dictionary(dir: Option<&Path>) -> Result<Dictionary, String> {
    let root = dir
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("RAVYN_DATA").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    let words = root.join("data/words.fst");
    let dict = Dictionary::load_from_fst(&words).map_err(|e| {
        format!(
            "{}: {e:#} (point --data at a directory holding data/)",
            words.display()
        )
    })?;
    let leaves = root.join("external/refengine/data/lexica/AMERICAN.klv2");
    Ok(match LeaveValues::load(&leaves) {
        Ok(table) => dict.with_leaves(table),
        Err(e) => {
            eprintln!("ravyn: no leave table ({e:#}); using heuristic leave values");
            dict
        }
    })
}

fn run(args: &[String]) -> Result<(), String> {
    let o = parse(args)?;
    let state = SolverState::from_cgp_with_rules(&o.cgp, o.rules)?;
    let dict = load_dictionary(o.data.as_deref())?.for_rules(o.rules);
    match o.command.as_str() {
        "gen" => gen(&o, &state, &dict),
        "sim" => sim(&o, &state, &dict),
        "endgame" => endgame(&o, &state, &dict),
        "preendgame" => preendgame(&o, &state, &dict),
        "infer" => infer(&o, &state, &dict),
        other => unreachable!("checked in parse: {other}"),
    }
}

fn leave(mv: &Move) -> String {
    mv.leave
        .iter()
        .map(|t| if t.is_blank { '?' } else { t.letter })
        .collect()
}

/// A move as `8H WORD` (blanks lowercase), `exch ABC` or `pass`.
fn label(mv: &Move) -> String {
    match &mv.kind {
        MoveKind::Play => {
            let (dr, dc) = mv.direction.delta();
            let word: String = mv
                .word
                .chars()
                .enumerate()
                .map(|(i, ch)| {
                    let r = (mv.row as i32 + dr * i as i32) as usize;
                    let c = (mv.col as i32 + dc * i as i32) as usize;
                    let blank = mv
                        .tiles_used
                        .iter()
                        .any(|&(pr, pc, t)| pr == r && pc == c && t.is_blank);
                    if blank {
                        ch.to_ascii_lowercase()
                    } else {
                        ch
                    }
                })
                .collect();
            format!("{} {}", mv.coords(), word)
        }
        MoveKind::Exchange(out) => format!(
            "exch {}",
            out.iter()
                .map(|t| if t.is_blank { '?' } else { t.letter })
                .collect::<String>()
        ),
        MoveKind::Pass => "pass".to_string(),
    }
}

fn gen(o: &Options, state: &SolverState, dict: &Dictionary) -> Result<(), String> {
    let moves = state.get_suggestions(dict);
    println!(
        "{} candidates ({} in the bag)",
        moves.len(),
        state.bag_size()
    );
    println!("{:<24} {:>5} {:>8}  leave", "move", "score", "equity");
    for mv in moves.iter().take(o.top) {
        println!(
            "{:<24} {:>5} {:>8.2}  {}",
            label(mv),
            mv.score,
            mv.equity,
            leave(mv)
        );
    }
    Ok(())
}

fn sim(o: &Options, state: &SolverState, dict: &Dictionary) -> Result<(), String> {
    if state.is_endgame() {
        return Err("the bag is empty: use `endgame`".into());
    }
    let mut candidates = state.get_suggestions(dict);
    candidates.truncate(o.candidates.max(1));
    candidates.push(Move::pass(&state.rack.tiles()));
    let settings = SimSettings::for_bag(
        state.bag_size(),
        16,
        Some(Duration::from_secs(o.seconds.max(1))),
    )
    .with_score_diff(score_diff(&o.cgp));
    let mut simulation = Simulation::new(settings, dict, state, &candidates, None, None);
    let snap = simulation.run(&AtomicBool::new(false), &mut |_| {});
    println!(
        "{} candidates, {} playouts in {:.1}s ({} in the bag)",
        candidates.len(),
        snap.playouts,
        snap.elapsed.as_secs_f32(),
        state.bag_size()
    );
    let mut rows: Vec<(usize, &ravyn::sim::ArmSummary)> = snap.arms.iter().enumerate().collect();
    rows.sort_by(|a, b| b.1.spread.total_cmp(&a.1.spread));
    println!(
        "{:<24} {:>5} {:>8} {:>13} {:>12} {:>6}  status",
        "move", "score", "equity", "spread", "win%", "n"
    );
    for (i, arm) in rows.into_iter().take(o.top) {
        let mv = &candidates[i];
        let status = match arm.status {
            ArmStatus::Pending => "pending",
            ArmStatus::Racing => "racing",
            ArmStatus::Stopped => "out",
            ArmStatus::Capped => "capped",
        };
        println!(
            "{:<24} {:>5} {:>8.2} {:>+7.1} ±{:>4.1} {:>5.1} ±{:>4.1} {:>6}  {}",
            label(mv),
            mv.score,
            mv.equity,
            arm.spread,
            arm.spread_se,
            arm.win * 100.0,
            arm.win_se * 100.0,
            arm.n,
            status
        );
    }
    Ok(())
}

fn endgame(o: &Options, state: &SolverState, dict: &Dictionary) -> Result<(), String> {
    if !state.is_endgame() || state.bag_size() > 0 {
        return Err(format!(
            "the bag holds {} tiles: endgame needs it empty",
            state.bag_size()
        ));
    }
    let theirs = opponent_rack(&o.cgp).unwrap_or_else(|| unseen(state));
    if o.rules.ending == Ending::GoOut {
        return go_out_endgame(o, state, dict, &theirs);
    }
    let results =
        EndgameSolver::new(dict).solve(&state.board, &state.rack.tiles(), &theirs, o.last);
    let diff = score_diff(&o.cgp);
    println!(
        "{} moves; value = our score minus their best reply",
        results.len()
    );
    println!(
        "{:<24} {:>5} {:>6} {:>7}  their reply",
        "move", "score", "value", "final"
    );
    for r in results.iter().take(o.top) {
        println!(
            "{:<24} {:>5} {:>+6} {:>+7}  {} {}",
            label(&r.mv),
            r.mv.score,
            r.endgame_value,
            diff + r.endgame_value,
            r.opponent_best_word,
            r.opponent_best_score
        );
    }
    Ok(())
}

/// The Scrabble endgame: negamax to the end of the game.
fn go_out_endgame(
    o: &Options,
    state: &SolverState,
    dict: &Dictionary,
    theirs: &[Tile],
) -> Result<(), String> {
    let settings = GoOutSettings {
        budget: Some(Duration::from_secs(o.seconds.max(1))),
        ..GoOutSettings::default()
    };
    let result = solve_go_out(
        &state.board,
        dict,
        &state.rack.tiles(),
        theirs,
        &settings,
        None,
    )
    .ok_or("no depth finished within the time budget")?;
    println!(
        "{}: depth {}, {} positions; value = spread change to the end of the game",
        if result.solved {
            "solved exactly"
        } else {
            "best estimate (not solved within the budget)"
        },
        result.depth,
        result.nodes
    );
    let diff = score_diff(&o.cgp);
    println!(
        "{:<24} {:>5} {:>8} {:>7}",
        "move", "score", "value", "final"
    );
    for line in result.lines.iter().take(o.top) {
        let value = if line.exact {
            format!("{:+}", line.value)
        } else {
            format!("<={:+}", line.value)
        };
        println!(
            "{:<24} {:>5} {:>8} {:>+7}",
            label(&line.mv),
            line.mv.score,
            value,
            diff + line.value
        );
    }
    Ok(())
}

fn preendgame(o: &Options, state: &SolverState, dict: &Dictionary) -> Result<(), String> {
    if o.rules.ending != Ending::EqualTurn {
        return Err("preendgame needs Crossplay's equal-turn ending".into());
    }
    let unseen = unseen(state);
    if unseen.len() != RACK_SIZE + 1 {
        return Err(format!(
            "{} tiles unseen: preendgame needs exactly one in the bag ({} unseen)",
            unseen.len(),
            RACK_SIZE + 1
        ));
    }
    let results = solve_one_in_bag(
        &state.board,
        dict,
        &state.rack.tiles(),
        &unseen,
        score_diff(&o.cgp),
        None,
    )
    .ok_or("no result")?;
    let solved = results.iter().filter(|r| r.exact).count();
    println!(
        "{} candidates, {solved} solved exactly over each possible last tile",
        results.len()
    );
    println!(
        "{:<24} {:>5} {:>6} {:>8}  by last tile",
        "move", "score", "win%", "spread"
    );
    for r in results.iter().filter(|r| r.exact).take(o.top) {
        let cases: Vec<String> = r
            .outcomes
            .iter()
            .map(|c| format!("{}{:+}", c.tile, c.value))
            .collect();
        println!(
            "{:<24} {:>5} {:>6.1} {:>+8.2}  {}",
            label(&r.mv),
            r.mv.score,
            r.win * 100.0,
            r.spread,
            cases.join(" ")
        );
    }
    Ok(())
}

fn infer(o: &Options, state: &SolverState, dict: &Dictionary) -> Result<(), String> {
    let played = tiles(o.played.as_deref().ok_or("infer needs --played")?);
    let score = o.score.ok_or("infer needs --score")?;
    let result = infer_leave(
        &state.board,
        dict,
        &state.rack.tiles(),
        &played,
        score,
        &InferenceSettings::default(),
        None,
    )
    .ok_or("no rack is consistent with that play")?;
    println!(
        "{} racks consistent with playing {} for {}",
        result.total_racks.unwrap_or(0),
        result.played_tiles,
        score
    );
    let mut marginals = result.letter_marginals.clone();
    marginals.sort_by(|a, b| b.p_at_least_one.total_cmp(&a.p_at_least_one));
    println!("{:<6} {:>8} {:>8}", "letter", "expected", "P(>=1)");
    for m in marginals.iter().take(o.top) {
        println!(
            "{:<6} {:>8.2} {:>7.1}%",
            m.letter, m.expected, m.p_at_least_one
        );
    }
    println!("most likely leaves:");
    for (leave, pct) in result.top_leaves.iter().take(o.top.min(10)) {
        println!("  {leave:<8} {pct:>5.2}%");
    }
    Ok(())
}
