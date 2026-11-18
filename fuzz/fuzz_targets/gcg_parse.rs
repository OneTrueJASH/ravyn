//! Fuzz the GCG reader: arbitrary text must give an error or a game that
//! writes back to GCG and reads back to the same game, and replaying it
//! must not panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ravyn::gcg::Game;
use ravyn::rules::CROSSPLAY;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(game) = Game::parse(text) {
        let written = game.to_gcg();
        let mut again = Game::parse(&written).expect("a written GCG reads back");
        let mut game = game;
        game.events.iter_mut().for_each(|e| e.line = 0);
        again.events.iter_mut().for_each(|e| e.line = 0);
        assert_eq!(again, game, "{written}");
        let _ = game.replay(&CROSSPLAY);
    }
});
