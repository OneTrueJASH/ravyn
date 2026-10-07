//! Fuzz the CGP reader: arbitrary text must give an error or a position
//! that writes back to CGP and reads back to the same position.
#![no_main]

use cpsolver2::game::SolverState;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(state) = SolverState::from_cgp(text) {
        let cgp = state.to_cgp((0, 0));
        let again = SolverState::from_cgp(&cgp).expect("a written CGP reads back");
        assert_eq!(again.to_cgp((0, 0)), cgp);
        let _ = state.unseen_tile_count();
    }
});
