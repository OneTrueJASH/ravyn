//! Fuzz the `.klv2` leave-table parser: arbitrary bytes must give an error
//! or a table whose lookups do not panic.
#![no_main]

use ravyn::leaves::LeaveValues;
use ravyn::tile::Tile;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(table) = LeaveValues::parse(data) {
        for leave in [
            vec![Tile::new('A')],
            vec![Tile::blank(' '), Tile::new('S')],
            "AEINRS".chars().map(Tile::new).collect::<Vec<_>>(),
        ] {
            let _ = table.value(&leave);
        }
    }
});
