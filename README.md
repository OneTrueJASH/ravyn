# RAVYN

**Racing Arms by Value: Ȳ over n**

A crossword-game analysis engine in Rust. RAVYN generates every legal play,
values each by score and leave, then races the candidates as bandit arms:
each playout samples an outcome *Y*, and arms are dropped once their mean
Ȳₙ is clearly behind the leader's.

> **Status:** early development. The engine is being extracted from
> [cpsolver2](https://github.com/OneTrueJASH/cpsolver2), where it currently
> lives as a workspace crate, and will move here once its API settles.

## License

[MIT](LICENSE)
