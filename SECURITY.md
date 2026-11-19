# Security

Please report security problems privately, not in a public issue: use
**Report a vulnerability** on this repository's
[Security tab](https://github.com/OneTrueJASH/ravyn/security/advisories/new).
You will get an answer there.

Only the latest release is supported. Data files are checked against
SHA-256 checksums built into the program before they are installed, so a
report that a data file was accepted without matching its checksum is in
scope, as is anything that reads past the end of a file, runs a command,
or writes outside the data directory.
