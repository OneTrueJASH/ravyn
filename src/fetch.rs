//! Download the data files (the word list and leave values) into a data
//! directory, kept apart from the code: `ravyn fetch-data` and
//! `ravyl --fetch-data` both call [`fetch`]. The files live in their own
//! repository, not in the engine's source or crate.
//!
//! Each data version is a directory `v<YYYYMMDD>/` holding the files. Their
//! SHA-256 checksums are built into the engine ([`FILES`]), so nothing is
//! installed unless it is exactly the file this version of the engine was
//! made with, whoever controls the source; each file is written to a
//! temporary name and renamed into place. Downloads go through `curl` (on
//! macOS, Linux and Windows 10 and later), keeping the engine free of an
//! HTTP stack. Each file is first fetched from the public repository with
//! no credentials; only if that fails is a token looked up and the GitHub
//! API tried, the token handed to curl on its standard input, never on its
//! command line, where other programs could read it. A token is only ever
//! sent to the GitHub API, never to a `$RAVYN_DATA_URL` source.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The data version this build of the engine uses.
pub const DATA_VERSION: &str = "20261118";

/// Where the data is fetched from: the public data repository's raw files
/// (`$RAVYN_DATA_URL` overrides it).
const PUBLIC_SOURCE: &str = "https://raw.githubusercontent.com/OneTrueJASH/ravyn-data/main";

/// The fallback when the public files cannot be reached: the GitHub
/// contents API, with a token, for a private data repository.
const API_SOURCE: &str = "https://api.github.com/repos/OneTrueJASH/ravyn-data/contents";

/// The files of the data version, installed under `<root>/data/`, with
/// their SHA-256 checksums.
pub const FILES: [(&str, &str); 3] = [
    (
        "words.fst",
        "7df3ad5c099fcaa1d10112855656f6d87732bc64adf07e23bb722686d5d41da5",
    ),
    (
        "leaves.rlv",
        "ead8de54d2b05038eebdcaf585b6d4f01423ed109c98168c54a366ecb5216ec0",
    ),
    (
        "crossplay-removed.txt",
        "81639bd55ea4e48d856a3681ee96bc95425a5cb3fe48fac2c41088c02cc54d35",
    ),
];

/// The default data root: `~/.ravyn` (the files go in `~/.ravyn/data`).
pub fn default_root() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join(".ravyn"))
}

/// A token for the data repository when it is private: `$RAVYN_DATA_TOKEN`,
/// `$GITHUB_TOKEN`, or the GitHub CLI's login (`gh auth token`).
fn token() -> Option<String> {
    for var in ["RAVYN_DATA_TOKEN", "GITHUB_TOKEN"] {
        if let Some(t) = std::env::var(var).ok().filter(|t| !t.trim().is_empty()) {
            return Some(t.trim().to_string());
        }
    }
    let out = Command::new("gh").args(["auth", "token"]).output().ok()?;
    let t = String::from_utf8(out.stdout).ok()?;
    (out.status.success() && !t.trim().is_empty()).then(|| t.trim().to_string())
}

/// What curl reads on its standard input (`--config -`): the headers, the
/// token among them, so the token never appears on a command line.
fn curl_config(token: Option<&str>) -> Result<String, String> {
    let mut config = String::from("header = \"Accept: application/vnd.github.raw\"\n");
    if let Some(t) = token {
        if t.chars().any(|c| c == '"' || c == '\\' || c.is_control()) {
            return Err("the token holds characters no token has".into());
        }
        config.push_str(&format!("header = \"Authorization: Bearer {t}\"\n"));
    }
    Ok(config)
}

/// Download `name` of the data version into `to`.
fn download(source: &str, token: Option<&str>, name: &str, to: &Path) -> Result<(), String> {
    let url = format!("{source}/v{DATA_VERSION}/{name}");
    let config = curl_config(token)?;
    let mut child = Command::new("curl")
        .args(["--fail", "--silent", "--show-error", "--location"])
        .args(["--config", "-", "--output"])
        .arg(to)
        .arg(&url)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run curl ({e}); install it or copy the files by hand"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(config.as_bytes())
            .map_err(|e| format!("curl: {e}"))?;
    }
    let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr);
        return Err(format!("{url}: {}", why.trim()));
    }
    Ok(())
}

/// Download `name` into `to`: from `$RAVYN_DATA_URL` if it is set, with no
/// token; otherwise from the public files, then, if that fails and a token
/// can be found, from the GitHub API with it. `token` caches the lookup.
fn download_any(
    custom: Option<&str>,
    token: &mut Option<Option<String>>,
    name: &str,
    to: &Path,
) -> Result<(), String> {
    if let Some(source) = custom {
        return download(source, None, name, to);
    }
    let public = match download(PUBLIC_SOURCE, None, name, to) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    match token.get_or_insert_with(self::token) {
        Some(t) => download(API_SOURCE, Some(t), name, to).map_err(|e| format!("{public}\n{e}")),
        None => Err(format!(
            "{public}\nif the data repository is private, set $RAVYN_DATA_TOKEN \
             (or $GITHUB_TOKEN), or log in with `gh auth login`"
        )),
    }
}

/// Fetch the data version into `<root>/data/`, verifying every file
/// against the checksum built into the engine. Progress is printed to
/// standard output; files already up to date are left alone.
pub fn fetch(root: &Path) -> Result<(), String> {
    let custom = std::env::var("RAVYN_DATA_URL")
        .ok()
        .filter(|u| !u.trim().is_empty());
    let mut token = None;
    let dir = root.join("data");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (name, want) in FILES {
        let target = dir.join(name);
        if std::fs::read(&target).is_ok_and(|bytes| sha256_hex(&bytes) == want) {
            println!("{name}: up to date");
            continue;
        }
        let temp = dir.join(format!(".{name}.download"));
        let fetched = download_any(custom.as_deref(), &mut token, name, &temp).and_then(|()| {
            let bytes = std::fs::read(&temp).map_err(|e| e.to_string())?;
            let got = sha256_hex(&bytes);
            if got == want {
                Ok(bytes.len())
            } else {
                Err(format!(
                    "{name}: checksum {got}, expected {want}; nothing installed"
                ))
            }
        });
        let size = match fetched {
            Ok(size) => size,
            Err(e) => {
                let _ = std::fs::remove_file(&temp);
                return Err(e);
            }
        };
        std::fs::rename(&temp, &target).map_err(|e| format!("{}: {e}", target.display()))?;
        println!("{name}: {size} bytes, checksum verified");
    }
    println!("data v{DATA_VERSION} in {}", dir.display());
    Ok(())
}

/// SHA-256 of `data`, in lowercase hexadecimal (FIPS 180-4).
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(v);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_token_stays_off_the_command_line() {
        let config = curl_config(Some("ghp_abc123")).unwrap();
        assert!(config.contains("Authorization: Bearer ghp_abc123"));
        assert!(curl_config(None).unwrap().lines().count() == 1);
        // Nothing that could break out of the quoted header.
        assert!(curl_config(Some("a\"b")).is_err());
        assert!(curl_config(Some("a\nheader = x")).is_err());
    }

    #[test]
    fn a_custom_source_never_gets_a_token() {
        // A source that cannot answer: the error must come from it alone,
        // without a token lookup or a fallback to the GitHub API.
        let dir = std::env::temp_dir().join(format!("ravyn-fetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut token = None;
        let err = download_any(
            Some("http://127.0.0.1:9"),
            &mut token,
            "words.fst",
            &dir.join("x"),
        )
        .unwrap_err();
        assert!(token.is_none(), "no token is looked up for a custom source");
        assert!(!err.contains("api.github.com"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn built_in_checksums_are_well_formed() {
        for (name, sum) in FILES {
            assert_eq!(sum.len(), 64, "{name}");
            assert!(sum
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        }
    }

    #[test]
    fn sha256_matches_the_standard_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // A million 'a's crosses many blocks.
        assert_eq!(
            sha256_hex(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
}
