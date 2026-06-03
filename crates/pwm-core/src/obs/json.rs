// SPDX-License-Identifier: Apache-2.0
//! A minimal, dependency-free JSON writer for observability records.
//!
//! `pwm-core` is the `no_std`, dependency-free trust root (INV-ARCH-01), so the
//! observability records are serialized by this small hand-rolled writer rather
//! than by pulling a JSON library into the root crate. The surface is
//! deliberately tiny — exactly the scalar shapes a [`super::value::LoggableValue`]
//! can take — which keeps the value-free guarantee easy to audit: there is no
//! code path here that can serialize a `BoundedInt`, `Tensor`, or `M31`.

use alloc::string::String;

/// Append a JSON string literal (with the surrounding quotes) for `s`, escaping
/// per RFC 8259.
pub(crate) fn push_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                // Other control characters: \u00XX.
                out.push_str("\\u");
                let code = c as u32;
                for shift in [12u32, 8, 4, 0] {
                    let nibble = (code >> shift) & 0xf;
                    out.push(char::from_digit(nibble, 16).unwrap_or('0'));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Append a JSON object key (a quoted string followed by `:`).
pub(crate) fn push_key(out: &mut String, key: &str) {
    push_string(out, key);
    out.push(':');
}

/// Append an unsigned integer.
pub(crate) fn push_u64(out: &mut String, n: u64) {
    out.push_str(itoa(n).as_str());
}

/// Append a finite `f64`. Non-finite values (`NaN`/`±inf`) cannot appear in
/// canonical JSON, so they are emitted as `null` — observability timings are
/// always finite, and a non-finite value is a bug surfaced as `null` rather than
/// invalid JSON.
pub(crate) fn push_f64(out: &mut String, x: f64) {
    if x.is_finite() {
        out.push_str(alloc::format!("{x}").as_str());
    } else {
        out.push_str("null");
    }
}

/// Append a boolean.
pub(crate) fn push_bool(out: &mut String, b: bool) {
    out.push_str(if b { "true" } else { "false" });
}

/// Append a lowercase-hex string literal for a 32-byte commitment/hash.
pub(crate) fn push_hex32(out: &mut String, bytes: &[u8; 32]) {
    out.push('"');
    for byte in bytes {
        out.push(hex_nibble(byte >> 4));
        out.push(hex_nibble(byte & 0xf));
    }
    out.push('"');
}

fn hex_nibble(n: u8) -> char {
    char::from_digit(n as u32, 16).unwrap_or('0')
}

/// Format a `u64` without `std`. Returns a small heap string.
fn itoa(mut n: u64) -> String {
    if n == 0 {
        return String::from("0");
    }
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    // SAFETY-free: the bytes are ASCII digits by construction.
    String::from_utf8_lossy(&buf[i..]).into_owned()
}
