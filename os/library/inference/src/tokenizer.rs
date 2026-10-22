//! Qwen3.5 NFC normalization, Unicode pre-tokenization, byte BPE, and chat framing.
use crate::{Error, Result};
use alloc::{
    collections::{BTreeMap, BinaryHeap},
    format,
    string::String,
    vec::Vec,
};
use core::cmp::Reverse;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

pub const MAGIC: &[u8; 8] = b"D3BPE\0\0\0";

pub struct Tokenizer<'a> {
    tokens: Vec<&'a [u8]>,
    merges: BTreeMap<(u32, u32), (usize, u32)>,
    added: Vec<(u32, bool)>,
    bytes: [u32; 256],
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(count).ok_or(Error::InvalidTokenizer)?;
        self.0 = tail;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| Error::InvalidTokenizer)?))
    }
}

impl<'a> Tokenizer<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let mut reader = Reader(data);
        if reader.take(8)? != MAGIC || reader.u32()? != 1 {
            return Err(Error::InvalidTokenizer);
        }
        let count = reader.u32()? as usize;
        let merge_count = reader.u32()? as usize;
        let special_count = reader.u32()? as usize;
        if !(256..=1_048_576).contains(&count) || merge_count > count || special_count > 256 {
            return Err(Error::InvalidTokenizer);
        }
        let mut tokens = Vec::new();
        let mut byte_ids = [u32::MAX; 256];
        for id in 0..count {
            let len = reader.u32()? as usize;
            if len == 0 || len > 1_048_576 {
                return Err(Error::InvalidTokenizer);
            }
            let bytes = reader.take(len)?;
            if let [byte] = bytes {
                if byte_ids[*byte as usize] != u32::MAX {
                    return Err(Error::InvalidTokenizer);
                }
                byte_ids[*byte as usize] = id as u32;
            }
            tokens.push(bytes);
        }
        if byte_ids.contains(&u32::MAX) {
            return Err(Error::InvalidTokenizer);
        }
        let mut merges = BTreeMap::new();
        for rank in 0..merge_count {
            let left = reader.u32()?;
            let right = reader.u32()?;
            let merged = reader.u32()?;
            let a = tokens.get(left as usize).ok_or(Error::InvalidTokenizer)?;
            let b = tokens.get(right as usize).ok_or(Error::InvalidTokenizer)?;
            let c = tokens.get(merged as usize).ok_or(Error::InvalidTokenizer)?;
            if c.len() != a.len() + b.len() || !c.starts_with(a) || !c.ends_with(b) || merges.insert((left, right), (rank, merged)).is_some() {
                return Err(Error::InvalidTokenizer);
            }
        }
        let mut added = Vec::new();
        for _ in 0..special_count {
            let id = reader.u32()?;
            let special = match reader.u32()? {
                0 => false,
                1 => true,
                _ => return Err(Error::InvalidTokenizer),
            };
            let token = tokens.get(id as usize).ok_or(Error::InvalidTokenizer)?;
            if token.len() < 2 || core::str::from_utf8(token).is_err() || added.iter().any(|&(other, _)| other == id) {
                return Err(Error::InvalidTokenizer);
            }
            added.push((id, special));
        }
        if !reader.0.is_empty() {
            return Err(Error::InvalidTokenizer);
        }
        Ok(Self {
            tokens,
            merges,
            added,
            bytes: byte_ids,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Recognize the upstream added tokens, normalize ordinary text to NFC, then apply BPE.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            let next = self
                .added
                .iter()
                .filter_map(|&(id, _)| {
                    let token = core::str::from_utf8(self.tokens[id as usize]).ok()?;
                    remaining.find(token).map(|position| (position, id, token.len()))
                })
                .min_by_key(|&(position, _, len)| (position, Reverse(len)));
            let (ordinary, tail) = remaining.split_at(next.map_or(remaining.len(), |(position, _, _)| position));
            let normalized: String = ordinary.nfc().collect();
            for piece in pieces(&normalized) {
                self.bpe(piece.as_bytes(), &mut output);
            }
            if let Some((_, id, len)) = next {
                output.push(id);
                remaining = &tail[len..];
            } else {
                break;
            }
        }
        Ok(output)
    }

    /// One user turn, with the official non-thinking assistant prefix.
    pub fn encode_chat(&self, user: &str) -> Result<Vec<u32>> {
        self.encode(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            user.trim()
        ))
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let token = self.tokens.get(id as usize).ok_or(Error::InvalidToken)?;
            if !skip_special || !self.added.iter().any(|&(other, special)| other == id && special) {
                bytes.extend_from_slice(token);
            }
        }
        // Byte BPE can end partway through a Unicode scalar when generation hits its token limit.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn bpe(&self, bytes: &[u8], output: &mut Vec<u32>) {
        if bytes.is_empty() {
            return;
        }
        struct Node {
            id: u32,
            prev: Option<usize>,
            next: Option<usize>,
            live: bool,
        }
        let mut nodes: Vec<_> = bytes
            .iter()
            .enumerate()
            .map(|(i, &byte)| Node {
                id: self.bytes[byte as usize],
                prev: i.checked_sub(1),
                next: (i + 1 < bytes.len()).then_some(i + 1),
                live: true,
            })
            .collect();
        let mut heap = BinaryHeap::new();
        let enqueue = |left: usize, nodes: &[Node], heap: &mut BinaryHeap<_>| {
            if let Some(right) = nodes[left].next
                && let Some(&(rank, merged)) = self.merges.get(&(nodes[left].id, nodes[right].id))
            {
                heap.push(Reverse((rank, left, right, nodes[left].id, nodes[right].id, merged)));
            }
        };
        for index in 0..nodes.len() {
            enqueue(index, &nodes, &mut heap);
        }
        while let Some(Reverse((_, left, right, a, b, merged))) = heap.pop() {
            if !nodes[left].live || !nodes[right].live || nodes[left].next != Some(right) || nodes[left].id != a || nodes[right].id != b {
                continue;
            }
            nodes[left].id = merged;
            nodes[left].next = nodes[right].next;
            nodes[right].live = false;
            if let Some(next) = nodes[right].next {
                nodes[next].prev = Some(left);
            }
            if let Some(prev) = nodes[left].prev {
                enqueue(prev, &nodes, &mut heap);
            }
            enqueue(left, &nodes, &mut heap);
        }
        output.extend(nodes.into_iter().filter(|node| node.live).map(|node| node.id));
    }
}

fn letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}
fn mark(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark | GeneralCategory::SpacingMark | GeneralCategory::EnclosingMark
    )
}
fn number(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::DecimalNumber | GeneralCategory::LetterNumber | GeneralCategory::OtherNumber
    )
}
fn word(c: char) -> bool {
    letter(c) || mark(c)
}
fn punctuation(c: char) -> bool {
    !c.is_whitespace() && !word(c) && !number(c)
}

// Implements the ordered alternatives in the pinned tokenizer's Unicode split expression.
// Positions are UTF-8 boundaries. A number is one scalar, including non-ASCII numbers.
fn pieces(mut text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    while !text.is_empty() {
        let first = text.chars().next().expect("nonempty text");
        let contraction = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
            .iter()
            .find(|suffix| text.get(..suffix.len()).is_some_and(|prefix| prefix.eq_ignore_ascii_case(suffix)))
            .map(|suffix| suffix.len())
            // Unicode case folding also maps long s to ASCII s in the upstream regex.
            .or_else(|| text.starts_with("'ſ").then_some("'ſ".len()));
        let length = if let Some(length) = contraction {
            length
        } else if word(first) {
            text.chars().take_while(|&c| word(c)).map(char::len_utf8).sum()
        } else if first != '\r' && first != '\n' && !letter(first) && !number(first) && text[first.len_utf8()..].chars().next().is_some_and(word) {
            first.len_utf8() + text[first.len_utf8()..].chars().take_while(|&c| word(c)).map(char::len_utf8).sum::<usize>()
        } else if number(first) {
            first.len_utf8()
        } else {
            let start = usize::from(first == ' ');
            let punct_len: usize = text[start..].chars().take_while(|&c| punctuation(c)).map(char::len_utf8).sum();
            if punct_len > 0 {
                start
                    + punct_len
                    + text[start + punct_len..]
                        .chars()
                        .take_while(|&c| c == '\r' || c == '\n')
                        .map(char::len_utf8)
                        .sum::<usize>()
            } else {
                let whitespace: usize = text.chars().take_while(|c| c.is_whitespace()).map(char::len_utf8).sum();
                if let Some((pos, c)) = text[..whitespace].char_indices().rfind(|(_, c)| *c == '\r' || *c == '\n') {
                    pos + c.len_utf8()
                } else if whitespace == text.len() {
                    whitespace
                } else {
                    let last = text[..whitespace].char_indices().next_back().map_or(0, |(pos, _)| pos);
                    if last > 0 { last } else { whitespace }
                }
            }
        };
        debug_assert!(length > 0);
        let (head, tail) = text.split_at(length);
        result.push(head);
        text = tail;
    }
    result
}

/// Verify independently encoded text from the pinned upstream tokenizer.
pub fn verify_fixture(tokenizer: &Tokenizer<'_>, bytes: &[u8]) -> Result<usize> {
    let mut reader = Reader(bytes);
    let count = reader.u32()? as usize;
    if count == 0 || count > 4096 {
        return Err(Error::InvalidTokenizer);
    }
    for _ in 0..count {
        let text_len = reader.u32()? as usize;
        let token_count = reader.u32()? as usize;
        let chat = match reader.u32()? {
            0 => false,
            1 => true,
            _ => return Err(Error::InvalidTokenizer),
        };
        let text = core::str::from_utf8(reader.take(text_len)?).map_err(|_| Error::InvalidTokenizer)?;
        let actual = if chat { tokenizer.encode_chat(text)? } else { tokenizer.encode(text)? };
        if actual.len() != token_count {
            return Err(Error::VerificationFailed);
        }
        for id in actual {
            if reader.u32()? != id {
                return Err(Error::VerificationFailed);
            }
        }
    }
    if !reader.0.is_empty() {
        return Err(Error::InvalidTokenizer);
    }
    Ok(count)
}
