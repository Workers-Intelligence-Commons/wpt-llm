//! Byte-level BPE: training, encoding and decoding.
//!
//! Token ids: 0..=255 are the bytes themselves, 256 is the end-of-document
//! token, 257..=512 are reserved for special tokens (the chat template's
//! tags, named in Python), and every learned merge adds the next id from 513
//! on. Every byte has a token, so encoding never drops anything; encoding
//! never produces END or a special token, and decoding gives them no bytes.
//!
//! Text is first cut into chunks so merges never cross a word boundary: a
//! chunk is an optional single leading space followed by bytes that are not
//! ASCII whitespace, or a run of whitespace. The rule looks at bytes only, so
//! results never depend on a Unicode version.
//!
//! Training is deterministic: it runs on one thread, and when two pairs are
//! equally frequent the pair with the smaller (left, right) ids wins.

use std::cmp::Reverse;
use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, ThreadId};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const END: u32 = 256;
const LAST_SPECIAL: u32 = 512;
const FIRST_MERGE: u32 = LAST_SPECIAL + 1;

type Pair = (u32, u32);

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Cut `data` into chunks (see the module comment).
fn chunks(data: &[u8]) -> Vec<&[u8]> {
    let n = data.len();
    // a space directly followed by a word starts that word's chunk
    let word_start = |i: usize| data[i] == b' ' && i + 1 < n && !is_space(data[i + 1]);
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let start = i;
        if word_start(i) || !is_space(data[i]) {
            i += 1;
            while i < n && !is_space(data[i]) {
                i += 1;
            }
        } else {
            while i < n && is_space(data[i]) && !word_start(i) {
                i += 1;
            }
        }
        out.push(&data[start..i]);
    }
    out
}

/// Replace every non-overlapping occurrence of `pair`, left to right.
fn merge_word(word: &[u32], pair: Pair, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && (word[i], word[i + 1]) == pair {
            out.push(new_id);
            i += 2;
        } else {
            out.push(word[i]);
            i += 1;
        }
    }
    out
}

fn pairs_of(word: &[u32]) -> impl Iterator<Item = Pair> + '_ {
    word.windows(2).map(|w| (w[0], w[1]))
}

/// Counts chunks across documents, then learns merges.
#[pyclass]
struct Trainer {
    counts: HashMap<Vec<u8>, u64>,
}

#[pymethods]
impl Trainer {
    #[new]
    fn new() -> Self {
        Trainer { counts: HashMap::new() }
    }

    /// Count the chunks of one document.
    fn add(&mut self, data: &[u8]) {
        for chunk in chunks(data) {
            *self.counts.entry(chunk.to_vec()).or_insert(0) += 1;
        }
    }

    /// Learn up to `n_merges` merges; stops early when no pair occurs twice.
    /// Returns the merges in the order learned, as (left id, right id).
    fn train(&self, py: Python<'_>, n_merges: usize) -> Vec<Pair> {
        py.detach(|| learn(&self.counts, n_merges))
    }
}

fn learn(counts: &HashMap<Vec<u8>, u64>, n_merges: usize) -> Vec<Pair> {
    // sort so nothing depends on hash order
    let mut entries: Vec<(&Vec<u8>, &u64)> = counts.iter().collect();
    entries.sort();
    let mut words: Vec<Vec<u32>> = entries.iter().map(|(w, _)| w.iter().map(|&b| b as u32).collect()).collect();
    let freq: Vec<i64> = entries.iter().map(|&(_, &c)| c as i64).collect();

    let mut pair_count: HashMap<Pair, i64> = HashMap::new();
    let mut where_: HashMap<Pair, HashSet<usize>> = HashMap::new();
    for (i, word) in words.iter().enumerate() {
        for p in pairs_of(word) {
            *pair_count.entry(p).or_insert(0) += freq[i];
            where_.entry(p).or_default().insert(i);
        }
    }
    // max count first; on a tie, the smallest pair
    let mut heap: BinaryHeap<(i64, Reverse<Pair>)> = pair_count.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

    let mut merges = Vec::new();
    while merges.len() < n_merges {
        let Some((count, Reverse(pair))) = heap.pop() else { break };
        if pair_count.get(&pair).copied().unwrap_or(0) != count {
            continue; // stale entry; the current count was pushed separately
        }
        if count < 2 {
            break;
        }
        let new_id = FIRST_MERGE + merges.len() as u32;
        merges.push(pair);

        let mut affected: Vec<usize> = where_.remove(&pair).unwrap_or_default().into_iter().collect();
        affected.sort_unstable();
        let mut changed: HashSet<Pair> = HashSet::new();
        for i in affected {
            if !pairs_of(&words[i]).any(|p| p == pair) {
                continue;
            }
            let merged = merge_word(&words[i], pair, new_id);
            for p in pairs_of(&words[i]) {
                *pair_count.get_mut(&p).unwrap() -= freq[i];
                changed.insert(p);
            }
            for p in pairs_of(&merged) {
                *pair_count.entry(p).or_insert(0) += freq[i];
                where_.entry(p).or_default().insert(i);
                changed.insert(p);
            }
            words[i] = merged;
        }
        pair_count.remove(&pair);
        let mut changed: Vec<Pair> = changed.into_iter().collect();
        changed.sort_unstable();
        for p in changed {
            if let Some(&c) = pair_count.get(&p)
                && c > 0
            {
                heap.push((c, Reverse(p)));
            }
        }
    }
    merges
}

/// The tokens of each chunk one thread has encoded.
type Cache = HashMap<Vec<u8>, Vec<u32>>;

/// Each thread's cache of one Codec, by the thread.
type Caches = Mutex<HashMap<ThreadId, Arc<Mutex<Cache>>>>;

/// Chunks one thread's cache holds before it starts again empty (about 100 MB): every thread has its own, and
/// a corpus has far more distinct chunks than that.
const CACHE_CHUNKS: usize = 1 << 20;

/// The Codecs one thread has a cache in, so that the thread's end removes its caches from them: a Codec kept
/// for a whole build (pools of threads that start and end) then holds the caches of the threads alive only.
struct Leaving {
    thread: ThreadId, // read when the thread starts: thread::current() is not there any more at its end
    codecs: RefCell<Vec<Weak<Caches>>>,
}

impl Drop for Leaving {
    fn drop(&mut self) {
        for caches in self.codecs.borrow().iter().filter_map(Weak::upgrade) {
            if let Ok(mut caches) = caches.lock() {
                caches.remove(&self.thread);
            }
        }
    }
}

thread_local! {
    static LEAVING: Leaving = Leaving { thread: thread::current().id(), codecs: RefCell::new(Vec::new()) };
}

/// Encodes and decodes with a fixed list of merges. Each thread keeps its
/// own cache of encoded chunks until it ends, so threads encoding at once
/// never wait for each other; a chunk's tokens are the same whichever cache
/// holds them, or none.
#[pyclass(frozen)]
struct Codec {
    ranks: HashMap<Pair, u32>,
    pieces: Vec<Vec<u8>>, // id -> bytes (END and the special tokens decode to nothing)
    // each living thread's cache (a thread's end removes its own, see Leaving); the map is locked only to
    // find a thread's own, and that cache's lock is never waited for, since only its thread takes it
    caches: Arc<Caches>,
}

impl Codec {
    fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        let mut word: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        loop {
            let best = pairs_of(&word).filter_map(|p| self.ranks.get(&p).map(|&r| (r, p))).min();
            let Some((rank, pair)) = best else { break };
            word = merge_word(&word, pair, FIRST_MERGE + rank);
        }
        word
    }

    /// The calling thread's cache, made on its first call and removed when the thread ends. A thread that is
    /// already ending (encoding from another thread-local's destructor) gets an empty cache it does not keep.
    fn own_cache(&self) -> Arc<Mutex<Cache>> {
        let mut caches = self.caches.lock().unwrap();
        if let Some(own) = caches.get(&thread::current().id()) {
            return own.clone();
        }
        let own = Arc::<Mutex<Cache>>::default();
        let registered = LEAVING.try_with(|leaving| {
            let mut codecs = leaving.codecs.borrow_mut();
            codecs.retain(|codec| codec.strong_count() > 0); // the Codecs dropped since
            codecs.push(Arc::downgrade(&self.caches));
        });
        if registered.is_ok() {
            caches.insert(thread::current().id(), own.clone());
        }
        own
    }

    fn encode_bytes(&self, data: &[u8]) -> Vec<u32> {
        let own = self.own_cache();
        let mut cache = own.lock().unwrap();
        let mut out = Vec::new();
        for chunk in chunks(data) {
            if let Some(ids) = cache.get(chunk) {
                out.extend_from_slice(ids);
            } else {
                let ids = self.encode_chunk(chunk);
                out.extend_from_slice(&ids);
                if cache.len() >= CACHE_CHUNKS {
                    cache.clear();
                }
                cache.insert(chunk.to_vec(), ids);
            }
        }
        out
    }
}

#[pymethods]
impl Codec {
    #[new]
    fn new(merges: Vec<Pair>) -> PyResult<Self> {
        let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        pieces.extend((END..=LAST_SPECIAL).map(|_| Vec::new())); // END and the special tokens
        let mut ranks = HashMap::new();
        for (rank, &(a, b)) in merges.iter().enumerate() {
            let id = FIRST_MERGE as usize + rank;
            if a as usize >= id || b as usize >= id || (END..=LAST_SPECIAL).contains(&a) || (END..=LAST_SPECIAL).contains(&b) {
                return Err(PyValueError::new_err(format!("merge {rank} refers to an id it cannot use: ({a}, {b})")));
            }
            if ranks.insert((a, b), rank as u32).is_some() {
                return Err(PyValueError::new_err(format!("merge {rank} repeats an earlier merge: ({a}, {b})")));
            }
            let mut piece = pieces[a as usize].clone();
            piece.extend_from_slice(&pieces[b as usize]);
            pieces.push(piece);
        }
        Ok(Codec { ranks, pieces, caches: Arc::default() })
    }

    fn encode(&self, py: Python<'_>, data: &[u8]) -> Vec<u32> {
        py.detach(|| self.encode_bytes(data))
    }

    fn decode<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let mut out = Vec::new();
        for id in ids {
            let piece = self.pieces.get(id as usize).ok_or_else(|| PyValueError::new_err(format!("unknown token id {id}")))?;
            out.extend_from_slice(piece);
        }
        Ok(PyBytes::new(py, &out))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Trainer>()?;
    m.add_class::<Codec>()?;
    m.add("END", END)?;
    m.add("FIRST_MERGE", FIRST_MERGE)?;
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("RUSTC_VERSION", env!("WPT_RUSTC_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_keep_every_byte_in_order() {
        let text = b"  hello world\n\nfoo  bar \tbaz ";
        let joined: Vec<u8> = chunks(text).concat();
        assert_eq!(joined, text);
        assert_eq!(chunks(b"a b"), vec![&b"a"[..], &b" b"[..]]);
        assert_eq!(chunks(b"a  b"), vec![&b"a"[..], &b" "[..], &b" b"[..]]);
    }

    #[test]
    fn a_thread_s_cache_leaves_the_codec_when_the_thread_ends() {
        let codec = Codec::new(vec![(b'a' as u32, b'b' as u32)]).unwrap();
        let ids = thread::scope(|scope| {
            let ids = (0..4).map(|_| scope.spawn(|| codec.encode_bytes(b"ab ab abc"))).collect::<Vec<_>>();
            ids.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(ids.iter().all(|one| *one == codec.encode_bytes(b"ab ab abc")));
        // the four threads ended: only this one's cache is left
        assert_eq!(codec.caches.lock().unwrap().keys().collect::<Vec<_>>(), vec![&thread::current().id()]);
    }
}
