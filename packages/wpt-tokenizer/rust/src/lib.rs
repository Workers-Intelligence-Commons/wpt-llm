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
//! Training is deterministic: the merges are the same on any machine and with any number of threads, and
//! when two pairs are equally frequent the pair with the smaller (left, right) ids wins. Chunks
//! are counted in a table of a fixed number of slots (Summary), so the memory
//! does not grow with the corpus.

use std::cmp::Reverse;
use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, ThreadId};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const END: u32 = 256;
const LAST_SPECIAL: u32 = 512;
const FIRST_MERGE: u32 = LAST_SPECIAL + 1;

/// Marks a token merged into the one before it.
const DEAD: u32 = u32::MAX;

/// From this many bytes a chunk is encoded with a heap rather than by rescanning its word.
const HEAP_FROM: usize = 12;

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

/// The slots a Trainer's table holds unless it is given another number. A fixed number, never measured from
/// the machine, so a vocabulary never depends on the machine that trained it. A slot costs about 120 bytes
/// (the chunk, its count, its error bound, its places in the heap and the index), and learning builds a few
/// words and pair tables over the slots' chunks: with this many, reading and learning together stay around
/// 25 GB, whatever the number of documents.
const SLOTS: usize = 50_000_000;

struct Slot {
    chunk: Arc<[u8]>,
    count: u64,
    /// The count of the chunk this slot held when this one took it over: the count is at most this much too
    /// high (0 for a chunk that never took over a slot).
    error: u64,
    at: usize, // the slot's place in the heap
}

/// The Space-Saving summary of Metwally, Agrawal and El Abbadi (2005): at most `capacity` chunks are counted.
/// A chunk in the table has its count raised; a new chunk takes a free slot with count 1; when the table is
/// full, the slot with the smallest count is evicted and the new chunk takes it with the evicted count plus 1,
/// recording the evicted count as its error. So a chunk's true count is between `count - error` and `count`,
/// and the chunks that occur more than the smallest count are all in the table. The slot to evict is the first
/// by (count, chunk bytes): the table depends on the chunks and their order alone, never on hash order. A min-heap
/// of slots finds it.
struct Summary {
    capacity: usize,
    slots: Vec<Slot>,
    index: HashMap<Arc<[u8]>, usize>, // chunk -> slot
    heap: Vec<usize>,                 // slot numbers, the smallest (count, chunk) first
}

impl Summary {
    fn new(capacity: usize) -> Self {
        Summary { capacity, slots: Vec::new(), index: HashMap::new(), heap: Vec::new() }
    }

    fn count(&mut self, chunk: &[u8]) {
        if let Some(&slot) = self.index.get(chunk) {
            self.slots[slot].count += 1;
            self.sift_down(self.slots[slot].at);
            return;
        }
        let chunk: Arc<[u8]> = Arc::from(chunk);
        if self.slots.len() < self.capacity {
            let slot = self.slots.len();
            let at = self.heap.len();
            self.slots.push(Slot { chunk: chunk.clone(), count: 1, error: 0, at });
            self.heap.push(slot);
            self.index.insert(chunk, slot);
            self.sift_up(at);
        } else {
            let slot = self.heap[0];
            let evicted = std::mem::replace(&mut self.slots[slot].chunk, chunk.clone());
            self.index.remove(&evicted[..]);
            self.index.insert(chunk, slot);
            let taken = &mut self.slots[slot];
            taken.error = taken.count;
            taken.count += 1;
            self.sift_down(0);
        }
    }

    fn before(&self, a: usize, b: usize) -> bool {
        let (a, b) = (&self.slots[self.heap[a]], &self.slots[self.heap[b]]);
        (a.count, &a.chunk) < (b.count, &b.chunk)
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.slots[self.heap[a]].at = a;
        self.slots[self.heap[b]].at = b;
    }

    fn sift_up(&mut self, mut at: usize) {
        while at > 0 && self.before(at, (at - 1) / 2) {
            self.swap(at, (at - 1) / 2);
            at = (at - 1) / 2;
        }
    }

    fn sift_down(&mut self, mut at: usize) {
        loop {
            let (left, right) = (2 * at + 1, 2 * at + 2);
            let mut first = at;
            if left < self.heap.len() && self.before(left, first) {
                first = left;
            }
            if right < self.heap.len() && self.before(right, first) {
                first = right;
            }
            if first == at {
                return;
            }
            self.swap(at, first);
            at = first;
        }
    }

    /// (chunk, count, error) of every slot, by chunk bytes.
    fn entries(&self) -> Vec<(&[u8], u64, u64)> {
        let mut entries: Vec<_> = self.slots.iter().map(|s| (&s.chunk[..], s.count, s.error)).collect();
        entries.sort_unstable();
        entries
    }
}

/// Counts chunks across documents, then learns merges.
#[pyclass]
struct Trainer {
    table: Summary,
    merged: AtomicUsize, // the merges learned so far, for whoever watches `train` from another thread
}

#[pymethods]
impl Trainer {
    /// `slots`: how many distinct chunks are counted at once. With at least as many slots as the documents
    /// have distinct chunks, every count is exact.
    #[new]
    #[pyo3(signature = (slots = SLOTS))]
    fn new(slots: usize) -> PyResult<Self> {
        if slots == 0 {
            return Err(PyValueError::new_err("a trainer needs at least 1 slot"));
        }
        Ok(Trainer { table: Summary::new(slots), merged: AtomicUsize::new(0) })
    }

    /// Count the chunks of one document.
    fn add(&mut self, data: &[u8]) {
        for chunk in chunks(data) {
            self.table.count(chunk);
        }
    }

    /// (chunk, count, error) of every chunk in the table, by chunk bytes: the chunk's true count is between
    /// `count - error` and `count`.
    fn entries(&self) -> Vec<(Vec<u8>, u64, u64)> {
        self.table.entries().into_iter().map(|(chunk, count, error)| (chunk.to_vec(), count, error)).collect()
    }

    /// Learn up to `n_merges` merges from the table's chunks and counts; stops early when no pair occurs
    /// twice. Returns the merges in the order learned, as (left id, right id). `threads`: how many threads
    /// work on a merge (default: the machine's); the merges are the same for any number.
    #[pyo3(signature = (n_merges, threads = None))]
    fn train(&self, py: Python<'_>, n_merges: usize, threads: Option<usize>) -> PyResult<Vec<Pair>> {
        let threads = match threads {
            Some(0) => return Err(PyValueError::new_err("training needs at least 1 thread")),
            Some(threads) => threads,
            None => thread::available_parallelism().map_or(1, |n| n.get()),
        };
        Ok(py.detach(|| {
            let entries = self.table.entries().into_iter().map(|(chunk, count, _)| (chunk, count)).collect();
            learn_on(entries, n_merges, threads, SLICE_MIN, &self.merged)
        }))
    }

    /// How many merges `train` has learned so far, readable from another thread while it runs.
    fn merged(&self) -> usize {
        self.merged.load(Ordering::Relaxed)
    }
}

/// Words per slice at least: fewer words than twice this stay on the calling thread, since a thread costs more
/// than the work it would take.
const SLICE_MIN: usize = 4096;

/// What one slice of words gives back for one merge: the change of each pair's count (a pair a word lost or
/// gained, zero included) and the words that now hold each new pair.
struct Rewritten {
    delta: HashMap<Pair, i64>,
    holders: HashMap<Pair, Vec<usize>>, // ascending
}

/// The slices `0..len` is cut into: as many as `threads`, none with fewer than `min_slice` items, one when the
/// work is small. They depend on `len`, `threads` and `min_slice` only.
fn slices(len: usize, threads: usize, min_slice: usize) -> Vec<Range<usize>> {
    let count = (len / min_slice.max(1)).clamp(1, threads.max(1));
    let size = len.div_ceil(count);
    (0..count).map(|k| (k * size).min(len)..((k + 1) * size).min(len)).collect()
}

/// `work` on each part, the results in part order. A single part runs on the calling thread.
fn per_part<P: Send, T: Send>(parts: Vec<P>, work: impl Fn(P) -> T + Sync) -> Vec<T> {
    if parts.len() == 1 {
        return parts.into_iter().map(work).collect();
    }
    thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = parts.into_iter().map(|part| scope.spawn(move || work(part))).collect();
        handles.into_iter().map(|handle| handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))).collect()
    })
}

/// Rewrite, in place, the words of `affected` that hold `pair`. `words` starts at the word `base`, and holds
/// all of `affected`, which is ascending.
fn rewrite(words: &mut [Vec<u32>], base: usize, freq: &[i64], affected: &[usize], pair: Pair, new_id: u32) -> Rewritten {
    let mut out = Rewritten { delta: HashMap::new(), holders: HashMap::new() };
    for &i in affected {
        let word = &mut words[i - base];
        if !pairs_of(word).any(|p| p == pair) {
            continue;
        }
        let merged = merge_word(word, pair, new_id);
        // only the pairs at a merge change: a pair of two tokens no merge took stays, with its count and its
        // holders, and adding them again for every merge of a long word would outgrow the memory
        let mut taken = vec![false; word.len()]; // the tokens a merge takes, left to right as merge_word does
        let mut at = 0;
        while at + 1 < word.len() {
            if (word[at], word[at + 1]) == pair {
                taken[at] = true;
                taken[at + 1] = true;
                at += 2;
            } else {
                at += 1;
            }
        }
        for at in 0..word.len() - 1 {
            if taken[at] || taken[at + 1] {
                *out.delta.entry((word[at], word[at + 1])).or_insert(0) -= freq[i];
            }
        }
        for p in pairs_of(&merged).filter(|p| p.0 == new_id || p.1 == new_id) {
            *out.delta.entry(p).or_insert(0) += freq[i];
            let held = out.holders.entry(p).or_default();
            if held.last() != Some(&i) {
                held.push(i);
            }
        }
        *word = merged;
    }
    out
}

/// Learn merges from distinct chunks and their counts, sorted by chunk bytes so nothing depends on hash order.
///
/// The merges come one after the other, since each depends on the counts the earlier ones left. The work
/// inside one merge is independent between words: it runs on `threads` threads over slices of the words, and
/// the slices' results are combined in slice order, so the merges are the same for any number of threads.
/// `min_slice`: the fewest words a slice holds.
fn learn_on(entries: Vec<(&[u8], u64)>, n_merges: usize, threads: usize, min_slice: usize, merged: &AtomicUsize) -> Vec<Pair> {
    let freq: Vec<i64> = entries.iter().map(|&(_, c)| c as i64).collect();
    let mut words: Vec<Vec<u32>> = per_part(slices(entries.len(), threads, min_slice), |slice| {
        entries[slice].iter().map(|(w, _)| w.iter().map(|&b| b as u32).collect::<Vec<u32>>()).collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();

    // each slice counts its own words; the slices' tables are added in slice order, so a pair's words stay in
    // ascending order
    let counted = per_part(slices(words.len(), threads, min_slice), |slice| {
        let mut count: HashMap<Pair, i64> = HashMap::new();
        let mut holders: HashMap<Pair, Vec<usize>> = HashMap::new();
        for i in slice {
            for p in pairs_of(&words[i]) {
                *count.entry(p).or_insert(0) += freq[i];
                let held = holders.entry(p).or_default();
                if held.last() != Some(&i) {
                    held.push(i);
                }
            }
        }
        (count, holders)
    });
    let mut pair_count: HashMap<Pair, i64> = HashMap::new();
    let mut where_: HashMap<Pair, Vec<usize>> = HashMap::new();
    for (count, holders) in counted {
        for (p, c) in count {
            *pair_count.entry(p).or_insert(0) += c;
        }
        for (p, mut held) in holders {
            where_.entry(p).or_default().append(&mut held);
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
        merged.store(merges.len(), Ordering::Relaxed);

        // the words listed under the pair, once each, in ascending order (a list may name a word more than
        // once, or one that no longer holds the pair)
        let mut affected = where_.remove(&pair).unwrap_or_default();
        affected.sort_unstable();
        affected.dedup();
        // each slice of `affected` gets the stretch of `words` it covers, so the threads rewrite in place
        let parts = slices(affected.len(), threads, min_slice);
        let mut rest = &mut words[..];
        let mut base = 0;
        let mut stretches = Vec::new();
        for (k, part) in parts.iter().enumerate() {
            let end = parts.get(k + 1).map_or(base + rest.len(), |next| affected[next.start]);
            let (stretch, tail) = rest.split_at_mut(end - base);
            stretches.push((stretch, base, &affected[part.clone()]));
            rest = tail;
            base = end;
        }
        let rewritten = per_part(stretches, |(stretch, base, affected)| rewrite(stretch, base, &freq, affected, pair, new_id));
        let mut changed: Vec<Pair> = Vec::new();
        for part in rewritten {
            for (p, d) in part.delta {
                *pair_count.entry(p).or_insert(0) += d;
                changed.push(p);
            }
            for (p, mut held) in part.holders {
                where_.entry(p).or_default().append(&mut held);
            }
        }
        pair_count.remove(&pair);
        changed.sort_unstable();
        changed.dedup();
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
    /// The tokens of one chunk: the best pair (lowest rank) merges everywhere, left to right, before any
    /// higher rank does. Short chunks rescan their word, which is cheapest for a few bytes; long ones use a
    /// heap, so the cost grows with length x log(length) instead of its square.
    fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        if chunk.len() < HEAP_FROM { self.encode_by_scans(chunk) } else { self.encode_by_heap(chunk) }
    }

    fn encode_by_scans(&self, chunk: &[u8]) -> Vec<u32> {
        let mut word: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        loop {
            let best = pairs_of(&word).filter_map(|p| self.ranks.get(&p).map(|&r| (r, p))).min();
            let Some((rank, pair)) = best else { break };
            word = merge_word(&word, pair, FIRST_MERGE + rank);
        }
        word
    }

    /// The tokens sit in a doubly linked list over arrays, each named by the place of its first byte. A heap
    /// holds the (rank, place) of every pair that can merge; an entry whose pair has changed since it was
    /// pushed is skipped when popped. A merge's rank is above the ranks of the merges that made its parts, so
    /// the pairs a merge creates rank above it and never come before the entries still waiting at its rank:
    /// popping by (rank, place) merges all occurrences of a rank left to right, as the scans do.
    fn encode_by_heap(&self, chunk: &[u8]) -> Vec<u32> {
        let n = chunk.len();
        let mut token: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        let mut next: Vec<usize> = (1..=n).collect(); // n: no next token
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect(); // usize::MAX: no previous token
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> =
            (0..n.saturating_sub(1)).filter_map(|i| self.ranks.get(&(token[i], token[i + 1])).map(|&r| Reverse((r, i)))).collect();
        while let Some(Reverse((rank, at))) = heap.pop() {
            let right = next[at];
            if token[at] == DEAD || right >= n || self.ranks.get(&(token[at], token[right])) != Some(&rank) {
                continue;
            }
            token[at] = FIRST_MERGE + rank;
            token[right] = DEAD;
            let after = next[right];
            next[at] = after;
            if after < n {
                prev[after] = at;
            }
            if after < n && let Some(&r) = self.ranks.get(&(token[at], token[after])) {
                heap.push(Reverse((r, at)));
            }
            let before = prev[at];
            if before != usize::MAX && let Some(&r) = self.ranks.get(&(token[before], token[at])) {
                heap.push(Reverse((r, before)));
            }
        }
        let mut out = Vec::new();
        let mut at = 0;
        while at < n {
            out.push(token[at]);
            at = next[at];
        }
        out
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
    m.add("SLOTS", SLOTS)?;
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

    fn counted(slots: usize, chunks: &[&str]) -> Vec<(String, u64, u64)> {
        let mut table = Summary::new(slots);
        for chunk in chunks {
            table.count(chunk.as_bytes());
        }
        table.entries().into_iter().map(|(c, n, e)| (String::from_utf8(c.to_vec()).unwrap(), n, e)).collect()
    }

    #[test]
    fn enough_slots_count_every_chunk_exactly() {
        let stream = ["a", "b", "a", "c", "a", "b"];
        let expected = vec![("a".to_string(), 3, 0), ("b".to_string(), 2, 0), ("c".to_string(), 1, 0)];
        assert_eq!(counted(3, &stream), expected);
        assert_eq!(counted(100, &stream), expected);
    }

    #[test]
    fn a_full_table_hands_the_smallest_slot_to_the_new_chunk_with_its_count_as_error() {
        // after a, a, b the table is full; c evicts b (count 1) and takes it with count 2, error 1
        let found = counted(2, &["a", "a", "b", "c"]);
        assert_eq!(found, vec![("a".to_string(), 2, 0), ("c".to_string(), 2, 1)]);
    }

    #[test]
    fn a_tie_between_equal_counts_evicts_the_smaller_chunk_whatever_the_hash_order() {
        // b and c both have count 1: the smaller bytes, b, goes
        assert_eq!(counted(2, &["c", "b", "d"]), vec![("c".to_string(), 1, 0), ("d".to_string(), 2, 1)]);
        assert_eq!(counted(2, &["b", "c", "d"]), vec![("c".to_string(), 1, 0), ("d".to_string(), 2, 1)]);
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

    /// Byte-pair merges by recounting every pair of every word after each merge (nothing shared with `learn_on`).
    fn recounted(entries: &[(&[u8], u64)], n_merges: usize) -> Vec<Pair> {
        let mut words: Vec<Vec<u32>> = entries.iter().map(|(w, _)| w.iter().map(|&b| b as u32).collect()).collect();
        let mut merges = Vec::new();
        while merges.len() < n_merges {
            let mut counts: HashMap<Pair, u64> = HashMap::new();
            for (word, (_, n)) in words.iter().zip(entries) {
                for p in word.windows(2) {
                    *counts.entry((p[0], p[1])).or_insert(0) += n;
                }
            }
            let Some((&pair, &best)) = counts.iter().min_by_key(|&(&p, &n)| (Reverse(n), p)) else { break };
            if best < 2 {
                break;
            }
            let new_id = FIRST_MERGE + merges.len() as u32;
            merges.push(pair);
            words = words.iter().map(|w| merge_word(w, pair, new_id)).collect();
        }
        merges
    }

    /// `count` distinct chunks of `alphabet` letters, up to `longest` long, with counts from 1 to `top`.
    fn random_chunks(seed: u64, count: usize, alphabet: u8, longest: usize, top: u64) -> Vec<(Vec<u8>, u64)> {
        let mut state = seed;
        let mut next = move |n: u64| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) % n
        };
        let mut seen = std::collections::BTreeMap::new();
        while seen.len() < count {
            let word: Vec<u8> = (0..1 + next(longest as u64)).map(|_| b'a' + next(alphabet as u64) as u8).collect();
            seen.insert(word, 1 + next(top));
        }
        seen.into_iter().collect()
    }

    fn borrowed(chunks: &[(Vec<u8>, u64)]) -> Vec<(&[u8], u64)> {
        chunks.iter().map(|(w, n)| (&w[..], *n)).collect()
    }

    #[test]
    fn every_thread_count_learns_the_merges_of_a_recount() {
        let handmade: Vec<(Vec<u8>, u64)> =
            [("aaab", 3), ("aab", 2), ("abab", 5), ("baba", 5), ("ab", 9), ("b", 1)].iter().map(|&(w, n)| (w.as_bytes().to_vec(), n)).collect();
        let ties = random_chunks(1, 3000, 4, 9, 1); // every count is 1: nearly every choice is a tie
        let mixed = random_chunks(2, 3000, 6, 12, 40);
        let repeated: Vec<(Vec<u8>, u64)> = (0..40).map(|k| (b"abcab".repeat(30 + k), 1 + k as u64 % 3)).collect();
        for (name, chunks) in [("handmade", handmade), ("ties", ties), ("mixed", mixed), ("repeated", repeated)] {
            let entries = borrowed(&chunks);
            let expected = recounted(&entries, 200);
            assert!(expected.len() > 3, "{name} learns something");
            for threads in [1, 2, 3, 8] {
                // a slice of 16 words, so that the slices and the threads really are used
                assert_eq!(learn_on(entries.clone(), 200, threads, 16, &AtomicUsize::new(0)), expected, "{name} on {threads} threads");
            }
        }
    }

    #[test]
    fn small_inputs_stay_on_one_slice_and_large_ones_split_in_order() {
        assert_eq!(slices(0, 8, 4096), vec![0..0]);
        assert_eq!(slices(8191, 8, 4096), vec![0..8191]);
        let split = slices(10, 4, 1);
        assert_eq!(split, vec![0..3, 3..6, 6..9, 9..10]);
        assert_eq!(slices(3, 4, 1).into_iter().flatten().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    /// The encoder as a plain loop: find the lowest-rank pair of the whole word, merge all of it, repeat.
    fn reference_encode(ranks: &HashMap<Pair, u32>, chunk: &[u8]) -> Vec<u32> {
        let mut word: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        while let Some((rank, pair)) = pairs_of(&word).filter_map(|p| ranks.get(&p).map(|&r| (r, p))).min() {
            word = merge_word(&word, pair, FIRST_MERGE + rank);
        }
        word
    }

    #[test]
    fn both_encoders_give_the_tokens_of_the_plain_loop() {
        let mut texts: Vec<Vec<u8>> = vec![
            b"aaa".to_vec(),
            b"aaaa".to_vec(),
            b"ab".repeat(100),
            b"a".repeat(HEAP_FROM * 20),
            b"abc".repeat(HEAP_FROM * 10),
            b"aab".repeat(700),
        ];
        for (seed, alphabet, longest) in [(11, 2, 40), (12, 3, 120), (13, 4, 300), (14, 6, 80)] {
            texts.extend(random_chunks(seed, 150, alphabet, longest, 1).into_iter().map(|(w, _)| w));
        }
        // lengths well past HEAP_FROM, and chunks made of repeated pieces
        texts.extend(random_chunks(15, 6, 3, 6000, 1).into_iter().map(|(w, _)| w));
        texts.extend(random_chunks(16, 40, 3, 6, 1).into_iter().map(|(w, _)| w.repeat(HEAP_FROM * 3)));
        for (name, learned_from) in [
            ("small alphabet", random_chunks(21, 1500, 3, 14, 20)),
            ("larger alphabet", random_chunks(22, 1500, 5, 20, 3)),
            ("repeats", (0..30).map(|k| (b"aab".repeat(5 + k), 1 + k as u64 % 4)).collect()),
        ] {
            let merges = recounted(&borrowed(&learned_from), 400);
            assert!(merges.len() > 20, "{name} learns merges");
            let codec = Codec::new(merges).unwrap();
            for text in &texts {
                let expected = reference_encode(&codec.ranks, text);
                assert_eq!(codec.encode_by_scans(text), expected, "{name}: scans on {} bytes", text.len());
                assert_eq!(codec.encode_by_heap(text), expected, "{name}: heap on {} bytes", text.len());
                assert_eq!(codec.encode_chunk(text), expected, "{name}: encode_chunk on {} bytes", text.len());
            }
        }
    }
}
