//! Compacted draws without indirect-count draws (Metal; any GPU without wgpu `MULTI_DRAW_INDIRECT_COUNT`).
//!
//! Without indirect-count draws every bin used to draw its whole capacity (`Bin::count`): every candidate, culled ones as
//! zero-instance draws, in every view. wgpu runs a multi-draw-indirect on Metal as one draw call per entry, so that was
//! tens of thousands of draw calls per view and frame, and the P15-A draw classes / pre-pass (which need compaction) were
//! off. Here the cull compacts as with indirect count (the view's args region is cleared first, so entries past the GPU's
//! count are empty draws), the per-(view, bin) draw counts are copied back to the CPU, and each bin is drawn with an
//! estimate: the largest count of its last read-backs x FH1_SW_DRAW_HEADROOM (1.5) + FH1_SW_DRAW_SLACK (8), at most the
//! capacity.
//!
//! - Views and bins are matched by identity (Bevy's retained view entity + Hi-Z phase; material group, variant, class),
//!   not by index: view slots are handed out per frame and the bin list is rebuilt while streaming.
//! - No history yet (a new view or bin) or a camera cut (the main camera moved more than 25 m or turned more than 25
//!   degrees in a frame) = the full capacity, until the read-backs (a few frames late) catch up.
//! - An under-estimate draws part of a bin for a frame or two (objects missing); the headroom keeps that rare.
//!
//! `FH1_SW_DRAW_ESTIMATE=0` = the old uncompacted draws; `=force` = estimated counts even with indirect-count draws (to
//! try the Metal path on a Windows / Linux PC).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering::Relaxed};
use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::RetainedViewEntity;

use super::{Bin, DrawLists, Variant, MAX_VIEWS};

/// Estimated draws wanted (FH1_SW_DRAW_ESTIMATE, default on); used only where indirect-count draws are missing.
pub(super) fn wanted() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| super::flag_on("FH1_SW_DRAW_ESTIMATE"))
}

/// `FH1_SW_DRAW_ESTIMATE=force`: estimated counts even where indirect-count draws exist.
pub(super) fn forced() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FH1_SW_DRAW_ESTIMATE").is_ok_and(|v| v.eq_ignore_ascii_case("force")))
}

fn headroom() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| super::env_f32("FH1_SW_DRAW_HEADROOM", 1.5).max(1.0))
}

fn slack() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| super::env_f32("FH1_SW_DRAW_SLACK", 8.0).max(0.0) as u32)
}

/// Read-backs kept per (view, bin).
const HISTORY: usize = 6;
/// Read-back buffers in flight.
const STAGING: usize = 3;
/// Frames of full draws after a camera cut (the read-backs land a few frames late).
const CUT_FRAMES: u32 = 5;
const CUT_DIST: f32 = 25.0;
const CUT_COS: f32 = 0.906; // cos 25 degrees

/// A view across frames: Bevy's retained view entity and the Hi-Z phase (0, or 2 for the phase-2 cull).
pub(super) type ViewKey = (RetainedViewEntity, u8);

/// A draw bin across list rebuilds.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BinKey {
    group: u32,
    variant: Variant,
    class: u8,
}

impl BinKey {
    fn of(b: &Bin) -> Self {
        Self { group: b.group, variant: b.variant, class: b.class }
    }
}

/// Slots whose draw counts the cull cleared (= culled) this frame, one bit per view slot.
pub(super) static CULLED: AtomicU32 = AtomicU32::new(0);

#[derive(Default)]
struct Hist {
    vals: [u32; HISTORY],
    at: usize,
    filled: usize,
}

impl Hist {
    fn push(&mut self, v: u32) {
        self.vals[self.at] = v;
        self.at = (self.at + 1) % HISTORY;
        self.filled = (self.filled + 1).min(HISTORY);
    }

    fn max(&self) -> u32 {
        self.vals[..self.filled].iter().copied().max().unwrap_or(0)
    }
}

const FREE: u8 = 0;
const MAPPING: u8 = 1;
const MAPPED: u8 = 2;
const FAILED: u8 = 3;

/// One read-back buffer and what its counts stand for.
struct Staging {
    buffer: Buffer,
    size: u64,
    /// Bytes copied and mapped (at most `size`).
    len: u64,
    state: Arc<AtomicU8>,
    /// (view, first count) of the views culled that frame.
    views: Vec<(ViewKey, u32)>,
    bins: Arc<Vec<BinKey>>,
}

#[derive(Resource, Default)]
pub(super) struct Estimates {
    history: HashMap<ViewKey, HashMap<BinKey, Hist>>,
    /// This frame: per view slot, the view and the draw count per bin (aligned with `DrawLists::bins`).
    slots: Vec<(ViewKey, Vec<u32>)>,
    /// `DrawLists::bins` as keys, for the read-backs (rebuilt with the list generation).
    bins: Arc<Vec<BinKey>>,
    bins_generation: u64,
    staging: Vec<Staging>,
    /// Frames of full draws left (camera cut).
    cut: u32,
    /// Main camera (position, forward) last frame.
    last_main: Option<(Vec3, Vec3)>,
}

impl Estimates {
    /// Start of `prepare_views`: no views yet; a camera cut when the main camera jumped.
    pub(super) fn begin_frame(&mut self, lists: &DrawLists, main: Option<(Vec3, Vec3)>) {
        self.slots.clear();
        CULLED.store(0, Relaxed);
        if self.bins_generation != lists.generation || self.bins.len() != lists.bins.len() {
            self.bins = Arc::new(lists.bins.iter().map(BinKey::of).collect());
            self.bins_generation = lists.generation;
        }
        self.cut = self.cut.saturating_sub(1);
        if let (Some((p, f)), Some((lp, lf))) = (main, self.last_main) {
            if p.distance(lp) > CUT_DIST || f.dot(lf) < CUT_COS {
                self.cut = CUT_FRAMES;
            }
        }
        if main.is_some() {
            self.last_main = main;
        }
    }

    /// `prepare_views`: the view given slot `slot` (slots are handed out in order) and its estimates.
    pub(super) fn add_view(&mut self, slot: u32, key: ViewKey, lists: &DrawLists) {
        debug_assert_eq!(slot as usize, self.slots.len());
        let (headroom, slack) = (headroom(), slack());
        let hist = (self.cut == 0).then(|| self.history.get(&key)).flatten();
        let est = lists
            .bins
            .iter()
            .map(|b| match hist.and_then(|h| h.get(&BinKey::of(b))) {
                Some(h) => ((h.max() as f32 * headroom).ceil() as u32 + slack).min(b.count),
                None => b.count,
            })
            .collect();
        self.slots.push((key, est));
    }

    /// The draw counts of the view in `slot` this frame (None = draw the capacity).
    pub(super) fn slot(&self, slot: u32) -> Option<&[u32]> {
        self.slots.get(slot as usize).map(|(_, e)| e.as_slice())
    }

    /// Read-backs that have landed: into the history.
    fn collect(&mut self) {
        for s in &mut self.staging {
            match s.state.load(Relaxed) {
                MAPPED => {
                    {
                        let data = s.buffer.slice(..s.len).get_mapped_range();
                        let words: Vec<u32> = data.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
                        for &(key, base) in &s.views {
                            let h = self.history.entry(key).or_default();
                            for (b, bin) in s.bins.iter().enumerate() {
                                if let Some(&n) = words.get(base as usize + b) {
                                    h.entry(*bin).or_default().push(n);
                                }
                            }
                        }
                    }
                    s.buffer.unmap();
                    s.state.store(FREE, Relaxed);
                }
                FAILED => s.state.store(FREE, Relaxed),
                _ => {}
            }
        }
    }
}

/// PrepareResources: read-backs that have landed go into the history.
pub(super) fn collect_readbacks(mut est: ResMut<Estimates>) {
    est.collect();
}

/// Cleanup (after the frame's submit): copies this frame's draw counts of the culled views into a free read-back
/// buffer and maps it.
pub(super) fn read_back_counts(mut est: ResMut<Estimates>, lists: Res<DrawLists>, cull: Option<Res<super::CullPipeline>>, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    if !cull.is_some_and(|c| c.estimated) || est.slots.is_empty() {
        return;
    }
    let Some(counts) = lists.counts.as_ref() else { return };
    let culled = CULLED.load(Relaxed);
    let views: Vec<(ViewKey, u32)> =
        est.slots.iter().enumerate().filter(|(i, _)| culled & (1 << i) != 0).map(|(i, (k, _))| (*k, i as u32 * lists.bins_cap)).collect();
    if views.is_empty() {
        return;
    }
    let size = MAX_VIEWS as u64 * lists.bins_cap as u64 * 4;
    if est.staging.len() < STAGING {
        est.staging.push(Staging {
            buffer: device.create_buffer(&BufferDescriptor { label: Some("static world count readback"), size, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false }),
            size,
            len: 0,
            state: Arc::new(AtomicU8::new(FREE)),
            views: Vec::new(),
            bins: Arc::default(),
        });
    }
    let bins = est.bins.clone();
    let Some(s) = est.staging.iter_mut().find(|s| s.state.load(Relaxed) == FREE) else { return };
    if s.size < size {
        s.buffer = device.create_buffer(&BufferDescriptor { label: Some("static world count readback"), size, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        s.size = size;
    }
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("static world count readback") });
    encoder.copy_buffer_to_buffer(counts, 0, &s.buffer, 0, size);
    queue.submit([encoder.finish()]);
    s.views = views;
    s.bins = bins;
    s.len = size;
    s.state.store(MAPPING, Relaxed);
    let state = s.state.clone();
    s.buffer.slice(..size).map_async(MapMode::Read, move |r| state.store(if r.is_ok() { MAPPED } else { FAILED }, Relaxed));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_max() {
        let mut h = Hist::default();
        assert_eq!(h.max(), 0);
        for v in [5, 9, 2] {
            h.push(v);
        }
        assert_eq!(h.max(), 9);
        for v in [1; HISTORY] {
            h.push(v);
        }
        assert_eq!(h.max(), 1, "older read-backs drop out");
    }
}
