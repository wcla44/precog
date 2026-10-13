//! Voice typing: microphone (cpal) -> local Whisper (whisper.cpp) -> keystrokes into the focused field.
//!
//! Text is typed phrase by phrase (cut on pauses) so it appears while you speak.
//! send() finishes and presses Enter, stop() finishes without Enter, cancel() erases what was typed.

use crate::{now, paths};
use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use enigo::{Direction, Key, Keyboard};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SR: u32 = 16_000;
const PAUSE_S: f64 = 0.7; // silence that ends a phrase
const MAX_PHRASE_S: f64 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Idle,
    Listening,
    Finishing,
}

#[derive(Clone, Debug)]
pub enum Event {
    Status(String),
    Listening,
    Text(String),
    Sent(String),
    Done(String),
    Cancelled,
    Error(String),
}

enum Cmd {
    Configure(String, String),
    Preload,
    Start,
    Send,
    Stop,
    Cancel,
}

pub struct Dictation {
    tx: Sender<Cmd>,
    pub events: Receiver<Event>,
    state: Arc<Mutex<State>>,
    last_voice: Arc<Mutex<f64>>,
}

impl Dictation {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        let (etx, events) = unbounded();
        let state = Arc::new(Mutex::new(State::Idle));
        let last_voice = Arc::new(Mutex::new(now()));
        let mut w = Worker {
            rx,
            events: etx,
            state: state.clone(),
            last_voice: last_voice.clone(),
            model: "base".into(),
            language: String::new(),
            ctx: None,
            stream: None,
            sr: SR,
            buf: Arc::new(Mutex::new(Vec::new())),
            typed: 0,
            text: String::new(),
        };
        std::thread::Builder::new().name("dictation".into()).spawn(move || w.run()).unwrap();
        Self { tx, events, state, last_voice }
    }

    pub fn state(&self) -> State {
        *self.state.lock()
    }
    pub fn configure(&self, model: &str, language: &str) {
        let _ = self.tx.send(Cmd::Configure(model.into(), language.into()));
    }
    pub fn preload(&self) {
        let _ = self.tx.send(Cmd::Preload);
    }
    /// Start listening from another thread (after the target window got focus).
    pub fn starter(&self) -> impl Fn() + Send + 'static {
        let tx = self.tx.clone();
        move || {
            let _ = tx.send(Cmd::Start);
        }
    }
    pub fn send(&self) {
        let _ = self.tx.send(Cmd::Send);
    }
    pub fn stop(&self) {
        let _ = self.tx.send(Cmd::Stop);
    }
    pub fn cancel(&self) {
        let _ = self.tx.send(Cmd::Cancel);
    }
    pub fn silent_for(&self) -> f64 {
        now() - *self.last_voice.lock()
    }
}

struct Worker {
    rx: Receiver<Cmd>,
    events: Sender<Event>,
    state: Arc<Mutex<State>>,
    last_voice: Arc<Mutex<f64>>,
    model: String,
    language: String,
    ctx: Option<(String, WhisperContext)>,
    stream: Option<cpal::Stream>,
    sr: u32,
    buf: Arc<Mutex<Vec<f32>>>, // mono, device sample rate
    typed: usize,
    text: String,
}

fn model_path(size: &str) -> std::path::PathBuf {
    paths::data_dir().join("whisper").join(format!("ggml-{size}.bin"))
}

impl Worker {
    fn emit(&self, e: Event) {
        let _ = self.events.send(e);
    }

    fn set_state(&self, s: State) {
        *self.state.lock() = s;
    }

    fn run(&mut self) {
        loop {
            let cmd = match self.rx.recv_timeout(Duration::from_millis(100)) {
                Ok(c) => Some(c),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return,
            };
            if let Err(e) = self.handle(cmd) {
                self.stream = None;
                self.set_state(State::Idle);
                self.emit(Event::Error(format!("{e:#}").chars().take(160).collect()));
            }
        }
    }

    fn handle(&mut self, cmd: Option<Cmd>) -> Result<()> {
        let state = *self.state.lock();
        match cmd {
            Some(Cmd::Configure(m, l)) => {
                self.model = m;
                self.language = l;
            }
            Some(Cmd::Preload) => {
                self.load()?;
                self.emit(Event::Status(String::new()));
            }
            Some(Cmd::Start) if state == State::Idle => self.start()?,
            Some(Cmd::Cancel) if state != State::Idle => {
                self.close();
                let mut e = crate::actions::enigo()?;
                for _ in 0..self.typed {
                    e.key(Key::Backspace, Direction::Click)?;
                }
                self.set_state(State::Idle);
                self.emit(Event::Cancelled);
            }
            Some(c @ (Cmd::Send | Cmd::Stop)) if state == State::Listening => {
                self.set_state(State::Finishing);
                let audio = self.close();
                if voiced(&audio, self.sr).is_some() {
                    let t = self.transcribe(&audio)?;
                    self.type_text(&t)?;
                }
                let send = matches!(c, Cmd::Send);
                if send {
                    std::thread::sleep(Duration::from_millis(50));
                    crate::actions::enigo()?.key(Key::Return, Direction::Click)?;
                }
                self.set_state(State::Idle);
                self.emit(if send { Event::Sent(self.text.clone()) } else { Event::Done(self.text.clone()) });
            }
            None if state == State::Listening => {
                if let Some(phrase) = self.cut_phrase() {
                    let t = self.transcribe(&phrase)?;
                    self.type_text(&t)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn load(&mut self) -> Result<&WhisperContext> {
        if self.ctx.as_ref().map(|c| &c.0) != Some(&self.model) {
            let path = model_path(&self.model);
            if !path.exists() {
                self.emit(Event::Status(format!("downloading speech model '{}' (first time only)…", self.model)));
                download(&self.model, &path)?;
            }
            self.emit(Event::Status(format!("loading speech model '{}'…", self.model)));
            let ctx = WhisperContext::new_with_params(&path, WhisperContextParameters::default())
                .map_err(|e| anyhow!("whisper: {e:?}"))?;
            self.ctx = Some((self.model.clone(), ctx));
        }
        Ok(&self.ctx.as_ref().unwrap().1)
    }

    fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let dev = host.default_input_device().context("no microphone")?;
        let cfg = dev.default_input_config()?;
        self.sr = cfg.sample_rate();
        let ch = cfg.channels() as usize;
        self.buf.lock().clear();
        self.typed = 0;
        self.text.clear();
        *self.last_voice.lock() = now();
        let buf = self.buf.clone();
        let err = |e| eprintln!("microphone error: {e}");
        let stream = match cfg.sample_format() {
            cpal::SampleFormat::F32 => dev.build_input_stream::<f32, _, _>(
                cfg.config(),
                move |d: &[f32], _| buf.lock().extend(d.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32)),
                err,
                None,
            )?,
            cpal::SampleFormat::I16 => dev.build_input_stream::<i16, _, _>(
                cfg.config(),
                move |d: &[i16], _| {
                    buf.lock().extend(d.chunks(ch).map(|f| f.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / ch as f32))
                },
                err,
                None,
            )?,
            f => return Err(anyhow!("unsupported microphone sample format {f:?}")),
        };
        stream.play()?;
        self.stream = Some(stream);
        self.set_state(State::Listening);
        self.emit(Event::Listening);
        Ok(())
    }

    fn close(&mut self) -> Vec<f32> {
        self.stream = None; // dropping the stream stops it
        std::mem::take(&mut *self.buf.lock())
    }

    /// A finished phrase (speech followed by a pause), taken out of the buffer.
    fn cut_phrase(&mut self) -> Option<Vec<f32>> {
        let audio = self.buf.lock().clone();
        let frame = (self.sr as f64 * 0.03) as usize;
        let n = audio.len() / frame;
        if n < 10 {
            return None;
        }
        let Some((last_voiced, _)) = voiced(&audio, self.sr) else {
            if audio.len() as f64 > self.sr as f64 * 3.0 {
                let keep = audio.len() - self.sr as usize / 2; // only silence: keep the buffer short
                self.buf.lock().drain(..keep);
            }
            return None;
        };
        if last_voiced + 3 >= n {
            *self.last_voice.lock() = now();
        }
        let trailing = (n - 1 - last_voiced) as f64 * 0.03;
        if trailing < PAUSE_S && (audio.len() as f64) < self.sr as f64 * MAX_PHRASE_S {
            return None;
        }
        self.buf.lock().drain(..audio.len()); // keep what arrived since
        Some(audio)
    }

    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let audio = resample(audio, self.sr, SR);
        if audio.len() < (SR as f64 * 0.3) as usize {
            return Ok(String::new());
        }
        let lang = if self.language.is_empty() { "auto".to_string() } else { self.language.clone() };
        let prompt: String = self.text.chars().rev().take(200).collect::<Vec<_>>().into_iter().rev().collect();
        let ctx = self.load()?;
        let mut st = ctx.create_state().map_err(|e| anyhow!("whisper: {e:?}"))?;
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_language(Some(&lang));
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        p.set_no_context(true);
        p.set_suppress_blank(true);
        if !prompt.is_empty() {
            p.set_initial_prompt(&prompt);
        }
        st.full(p, &audio).map_err(|e| anyhow!("whisper: {e:?}"))?;
        let mut out = String::new();
        for i in 0..st.full_n_segments() {
            if let Some(seg) = st.get_segment(i) {
                out.push_str(&seg.to_str_lossy().map_err(|e| anyhow!("whisper: {e:?}"))?);
            }
        }
        Ok(clean(&out))
    }

    fn type_text(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let t = if self.typed > 0 { format!(" {text}") } else { text.to_string() };
        crate::actions::enigo()?.text(&t)?;
        self.typed += t.chars().count();
        self.text.push_str(&t);
        self.emit(Event::Text(self.text.clone()));
        Ok(())
    }
}

/// Drop whisper's non-speech annotations ("[BLANK_AUDIO]", "(music)") and tidy spaces.
fn clean(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Energy VAD on 30 ms frames -> (index of last voiced frame, number of voiced frames).
fn voiced(audio: &[f32], sr: u32) -> Option<(usize, usize)> {
    let frame = (sr as f64 * 0.03) as usize;
    let rms: Vec<f32> = audio.chunks_exact(frame).map(|f| (f.iter().map(|x| x * x).sum::<f32>() / frame as f32).sqrt()).collect();
    if rms.is_empty() {
        return None;
    }
    let mut sorted = rms.clone();
    sorted.sort_by(f32::total_cmp);
    let thresh = (sorted[sorted.len() / 5] * 3.0).max(0.01);
    let idx: Vec<usize> = rms.iter().enumerate().filter(|(_, &r)| r > thresh).map(|(i, _)| i).collect();
    // ignore isolated clicks: need ~150 ms of voice
    (idx.len() >= 5).then(|| (*idx.last().unwrap(), idx.len()))
}

fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as f64 * to as f64 / from as f64) as usize;
    (0..n)
        .map(|i| {
            let p = i as f64 * from as f64 / to as f64;
            let (j, f) = (p as usize, (p.fract()) as f32);
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

fn download(size: &str, path: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{size}.bin");
    let mut resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(1800))
        .build()?
        .get(&url)
        .send()?
        .error_for_status()
        .with_context(|| format!("downloading {url}"))?;
    let tmp = path.with_extension("part");
    let mut f = std::fs::File::create(&tmp)?;
    std::io::copy(&mut resp, &mut f)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}
