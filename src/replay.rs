//! Offline evaluation of gaze model variants on a recorded calibration (`--replay FILE`): the dots train, the
//! pursuit (never seen in training) tests. Reports the error overall, near the screen edges and in the middle, and
//! the "reach" per axis: the slope of the predicted vs the true position (1 = reaches the edges, < 1 = shrunk towards
//! the middle).

use crate::gaze::{FitOpts, GazeModel, NF};
use crate::monitors::{self, Monitor};
use anyhow::{bail, Context, Result};
use std::path::Path;

struct Sample {
    f: [f64; NF],
    target: [f64; 2],
    group: i64,
    weight: f64,
}

fn load(path: &Path) -> Result<(Vec<Sample>, Vec<Sample>, Vec<Vec<Sample>>)> {
    let d: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let points = d["points"].as_array().context("no points")?;
    // the camera focal length the calibration settled on: dot features are recomputed with it
    if let Some(f) = points.iter().find_map(|p| p["focal"].as_f64()) {
        crate::unigaze::set_focal(f);
        println!("camera focal from the recording: {f:.0} px at 1280 px width");
    }
    let (mut dots, mut pursuit, mut validation) = (Vec::new(), Vec::new(), Vec::new());
    for (gi, p) in points.iter().enumerate() {
        if let Some(t) = p["target"].as_array() {
            let target = [t[0].as_f64().unwrap_or(0.0), t[1].as_f64().unwrap_or(0.0)];
            // recorded features when present (they hold the eye patches), else recomputed from the landmarks
            let feats: Vec<Vec<f64>> = p["feats"].as_array().into_iter().flatten().map(|f| f.as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect()).filter(|v: &Vec<f64>| v.len() == NF).collect();
            if !feats.is_empty() {
                // recorded at the click, possibly with an earlier focal length: landmark features recomputed from the
                // raw landmarks (with the recording's focal), eye patches kept
                let raws: Vec<Vec<f32>> = p["frames"].as_array().into_iter().flatten().map(|fr| fr.as_array().into_iter().flatten().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()).collect();
                for (k, v) in feats.iter().enumerate() {
                    let mut f = [0.0; NF];
                    f.copy_from_slice(v);
                    if let Some(raw) = raws.get(k).filter(|r| r.len() >= 478 * 3 + 2) {
                        let (w, h) = (raw[raw.len() - 2] as usize, raw[raw.len() - 1] as usize);
                        let mesh: Vec<[f32; 3]> = raw[..478 * 3].chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                        let land = crate::tracker::features(&mesh, w, h);
                        f[..crate::gaze::LAND].copy_from_slice(&land[..crate::gaze::LAND]);
                    }
                    dots.push(Sample { f, target, group: gi as i64, weight: 1.0 });
                }
                continue;
            }
            for fr in p["frames"].as_array().into_iter().flatten() {
                let v: Vec<f32> = fr.as_array().into_iter().flatten().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
                if v.len() < 478 * 3 + 2 {
                    continue;
                }
                let (w, h) = (v[v.len() - 2] as usize, v[v.len() - 1] as usize);
                let mesh: Vec<[f32; 3]> = v[..478 * 3].chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                dots.push(Sample { f: crate::tracker::features(&mesh, w, h), target, group: gi as i64, weight: 1.0 });
            }
        } else if let Some(val) = p["validation"].as_array() {
            // validation dots (never learned from in the calibration): the most honest test set
            for d in val {
                let t = d["target"].as_array().map(|t| [t[0].as_f64().unwrap_or(0.0), t[1].as_f64().unwrap_or(0.0)]).unwrap_or([0.0; 2]);
                let frames = d["frames"].as_array().into_iter().flatten().filter_map(|fr| {
                    let v: Vec<f64> = fr.as_array()?.iter().filter_map(|x| x.as_f64()).collect();
                    let mut f = [0.0; NF];
                    (v.len() == NF).then(|| {
                        f.copy_from_slice(&v);
                        Sample { f, target: t, group: 0, weight: 0.0 }
                    })
                });
                validation.push(frames.collect());
            }
        } else if let Some(moves) = p["moves"].as_array() {
            // phase 1 pointer paths (already aligned and filtered): training samples, like in the calibration
            for mv in moves {
                let v: Vec<f64> = mv.as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect();
                if v.len() < 4 + NF {
                    continue;
                }
                let mut f = [0.0; NF];
                f.copy_from_slice(&v[4..4 + NF]);
                dots.push(Sample { f, target: [v[1], v[2]], group: 200 + v[3] as i64, weight: 0.5 });
            }
        } else if p["pursuit"].is_number() {
            let lag = p["lag"].as_f64().unwrap_or(0.0);
            let path: Vec<[f64; 2]> =
                p["path"].as_array().into_iter().flatten().filter_map(|q| Some([q[0].as_f64()?, q[1].as_f64()?])).collect();
            for fr in p["frames"].as_array().into_iter().flatten() {
                let v: Vec<f64> = fr.as_array().into_iter().flatten().filter_map(|x| x.as_f64()).collect();
                if v.len() < 1 + NF || path.is_empty() {
                    continue;
                }
                let k = (((v[0] - lag) * 30.0).round().max(0.0) as usize).min(path.len() - 1);
                let mut f = [0.0; NF];
                f.copy_from_slice(&v[1..1 + NF]);
                pursuit.push(Sample { f, target: path[k], group: 1000 + gi as i64 * 50 + v[0] as i64, weight: 0.5 });
            }
        }
    }
    Ok((dots, pursuit, validation))
}

/// Distance to the nearest edge of its screen, as a fraction of that screen's smaller side.
fn edge_frac(mons: &[Monitor], x: f64, y: f64) -> f64 {
    let m = monitors::at(mons, x, y);
    let d = (x - m.left as f64).min(m.right() as f64 - x).min(y - m.top as f64).min(m.bottom() as f64 - y);
    d / (m.width.min(m.height) as f64)
}

fn slope(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let (mt, mp) = (pairs.iter().map(|p| p.0).sum::<f64>() / n, pairs.iter().map(|p| p.1).sum::<f64>() / n);
    let cov: f64 = pairs.iter().map(|p| (p.0 - mt) * (p.1 - mp)).sum();
    let var: f64 = pairs.iter().map(|p| (p.0 - mt).powi(2)).sum();
    cov / var.max(1e-9)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

pub fn run(path: &Path) -> Result<()> {
    let mons = monitors::list();
    let (dots, pursuit, validation) = load(path)?;
    if dots.len() < 30 || pursuit.is_empty() {
        bail!("need dots and a pursuit ({} dot frames, {} pursuit frames)", dots.len(), pursuit.len());
    }
    println!("{} dot frames ({} dots), {} pursuit frames", dots.len(), dots.iter().map(|s| s.group).collect::<std::collections::HashSet<_>>().len(), pursuit.len());
    let o = |stretch_angles, bounded, eye_patches| Some(FitOpts { stretch_angles, bounded, eye_patches });
    let variants: [(&str, Option<FitOpts>); 5] = [
        ("pixel regression (upstream)", None),
        ("geometric", o(false, false, false)),
        ("geometric, bounded", o(false, true, false)),
        ("geometric, bounded + eye patches", o(false, true, true)),
        ("bounded + patches + stretch", o(true, true, true)),
    ];
    println!("({} validation dots: \"val\" column = median over dots, trained on dots + pointer paths + pursuits)", validation.len());
    println!("{:<30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}   geometry", "variant", "pursuit", "edges", "middle", "reach x", "reach y", "val");
    for (name, opts) in variants {
        let mut m = GazeModel::landmarks();
        m.layout = mons.iter().map(|m| m.layout()).collect();
        for s in &dots {
            m.add(&[s.f], s.target, s.group, s.weight);
        }
        let mut what = String::new();
        if let Some(o) = opts {
            m.opts = o;
            what = m.fit_geometry(&mons)?.1;
        }
        m.fit()?;
        let (mut all, mut edges, mut middle, mut px, mut py) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for s in &pursuit {
            let Some((x, y)) = m.predict(&s.f) else { continue };
            let (x, y) = monitors::clamp(&mons, x, y);
            let e = (x - s.target[0]).hypot(y - s.target[1]);
            all.push(e);
            if edge_frac(&mons, s.target[0], s.target[1]) < 0.15 { edges.push(e) } else { middle.push(e) }
            px.push((s.target[0], x));
            py.push((s.target[1], y));
        }
        // validation: a model trained on everything (dots, paths, pursuits), as at the end of the calibration
        let mut full = GazeModel::landmarks();
        full.layout = m.layout.clone();
        for s in dots.iter().chain(&pursuit) {
            full.add(&[s.f], s.target, s.group, s.weight);
        }
        let val = if validation.is_empty() {
            f64::NAN
        } else {
            if let Some(o) = opts {
                full.opts = o;
                full.fit_geometry(&mons)?;
            }
            full.fit()?;
            median(
                validation
                    .iter()
                    .filter_map(|d| {
                        let e: Vec<f64> = d
                            .iter()
                            .filter_map(|s| full.predict(&s.f))
                            .map(|(x, y)| monitors::clamp(&mons, x, y))
                            .map(|(x, y)| (x - d[0].target[0]).hypot(y - d[0].target[1]))
                            .collect();
                        (!e.is_empty()).then(|| median(e))
                    })
                    .collect(),
            )
        };
        if std::env::var_os("PRECOG_REPLAY_DOTS").is_some() && name == "geometric, bounded" {
            let eye = |v: &[&Sample]| [17, 18, 19].map(|i| median(v.iter().map(|s| s.f[i]).collect()));
            let train: Vec<&Sample> = dots.iter().chain(&pursuit).collect();
            println!("  training eye position (mm, camera frame): {:?}", eye(&train).map(|v| v.round()));
            for d in &validation {
                let p: Vec<(f64, f64)> = d.iter().filter_map(|s| full.predict(&s.f)).map(|(x, y)| monitors::clamp(&mons, x, y)).collect();
                if p.is_empty() {
                    continue;
                }
                let (mx, my) = (median(p.iter().map(|q| q.0).collect()), median(p.iter().map(|q| q.1).collect()));
                let all: Vec<&Sample> = d.iter().collect();
                println!(
                    "  dot ({:5.0},{:5.0}) -> predicted ({mx:5.0},{my:5.0})  offset ({:+5.0},{:+5.0})  eye {:?}",
                    d[0].target[0],
                    d[0].target[1],
                    mx - d[0].target[0],
                    my - d[0].target[1],
                    eye(&all).map(|v| v.round())
                );
            }
        }
        println!(
            "{name:<30} {:>8.0} {:>8.0} {:>8.0} {:>8.2} {:>8.2} {:>8.0}   {what}",
            median(all),
            median(edges),
            median(middle),
            slope(&px),
            slope(&py),
            val
        );
    }
    Ok(())
}
