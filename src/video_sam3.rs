use anyhow::{Context, Result};
use argh::FromArgs;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use usls::{
    Annotator, Config, Task, Viewer,
    models::{SAM3, Sam3Prompt},
};

#[derive(FromArgs)]
/// SAM3 video-file inference (text prompts via `usls`).
pub struct Args {
    /// input video path (mp4, mov, etc.; decoded via `ffmpeg`)
    #[argh(positional)]
    input: String,

    /// task (sam3-image, sam3-tracker)
    #[argh(option, default = "String::from(\"sam3-image\")")]
    task: String,

    /// device (cpu:0, cuda:0, etc.)
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype (q4f16, fp16, fp32, etc.)
    #[argh(option, default = "String::from(\"q4f16\")")]
    dtype: String,

    /// scale output width (requires --height too)
    #[argh(option)]
    width: Option<u32>,

    /// scale output height (requires --width too)
    #[argh(option)]
    height: Option<u32>,

    /// override playback FPS (default: probed from input, fallback 30)
    #[argh(option)]
    fps: Option<f32>,

    /// prompts (repeatable): `-p shoe` or `-p \"pos:480,290,110,360\"`
    #[argh(option, short = 'p')]
    prompt: Vec<String>,

    /// confidence threshold (default: 0.5)
    #[argh(option, default = "0.5")]
    conf: f32,

    /// show mask
    #[argh(option, default = "false")]
    show_mask: bool,

    /// run inference every N frames (set 0 to disable)
    #[argh(option, default = "3")]
    infer_every: u32,

    /// window scale (1.0 = native resolution)
    #[argh(option, default = "1.0")]
    window_scale: f32,

    /// tensorrt: enable FP16 in EP
    #[argh(option, default = "true")]
    trt_fp16: bool,

    /// tensorrt: enable engine cache
    #[argh(option, default = "true")]
    trt_engine_cache: bool,

    /// tensorrt: enable timing cache
    #[argh(option, default = "true")]
    trt_timing_cache: bool,

    /// save directory (default: ./runs/<model-spec>/)
    #[argh(option)]
    save_dir: Option<String>,

    /// save annotated video to path (disables display window)
    #[argh(option)]
    save_video: Option<String>,
}

fn parse_prompts(raw: &[String]) -> Result<Vec<Sam3Prompt>> {
    if raw.is_empty() {
        anyhow::bail!("No prompt. Use -p \"text\" or -p \"visual;pos:x,y,w,h\"");
    }
    raw.iter()
        .map(|s| s.parse())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("{}", e))
}

fn prompt_update_loop() -> Result<Option<Vec<Sam3Prompt>>> {
    eprint!("New prompt(s) (split with `|`, empty keeps current): ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .context("failed to read prompt from stdin")?;
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let parts: Vec<String> = line
        .split('|')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Ok(Some(parse_prompts(&parts)?))
}

#[derive(Clone, Copy, Debug)]
struct VideoInfo {
    width: u32,
    height: u32,
    fps: f32,
}

fn ffprobe_single_value(args: &[&str], input: &str) -> Result<Option<String>> {
    let output = Command::new("ffprobe")
        .args(["-v", "error"])
        .args(args)
        .args(["-of", "default=noprint_wrappers=1:nokey=1"])
        .arg(input)
        .output()
        .with_context(|| "failed to run `ffprobe` (is FFmpeg installed?)")?;

    if !output.status.success() {
        return Ok(None);
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.lines().map(str::trim).find(|l| !l.is_empty()).map(|s| s.to_string()))
}

fn ffprobe_duration_seconds(input: &str) -> Result<Option<f64>> {
    let Some(v) = ffprobe_single_value(&["-show_entries", "format=duration"], input)? else {
        return Ok(None);
    };
    let v = v.trim();
    if v.is_empty() || v == "N/A" {
        return Ok(None);
    }
    Ok(v.parse::<f64>().ok().filter(|d| d.is_finite() && *d > 0.0))
}

fn ffprobe_nb_frames(input: &str) -> Result<Option<u64>> {
    let Some(v) = ffprobe_single_value(&["-select_streams", "v:0", "-show_entries", "stream=nb_frames"], input)?
    else {
        return Ok(None);
    };
    let v = v.trim();
    if v.is_empty() || v == "N/A" {
        return Ok(None);
    }
    Ok(v.parse::<u64>().ok().filter(|n| *n > 0))
}

fn parse_rate(s: &str) -> Option<f32> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some((num, den)) = s.split_once('/') {
        let num: f32 = num.trim().parse().ok()?;
        let den: f32 = den.trim().parse().ok()?;
        if den == 0.0 {
            return None;
        }
        return Some(num / den);
    }
    s.parse().ok()
}

fn ffprobe_video_info(input: &str) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input,
        ])
        .output()
        .with_context(|| "failed to run `ffprobe` (is FFmpeg installed?)")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffprobe failed: {}", stderr.trim());
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let width: u32 = lines
        .next()
        .context("ffprobe output missing width")?
        .parse()
        .context("failed to parse width from ffprobe")?;
    let height: u32 = lines
        .next()
        .context("ffprobe output missing height")?
        .parse()
        .context("failed to parse height from ffprobe")?;
    let fps = lines
        .next()
        .and_then(parse_rate)
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(30.0);

    Ok(VideoInfo { width, height, fps })
}

fn fmt_hms(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

struct Progress {
    enabled: bool,
    tty: bool,
    total_frames: Option<u64>,
    fps: f32,
    started: Instant,
    last_update: Instant,
}

impl Progress {
    fn new(enabled: bool, fps: f32, total_frames: Option<u64>) -> Self {
        Self {
            enabled,
            tty: std::io::stderr().is_terminal(),
            total_frames,
            fps,
            started: Instant::now(),
            last_update: Instant::now(),
        }
    }

    fn maybe_update(&mut self, frame_idx: u64) {
        if !self.enabled {
            return;
        }

        let now = Instant::now();
        if frame_idx != 1 && now.duration_since(self.last_update) < Duration::from_millis(500) {
            return;
        }
        self.last_update = now;

        let elapsed_s = now.duration_since(self.started).as_secs_f64();
        let speed_fps = if elapsed_s > 0.0 {
            frame_idx as f64 / elapsed_s
        } else {
            0.0
        };
        let pos_s = frame_idx as f64 / (self.fps.max(0.001) as f64);

        let (pct, eta_s) = match (self.total_frames, speed_fps > 0.0) {
            (Some(total), true) if total > 0 => {
                let remaining = total.saturating_sub(frame_idx) as f64;
                (Some((frame_idx as f64 / total as f64) * 100.0), Some(remaining / speed_fps))
            }
            _ => (None, None),
        };

        if self.tty {
            match (self.total_frames, pct, eta_s) {
                (Some(total), Some(p), Some(eta)) => {
                    eprint!(
                        "\rframe {frame_idx}/{total} ({p:5.1}%) pos {} elapsed {} speed {:5.1} fps ETA {}",
                        fmt_hms(pos_s),
                        fmt_hms(elapsed_s),
                        speed_fps,
                        fmt_hms(eta)
                    );
                }
                (Some(total), _, _) => {
                    eprint!(
                        "\rframe {frame_idx}/{total} pos {} elapsed {} speed {:5.1} fps",
                        fmt_hms(pos_s),
                        fmt_hms(elapsed_s),
                        speed_fps
                    );
                }
                _ => {
                    eprint!(
                        "\rframe {frame_idx} pos {} elapsed {} speed {:5.1} fps",
                        fmt_hms(pos_s),
                        fmt_hms(elapsed_s),
                        speed_fps
                    );
                }
            }
            let _ = std::io::stderr().flush();
        } else {
            match (self.total_frames, pct, eta_s) {
                (Some(total), Some(p), Some(eta)) => {
                    tracing::info!(
                        "Progress: frame {}/{} ({:.1}%) pos {} elapsed {} speed {:.1} fps ETA {}",
                        frame_idx,
                        total,
                        p,
                        fmt_hms(pos_s),
                        fmt_hms(elapsed_s),
                        speed_fps,
                        fmt_hms(eta)
                    );
                }
                (Some(total), _, _) => {
                    tracing::info!(
                        "Progress: frame {}/{} pos {} elapsed {} speed {:.1} fps",
                        frame_idx,
                        total,
                        fmt_hms(pos_s),
                        fmt_hms(elapsed_s),
                        speed_fps
                    );
                }
                _ => {
                    tracing::info!(
                        "Progress: frame {} pos {} elapsed {} speed {:.1} fps",
                        frame_idx,
                        fmt_hms(pos_s),
                        fmt_hms(elapsed_s),
                        speed_fps
                    );
                }
            }
        }
    }

    fn finish(&mut self, frame_idx: u64) {
        if !self.enabled {
            return;
        }
        if self.tty {
            self.last_update = Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or(self.last_update);
            self.maybe_update(frame_idx);
            eprintln!();
        }
    }
}

struct FfmpegRawRgb24 {
    child: Child,
    width: u32,
    height: u32,
}

impl FfmpegRawRgb24 {
    fn spawn(input: &str, width: u32, height: u32, scale: bool) -> Result<Self> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error"]);
        cmd.args(["-i", input]);
        cmd.args(["-map", "0:v:0", "-an", "-sn", "-dn"]);

        if scale {
            cmd.args(["-vf", &format!("scale={width}:{height}")]);
        }

        cmd.args(["-vsync", "0"]);
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);

        let child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| "failed to run `ffmpeg` (is FFmpeg installed?)")?;

        Ok(Self {
            child,
            width,
            height,
        })
    }

    fn frame_size(&self) -> Result<usize> {
        let size = self
            .width
            .checked_mul(self.height)
            .and_then(|px| px.checked_mul(3))
            .context("width*height overflow")?;
        Ok(size as usize)
    }

    fn read_frame(&mut self) -> Result<Option<image::RgbImage>> {
        let frame_size = self.frame_size()?;
        let Some(stdout) = self.child.stdout.as_mut() else {
            anyhow::bail!("ffmpeg stdout missing");
        };

        let mut buf = vec![0u8; frame_size];
        match stdout.read_exact(&mut buf) {
            Ok(()) => {
                let img = image::RgbImage::from_raw(self.width, self.height, buf)
                    .context("failed to construct RgbImage")?;
                Ok(Some(img))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e).context("failed to read frame bytes from ffmpeg"),
        }
    }

    fn finish(mut self) -> Result<()> {
        let status = self
            .child
            .wait()
            .context("failed to wait for ffmpeg")?;
        if status.success() {
            return Ok(());
        }
        let mut err = String::new();
        if let Some(mut stderr) = self.child.stderr.take() {
            stderr.read_to_string(&mut err).ok();
        }
        anyhow::bail!("ffmpeg exited with {status}: {}", err.trim());
    }
}

impl Drop for FfmpegRawRgb24 {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

struct FfmpegVideoWriter {
    child: Child,
}

impl FfmpegVideoWriter {
    fn spawn(output: &Path, width: u32, height: u32, fps: f32) -> Result<Self> {
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create output directory: {}", parent.display()))?;
            }
        }

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24"]);
        cmd.args(["-video_size", &format!("{width}x{height}")]);
        cmd.args(["-framerate", &format!("{fps:.3}")]);
        cmd.args(["-i", "-"]);
        cmd.args(["-an", "-sn", "-dn"]);
        cmd.args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]);
        cmd.args(["-pix_fmt", "yuv420p"]);
        cmd.arg(output);

        let child = cmd
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| "failed to run `ffmpeg` for encoding (is FFmpeg installed?)")?;

        Ok(Self { child })
    }

    fn write_frame(&mut self, img: &usls::Image) -> Result<()> {
        let Some(stdin) = self.child.stdin.as_mut() else {
            anyhow::bail!("ffmpeg stdin missing");
        };
        stdin
            .write_all(img.as_raw())
            .context("failed to write frame bytes to ffmpeg")?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        drop(self.child.stdin.take());
        let status = self
            .child
            .wait()
            .context("failed to wait for ffmpeg (encoder)")?;
        if status.success() {
            return Ok(());
        }
        let mut err = String::new();
        if let Some(mut stderr) = self.child.stderr.take() {
            stderr.read_to_string(&mut err).ok();
        }
        anyhow::bail!(
            "ffmpeg (encoder) exited with {status}: {}",
            err.trim()
        );
    }
}

impl Drop for FfmpegVideoWriter {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

pub fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();
    let mut prompts = parse_prompts(&args.prompt)?;

    let probed = ffprobe_video_info(&args.input)?;
    let (out_w, out_h, scale) = match (args.width, args.height) {
        (None, None) => (probed.width, probed.height, false),
        (Some(w), Some(h)) => (w, h, true),
        _ => anyhow::bail!("Specify both --width and --height (or neither)."),
    };
    let fps = args.fps.unwrap_or(probed.fps).max(0.1);
    let delay_ms: u64 = ((1000.0 / fps).round() as u64).clamp(1, 1000);

    tracing::info!(
        "Video: {} ({}x{}, {:.3} fps)",
        args.input,
        out_w,
        out_h,
        fps
    );

    let nb_frames = ffprobe_nb_frames(&args.input)?;
    let duration_s = ffprobe_duration_seconds(&args.input)?;
    let total_frames = nb_frames.or_else(|| duration_s.map(|d| (d * fps as f64).round() as u64).filter(|n| *n > 0));
    if let Some(total) = total_frames {
        tracing::info!("Frames: ~{total}");
    }

    let config = match args.task.parse()? {
        Task::Sam3Image => Config::sam3_image(),
        Task::Sam3Tracker => Config::sam3_tracker(),
        _ => anyhow::bail!(
            "Sam3 Task now only support: {}, {}",
            Task::Sam3Image,
            Task::Sam3Tracker
        ),
    }
    .with_tensorrt_fp16_all(args.trt_fp16)
    .with_tensorrt_engine_cache_all(args.trt_engine_cache)
    .with_tensorrt_timing_cache_all(args.trt_timing_cache)
    .with_dtype_all(args.dtype.parse()?)
    .with_class_confs(&[args.conf])
    .with_device_all(args.device.parse()?)
    .commit()?;

    let mut model = SAM3::new(config)?;
    let annotator = Annotator::default()
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(args.show_mask)
                .with_cutout(true)
                .with_draw_polygon_largest(true),
        )
        .with_polygon_style(usls::PolygonStyle::default().with_thickness(2));

    let save_video_path: Option<PathBuf> = args.save_video.as_deref().map(PathBuf::from);
    let mut viewer = save_video_path
        .is_none()
        .then(|| Viewer::new("sam3-video").with_window_scale(args.window_scale));

    let save_base = match args.save_dir {
        Some(dir) => std::path::PathBuf::from(dir),
        None => usls::Dir::Current.base_dir_with_subs(&["runs", model.spec()])?,
    };

    if let Some(path) = &save_video_path {
        tracing::info!("Writing annotated video to: {}", path.display());
    } else {
        tracing::info!("Controls: ESC/Q quit, P update prompt, S save frame");
    }

    let mut decoder = FfmpegRawRgb24::spawn(&args.input, out_w, out_h, scale)?;
    let mut encoder = match &save_video_path {
        Some(path) => Some(FfmpegVideoWriter::spawn(path, out_w, out_h, fps)?),
        None => None,
    };

    let mut last_displayed: Option<usls::Image> = None;
    let mut frame_idx: u64 = 0;
    let mut stopped_early = false;
    let mut progress = Progress::new(save_video_path.is_some(), fps, total_frames);
    loop {
        let Some(rgb8) = decoder.read_frame()? else {
            break;
        };
        frame_idx += 1;
        progress.maybe_update(frame_idx);
        let img = usls::Image::from(rgb8);

        let run_infer = args.infer_every > 0 && frame_idx.is_multiple_of(args.infer_every as u64);
        if run_infer {
            let batch = vec![img.clone()];
            let ys = model.forward(&batch, &prompts)?;

            let mut annotated = annotator.annotate(&img, &ys[0])?;
            for prompt in &prompts {
                annotated = annotator.annotate(&annotated, &prompt.boxes)?;
                annotated = annotator.annotate(&annotated, &prompt.points)?;
            }
            last_displayed = Some(annotated);
        }

        let display = match &last_displayed {
            Some(img) => img,
            None => &img,
        };

        if let Some(encoder) = encoder.as_mut() {
            encoder.write_frame(display)?;
        }

        if let Some(viewer) = viewer.as_mut() {
            if viewer.is_window_exist_and_closed() {
                stopped_early = true;
                break;
            }

            viewer.imshow(display)?;
            if let Some(key) = viewer.wait_key(delay_ms) {
                match key {
                    usls::Key::Escape | usls::Key::Q => {
                        stopped_early = true;
                        break;
                    }
                    usls::Key::S => {
                        if let Some(img) = &last_displayed {
                            let path = save_base.join(format!("{}.jpg", usls::timestamp(None)));
                            img.save(&path)?;
                            tracing::info!("Saved: {}", path.display());
                        }
                    }
                    usls::Key::P => {
                        if let Some(new_prompts) = prompt_update_loop()? {
                            prompts = new_prompts;
                            tracing::info!("Updated prompts: {:?}", prompts);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(encoder) = encoder {
        encoder.finish()?;
    }

    progress.finish(frame_idx);

    if stopped_early {
        drop(decoder);
    } else {
        decoder.finish()?;
    }
    usls::perf(false);
    Ok(())
}
