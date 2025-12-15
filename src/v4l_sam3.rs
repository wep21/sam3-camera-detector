use anyhow::{Context, Result};
use argh::FromArgs;
use std::io::Write;
use usls::{
    Annotator, Config, Task, Viewer,
    models::{SAM3, Sam3Prompt},
};

#[derive(FromArgs)]
/// SAM3 webcam inference (text prompts via `usls`).
pub struct Args {
    /// task (sam3-image, sam3-tracker)
    #[argh(option, default = "String::from(\"sam3-image\")")]
    task: String,

    /// device (cpu:0, cuda:0, etc.)
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype (q4f16, fp16, fp32, etc.)
    #[argh(option, default = "String::from(\"q4f16\")")]
    dtype: String,

    /// camera index (usually 0)
    #[argh(option, default = "0")]
    camera: usize,

    /// capture width (best-effort; may be overridden by the driver)
    #[argh(option, default = "640")]
    width: u32,

    /// capture height (best-effort; may be overridden by the driver)
    #[argh(option, default = "480")]
    height: u32,

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

#[cfg(not(target_os = "linux"))]
pub fn run() -> Result<()> {
    anyhow::bail!("`v4l_sam3` currently supports only Linux (V4L2).")
}

#[cfg(target_os = "linux")]
pub fn run() -> Result<()> {
    use v4l::io::traits::CaptureStream;
    use v4l::video::Capture;
    use v4l::{Device, FourCC, buffer::Type, prelude::*};

    fn clamp_u8(x: i32) -> u8 {
        x.clamp(0, 255) as u8
    }

    fn yuyv_to_rgb8(width: u32, height: u32, yuyv: &[u8]) -> Result<image::RgbImage> {
        let expected_len = width
            .checked_mul(height)
            .and_then(|px| px.checked_mul(2))
            .context("width*height overflow")? as usize;
        if yuyv.len() < expected_len {
            anyhow::bail!(
                "YUYV buffer too small: got {}, expected {}",
                yuyv.len(),
                expected_len
            );
        }

        let mut rgb = vec![0u8; (width as usize) * (height as usize) * 3];
        let mut di = 0usize;

        for si in (0..expected_len).step_by(4) {
            let y0 = yuyv[si] as i32;
            let u = yuyv[si + 1] as i32;
            let y1 = yuyv[si + 2] as i32;
            let v = yuyv[si + 3] as i32;

            for y in [y0, y1] {
                let c = y - 16;
                let d = u - 128;
                let e = v - 128;

                let r = (298 * c + 409 * e + 128) >> 8;
                let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
                let b = (298 * c + 516 * d + 128) >> 8;

                rgb[di] = clamp_u8(r);
                rgb[di + 1] = clamp_u8(g);
                rgb[di + 2] = clamp_u8(b);
                di += 3;
            }
        }

        image::RgbImage::from_raw(width, height, rgb).context("failed to construct RgbImage")
    }

    fn decode_frame_to_rgb8(
        width: u32,
        height: u32,
        fourcc: FourCC,
        bytes: &[u8],
    ) -> Result<image::RgbImage> {
        if fourcc == FourCC::new(b"YUYV") {
            return yuyv_to_rgb8(width, height, bytes);
        }

        if fourcc == FourCC::new(b"MJPG") || fourcc == FourCC::new(b"JPEG") {
            let img = image::load_from_memory(bytes).context("failed to decode MJPEG frame")?;
            return Ok(img.to_rgb8());
        }

        anyhow::bail!(
            "Unsupported camera pixel format: {:?} (expected YUYV or MJPG)",
            fourcc
        );
    }

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();
    let mut prompts = parse_prompts(&args.prompt)?;

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

    let mut viewer = Viewer::new("sam3-v4l").with_window_scale(args.window_scale);

    let dev = Device::new(args.camera).context("failed to open camera device")?;
    let mut fmt = dev.format().context("failed to read camera format")?;
    fmt.width = args.width;
    fmt.height = args.height;
    fmt.fourcc = FourCC::new(b"YUYV");
    let fmt = dev
        .set_format(&fmt)
        .context("failed to set camera format")?;
    tracing::info!(
        "Camera format: {}x{} {:?}",
        fmt.width,
        fmt.height,
        fmt.fourcc
    );

    let mut stream =
        MmapStream::with_buffers(&dev, Type::VideoCapture, 4).context("failed to start stream")?;

    let save_base = match args.save_dir {
        Some(dir) => std::path::PathBuf::from(dir),
        None => usls::Dir::Current.base_dir_with_subs(&["runs", model.spec()])?,
    };

    tracing::info!("Controls: ESC/Q quit, P update prompt, S save frame");

    let mut last_displayed: Option<usls::Image> = None;
    let mut frame_idx: u64 = 0;
    loop {
        if viewer.is_window_exist_and_closed() {
            break;
        }

        let (data, meta) = stream.next().context("failed to capture frame")?;
        let bytes_used = (meta.bytesused as usize).min(data.len());
        let rgb8 = decode_frame_to_rgb8(fmt.width, fmt.height, fmt.fourcc, &data[..bytes_used])?;
        let img = usls::Image::from(rgb8);

        frame_idx += 1;
        let run_infer = args.infer_every > 0 && (frame_idx % args.infer_every as u64 == 0);
        let display = if run_infer {
            let batch = vec![img.clone()];
            let ys = model.forward(&batch, &prompts)?;

            let mut annotated = annotator.annotate(&img, &ys[0])?;
            for prompt in &prompts {
                annotated = annotator.annotate(&annotated, &prompt.boxes)?;
                annotated = annotator.annotate(&annotated, &prompt.points)?;
            }
            last_displayed = Some(annotated.clone());
            annotated
        } else {
            last_displayed.clone().unwrap_or(img)
        };

        viewer.imshow(&display)?;

        if let Some(key) = viewer.wait_key(1) {
            match key {
                usls::Key::Escape | usls::Key::Q => break,
                usls::Key::S => {
                    if let Some(img) = &last_displayed {
                        let path = save_base.join(format!("{}.jpg", usls::timestamp(None)));
                        img.save(&path)?;
                        tracing::info!("Saved: {}", path.display());
                    }
                }
                usls::Key::P => match prompt_update_loop()? {
                    Some(new_prompts) => {
                        prompts = new_prompts;
                        tracing::info!("Updated prompts: {:?}", prompts);
                    }
                    None => {}
                },
                _ => {}
            }
        }
    }

    usls::perf(false);
    Ok(())
}
