use anyhow::Result;

#[cfg(not(all(target_os = "linux", feature = "hikvision")))]
pub fn run() -> Result<()> {
    anyhow::bail!(
        "`hikvision_sam3` requires Linux and `--features hikvision` (and /opt/MVS installed)."
    )
}

#[cfg(all(target_os = "linux", feature = "hikvision"))]
pub fn run() -> Result<()> {
    use anyhow::{Context, Result};
    use argh::FromArgs;
    use std::ffi::{CStr, CString};
    use std::io::Write;
    use std::ptr;
    use usls::{
        Annotator, Config, Task, Viewer,
        models::{SAM3, Sam3Prompt},
    };

    use hikvision_mvs_sys as mvs;

    const PIXEL_TYPE_RGB8_PACKED: u64 = 0x02180014;

    #[derive(FromArgs)]
    /// SAM3 inference from Hikvision MVS camera (RGB8Packed).
    struct Args {
        /// list connected camera user-defined names and exit
        #[argh(switch)]
        list: bool,

        /// camera user-defined name (from `--list`)
        #[argh(option)]
        camera_name: Option<String>,

        /// set Width (best-effort; depends on camera)
        #[argh(option)]
        width: Option<u32>,

        /// set Height (best-effort; depends on camera)
        #[argh(option)]
        height: Option<u32>,

        /// frame grab timeout in ms
        #[argh(option, default = "1000")]
        timeout_ms: u32,

        /// task (sam3-image, sam3-tracker)
        #[argh(option, default = "String::from(\"sam3-image\")")]
        task: String,

        /// device (cpu:0, cuda:0, etc.)
        #[argh(option, default = "String::from(\"cpu:0\")")]
        device: String,

        /// dtype (q4f16, fp16, fp32, etc.)
        #[argh(option, default = "String::from(\"q4f16\")")]
        dtype: String,

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

    struct HikCamera {
        handle: *mut std::ffi::c_void,
    }

    impl HikCamera {
        fn enumerate_names() -> Result<Vec<String>> {
            unsafe {
                let mut device_list: mvs::MV_CC_DEVICE_INFO_LIST = std::mem::zeroed();
                let status = mvs::MV_CC_EnumDevices(
                    mvs::MV_GIGE_DEVICE | mvs::MV_USB_DEVICE,
                    &mut device_list,
                );
                if status != mvs::MV_OK as i32 {
                    anyhow::bail!("MV_CC_EnumDevices failed: {}", status);
                }
                let mut names = Vec::new();
                for i in 0..device_list.nDeviceNum as usize {
                    let device_info = &*device_list.pDeviceInfo[i];
                    let name_ptr = if device_info.nTLayerType == mvs::MV_GIGE_DEVICE {
                        device_info
                            .SpecialInfo
                            .stGigEInfo
                            .chUserDefinedName
                            .as_ptr()
                    } else {
                        device_info
                            .SpecialInfo
                            .stUsb3VInfo
                            .chUserDefinedName
                            .as_ptr()
                    };
                    let name = CStr::from_ptr(name_ptr as *const i8)
                        .to_string_lossy()
                        .to_string();
                    if !name.trim().is_empty() {
                        names.push(name);
                    }
                }
                Ok(names)
            }
        }

        fn open_by_name(name: &str) -> Result<Self> {
            unsafe {
                let mut device_list: mvs::MV_CC_DEVICE_INFO_LIST = std::mem::zeroed();
                let status = mvs::MV_CC_EnumDevices(
                    mvs::MV_GIGE_DEVICE | mvs::MV_USB_DEVICE,
                    &mut device_list,
                );
                if status != mvs::MV_OK as i32 {
                    anyhow::bail!("MV_CC_EnumDevices failed: {}", status);
                }

                for i in 0..device_list.nDeviceNum as usize {
                    let device_info = &*device_list.pDeviceInfo[i];
                    let name_ptr = if device_info.nTLayerType == mvs::MV_GIGE_DEVICE {
                        device_info
                            .SpecialInfo
                            .stGigEInfo
                            .chUserDefinedName
                            .as_ptr()
                    } else {
                        device_info
                            .SpecialInfo
                            .stUsb3VInfo
                            .chUserDefinedName
                            .as_ptr()
                    };

                    let found_name = CStr::from_ptr(name_ptr as *const i8).to_string_lossy();
                    if found_name != name {
                        continue;
                    }

                    let mut handle: *mut std::ffi::c_void = ptr::null_mut();
                    let status = mvs::MV_CC_CreateHandle(&mut handle, device_list.pDeviceInfo[i]);
                    if status != mvs::MV_OK as i32 {
                        anyhow::bail!("MV_CC_CreateHandle failed: {}", status);
                    }

                    let status = mvs::MV_CC_OpenDevice(handle, mvs::MV_ACCESS_Exclusive, 0);
                    if status != mvs::MV_OK as i32 {
                        mvs::MV_CC_DestroyHandle(handle);
                        anyhow::bail!("MV_CC_OpenDevice failed: {}", status);
                    }

                    return Ok(HikCamera { handle });
                }

                anyhow::bail!("Camera not found by name: {}", name);
            }
        }

        fn set_int(&self, key: &str, value: u32) -> Result<()> {
            unsafe {
                let c_key = CString::new(key).context("key contains NUL")?;
                let status = mvs::MV_CC_SetIntValueEx(self.handle, c_key.as_ptr(), value as i64);
                if status != mvs::MV_OK as i32 {
                    anyhow::bail!("MV_CC_SetIntValue({key}={value}) failed: {}", status);
                }
                Ok(())
            }
        }

        fn start_grabbing(&self) -> Result<()> {
            unsafe {
                let status = mvs::MV_CC_StartGrabbing(self.handle);
                if status != mvs::MV_OK as i32 {
                    anyhow::bail!("MV_CC_StartGrabbing failed: {}", status);
                }
                Ok(())
            }
        }

        fn stop_grabbing(&self) {
            unsafe {
                mvs::MV_CC_StopGrabbing(self.handle);
            }
        }

        fn get_int_param(&self, key: &str) -> Result<u32> {
            unsafe {
                let c_key = CString::new(key).context("key contains NUL")?;
                let mut value: mvs::MVCC_INTVALUE_EX = std::mem::zeroed();
                let status = mvs::MV_CC_GetIntValueEx(self.handle, c_key.as_ptr(), &mut value);
                if status != mvs::MV_OK as i32 {
                    anyhow::bail!("MV_CC_GetIntValue({key}) failed: {}", status);
                }
                Ok(value.nCurValue as u32)
            }
        }

        fn get_frame_rgb8(&self, timeout_ms: u32) -> Result<(Vec<u8>, u32, u32)> {
            unsafe {
                let payload_size = self.get_int_param("PayloadSize").unwrap_or(0);
                let mut buffer = vec![0u8; payload_size.max(1) as usize];
                let mut frame_info: mvs::MV_FRAME_OUT_INFO_EX = std::mem::zeroed();
                let status = mvs::MV_CC_GetOneFrameTimeout(
                    self.handle,
                    buffer.as_mut_ptr(),
                    buffer.len() as u32,
                    &mut frame_info,
                    timeout_ms,
                );
                if status != mvs::MV_OK as i32 {
                    anyhow::bail!("MV_CC_GetOneFrameTimeout failed: {}", status);
                }

                let width = frame_info.nWidth as u32;
                let height = frame_info.nHeight as u32;
                let pixel_type = frame_info.enPixelType as u64;
                if pixel_type != PIXEL_TYPE_RGB8_PACKED {
                    anyhow::bail!(
                        "Unsupported pixel format: 0x{:X} (expected RGB8Packed). Configure the camera PixelFormat in MVS (persistent/default settings).",
                        pixel_type
                    );
                }

                let required = (width as usize)
                    .checked_mul(height as usize)
                    .and_then(|px| px.checked_mul(3))
                    .context("width*height overflow")?;
                if buffer.len() < required {
                    anyhow::bail!(
                        "Frame buffer too small: got {}, expected {}",
                        buffer.len(),
                        required
                    );
                }

                buffer.truncate(required);
                Ok((buffer, width, height))
            }
        }
    }

    impl Drop for HikCamera {
        fn drop(&mut self) {
            unsafe {
                mvs::MV_CC_CloseDevice(self.handle);
                mvs::MV_CC_DestroyHandle(self.handle);
            }
        }
    }

    fn initialize_sdk() -> Result<()> {
        let status = unsafe { mvs::MV_CC_Initialize() };
        if status != mvs::MV_OK as i32 {
            anyhow::bail!("MV_CC_Initialize failed: {}", status);
        }
        Ok(())
    }

    initialize_sdk()?;

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    if args.list {
        for name in HikCamera::enumerate_names()? {
            println!("{name}");
        }
        return Ok(());
    }

    let camera_name = args
        .camera_name
        .clone()
        .context("Missing --camera-name (use --list to see available names)")?;

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

    let mut viewer = Viewer::new("sam3-hikvision").with_window_scale(args.window_scale);

    let camera = HikCamera::open_by_name(&camera_name)?;

    // Use the camera's persisted/default settings; ensure output is RGB8Packed.

    if let Some(width) = args.width {
        if let Err(e) = camera.set_int("Width", width) {
            tracing::warn!("Failed to set Width={width}: {e}");
        }
    }
    if let Some(height) = args.height {
        if let Err(e) = camera.set_int("Height", height) {
            tracing::warn!("Failed to set Height={height}: {e}");
        }
    }

    camera.start_grabbing()?;

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

        let (rgb, width, height) = match camera.get_frame_rgb8(args.timeout_ms) {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!("Frame grab failed: {e}");
                continue;
            }
        };

        let rgb8 = image::RgbImage::from_raw(width, height, rgb)
            .context("failed to construct RgbImage")?;
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

    camera.stop_grabbing();
    usls::perf(false);
    Ok(())
}
