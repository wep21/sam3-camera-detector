# sam3-card-detector

Rust webcam inference app using `SAM3` (text prompts) via the `usls` crate pinned to a git revision.

Camera backends:

- V4L2 (`v4l_sam3` / `webcam_sam3`): Linux only.
- Hikvision MVS (`hikvision_sam3`): Linux + `/opt/MVS` + `--features hikvision`.

## Run

V4L2 (CPU):

```bash
cargo run -r --bin v4l_sam3 -- \
  --device cpu:0 --dtype q4f16 \
  --camera 0 --width 640 --height 480 \
  -p "playing card"
```

CUDA (ONNX Runtime CUDA EP):

```bash
cargo run -r --features cuda --bin v4l_sam3 -- \
  --device cuda:0 --dtype fp16 \
  -p "playing card"
```

TensorRT (ONNX Runtime TensorRT EP):

```bash
cargo run -r --features tensorrt --bin v4l_sam3 -- \
  --device tensorrt:0 --dtype fp16 \
  -p "playing card"
```

Hikvision MVS:

```bash
cargo run -r --features hikvision --bin hikvision_sam3 -- --list
cargo run -r --features hikvision --bin hikvision_sam3 -- \
  --camera-name "<YOUR_CAMERA_NAME>" \
  -p "playing card"
```

Note: this backend expects the camera's current PixelFormat to be `RGB8Packed` (set it persistently in MVS).

## Controls

- `ESC` / `Q`: quit
- `P`: update prompt(s) (split multiple prompts with `|`)
- `S`: save the last displayed frame to `./runs/<model-spec>/`
