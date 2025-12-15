# sam3-card-detector

Rust webcam inference app using `SAM3` (text prompts) from the local `../usls` crate.

`webcam_sam3` currently uses V4L2 (`v4l` crate), so it targets Linux.

## Run

CPU:

```bash
cargo run -r --bin webcam_sam3 -- \
  --device cpu:0 --dtype q4f16 \
  --camera 0 --width 640 --height 480 \
  -p "playing card"
```

CUDA (ONNX Runtime CUDA EP):

```bash
cargo run -r --features cuda --bin webcam_sam3 -- \
  --device cuda:0 --dtype fp16 \
  -p "playing card"
```

TensorRT (ONNX Runtime TensorRT EP):

```bash
cargo run -r --features tensorrt --bin webcam_sam3 -- \
  --device tensorrt:0 --dtype fp16 \
  -p "playing card"
```

## Controls

- `ESC` / `Q`: quit
- `P`: update prompt(s) (split multiple prompts with `|`)
- `S`: save the last displayed frame to `./runs/<model-spec>/`
