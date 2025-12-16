fn main() {
    eprintln!("V4L2: `cargo run -r --bin v4l-sam3 -- -p \"playing card\"`");
    eprintln!("Video: `cargo run -r --bin video-sam3 -- <video.mp4> -p \"playing card\"`");
    eprintln!("Hikvision: `cargo run -r --features hikvision --bin hikvision-sam3 -- --list`");
}
