# audiopipe

Fast speech-to-text in Rust. Supports ONNX Runtime and GGML backends — runs on Metal (macOS GPU), Vulkan (Windows/Linux GPU), CUDA, CoreML, DirectML, or CPU. No Python, no ffmpeg.

```rust
use audiopipe::{Model, TranscribeOptions};

let mut model = Model::from_pretrained("qwen3-asr-0.6b-ggml")?;
let result = model.transcribe(&audio_f32, 16000, TranscribeOptions::default())?;
println!("{}", result.text);
```

## Models

| Model | Backend | Params | Languages | Notes |
|-------|---------|--------|-----------|-------|
| `qwen3-asr-0.6b-antirez` | Pure C | 0.6B | 6000+ | Fastest on CPU. BLAS + AVX2/NEON |
| `qwen3-asr-0.6b-ggml` | GGML | 0.6B | 6000+ | Metal/Vulkan/CUDA GPU support |
| `qwen3-asr-0.6b` | ONNX | 0.6B | 6000+ | CoreML/DirectML/CUDA via ONNX Runtime |
| `parakeet-tdt-0.6b-v2` | ONNX | 0.6B | English | NVIDIA Parakeet TDT |
| `parakeet-tdt-0.6b-v3` | ONNX | 0.6B | 25 languages | NVIDIA Parakeet TDT |
| `whisper-*` | whisper.cpp | varies | 99 languages | e.g. `whisper-large-v3-turbo` |

## Usage

### Qwen3-ASR antirez (fastest CPU)

Pure C implementation by [antirez](https://github.com/antirez/qwen-asr). Faster-than-realtime on low-end CPUs using BLAS and custom AVX2/NEON kernels. Uses safetensors weights from `Qwen/Qwen3-ASR-0.6B`.

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-antirez"] }
```

For best CPU performance on Windows/Linux, enable OpenBLAS (`~10x faster` matmul):

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-antirez-blas"] }
```

On macOS, Accelerate is used automatically. On Windows, set up OpenBLAS:

```powershell
# download and extract pre-built OpenBLAS
Invoke-WebRequest "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.31/OpenBLAS-0.3.31-x64.zip" -OutFile OpenBLAS.zip
Expand-Archive OpenBLAS.zip -DestinationPath C:\OpenBLAS
Remove-Item OpenBLAS.zip

# set env var (persistent across sessions)
[System.Environment]::SetEnvironmentVariable("OPENBLAS_PATH", "C:\OpenBLAS\win64", "User")
$env:OPENBLAS_PATH = "C:\OpenBLAS\win64"

# build with BLAS
cargo run --release --example bench_asr --features qwen3-asr-antirez-blas --no-default-features -- audio.wav
```

### Qwen3-ASR GGML

Best balance of speed, accuracy, and language coverage. Uses GGML backend with native GPU support.

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml"] }
```

GPU acceleration:

```toml
# macOS — Metal GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "metal"] }

# Windows/Linux — Vulkan GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "vulkan-ggml"] }

# NVIDIA — CUDA GPU
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr-ggml", "cuda-ggml"] }
```

### Qwen3-ASR ONNX

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr"] }

# With CoreML (macOS)
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr", "coreml"] }

# With DirectML (Windows)
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["qwen3-asr", "directml"] }
```

### Parakeet

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["parakeet"] }
```

### Whisper

```toml
[dependencies]
audiopipe = { git = "https://github.com/screenpipe/audiopipe.git", features = ["whisper"] }
```

## Feature flags

| Feature | Description |
|---------|-------------|
| `qwen3-asr-antirez` | Qwen3-ASR pure C (fastest CPU) |
| `qwen3-asr-antirez-blas` | Above + OpenBLAS for ~10x faster matmul |
| `qwen3-asr-ggml` | Qwen3-ASR via GGML |
| `qwen3-asr` | Qwen3-ASR via ONNX Runtime |
| `parakeet` | NVIDIA Parakeet TDT via ONNX Runtime |
| `whisper` | Whisper via whisper.cpp |
| `metal` | Metal GPU for GGML models (macOS) |
| `vulkan-ggml` | Vulkan GPU for GGML models (Windows/Linux) |
| `cuda-ggml` | CUDA GPU for GGML models (NVIDIA) |
| `coreml` | CoreML for ONNX models (macOS) |
| `directml` | DirectML for ONNX models (Windows) |
| `cuda` | CUDA for ONNX models (NVIDIA) |

## Examples

```sh
# Transcribe with GGML Qwen3-ASR (model auto-downloads from HuggingFace)
cargo run --release --example test_ggml --features qwen3-asr-ggml -- audio.wav

# Use quantized model (smaller, faster on CPU)
cargo run --release --example test_ggml --features qwen3-asr-ggml -- audio.wav qwen3-asr-0.6b-ggml-q8

# Transcribe with Parakeet
cargo run --release --example transcribe --features parakeet -- audio.wav

# Benchmark antirez/qwen-asr (pure C, fast CPU)
cargo run --release --example bench_asr --features qwen3-asr-antirez -- audio.wav

# Benchmark with OpenBLAS (fastest on Windows/Linux)
cargo run --release --example bench_asr --features qwen3-asr-antirez-blas -- audio.wav

# Benchmark
cargo run --release --example benchmark --features qwen3-asr-ggml
```

## Used by

- [screenpipe](https://github.com/screenpipe/screenpipe) — AI screen + audio memory for your desktop

## License

MIT
