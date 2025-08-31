# Modular AI Music-Video Generator

Turn a free-form idea in `song_brief.txt` into a beat-synced music video (`out/final_cut.mp4`) via purely modular micro-services.

## Notice: 04->07 are still WIP


## 🚀 Quick Start

```bash
# Setup
pip install -r requirements.txt

# Start model servers (in separate terminals)
python -m model_servers.llm_server --port 8001
python -m model_servers.clap_server --port 8002
python -m model_servers.image_server --port 8005 --model-name "Qwen/Qwen-Image"
python -m model_servers.video_server --port 8004 --model-name "Wan-AI/Wan2.2-S2V-14B"

# Run the full pipeline
python orchestrate.py

# Or run individual services
python -m services.00_prompt_compactor --config configs/00.yaml
python -m services.01_music_generator --config configs/01.yaml
# ... etc
```

## 📁 Project Structure

```
video_gen/
├─ services/           # Workflow micro-services (00-07)
├─ model_servers/      # FastAPI model servers
│ ├─ llm_server/      # Local LLM server (Llama/Mistral)
│ └─ clap_server/     # CLAP audio analysis server
├─ configs/           # YAML configs per service
├─ data/              # Input/output artefacts
├─ tests/             # Unit tests
├─ scripts/           # Helper scripts
├─ orchestrate.py     # Pipeline orchestrator
└─ TODO.md           # Living checklist
```

## 🔧 Architecture

### Workflow Services
- **00_prompt_compactor**: Converts song brief to Suno prompt
- **01_music_generator**: Generates music via Suno API
- **02_audio_analyser**: Extracts lyrics, beats, and vibe tags
- **03_storyboard_gen**: Creates video storyboard from lyrics
- **03.5_image_generator**: Generates reference images from storyboard prompts
- **04_video_renderer**: Renders video clips via video server (Wan2.2-S2V-14B)
- **05_video_assembler**: Assembles final beat-synced video
- **06_style_lora**: (Optional) Custom style training
- **07_rlhf_evaluator**: (Optional) RLHF evaluation

### Model Servers (FastAPI)
- **llm_server**: Serves local LLMs (Llama/Mistral) for text generation
- **clap_server**: Serves CLAP models for audio analysis and vibe detection
- **image_server**: Serves image generation models (Qwen-Image) for text-to-image
- **video_server**: Serves video generation models (Wan2.2-S2V-14B) for image+audio-to-video

## 🎯 Features

- **Modular**: Swap any service by editing one folder
- **Beat-synced**: Video clips sync to audio beats
- **Client-Server**: Model servers separate from workflow services
- **Multi-model**: Support for local and OpenAI models
- **Type-safe**: Full Python typing with mypy
- **Tested**: Unit tests for each service

## 📋 Requirements

- Python ≥ 3.10
- See `requirements.txt` for dependencies

## 🧪 Testing

```bash
pytest tests/
```

## 📝 License

MIT License 