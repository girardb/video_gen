# TODO: Modular AI Music-Video Generator

## ✅ Completed
- [x] **Repo init** (`README`, `.gitignore`, `TODO.md`)

## 🔄 In Progress
- [x] Generate `configs/00.yaml`…`05.yaml` with sensible defaults (include model switching for 00 & 03)
- [x] Scaffold each service with `__main__.py`, argument parsing, and dummy return codes
- [x] **Model Servers**: Create FastAPI servers for local models
- [x] **Service Updates**: Update services to query model servers via HTTP
- [ ] Implement & unit-test Services **02 → 05** end-to-end (use 1-sec sine-wave MP3 for tests)
- [ ] Create `orchestrate.py` MVP and a `run_demo.sh` wrapper

## 📋 Pending Tasks

### Model Servers (FastAPI)
- [x] **llm_server**
  - [x] Create FastAPI server for local LLMs (Llama/Mistral)
  - [x] Add model loading and inference endpoints
  - [x] Add health check and model info endpoints
  - [x] Add multi-model loading strategies (llama-cpp, transformers)
  - [ ] Write unit tests

- [x] **clap_server**
  - [x] Create FastAPI server for CLAP audio analysis
  - [x] Add audio classification and embedding endpoints
  - [x] Add vibe tag extraction
  - [x] Add multi-model loading strategies (CLAP, WavLM, Wav2Vec)
  - [ ] Write unit tests

### Service Implementation
- [x] **00_prompt_compactor**
  - [x] Implement local model support (Llama/Mistral)
  - [x] Implement OpenAI API support
  - [x] Add input validation and sanitization
  - [x] Update to query llm_server via HTTP
  - [ ] Write unit tests

- [x] **01_music_generator**
  - [x] Implement Suno API integration
  - [x] Add offline mode with silent MP3 generation
  - [x] Add error handling for missing MP3
  - [x] Add WAV fallback support
  - [ ] Write unit tests

- [x] **02_audio_analyser**
  - [x] Extract lyrics from Suno API response
  - [x] Query clap_server for vibe analysis
  - [x] Implement beat detection with librosa
  - [ ] Write unit tests

- [x] **03_storyboard_gen**
  - [x] Update to query llm_server via HTTP
  - [x] Implement OpenAI API support
  - [x] Add JSON schema validation
  - [x] Write unit tests

- [ ] **04_video_renderer**
  - [ ] Implement AnimateDiff-SDXL integration
  - [ ] Add support for multiple engines
  - [ ] Implement seed and ref_image handling
  - [ ] Write unit tests

- [ ] **05_video_assembler**
  - [ ] Implement moviepy + ffmpeg integration
  - [ ] Add beat synchronization
  - [ ] Add post-processing effects
  - [ ] Write unit tests

### Infrastructure
- [ ] **Orchestrator**
  - [ ] Create `orchestrate.py` with YAML config loading
  - [ ] Implement service execution pipeline
  - [ ] Add error handling and logging
  - [ ] Create `run_demo.sh` wrapper

- [ ] **Configuration**
  - [ ] Create `configs/pipeline.yaml` for orchestrator
  - [ ] Create individual service configs (00-05)
  - [ ] Add storyboard schema validation

- [ ] **Testing**
  - [ ] Set up pytest framework
  - [ ] Create test data and fixtures
  - [ ] Write integration tests
  - [ ] Add CI/CD pipeline

- [ ] **Documentation**
  - [ ] Add detailed service documentation
  - [ ] Create API reference
  - [ ] Add troubleshooting guide

### Optional Services (Future)
- [ ] **06_style_lora** - Custom style training
- [ ] **07_rlhf_evaluator** - RLHF evaluation system

## 🐛 Known Issues
None yet

## 💡 Ideas for Future
- Add support for more video generation engines
- Implement real-time preview during generation
- Add support for custom music generation models
- Create web UI for easier interaction 