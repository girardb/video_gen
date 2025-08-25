"""
Audio analyser service implementation.

Analyzes audio to extract lyrics, vibe tags, and beat information.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import requests

from ..base import BaseService, run_service
from .config import AudioAnalyserConfig


class AudioAnalyserService(BaseService):
    """Service for analyzing audio to extract lyrics, vibes, and beats."""
    
    def _create_config(self, config_data: dict) -> AudioAnalyserConfig:
        """Create configuration object from data."""
        return AudioAnalyserConfig(**config_data)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        # Check audio file
        audio_file = Path(self.config.io.inputs["audio"])
        if not audio_file.exists():
            self.logger.error(f"Audio file not found: {audio_file}")
            return False
        
        # Check lyrics file (required)
        lyrics_file = Path(self.config.io.inputs["lyrics"])
        if not lyrics_file.exists():
            self.logger.error(f"Lyrics file not found: {lyrics_file}")
            return False
        
        return True
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created successfully."""
        outputs = self.config.io.outputs
        
        for output_name, output_path in outputs.items():
            output_file = Path(output_path)
            if not output_file.exists():
                self.logger.error(f"Output file not created: {output_file}")
                return False
            
            # Check file size
            if output_file.stat().st_size == 0:
                self.logger.error(f"Output file is empty: {output_file}")
                return False
        
        return True
    
    def _load_audio(self) -> tuple:
        """Load and preprocess audio."""
        audio_file = self.config.io.inputs["audio"]
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.config.processing.resample_rate)
        
        # Normalize if requested
        if self.config.processing.normalize:
            audio = librosa.util.normalize(audio)
        
        # Trim silence if requested
        if self.config.processing.trim_silence:
            audio, _ = librosa.effects.trim(audio)
        
        return audio, sr
    
    def _load_lyrics(self) -> Dict:
        """Load lyrics from file."""
        lyrics_file = Path(self.config.io.inputs["lyrics"])
        
        if not lyrics_file.exists():
            raise FileNotFoundError(f"Lyrics file not found: {lyrics_file}")
            
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _analyze_vibes_with_clap(self, audio_file: str) -> List[Dict]:
        """Analyze audio vibes using CLAP server."""
        try:
            # Prepare request payload
            payload = {
                "audio_file": audio_file,
                "top_k": self.config.clap.top_k,
                "threshold": self.config.clap.threshold
            }
            
            # Make request to CLAP server
            response = requests.post(
                f"{self.config.clap.server_url}/analyze",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["vibe_tags"]
            else:
                self.logger.error(f"CLAP server request failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"CLAP server failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Failed to call CLAP server: {e}")
            raise RuntimeError(f"CLAP server communication failed: {e}")
    
    def _analyze_music_characteristics(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze musical characteristics and emotional content."""
        # Tempo and rhythm analysis
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Energy and dynamics
        rms_energy = librosa.feature.rms(y=audio)[0]
        
        # Key and harmony (basic)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Calculate averages
        avg_spectral_centroid = float(np.mean(spectral_centroids))
        avg_spectral_rolloff = float(np.mean(spectral_rolloff))
        avg_zero_crossing = float(np.mean(zero_crossing_rate))
        avg_energy = float(np.mean(rms_energy))
        
        # Emotional and musical interpretation
        analysis = {
            "_note": "Analysis based on basic audio features. Emotional interpretations are heuristic and not ML-validated.",
            "tempo": {
                "bpm": float(tempo),
                "description": self._interpret_tempo(tempo)
            },
            "energy": {
                "level": avg_energy,
                "description": self._interpret_energy(avg_energy)
            },
            "brightness": {
                "spectral_centroid": avg_spectral_centroid,
                "description": self._interpret_brightness(avg_spectral_centroid)
            },
            "dynamics": {
                "variability": float(np.std(rms_energy)),
                "description": self._interpret_dynamics(np.std(rms_energy))
            },
            "emotional_tone": self._analyze_emotional_tone(tempo, avg_energy, avg_spectral_centroid),
            "musical_style": self._analyze_musical_style(tempo, avg_energy, avg_spectral_centroid, avg_zero_crossing),
            "beat_count": len(beats),
            "duration_seconds": float(len(audio) / sr)
        }
        
        self.logger.info(f"Analyzed music: {analysis['tempo']['description']}, {analysis['emotional_tone']}, {analysis['energy']['description']}")
        
        return analysis
    
    def _interpret_tempo(self, tempo: float) -> str:
        """Interpret tempo into musical description."""
        if tempo < 60:
            return "very slow, ballad-like"
        elif tempo < 80:
            return "slow, contemplative"
        elif tempo < 100:
            return "moderate, walking pace"
        elif tempo < 120:
            return "moderately fast, upbeat"
        elif tempo < 140:
            return "fast, energetic"
        elif tempo < 160:
            return "very fast, driving"
        else:
            return "extremely fast, intense"
    
    def _interpret_energy(self, energy: float) -> str:
        """Interpret energy level into description (rough heuristic)."""
        # Note: RMS values are typically 0.001-0.5 for normalized audio
        if energy < 0.05:
            return "quiet, low energy"
        elif energy < 0.15:
            return "moderate energy"
        elif energy < 0.25:
            return "high energy"
        else:
            return "very high energy"
    
    def _interpret_brightness(self, spectral_centroid: float) -> str:
        """Interpret spectral brightness (rough heuristic)."""
        # Note: Spectral centroid typically ranges 500-8000 Hz for music
        if spectral_centroid < 1500:
            return "bass-heavy, darker tone"
        elif spectral_centroid < 2500:
            return "balanced frequency content"
        elif spectral_centroid < 4000:
            return "bright, clear highs"
        else:
            return "very bright, treble-heavy"
    
    def _interpret_dynamics(self, variability: float) -> str:
        """Interpret dynamic range."""
        if variability < 0.01:
            return "consistent volume"
        elif variability < 0.02:
            return "some dynamic variation"
        else:
            return "highly dynamic, varied volume"
    
    def _analyze_emotional_tone(self, tempo: float, energy: float, brightness: float) -> str:
        """Analyze overall emotional tone (basic heuristic - not ML-based)."""
        # Simple rule-based emotional inference - NOT scientifically validated
        if tempo > 120 and energy > 0.15:
            if brightness > 2500:
                return "likely upbeat, energetic"
            else:
                return "intense, driving"
        elif tempo < 80:
            if energy < 0.1:
                return "possibly melancholic, contemplative"
            else:
                return "dramatic, slow but powerful"
        elif energy < 0.1:
            return "calm, subdued"
        else:
            return "moderate energy, neutral mood"
    
    def _analyze_musical_style(self, tempo: float, energy: float, brightness: float, zcr: float) -> str:
        """Analyze musical style characteristics (basic heuristics)."""
        style_elements = []
        
        # Electronic vs acoustic indicators (rough guess)
        if zcr > 0.1:
            style_elements.append("possibly electronic/digital")
        else:
            style_elements.append("smoother texture")
        
        # Basic tempo-based genre hints
        if tempo > 140 and energy > 0.2:
            style_elements.append("fast, high-energy")
        elif tempo > 120:
            style_elements.append("upbeat tempo")
        elif tempo < 80:
            style_elements.append("slow/ballad tempo")
        else:
            style_elements.append("moderate tempo")
        
        return ", ".join(style_elements)
    
    def _enhance_emotional_analysis(self, vibe_tags: List[Dict], music_analysis: Dict) -> Dict:
        """Enhance emotional analysis using CLAP tags and basic heuristics."""
        
        # Extract tag names from CLAP
        tag_names = [tag["tag"].lower() for tag in vibe_tags if "tag" in tag]
        
        # Emotion-related keywords mapping
        emotion_keywords = {
            "happy": ["happy", "joyful", "upbeat", "cheerful", "bright", "fun", "energetic", "positive"],
            "sad": ["sad", "melancholic", "depressing", "somber", "mournful", "tragic", "dark"],
            "angry": ["aggressive", "angry", "intense", "furious", "violent", "harsh", "brutal"],
            "calm": ["calm", "peaceful", "relaxing", "serene", "gentle", "soft", "ambient", "chill"],
            "nostalgic": ["nostalgic", "vintage", "retro", "memory", "longing", "wistful"],
            "romantic": ["romantic", "love", "tender", "intimate", "passionate", "sensual"],
            "mysterious": ["mysterious", "eerie", "dark", "mysterious", "enigmatic", "haunting"],
            "energetic": ["energetic", "dynamic", "powerful", "driving", "exciting", "pumping"]
        }
        
        # Find emotional matches
        detected_emotions = []
        for emotion, keywords in emotion_keywords.items():
            matches = [tag for tag in tag_names if any(keyword in tag for keyword in keywords)]
            if matches:
                detected_emotions.append({
                    "emotion": emotion,
                    "evidence": matches,
                    "confidence": len(matches) / len(tag_names) if tag_names else 0
                })
        
        # Sort by confidence
        detected_emotions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Create enhanced emotional description
        if detected_emotions:
            primary_emotion = detected_emotions[0]["emotion"]
            evidence_tags = detected_emotions[0]["evidence"]
            
            enhanced_emotional_tone = f"{primary_emotion} (based on tags: {', '.join(evidence_tags[:3])})"
        else:
            enhanced_emotional_tone = music_analysis.get("emotional_tone", "neutral")
        
        # Human-like overall description
        tempo_desc = music_analysis["tempo"]["description"]
        energy_desc = music_analysis["energy"]["description"]
        
        human_description = self._generate_human_description(
            primary_emotion if detected_emotions else "neutral",
            tempo_desc,
            energy_desc,
            tag_names[:5]  # Top 5 CLAP tags
        )
        
        return {
            "detected_emotions": detected_emotions,
            "enhanced_emotional_tone": enhanced_emotional_tone,
            "human_description": human_description,
            "clap_evidence": tag_names[:10]  # Top 10 tags for reference
        }
    
    def _generate_human_description(self, emotion: str, tempo: str, energy: str, tags: List[str]) -> str:
        """Generate a human-like description of how the song feels."""
        
        descriptions = {
            "happy": f"This song feels uplifting and positive. {tempo.capitalize()} with {energy}. The overall vibe suggests it would make people feel good.",
            "sad": f"This song has a melancholic, emotional quality. {tempo.capitalize()} with {energy}. It might evoke feelings of sadness or reflection.", 
            "angry": f"This song feels intense and aggressive. {tempo.capitalize()} with {energy}. It has a powerful, driving energy.",
            "calm": f"This song feels peaceful and relaxing. {tempo.capitalize()} with {energy}. It would likely help people unwind.",
            "energetic": f"This song is full of energy and excitement. {tempo.capitalize()} with {energy}. It would get people moving.",
            "romantic": f"This song has a romantic, intimate feeling. {tempo.capitalize()} with {energy}. It conveys tenderness and emotion.",
            "nostalgic": f"This song evokes memories and nostalgia. {tempo.capitalize()} with {energy}. It has a wistful, reflective quality.",
            "mysterious": f"This song has an enigmatic, mysterious atmosphere. {tempo.capitalize()} with {energy}. It creates intrigue and depth."
        }
        
        base_desc = descriptions.get(emotion, f"This song has a {emotion} feeling. {tempo.capitalize()} with {energy}.")
        
        if tags:
            tag_context = f" The audio analysis suggests: {', '.join(tags[:3])}."
            return base_desc + tag_context
        
        return base_desc

    def _save_vibe_tags(self, vibe_tags: List[Dict]) -> None:
        """Save vibe tags to file."""
        output_path = Path(self.config.io.outputs["vibe_tags"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format based on config
        if self.config.output.vibe_format == "top3":
            output_data = {"top3": [tag["tag"] for tag in vibe_tags[:3]]}
        elif self.config.output.vibe_format == "all":
            output_data = {"tags": vibe_tags}
        else:  # threshold
            threshold_tags = [tag for tag in vibe_tags if tag["confidence"] >= self.config.clap.threshold]
            output_data = {"tags": threshold_tags}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    
    def _save_music_analysis(self, analysis: Dict) -> None:
        """Save musical analysis to file."""
        output_path = Path(self.config.io.outputs["beats"])  # Reusing the beats output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Always save as JSON for readability
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def run(self) -> int:
        """Run the audio analyser service."""
        try:
            self.logger.info("Starting audio analyser service")
            
            # Validate inputs
            if not self.validate_inputs():
                return 1
            
            # Load audio
            audio, sr = self._load_audio()
            self.logger.info(f"Loaded audio: {len(audio)/sr:.1f}s at {sr}Hz")
            
            # Load lyrics
            lyrics = self._load_lyrics()
            self.logger.info(f"Loaded lyrics: {len(lyrics['text'])} characters")
            
            # Analyze vibes with CLAP
            audio_file = self.config.io.inputs["audio"]
            vibe_tags = self._analyze_vibes_with_clap(audio_file)
            self.logger.info(f"Analyzed vibes: {len(vibe_tags)} tags")
            
            # Analyze musical characteristics  
            music_analysis = self._analyze_music_characteristics(audio, sr)
            
            # Try to get better emotional analysis from CLAP tags
            emotional_analysis = self._enhance_emotional_analysis(vibe_tags, music_analysis)
            music_analysis.update(emotional_analysis)
            
            # Save outputs
            self._save_vibe_tags(vibe_tags)
            self._save_music_analysis(music_analysis)
            
            # Validate outputs
            if not self.validate_outputs():
                return 1
            
            self.logger.info("Audio analyser service completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Audio analyser service failed: {e}")
            return 1


def main():
    """Entry point for the audio analyser service."""
    run_service(AudioAnalyserService, "Audio Analyser")


if __name__ == "__main__":
    main() 