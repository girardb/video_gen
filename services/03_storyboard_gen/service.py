"""
Storyboard generator service implementation.

Creates video storyboards from lyrics and vibe tags using LLM.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import jsonschema
import requests

from ..base import BaseService, run_service
from .config import StoryboardGeneratorConfig


class StoryboardGeneratorService(BaseService):
    """Service for generating video storyboards from lyrics and vibe tags."""
    
    def _create_config(self, config_data: dict) -> StoryboardGeneratorConfig:
        """Create configuration object from data."""
        return StoryboardGeneratorConfig(**config_data)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        inputs = self.config.io.inputs
        
        for input_name, input_path in inputs.items():
            input_file = Path(input_path)
            if not input_file.exists():
                self.logger.error(f"Input file not found: {input_file}")
                return False
        
        return True
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created successfully."""
        output_file = Path(self.config.io.output)
        if not output_file.exists():
            self.logger.error(f"Output file not created: {output_file}")
            return False
        
        # Validate JSON schema
        try:
            with open(output_file, 'r') as f:
                storyboard = json.load(f)
            
            # Load schema
            schema_file = Path(self.config.validation.schema_file)
            if schema_file.exists():
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                
                jsonschema.validate(storyboard, schema)
                self.logger.info("Storyboard validated against schema")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False
    
    def _load_lyrics(self) -> Dict:
        """Load lyrics from file."""
        lyrics_file = Path(self.config.io.inputs["lyrics"])
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_vibe_tags(self) -> Dict:
        """Load vibe tags from file."""
        vibe_file = Path(self.config.io.inputs["vibe_tags"])
        with open(vibe_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_json_array(self, text: str) -> Optional[str]:
        """Extract JSON array from text using multiple strategies."""
        import json
        
        # Strategy 1: Fix malformed JSON and find complete array
        # First, try to fix common LLM JSON issues
        fixed_text = self._fix_missing_field_names(text)
        fixed_text = self._fix_malformed_json(fixed_text)
        
        bracket_count = 0
        start_pos = -1
        end_pos = -1
        
        for i, char in enumerate(fixed_text):
            if char == '[':
                if bracket_count == 0:
                    start_pos = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0 and start_pos != -1:
                    end_pos = i
                    break
        
        if start_pos != -1 and end_pos != -1:
            candidate = text[start_pos:end_pos+1]
            try:
                # Test if it's valid JSON
                json.loads(candidate)
                return candidate
            except:
                pass
        
        # Strategy 2: Regex search for JSON array
        patterns = [
            r'\[[\s\S]*?\]',  # Basic array pattern
            r'\[[\s\S]*?\](?=\s*$)',  # Array at end of text
            r'```json\s*(\[[\s\S]*?\])\s*```',  # JSON in code blocks
            r'```\s*(\[[\s\S]*?\])\s*```',  # Generic code blocks
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                candidate = match if isinstance(match, str) else match[0] if match else ""
                try:
                    json.loads(candidate)
                    return candidate
                except:
                    continue
        
        # Strategy 3: Extract between keywords
        keywords = ['JSON Array:', 'json:', '[', 'storyboard:', 'shots:']
        for keyword in keywords:
            pos = text.lower().find(keyword.lower())
            if pos != -1:
                subset = text[pos:]
                start = subset.find('[')
                if start != -1:
                    # Find matching closing bracket
                    bracket_count = 0
                    end = -1
                    for i in range(start, len(subset)):
                        if subset[i] == '[':
                            bracket_count += 1
                        elif subset[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end = i
                                break
                    
                    if end != -1:
                        candidate = subset[start:end+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except:
                            continue
        
        # Strategy 4: Clean up common LLM artifacts and try again
        cleaned_text = text
        # Remove common prefixes/suffixes
        artifacts = [
            "Here's the JSON:", "JSON:", "Array:", "Storyboard:",
            "```json", "```", "```python", "```javascript",
            "The storyboard is:", "Response:", "Output:",
        ]
        
        for artifact in artifacts:
            cleaned_text = cleaned_text.replace(artifact, "")
        
        # Try to find array in cleaned text and truncate after the last ]
        start = cleaned_text.find('[')
        end = cleaned_text.rfind(']')
        
        if start != -1 and end != -1 and end > start:
            # Truncate everything after the last ] to remove trailing text
            candidate = cleaned_text[start:end+1]
            try:
                json.loads(candidate)
                return candidate
            except:
                pass
        
        return None
    
    def _fix_missing_field_names(self, text: str) -> str:
        """Fix shots that are missing field names for any field."""
        import re
        
        # Define the expected field order and types
        field_patterns = [
            # Pattern: "start": number, "number", next_field
            (r'("start":\s*[\d.]+),\s*"([\d.]+)",\s*("(?:end|prompt|motion|seed|model|ref_image)":)', r'\1, "end": \2, \3'),
            
            # Pattern: "end": number, "text", next_field  
            (r'("end":\s*[\d.]+),\s*"([^"]+)",\s*("(?:motion|seed|model|ref_image)":)', r'\1, "prompt": "\2", \3'),
            
            # Pattern: "prompt": "text", "text", next_field
            (r'("prompt":\s*"[^"]+"),\s*"([^"]+)",\s*("(?:seed|model|ref_image)":)', r'\1, "motion": "\2", \3'),
            
            # Pattern: "motion": "text", number, next_field
            (r'("motion":\s*"[^"]+"),\s*(\d+),\s*("(?:model|ref_image)":)', r'\1, "seed": \2, \3'),
            
            # Pattern: "seed": number, "text", next_field
            (r'("seed":\s*\d+),\s*"([^"]+)",\s*("ref_image":)', r'\1, "model": "\2", \3'),
        ]
        
        # Apply each pattern fix
        for pattern, replacement in field_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text

    def _fix_malformed_json(self, text: str) -> str:
        """Fix common JSON formatting issues from LLM responses."""
        import re
        
        # Fix any malformed entries like {"main": ...} instead of {"start": ...}
        text = re.sub(r'{"main":', '{"start": 999.0, "end": 999.0, "prompt":', text)
        
        # Remove any trailing commas before closing brackets/braces
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix common bracket issues
        if text.count('[') > text.count(']'):
            text = text + ']'
        elif text.count(']') > text.count('['):
            text = '[' + text
            
        return text
    
    def _fix_json_syntax_errors(self, text: str) -> str:
        """Fix common JSON syntax errors from LLM responses."""
        import re
        
        # Fix missing quotes before field names (e.g., end": -> "end":)
        text = re.sub(r'(\w+)":', r'"\1":', text)
        
        # Fix cases where there might be extra text after the JSON
        # Find the last } and add ] after it if needed
        last_brace = text.rfind('}')
        if last_brace != -1:
            # Check if there's already a ] after the last }
            after_brace = text[last_brace:].strip()
            if not after_brace.endswith(']'):
                text = text[:last_brace+1] + '\n]'
        
        return text
    
    def _generate_storyboard_with_llm(self, lyrics: Dict, vibe_tags: Dict, music_analysis: Dict, song_brief: str, suno_prompt: str) -> List[Dict]:
        """Generate storyboard using LLM server."""
        try:
            # Prepare system prompt
            system_prompt = self._create_system_prompt()
            
            # Prepare user prompt
            user_prompt = self._create_user_prompt(lyrics, vibe_tags, music_analysis, song_brief, suno_prompt)
            
            # Make request to LLM server
            payload = {
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_tokens": self.config.llm.max_tokens,
                "temperature": self.config.llm.temperature,
                "top_p": self.config.llm.top_p
            }
            
            response = requests.post(
                f"{self.config.llm.server_url}/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                data = response.json()
                storyboard_text = data["text"]
                
                # Parse the generated storyboard by combining our prompt with the response
                user_prompt = self._create_user_prompt(lyrics, vibe_tags, music_analysis)
                return self._parse_storyboard(storyboard_text, lyrics, vibe_tags, user_prompt)
            else:
                self.logger.error(f"LLM server request failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"LLM server failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Failed to call LLM server: {e}")
            raise RuntimeError(f"LLM server communication failed: {e}")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for storyboard generation."""
        return ""
    
    def _create_user_prompt(self, lyrics: Dict, vibe_tags: Dict, music_analysis: Dict, song_brief: str, suno_prompt: str) -> str:
        """Create user prompt with lyrics, vibe tags, music analysis, and original context."""
        # Extract vibe words
        vibe_words = []
        if "top3" in vibe_tags:
            vibe_words = vibe_tags["top3"]
        elif "tags" in vibe_tags:
            vibe_words = [tag["tag"] for tag in vibe_tags["tags"]]
        
        # Get music characteristics
        tempo_desc = music_analysis.get("tempo", {}).get("description", "moderate tempo")
        emotional_tone = music_analysis.get("enhanced_emotional_tone", music_analysis.get("emotional_tone", "neutral"))
        human_description = music_analysis.get("human_description", "")
        duration = music_analysis.get("duration_seconds", 30)
        
        # Calculate number of shots based on actual duration
        # For a ~148s song, we want more shots for better coverage
        num_shots = max(15, int(duration / 60 * self.config.generation.shots_per_minute * 2))
        
        # Get structured lyrics sections
        structured_lyrics = lyrics.get("structured_lyrics", [])
        lyrics_summary = ""
        if structured_lyrics:
            sections = [f"[{s['type']}] {s['text'][:50]}..." for s in structured_lyrics[:3]]
            lyrics_summary = " | ".join(sections)
        else:
            lyrics_summary = lyrics.get("text", "")[:200] + "..."
        
        return f"""Music video storyboard: {duration:.1f}s {emotional_tone} {tempo_desc}.

Concept: {song_brief[:150]}...
Suno: {suno_prompt}
Lyrics: {lyrics_summary}
Style: {', '.join(vibe_words)}

Continue this JSON:

[
  {{"start": 0.0, "end": 4.0, "prompt": "wide shot matching the song's opening mood with {vibe_words[0] if vibe_words else 'atmospheric'} lighting", "motion": "slow dolly", "seed": 12345, "model": "sdxl", "ref_image": null}},
"""
    
    def _parse_storyboard(self, storyboard_text: str, lyrics: Dict, vibe_tags: Dict, user_prompt: str = "") -> List[Dict]:
        """Parse the generated storyboard text into structured data."""
        try:
            self.logger.info(f"Raw LLM response: {storyboard_text[:200]}...")
            
            # Combine our prompt (which has opening bracket + first shot) with LLM response
            combined_text = user_prompt + storyboard_text + "\n]"
            self.logger.info(f"Combined prompt + response: {combined_text[:500]}...")
            
            # Also log the end to see if we're getting the full response
            self.logger.info(f"End of combined text: ...{combined_text[-200:]}")
            
            # Try to fix common JSON syntax errors before extraction
            combined_text = self._fix_json_syntax_errors(combined_text)
            
            storyboard_json = self._extract_json_array(combined_text)
            
            if storyboard_json:
                self.logger.info(f"Extracted JSON: {storyboard_json[:100]}...")
                storyboard = json.loads(storyboard_json)
            else:
                raise ValueError("Could not extract valid JSON from LLM response")
            
            # Validate and clean up the storyboard
            cleaned_storyboard = self._validate_and_clean_storyboard(storyboard)
            self.logger.info(f"Cleaned storyboard has {len(cleaned_storyboard)} shots")
            return cleaned_storyboard
            
        except Exception as e:
            self.logger.error(f"Failed to parse storyboard: {e}")
            raise RuntimeError(f"Storyboard parsing failed: {e}")
    
    def _load_music_analysis(self) -> Dict:
        """Load music analysis from file."""
        analysis_file = Path(self.config.io.inputs["music_analysis"])
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_song_brief(self) -> str:
        """Load original song brief from file."""
        brief_file = Path(self.config.io.inputs["song_brief"])
        with open(brief_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _load_suno_prompt(self) -> str:
        """Load compacted Suno prompt from file."""
        prompt_file = Path(self.config.io.inputs["suno_prompt"])
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _validate_and_clean_storyboard(self, storyboard: List[Dict]) -> List[Dict]:
        """Validate and clean up the storyboard."""
        cleaned_storyboard = []
        current_time = 0.0
        
        for i, shot in enumerate(storyboard):
            cleaned_shot = {}
            
            # Remove any invalid fields (only keep schema-allowed fields)
            allowed_fields = ["start", "end", "prompt", "motion", "seed", "model", "ref_image"]
            for field in allowed_fields:
                if field in shot:
                    cleaned_shot[field] = shot[field]
            
            # Fix missing start/end times
            if "start" not in cleaned_shot or "end" not in cleaned_shot:
                # Try to extract duration if available
                duration = shot.get("duration", 4.0)
                cleaned_shot["start"] = current_time
                cleaned_shot["end"] = current_time + duration
                current_time = cleaned_shot["end"]
            else:
                current_time = max(current_time, cleaned_shot["end"])
            
            # Ensure required fields exist
            required_fields = ["start", "end", "prompt", "motion", "seed", "model"]
            if not all(field in cleaned_shot for field in required_fields):
                self.logger.warning(f"Shot {i} missing required fields, skipping: {cleaned_shot}")
                continue
            
            # Validate timing
            if cleaned_shot["start"] >= cleaned_shot["end"]:
                self.logger.warning(f"Shot {i} has invalid timing, skipping: {cleaned_shot['start']} >= {cleaned_shot['end']}")
                continue
            
            # Validate prompt length
            prompt_words = len(cleaned_shot["prompt"].split())
            if prompt_words < self.config.validation.min_prompt_words or prompt_words > self.config.validation.max_prompt_words:
                # Adjust prompt length
                words = cleaned_shot["prompt"].split()
                if prompt_words > self.config.validation.max_prompt_words:
                    cleaned_shot["prompt"] = " ".join(words[:self.config.validation.max_prompt_words])
                else:
                    cleaned_shot["prompt"] = " ".join(words) + " with beautiful lighting and atmosphere"
            
            # Ensure seed is valid
            if not isinstance(cleaned_shot["seed"], int) or cleaned_shot["seed"] < 0 or cleaned_shot["seed"] > 999999:
                cleaned_shot["seed"] = random.randint(1000, 999999)
            
            # Ensure ref_image is null if not provided
            if "ref_image" not in cleaned_shot:
                cleaned_shot["ref_image"] = None
            
            cleaned_storyboard.append(cleaned_shot)
        
        return cleaned_storyboard
    

    
    def _save_storyboard(self, storyboard: List[Dict]) -> None:
        """Save storyboard to file."""
        output_path = Path(self.config.io.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(storyboard, f, indent=2)
    
    def run(self) -> int:
        """Run the storyboard generator service."""
        try:
            self.logger.info("Starting storyboard generator service")
            
            # Validate inputs
            if not self.validate_inputs():
                return 1
            
            # Load inputs
            lyrics = self._load_lyrics()
            vibe_tags = self._load_vibe_tags()
            music_analysis = self._load_music_analysis()
            song_brief = self._load_song_brief()
            suno_prompt = self._load_suno_prompt()
            
            self.logger.info(f"Loaded lyrics: {len(lyrics.get('text', ''))} characters")
            self.logger.info(f"Loaded vibe tags: {len(vibe_tags)} tags")
            self.logger.info(f"Loaded music analysis: {music_analysis.get('emotional_tone', 'unknown')} tone")
            self.logger.info(f"Loaded song brief: {len(song_brief)} characters")
            self.logger.info(f"Loaded suno prompt: {suno_prompt[:50]}...")
            
            # Generate storyboard
            storyboard = self._generate_storyboard_with_llm(lyrics, vibe_tags, music_analysis, song_brief, suno_prompt)
            self.logger.info(f"Generated storyboard: {len(storyboard)} shots")
            
            # Save storyboard
            self._save_storyboard(storyboard)
            
            # Validate outputs
            if not self.validate_outputs():
                return 1
            
            self.logger.info("Storyboard generator service completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Storyboard generator service failed: {e}")
            return 1


def main():
    """Entry point for the storyboard generator service."""
    run_service(StoryboardGeneratorService, "Storyboard Generator")


if __name__ == "__main__":
    main() 