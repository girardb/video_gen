#!/usr/bin/env python3
"""Test script to validate JSON array extraction from storyboard output."""

import json
import re
from pathlib import Path

def _fix_malformed_json(text):
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

def _fix_missing_field_names(text):
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

def _fix_json_syntax_errors(text):
    """Fix common JSON syntax errors from LLM responses."""
    import re
    
    # Fix missing quotes before field names ONLY if there's no quote already (e.g., end": -> "end":)
    # Look for word followed by ": but not preceded by "
    text = re.sub(r'(?<!")(\b\w+)":', r'"\1":', text)
    
    # Fix cases where there might be extra text after the JSON
    # Find the last } and add ] after it if needed
    last_brace = text.rfind('}')
    if last_brace != -1:
        # Check if there's already a ] after the last }
        after_brace = text[last_brace:].strip()
        if not after_brace.endswith(']'):
            text = text[:last_brace+1] + '\n]'
    
    return text

def _extract_json_array(text):
    """Extract JSON array from text using multiple strategies."""
    import json
    
    # Strategy 1: Fix malformed JSON and find complete array
    # First, try to fix common LLM JSON issues
    fixed_text = _fix_missing_field_names(text)
    fixed_text = _fix_malformed_json(fixed_text)

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
        candidate = fixed_text[start_pos:end_pos+1]
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
        print("potential clean text (truncated): ", candidate)
        try:
            json.loads(candidate)
            return candidate
        except Exception as e:
            print(f"JSON parse error: {e}")
            pass
    
    return None

def test_json_extraction(input_file):
    """Test JSON extraction on a file."""
    
    # Read the input file
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File {input_file} not found")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"=== TESTING JSON EXTRACTION ===")
    print(f"Input file: {input_file}")
    print(f"Input length: {len(text)} characters")
    print(f"First 200 chars: {text[:200]}...")
    print()
    
    # Skip syntax fixes for now since they're causing issues
    print("=== EXTRACTING JSON ARRAY (without syntax fixes) ===")
    extracted_json = _extract_json_array(text)
    
    if extracted_json:
        print("✅ JSON extraction successful!")
        print(f"Extracted JSON length: {len(extracted_json)} characters")
        
        # Try to parse it
        try:
            parsed = json.loads(extracted_json)
            print(f"✅ JSON parsing successful!")
            print(f"Number of shots: {len(parsed)}")
            
            # Show first few shots
            print("\n=== FIRST FEW SHOTS ===")
            for i, shot in enumerate(parsed[:3]):
                print(f"Shot {i+1}: {shot}")
            
            # Show last shot
            if len(parsed) > 3:
                print(f"\n=== LAST SHOT ===")
                print(f"Shot {len(parsed)}: {parsed[-1]}")
            
            # Save cleaned JSON
            output_file = input_path.with_suffix('.cleaned.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=2)
            print(f"\n✅ Cleaned JSON saved to: {output_file}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Extracted JSON (first 500 chars): {extracted_json[:500]}...")
    else:
        print("❌ JSON extraction failed!")
        print("No valid JSON array found in the text.")
        
        # Let's see what the issue might be
        print("\n=== DEBUGGING INFO ===")
        print(f"Text starts with '[': {text.strip().startswith('[')}")
        print(f"Text ends with ']': {text.strip().endswith(']')}")
        print(f"Bracket count: [ = {text.count('[')} ] = {text.count(']')}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_json_extraction.py <input_file>")
        print("Example: python test_json_extraction.py storyboard_output.txt")
        sys.exit(1)
    
    test_json_extraction(sys.argv[1])