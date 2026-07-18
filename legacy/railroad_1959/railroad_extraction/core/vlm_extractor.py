import base64
import json
import os
from typing import List, Dict, Any
from pathlib import Path
import anthropic
from dotenv import load_dotenv
from ..schemas.railroad_rule import RailroadRule

# Load environment variables from the root .env file
root_env = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(root_env)

class VLMExtractor:
    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929"):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_rules_from_image(self, image_path: str, page_number: int = None) -> List[RailroadRule]:
        """
        Extract railroad rules from a single image using VLM.
        """
        print(f"Extracting rules from {image_path}...")
        base64_image = self._encode_image(image_path)
        
        system_prompt = """You are an expert archivist and railroad safety engineer. 
        Your task is to extract operating rules from images of the 1959 Consolidated Code of Operating Rules.
        
        Extract every rule visible on the page. Maintain strict fidelity to the text.
        If a rule has sub-parts (like A, B, C), capture them as sub_rules or within the main text if they are short.
        
        Output strictly valid JSON in the following format:
        {
            "rules": [
                {
                    "rule_id": "Rule Number (e.g., '99', '251', 'M')",
                    "text": "The full text of the rule.",
                    "category": "The section header (e.g., 'General Rules', 'Signals')",
                    "section": "Optional subsection",
                    "sub_rules": []
                }
            ]
        }
        """

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": "Extract all railroad operating rules from this page. Return strictly JSON."
                            }
                        ]
                    }
                ]
            )
            
            # print(f"DEBUG: Message object: {message}") # Commented out to avoid clutter
            if not message.content:
                print("DEBUG: No content in message")
                return []
                
            response_text = message.content[0].text
            with open("last_vlm_response.txt", "w", encoding="utf-8") as f:
                f.write(response_text)
            
            # Basic cleanup to find JSON if there's wrapper text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(response_text)
            rules = []
            
            def process_rule(rule_data, parent_category=None, parent_section=None):
                # Inherit category/section if missing
                if "category" not in rule_data and parent_category:
                    rule_data["category"] = parent_category
                if "section" not in rule_data and parent_section:
                    rule_data["section"] = parent_section
                
                # Inject page number
                if page_number:
                    rule_data["page_number"] = page_number
                
                # Process sub-rules recursively
                if "sub_rules" in rule_data and rule_data["sub_rules"]:
                    processed_subs = []
                    for sub in rule_data["sub_rules"]:
                        processed_subs.append(process_rule(sub, rule_data.get("category"), rule_data.get("section")))
                    rule_data["sub_rules"] = processed_subs
                
                return RailroadRule(**rule_data)

            for r in data.get("rules", []):
                try:
                    rules.append(process_rule(r))
                except Exception as e:
                    print(f"Skipping invalid rule {r.get('rule_id', '?')}: {e}")
                
            return rules

        except Exception as e:
            error_msg = f"Error extracting from {image_path}: {e}"
            print(error_msg)
            with open("vlm_error.txt", "w") as f:
                f.write(error_msg)
            return []

if __name__ == "__main__":
    # Test run
    extractor = VLMExtractor()
    # Find a processed image to test
    base_dir = Path(__file__).parent.parent.parent
    img_dir = base_dir / "data" / "processed_images"
    
    if img_dir.exists():
        images = list(img_dir.glob("*.png"))
        if images:
            test_image = images[40] # Pick page 41, likely to have rules
            print(f"Testing on {test_image}")
            rules = extractor.extract_rules_from_image(str(test_image), page_number=41)
            print(f"Extracted {len(rules)} rules:")
            for r in rules:
                print(f"- {r.rule_id}: {r.text[:50]}...")
