import json
import logging
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Generator
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RailroadSFTGenerator:
    def __init__(self, rules_path: str, output_path: str):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        # Use Gemini 2.5 Flash as specified in the reference implementation
        model_name = os.getenv('GEMINI_MODEL', 'gemini-3-pro-preview')
        self.model = genai.GenerativeModel(model_name)
        
        self.rules_path = Path(rules_path)
        self.output_path = Path(output_path)
        
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_path}")
            
        # Rate limiting configuration
        self.base_delay = 1.0
        self.max_retries = 5
        self.max_retry_delay = 300

    def load_rules(self) -> List[Dict]:
        """Load railroad rules from the JSON file."""
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both list and dict wrapper formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'rules' in data:
                return data['rules']
            else:
                raise ValueError("Unexpected JSON format in rules file")

    def create_context_prompt(self, rules: List[Dict]) -> str:
        """Create a context string containing the rules for the LLM."""
        context = """You are an expert Railroad Rules Examiner for the 1959 Consolidated Code of Operating Rules.
        Your task is to generate realistic, complex scenarios that test a train crew's understanding and application of specific rules.
        
        Guidelines for Scenarios:
        1. **Realism**: Use realistic terminology (e.g., "Extra 400 West", "Siding at Mill Creek", "Dispatcher", "Conductor").
        2. **Complexity**: Create situations with ambiguity, multiple factors (weather, mechanical issues), or conflicting priorities where the rule provides the clear answer.
        3. **Format**: 
           - The 'Question' should be the scenario description ending with "What is the correct action?" or similar.
           - The 'Answer' should be the precise action required, explicitly citing the rule number (e.g., "Per Rule 99, the flagman must...").
        4. **Diversity**: Vary the situations (freight vs passenger, day vs night, main track vs yard).
        
        Rules to cover in this batch:
        """
        
        for rule in rules:
            rule_text = f"Rule {rule.get('rule_id', 'Unknown')}: {rule.get('text', '')}"
            # Add sub-rules if present
            if 'sub_rules' in rule and rule['sub_rules']:
                for sub in rule['sub_rules']:
                    rule_text += f"\n  - {sub.get('rule_id', '')}: {sub.get('text', '')}"
            context += f"\n{rule_text}\n"
            
        return context

    def generate_qa_pairs(self, rules_batch: List[Dict], num_pairs: int = 5) -> Generator[Dict, None, None]:
        """Generate Q&A pairs for a batch of rules."""
        
        context = self.create_context_prompt(rules_batch)
        
        prompt = f"""Based on the rules provided above, generate {num_pairs} diverse, complex question-answer pairs.
        
        Format your response EXACTLY as a JSON array:
        [
            {{
                "question": "Scenario description...",
                "answer": "Correct action citing the rule..."
            }}
        ]
        
        Ensure the response is valid JSON. Do not include markdown formatting like ```json.
        """
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(context + "\n" + prompt)
                response_text = response.text.strip()
                
                # Clean up markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                qa_pairs = json.loads(response_text)
                
                for pair in qa_pairs:
                    if 'question' in pair and 'answer' in pair:
                        # Add metadata about which rules were used
                        pair['related_rules'] = [r.get('rule_id') for r in rules_batch]
                        yield pair
                
                time.sleep(self.base_delay)
                return

            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(min(self.base_delay * (2 ** attempt), self.max_retry_delay))
        
        logger.error("Failed to generate pairs after all retries.")

    def generate_dataset(self, target_pairs: int = 1000, batch_size: int = 3):
        """
        Generate the full synthetic dataset.
        
        Args:
            target_pairs: Total number of pairs to generate.
            batch_size: Number of rules to include in each prompt context.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rules = self.load_rules()
        logger.info(f"Loaded {len(rules)} rules.")
        
        generated_count = 0
        
        # Open in append mode to allow resuming/accumulating
        with open(self.output_path, 'a', encoding='utf-8') as f:
            
            # We loop through rules, wrapping around if needed to hit target_pairs
            pbar = tqdm(total=target_pairs, desc="Generating SFT Data")
            
            while generated_count < target_pairs:
                # Pick a random or sequential batch of rules
                # For now, let's just iterate sequentially and wrap around
                for i in range(0, len(rules), batch_size):
                    if generated_count >= target_pairs:
                        break
                        
                    batch = rules[i : i + batch_size]
                    
                    # Generate pairs for this batch
                    pairs_in_batch = 0
                    for pair in self.generate_qa_pairs(batch, num_pairs=5):
                        if generated_count >= target_pairs:
                            break
                            
                        pair['id'] = generated_count + 1
                        pair['generated_at'] = datetime.now().isoformat()
                        
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                        f.flush()
                        
                        generated_count += 1
                        pairs_in_batch += 1
                        pbar.update(1)
                    
                    if pairs_in_batch == 0:
                        logger.warning("No pairs generated for batch, skipping...")

        logger.info(f"Generation complete. Saved {generated_count} pairs to {self.output_path}")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    rules_file = base_dir / "RailroadEngineer1959" / "data" / "railroad_extracted" / "railroad_rules_complete.json"
    output_file = base_dir / "RailroadEngineer1959" / "data" / "railroad_sft_data.jsonl"
    
    generator = RailroadSFTGenerator(str(rules_file), str(output_file))
    
    # Generate 5000 pairs as requested
    generator.generate_dataset(target_pairs=5000, batch_size=3)

if __name__ == "__main__":
    main()
