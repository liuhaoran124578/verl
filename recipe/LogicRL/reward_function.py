import re
from typing import Any, Dict, List, Optional, Tuple


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string."""
    processed_str = solution_str
    
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid <answer> tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    
    # Handle literal \n for compatibility
    if '\\n' in final_answer and '\n' not in final_answer:
        final_answer = final_answer.replace('\\n', '\n')
        print("  [Info] Converted literal \\n in answer to real newline")
    
    return final_answer, processed_str

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary."""
    status_dict = {}
    print("\n[Ground Truth Parsing]")
    print(f"  Raw solution_text: {repr(solution_text)}")
    
    # Handle literal \n characters
    if '\\n' in solution_text and '\n' not in solution_text:
        solution_text = solution_text.replace('\\n', '\n')
        print("  [Info] Converted literal \\n to real newline")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Match "Name is a knight/knave"
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_text: str, expected_names: List[str]) -> Optional[Dict[str, str]]:
    """Parses model's answer into a status dictionary."""
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Raw answer_text: {repr(answer_text)}")
    print(f"  Expected characters: {expected_names}")
    
    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
    
    # Check if all expected names are accounted for
    if len(status_dict) != len(expected_names):
        print("  [Error] Missing or incomplete roles")
        return None
    
    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """Validates the structural requirements of the response."""
    print("\n[Structure Validation]")
    validation_passed = True

    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        print(f"  {tag_str}: count={count}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    return validation_passed


def compute_score(solution_str: str,
                  ground_truth: Dict[str, Any],
                  extra_info: Optional[Dict[str, Any]] = None,
                  **kwargs) -> float:
    """Computes the total score for model's response."""
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))

    # Parse ground truth data
    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")

    if not gt_status:
        print("[Error] Failed to parse ground truth!")
        return -3

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = 1 if format_correct else -1
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    if not format_correct or not answer_text:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")
    else:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status is None:
            answer_score = -2
            print("\n[Content Validation] FAIL - Answer cannot be parsed")
        else:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")

            # Calculate answer score based on correctness
            correct_count = sum(
                1 for name in expected_names 
                if pred_status.get(name) == gt_status.get(name)
            )
            incorrect_count = len(expected_names) - correct_count

            if incorrect_count == 0:  # All correct
                answer_score = 2
                print("  ✓ Fully correct answer")
            elif correct_count > 0:  # Partially correct
                answer_score = -1
                print(f"  ✗ Partially correct: {correct_count}/{len(expected_names)} correct")
            else:  # Completely incorrect
                answer_score = -1.5
                print("  ✗ Completely incorrect answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score:+.1f}")
    print(f"  Answer: {answer_score:+.1f}")
    print(f"  Total: {total_score:+.1f}")
    print("="*80 + "\n")

    return total_score