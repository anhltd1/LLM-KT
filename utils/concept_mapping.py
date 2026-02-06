"""
Concept Mapping Utilities

Maps concept names (especially Chinese) to numeric IDs.
Since the MOOCRadar dataset has concept names as text, not IDs,
we need to build a mapping from concept names to integer IDs.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pickle


class ConceptMapper:
    """
    Maps concept names to numeric IDs and vice versa.
    
    Supports:
    - Chinese concept names (MOOCRadar dataset)
    - Building vocabulary from dataset
    - Saving/loading mappings for consistency
    """
    
    def __init__(self, start_id: int = 1):
        """
        Initialize the ConceptMapper.
        
        Args:
            start_id: Starting ID for concepts (default 1, 0 reserved for padding)
        """
        self.start_id = start_id
        self.concept_to_id: Dict[str, int] = {}
        self.id_to_concept: Dict[int, str] = {}
        self.concept_count: Dict[str, int] = defaultdict(int)
        self.next_id = start_id
        
        # Special tokens
        self.PAD_ID = 0
        self.UNK_ID = -1
    
    def add_concept(self, concept_name: str) -> int:
        """
        Add a concept and get its ID.
        
        Args:
            concept_name: The concept name (can be Chinese)
            
        Returns:
            The numeric ID for this concept
        """
        # Normalize the concept name
        concept_name = concept_name.strip()
        
        if not concept_name:
            return self.PAD_ID
        
        if concept_name not in self.concept_to_id:
            self.concept_to_id[concept_name] = self.next_id
            self.id_to_concept[self.next_id] = concept_name
            self.next_id += 1
        
        self.concept_count[concept_name] += 1
        return self.concept_to_id[concept_name]
    
    def get_id(self, concept_name: str, add_if_missing: bool = True) -> int:
        """
        Get the ID for a concept name.
        
        Args:
            concept_name: The concept name
            add_if_missing: Whether to add the concept if not found
            
        Returns:
            The concept ID
        """
        concept_name = concept_name.strip()
        
        if not concept_name:
            return self.PAD_ID
        
        if concept_name in self.concept_to_id:
            return self.concept_to_id[concept_name]
        elif add_if_missing:
            return self.add_concept(concept_name)
        else:
            return self.UNK_ID
    
    def get_name(self, concept_id: int) -> str:
        """
        Get the concept name for an ID.
        
        Args:
            concept_id: The concept ID
            
        Returns:
            The concept name, or empty string if not found
        """
        if concept_id == self.PAD_ID:
            return ""
        return self.id_to_concept.get(concept_id, "")
    
    def get_ids(self, concept_names: List[str], add_if_missing: bool = True) -> List[int]:
        """
        Get IDs for a list of concept names.
        
        Args:
            concept_names: List of concept names
            add_if_missing: Whether to add missing concepts
            
        Returns:
            List of concept IDs
        """
        return [self.get_id(name, add_if_missing) for name in concept_names]
    
    def get_names(self, concept_ids: List[int]) -> List[str]:
        """
        Get concept names for a list of IDs.
        
        Args:
            concept_ids: List of concept IDs
            
        Returns:
            List of concept names
        """
        return [self.get_name(cid) for cid in concept_ids]
    
    @property
    def num_concepts(self) -> int:
        """Get the total number of concepts."""
        return len(self.concept_to_id)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including padding)."""
        return self.next_id
    
    def build_from_problems(self, problems: List[Dict]) -> 'ConceptMapper':
        """
        Build concept vocabulary from problem data.
        
        Args:
            problems: List of problem dictionaries, each with 'concepts' field
            
        Returns:
            self (for chaining)
        """
        print(f"  → Building concept vocabulary from {len(problems)} problems...")
        
        for problem in problems:
            concepts = problem.get('concepts', [])
            if isinstance(concepts, str):
                # Handle case where concepts is a string
                concepts = [concepts]
            
            for concept in concepts:
                if concept:
                    self.add_concept(concept)
        
        print(f"  → Found {self.num_concepts} unique concepts")
        return self
    
    def save(self, filepath: str):
        """
        Save the mapping to a file.
        
        Args:
            filepath: Path to save the mapping
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'concept_to_id': self.concept_to_id,
            'id_to_concept': self.id_to_concept,
            'concept_count': dict(self.concept_count),
            'next_id': self.next_id,
            'start_id': self.start_id
        }
        
        # Save as pickle for efficiency with Chinese characters
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  → Saved concept mapping ({self.num_concepts} concepts) to {filepath}")
    
    def load(self, filepath: str) -> 'ConceptMapper':
        """
        Load the mapping from a file.
        
        Args:
            filepath: Path to load the mapping from
            
        Returns:
            self (for chaining)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Concept mapping not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.concept_to_id = data['concept_to_id']
        self.id_to_concept = data['id_to_concept']
        self.concept_count = defaultdict(int, data.get('concept_count', {}))
        self.next_id = data['next_id']
        self.start_id = data.get('start_id', 1)
        
        print(f"  → Loaded concept mapping ({self.num_concepts} concepts) from {filepath}")
        return self
    
    def export_to_json(self, filepath: str):
        """
        Export the mapping to a JSON file (for inspection).
        
        Args:
            filepath: Path to save the JSON
        """
        data = {
            'num_concepts': self.num_concepts,
            'concepts': [
                {
                    'id': cid,
                    'name': name,
                    'count': self.concept_count.get(name, 0)
                }
                for name, cid in sorted(self.concept_to_id.items(), key=lambda x: x[1])
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  → Exported concept mapping to {filepath}")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the concept vocabulary.
        
        Returns:
            Dictionary of statistics
        """
        counts = list(self.concept_count.values())
        
        return {
            'num_concepts': self.num_concepts,
            'total_occurrences': sum(counts),
            'min_count': min(counts) if counts else 0,
            'max_count': max(counts) if counts else 0,
            'avg_count': sum(counts) / len(counts) if counts else 0,
            'top_concepts': sorted(
                self.concept_count.items(), 
                key=lambda x: -x[1]
            )[:10]
        }


class QuestionMapper:
    """
    Maps question IDs (string format like "Pm_2046133") to numeric IDs.
    
    Similar to ConceptMapper but for questions.
    """
    
    def __init__(self, start_id: int = 1):
        """
        Initialize the QuestionMapper.
        
        Args:
            start_id: Starting ID for questions (default 1, 0 reserved for padding)
        """
        self.start_id = start_id
        self.question_to_id: Dict[str, int] = {}
        self.id_to_question: Dict[int, str] = {}
        self.next_id = start_id
        
        self.PAD_ID = 0
        self.UNK_ID = -1
    
    def add_question(self, question_id: str) -> int:
        """
        Add a question and get its numeric ID.
        
        Args:
            question_id: The original question ID (e.g., "Pm_2046133")
            
        Returns:
            The numeric ID
        """
        question_id = question_id.strip()
        
        if not question_id:
            return self.PAD_ID
        
        if question_id not in self.question_to_id:
            self.question_to_id[question_id] = self.next_id
            self.id_to_question[self.next_id] = question_id
            self.next_id += 1
        
        return self.question_to_id[question_id]
    
    def get_id(self, question_id: str, add_if_missing: bool = True) -> int:
        """
        Get the numeric ID for a question ID string.
        
        Args:
            question_id: The original question ID
            add_if_missing: Whether to add if not found
            
        Returns:
            The numeric ID
        """
        question_id = question_id.strip()
        
        if not question_id:
            return self.PAD_ID
        
        if question_id in self.question_to_id:
            return self.question_to_id[question_id]
        elif add_if_missing:
            return self.add_question(question_id)
        else:
            return self.UNK_ID
    
    def get_original_id(self, numeric_id: int) -> str:
        """
        Get the original question ID for a numeric ID.
        
        Args:
            numeric_id: The numeric ID
            
        Returns:
            The original question ID string
        """
        if numeric_id == self.PAD_ID:
            return ""
        return self.id_to_question.get(numeric_id, "")
    
    @property
    def num_questions(self) -> int:
        """Get the total number of questions."""
        return len(self.question_to_id)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including padding)."""
        return self.next_id
    
    def build_from_problems(self, problems: List[Dict]) -> 'QuestionMapper':
        """
        Build question vocabulary from problem data.
        
        Args:
            problems: List of problem dictionaries with 'problem_id' field
            
        Returns:
            self (for chaining)
        """
        print(f"  → Building question vocabulary from {len(problems)} problems...")
        
        for problem in problems:
            qid = problem.get('problem_id', '')
            if qid:
                self.add_question(qid)
        
        print(f"  → Found {self.num_questions} unique questions")
        return self
    
    def save(self, filepath: str):
        """Save the mapping to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'question_to_id': self.question_to_id,
            'id_to_question': self.id_to_question,
            'next_id': self.next_id,
            'start_id': self.start_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  → Saved question mapping ({self.num_questions} questions) to {filepath}")
    
    def load(self, filepath: str) -> 'QuestionMapper':
        """Load the mapping from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Question mapping not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.question_to_id = data['question_to_id']
        self.id_to_question = data['id_to_question']
        self.next_id = data['next_id']
        self.start_id = data.get('start_id', 1)
        
        print(f"  → Loaded question mapping ({self.num_questions} questions) from {filepath}")
        return self


def build_mappings_from_dataset(
    problems_path: str,
    save_dir: str = "./mappings"
) -> Tuple[QuestionMapper, ConceptMapper]:
    """
    Build question and concept mappings from the dataset.
    
    Args:
        problems_path: Path to problem.json file
        save_dir: Directory to save mappings
        
    Returns:
        Tuple of (QuestionMapper, ConceptMapper)
    """
    print("\n[Building Mappings from Dataset]")
    
    # Load problems (JSONL format)
    problems = []
    with open(problems_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    
    print(f"  → Loaded {len(problems)} problems")
    
    # Build mappers
    question_mapper = QuestionMapper().build_from_problems(problems)
    concept_mapper = ConceptMapper().build_from_problems(problems)
    
    # Save mappings
    os.makedirs(save_dir, exist_ok=True)
    question_mapper.save(os.path.join(save_dir, "question_mapping.pkl"))
    concept_mapper.save(os.path.join(save_dir, "concept_mapping.pkl"))
    
    # Export JSON for inspection
    concept_mapper.export_to_json(os.path.join(save_dir, "concepts.json"))
    
    print(f"\n  → Mapping Summary:")
    print(f"     Questions: {question_mapper.num_questions}")
    print(f"     Concepts:  {concept_mapper.num_concepts}")
    
    return question_mapper, concept_mapper


def load_mappings(
    save_dir: str = "./mappings"
) -> Tuple[QuestionMapper, ConceptMapper]:
    """
    Load pre-built mappings.
    
    Args:
        save_dir: Directory containing saved mappings
        
    Returns:
        Tuple of (QuestionMapper, ConceptMapper)
    """
    question_mapper = QuestionMapper().load(os.path.join(save_dir, "question_mapping.pkl"))
    concept_mapper = ConceptMapper().load(os.path.join(save_dir, "concept_mapping.pkl"))
    
    return question_mapper, concept_mapper
