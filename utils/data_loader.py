"""
Data Loading Utilities for Knowledge Tracing

Handles loading and preprocessing of MOOCRadar dataset including:
- Problem data (questions, concepts, text content)
- Student interaction logs
- Creating train/val/test splits
- Building sequences for the embedding model
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .concept_mapping import ConceptMapper, QuestionMapper


class ProblemDatabase:
    """
    Database for problem information.
    
    Stores:
    - Problem text content (Chinese)
    - Problem concepts
    - Problem metadata
    """
    
    def __init__(self):
        self.problems: Dict[str, Dict] = {}
        self.problem_concepts: Dict[str, List[str]] = {}
        self.problem_text: Dict[str, str] = {}
    
    def load_from_file(self, filepath: str) -> 'ProblemDatabase':
        """
        Load problem data from JSONL file.
        
        Args:
            filepath: Path to problem.json
            
        Returns:
            self (for chaining)
        """
        print(f"  → Loading problems from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                problem = json.loads(line)
                pid = problem.get('problem_id', '')
                
                if pid:
                    self.problems[pid] = problem
                    
                    # Extract concepts
                    concepts = problem.get('concepts', [])
                    if isinstance(concepts, str):
                        concepts = [concepts]
                    self.problem_concepts[pid] = concepts
                    
                    # Extract text content
                    detail = problem.get('detail', '')
                    if isinstance(detail, str):
                        try:
                            detail = json.loads(detail.replace("'", '"'))
                        except:
                            pass
                    
                    if isinstance(detail, dict):
                        text = detail.get('content', '')
                        options = detail.get('option', {})
                        if options:
                            for key, value in options.items():
                                text += f" {key}: {value}"
                    else:
                        text = str(detail)
                    
                    self.problem_text[pid] = text
        
        print(f"  → Loaded {len(self.problems)} problems")
        return self
    
    def get_problem(self, problem_id: str) -> Optional[Dict]:
        """Get problem data by ID."""
        return self.problems.get(problem_id)
    
    def get_concepts(self, problem_id: str) -> List[str]:
        """Get concepts for a problem."""
        return self.problem_concepts.get(problem_id, [])
    
    def get_text(self, problem_id: str) -> str:
        """Get text content for a problem."""
        return self.problem_text.get(problem_id, "")
    
    def get_concept_description(self, concept_name: str) -> str:
        """
        Get a description for a concept.
        
        For now, we just return the concept name itself.
        Could be extended with a concept description database.
        """
        return concept_name
    
    @property
    def num_problems(self) -> int:
        return len(self.problems)


class StudentInteractions:
    """
    Stores student interaction logs.
    
    Groups interactions by student for sequence construction.
    """
    
    def __init__(self):
        self.interactions: List[Dict] = []
        self.student_interactions: Dict[str, List[Dict]] = defaultdict(list)
        self.problem_database: Optional[ProblemDatabase] = None
    
    def load_from_file(self, filepath: str) -> 'StudentInteractions':
        """
        Load interaction data from JSON file.
        
        Args:
            filepath: Path to student-problem-coarse-flattened.json
            
        Returns:
            self (for chaining)
        """
        print(f"  → Loading interactions from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.interactions = data
        else:
            self.interactions = [data]
        
        # Group by student
        for interaction in self.interactions:
            user_id = interaction.get('user_id', '')
            if user_id:
                self.student_interactions[user_id].append(interaction)
        
        # Sort each student's interactions by timestamp if available
        for user_id in self.student_interactions:
            self.student_interactions[user_id].sort(
                key=lambda x: x.get('timestamp', x.get('log_id', ''))
            )
        
        print(f"  → Loaded {len(self.interactions)} interactions from {len(self.student_interactions)} students")
        return self
    
    def set_problem_database(self, problem_db: ProblemDatabase):
        """Set the problem database for concept lookup."""
        self.problem_database = problem_db
    
    def get_student_sequence(
        self,
        user_id: str,
        max_length: Optional[int] = None
    ) -> List[Dict]:
        """
        Get interaction sequence for a student.
        
        Args:
            user_id: Student ID
            max_length: Maximum sequence length
            
        Returns:
            List of interaction dictionaries
        """
        interactions = self.student_interactions.get(user_id, [])
        
        if max_length and len(interactions) > max_length:
            interactions = interactions[-max_length:]
        
        return interactions
    
    def get_all_students(self) -> List[str]:
        """Get list of all student IDs."""
        return list(self.student_interactions.keys())
    
    @property
    def num_interactions(self) -> int:
        return len(self.interactions)
    
    @property
    def num_students(self) -> int:
        return len(self.student_interactions)


class KTDataset(Dataset):
    """
    PyTorch Dataset for Knowledge Tracing.
    
    Each sample contains:
    - History of questions, concepts, and responses
    - Target question and concept
    - Target response (label)
    - Text content for context encoding
    """
    
    def __init__(
        self,
        student_interactions: StudentInteractions,
        problem_database: ProblemDatabase,
        question_mapper: QuestionMapper,
        concept_mapper: ConceptMapper,
        max_seq_len: int = 200,
        min_seq_len: int = 5,
        train: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            student_interactions: Student interaction data
            problem_database: Problem information database
            question_mapper: Question ID mapper
            concept_mapper: Concept name mapper
            max_seq_len: Maximum sequence length
            min_seq_len: Minimum sequence length
            train: Whether this is training data
        """
        self.interactions = student_interactions
        self.problems = problem_database
        self.question_mapper = question_mapper
        self.concept_mapper = concept_mapper
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.train = train
        
        # Build samples
        self.samples = self._build_samples()
        print(f"  → Created {len(self.samples)} samples (train={train})")
    
    def _build_samples(self) -> List[Dict]:
        """
        Build training samples from student sequences.
        
        Each sample predicts one interaction given history.
        """
        samples = []
        
        for user_id in self.interactions.get_all_students():
            user_interactions = self.interactions.get_student_sequence(user_id)
            
            if len(user_interactions) < self.min_seq_len:
                continue
            
            # Create samples for each position (after min_seq_len)
            for i in range(self.min_seq_len, len(user_interactions)):
                # History is everything before position i
                history = user_interactions[:i]
                target = user_interactions[i]
                
                # Limit history length
                if len(history) > self.max_seq_len:
                    history = history[-self.max_seq_len:]
                
                sample = {
                    'user_id': user_id,
                    'history': history,
                    'target': target,
                    'position': i
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample.
        
        Returns:
            Dictionary with:
            - question_ids: History question IDs [seq_len]
            - concept_ids: History concept IDs [seq_len]
            - responses: History responses [seq_len]
            - mask: Padding mask [seq_len]
            - target_question_id: Target question ID
            - target_concept_ids: Target concept IDs
            - target_response: Target response (label)
            - target_question_text: Question text for context encoding
            - target_concept_texts: Concept texts for context encoding
        """
        sample = self.samples[idx]
        history = sample['history']
        target = sample['target']
        
        # Process history
        question_ids = []
        concept_ids = []
        responses = []
        
        for interaction in history:
            pid = interaction.get('problem_id', '')
            
            # Get numeric question ID
            qid = self.question_mapper.get_id(pid, add_if_missing=False)
            question_ids.append(qid if qid != -1 else 0)
            
            # Get concept IDs (use first concept)
            concepts = self.problems.get_concepts(pid)
            if concepts:
                cid = self.concept_mapper.get_id(concepts[0], add_if_missing=False)
                concept_ids.append(cid if cid != -1 else 0)
            else:
                concept_ids.append(0)
            
            # Get response
            response = interaction.get('score', interaction.get('correct', 0))
            if isinstance(response, float):
                response = 1 if response > 0.5 else 0
            responses.append(response)
        
        # Pad sequences
        seq_len = len(question_ids)
        pad_len = self.max_seq_len - seq_len
        
        if pad_len > 0:
            question_ids = [0] * pad_len + question_ids
            concept_ids = [0] * pad_len + concept_ids
            responses = [0] * pad_len + responses
            mask = [0] * pad_len + [1] * seq_len
        else:
            question_ids = question_ids[-self.max_seq_len:]
            concept_ids = concept_ids[-self.max_seq_len:]
            responses = responses[-self.max_seq_len:]
            mask = [1] * self.max_seq_len
        
        # Process target
        target_pid = target.get('problem_id', '')
        target_qid = self.question_mapper.get_id(target_pid, add_if_missing=False)
        target_qid = target_qid if target_qid != -1 else 0
        
        target_concepts = self.problems.get_concepts(target_pid)
        target_concept_ids = []
        target_concept_texts = []
        for concept in target_concepts[:3]:  # Limit to 3 concepts
            cid = self.concept_mapper.get_id(concept, add_if_missing=False)
            target_concept_ids.append(cid if cid != -1 else 0)
            target_concept_texts.append(self.problems.get_concept_description(concept))
        
        # Pad concept IDs
        while len(target_concept_ids) < 3:
            target_concept_ids.append(0)
            target_concept_texts.append("")
        
        target_response = target.get('score', target.get('correct', 0))
        if isinstance(target_response, float):
            target_response = 1 if target_response > 0.5 else 0
        
        target_question_text = self.problems.get_text(target_pid)
        
        return {
            'question_ids': torch.tensor(question_ids, dtype=torch.long),
            'concept_ids': torch.tensor(concept_ids, dtype=torch.long),
            'responses': torch.tensor(responses, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target_question_id': target_qid,
            'target_question_id_str': target_pid,
            'target_concept_ids': torch.tensor(target_concept_ids, dtype=torch.long),
            'target_response': target_response,
            'target_question_text': target_question_text,
            'target_concept_texts': target_concept_texts,
            'user_id': sample['user_id']
        }


def create_data_loaders(
    problems_path: str,
    interactions_path: str,
    mappings_dir: str,
    batch_size: int = 32,
    max_seq_len: int = 200,
    min_seq_len: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        problems_path: Path to problem.json
        interactions_path: Path to student-problem-coarse-flattened.json
        mappings_dir: Directory with saved mappings
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        min_seq_len: Minimum sequence length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        num_workers: Number of data loader workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata)
    """
    print("\n[Creating Data Loaders]")
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load problem database
    problem_db = ProblemDatabase().load_from_file(problems_path)
    
    # Load interactions
    interactions = StudentInteractions().load_from_file(interactions_path)
    interactions.set_problem_database(problem_db)
    
    # Load or build mappings
    from .concept_mapping import load_mappings, build_mappings_from_dataset
    
    try:
        question_mapper, concept_mapper = load_mappings(mappings_dir)
    except FileNotFoundError:
        question_mapper, concept_mapper = build_mappings_from_dataset(
            problems_path, mappings_dir
        )
    
    # Split students into train/val/test
    all_students = interactions.get_all_students()
    random.shuffle(all_students)
    
    n_train = int(len(all_students) * train_ratio)
    n_val = int(len(all_students) * val_ratio)
    
    train_students = set(all_students[:n_train])
    val_students = set(all_students[n_train:n_train + n_val])
    test_students = set(all_students[n_train + n_val:])
    
    print(f"  → Split: Train={len(train_students)}, Val={len(val_students)}, Test={len(test_students)}")
    
    # Create filtered interactions for each split
    def filter_interactions(student_set):
        filtered = StudentInteractions()
        for user_id in student_set:
            filtered.student_interactions[user_id] = interactions.student_interactions[user_id]
        filtered.problem_database = problem_db
        return filtered
    
    train_interactions = filter_interactions(train_students)
    val_interactions = filter_interactions(val_students)
    test_interactions = filter_interactions(test_students)
    
    # Create datasets
    train_dataset = KTDataset(
        train_interactions, problem_db, question_mapper, concept_mapper,
        max_seq_len, min_seq_len, train=True
    )
    val_dataset = KTDataset(
        val_interactions, problem_db, question_mapper, concept_mapper,
        max_seq_len, min_seq_len, train=False
    )
    test_dataset = KTDataset(
        test_interactions, problem_db, question_mapper, concept_mapper,
        max_seq_len, min_seq_len, train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    metadata = {
        'num_questions': question_mapper.vocab_size,
        'num_concepts': concept_mapper.vocab_size,
        'problem_database': problem_db,
        'question_mapper': question_mapper,
        'concept_mapper': concept_mapper
    }
    
    return train_loader, val_loader, test_loader, metadata


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Handles batching of variable-length sequences and text data.
    """
    # Stack tensor fields
    question_ids = torch.stack([b['question_ids'] for b in batch])
    concept_ids = torch.stack([b['concept_ids'] for b in batch])
    responses = torch.stack([b['responses'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    target_concept_ids = torch.stack([b['target_concept_ids'] for b in batch])
    
    # Gather scalar fields
    target_question_ids = [b['target_question_id'] for b in batch]
    target_responses = [b['target_response'] for b in batch]
    
    # Gather text fields
    target_question_texts = [b['target_question_text'] for b in batch]
    target_concept_texts = [b['target_concept_texts'] for b in batch]
    user_ids = [b['user_id'] for b in batch]
    target_question_ids_str = [b['target_question_id_str'] for b in batch]
    
    return {
        'question_ids': question_ids,
        'concept_ids': concept_ids,
        'responses': responses,
        'mask': masks,
        'target_question_id': torch.tensor(target_question_ids, dtype=torch.long),
        'target_question_id_str': target_question_ids_str,
        'target_concept_ids': target_concept_ids,
        'target_response': torch.tensor(target_responses, dtype=torch.long),
        'target_question_text': target_question_texts,
        'target_concept_texts': target_concept_texts,
        'user_id': user_ids
    }


class ProcessedKTDataset(Dataset):
    """
    PyTorch Dataset for pre-processed Knowledge Tracing data.
    
    Loads samples from .pt files created by prepare_data.py.
    This is faster than processing raw JSON data on-the-fly.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        max_seq_len: int = 200
    ):
        """
        Initialize from pre-processed samples.
        
        Args:
            samples: List of sample dictionaries from .pt file
            max_seq_len: Maximum sequence length for padding
        """
        self.samples = samples
        self.max_seq_len = max_seq_len
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample with proper padding."""
        sample = self.samples[idx]
        
        # Get sequences
        question_ids = sample['question_ids']
        concept_ids = sample['concept_ids']
        responses = sample['responses']
        seq_len = sample['seq_len']
        
        # Pad sequences
        pad_len = self.max_seq_len - seq_len
        
        if pad_len > 0:
            question_ids = [0] * pad_len + question_ids
            concept_ids = [0] * pad_len + concept_ids
            responses = [0] * pad_len + responses
            mask = [0] * pad_len + [1] * seq_len
        else:
            question_ids = question_ids[-self.max_seq_len:]
            concept_ids = concept_ids[-self.max_seq_len:]
            responses = responses[-self.max_seq_len:]
            mask = [1] * self.max_seq_len
        
        return {
            'question_ids': torch.tensor(question_ids, dtype=torch.long),
            'concept_ids': torch.tensor(concept_ids, dtype=torch.long),
            'responses': torch.tensor(responses, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target_question_id': sample['target_question_id'],
            'target_question_id_str': sample['target_question_id_str'],
            'target_concept_ids': torch.tensor(sample['target_concept_ids'], dtype=torch.long),
            'target_response': sample['target_response'],
            'target_question_text': sample['target_question_text'],
            'target_concept_texts': sample['target_concept_texts'],
            'user_id': sample['user_id']
        }


def load_processed_data(processed_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Load pre-processed data from directory.
    
    Args:
        processed_dir: Directory containing train.pt, val.pt, test.pt, metadata.json
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples, metadata)
    """
    import json
    
    print(f"  → Loading processed data from {processed_dir}")
    
    train_samples = torch.load(os.path.join(processed_dir, "train.pt"))
    val_samples = torch.load(os.path.join(processed_dir, "val.pt"))
    test_samples = torch.load(os.path.join(processed_dir, "test.pt"))
    
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    print(f"  → Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
    
    return train_samples, val_samples, test_samples, metadata


def create_data_loaders_from_processed(
    processed_dir: str,
    batch_size: int = 32,
    max_seq_len: int = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create data loaders from pre-processed data.
    
    Args:
        processed_dir: Directory with processed .pt files
        batch_size: Batch size
        max_seq_len: Maximum sequence length (uses metadata default if None)
        num_workers: Number of data loader workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata)
    """
    print("\n[Loading Pre-processed Data]")
    
    train_samples, val_samples, test_samples, metadata = load_processed_data(processed_dir)
    
    # Use max_seq_len from metadata if not specified
    if max_seq_len is None:
        max_seq_len = metadata.get('max_seq_len', 200)
    
    # Load mappings
    mappings_dir = os.path.join(processed_dir, "mappings")
    from .concept_mapping import load_mappings
    question_mapper, concept_mapper = load_mappings(mappings_dir)
    
    # Create datasets
    train_dataset = ProcessedKTDataset(train_samples, max_seq_len)
    val_dataset = ProcessedKTDataset(val_samples, max_seq_len)
    test_dataset = ProcessedKTDataset(test_samples, max_seq_len)
    
    print(f"  → Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Combine metadata with mappers
    metadata['question_mapper'] = question_mapper
    metadata['concept_mapper'] = concept_mapper
    
    return train_loader, val_loader, test_loader, metadata
