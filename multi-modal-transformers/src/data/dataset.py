import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch.utils.data import Dataset
import logging


class MultiModalDataset(Dataset):
    """
    Base class for multi-modal datasets.
    Dispatches to task-specific dataset implementations.
    """
    
    def __init__(
        self,
        data_path: str,
        task: str = "retrieval",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_length: int = 512,
        image_size: int = 224
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_path: Path to data directory or file
            task: Task type (retrieval, vqa, captioning)
            split: Data split (train, val, test)
            transform: Image transformation function
            max_length: Maximum sequence length for text
            image_size: Size of images (height and width)
        """
        self.data_path = data_path
        self.task = task
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.image_size = image_size
        
        # Create task-specific dataset
        if task == "retrieval":
            self.dataset = ImageTextDataset(
                data_path=data_path,
                split=split,
                transform=transform,
                max_length=max_length,
                image_size=image_size
            )
        elif task == "vqa":
            self.dataset = VQADataset(
                data_path=data_path,
                split=split,
                transform=transform,
                max_length=max_length,
                image_size=image_size
            )
        elif task == "captioning":
            self.dataset = CaptioningDataset(
                data_path=data_path,
                split=split,
                transform=transform,
                max_length=max_length,
                image_size=image_size
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        return self.dataset[idx]


class ImageTextDataset(Dataset):
    """
    Dataset for image-text retrieval tasks.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_length: int = 512,
        image_size: int = 224
    ):
        """
        Initialize image-text dataset.
        
        Args:
            data_path: Path to data directory or file
            split: Data split (train, val, test)
            transform: Image transformation function
            max_length: Maximum sequence length for text
            image_size: Size of images (height and width)
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.image_size = image_size
        
        # Load data
        self.data = self._load_data()
        
        # Log dataset information
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(self.data)} image-text pairs for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Returns:
            List of data samples
        """
        # Determine file path based on split
        if os.path.isdir(self.data_path):
            file_path = os.path.join(self.data_path, f"{self.split}.json")
        else:
            file_path = self.data_path
        
        # Load data from file
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformation if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = image.resize((self.image_size, self.image_size))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - image: Image tensor [3, H, W]
                - text_input_ids: Text token IDs [seq_len]
                - text_attention_mask: Text attention mask [seq_len]
        """
        item = self.data[idx]
        
        # Load image
        image_path = item["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.data_path), image_path)
        
        image = self._load_image(image_path)
        
        # Process text
        text = item["text"]
        
        # In a real implementation, we would tokenize the text here
        # For this example, we'll just create dummy token IDs
        text_input_ids = torch.randint(0, 1000, (self.max_length,))
        text_attention_mask = torch.ones_like(text_input_ids)
        
        # Add random padding to text
        padding_length = np.random.randint(0, self.max_length // 4)
        if padding_length > 0:
            text_attention_mask[-padding_length:] = 0
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text": text  # Include raw text for debugging
        }


class VQADataset(Dataset):
    """
    Dataset for visual question answering tasks.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_length: int = 512,
        image_size: int = 224,
        answer_vocab_path: Optional[str] = None
    ):
        """
        Initialize VQA dataset.
        
        Args:
            data_path: Path to data directory or file
            split: Data split (train, val, test)
            transform: Image transformation function
            max_length: Maximum sequence length for text
            image_size: Size of images (height and width)
            answer_vocab_path: Path to answer vocabulary file
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.image_size = image_size
        
        # Load data
        self.data = self._load_data()
        
        # Load answer vocabulary
        self.answer_vocab = self._load_answer_vocab(answer_vocab_path)
        
        # Log dataset information
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(self.data)} VQA samples for {split} split")
        logger.info(f"Answer vocabulary size: {len(self.answer_vocab)}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Returns:
            List of data samples
        """
        # Determine file path based on split
        if os.path.isdir(self.data_path):
            file_path = os.path.join(self.data_path, f"{self.split}.json")
        else:
            file_path = self.data_path
        
        # Load data from file
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return data
    
    def _load_answer_vocab(self, answer_vocab_path: Optional[str]) -> Dict[str, int]:
        """
        Load answer vocabulary.
        
        Args:
            answer_vocab_path: Path to answer vocabulary file
            
        Returns:
            Dictionary mapping answers to indices
        """
        if answer_vocab_path is None:
            # If no vocabulary file is provided, create one from the data
            answers = set()
            for item in self.data:
                if "answer" in item:
                    answers.add(item["answer"])
            
            # Create vocabulary
            answer_vocab = {answer: idx for idx, answer in enumerate(sorted(answers))}
        else:
            # Load vocabulary from file
            with open(answer_vocab_path, "r") as f:
                answer_vocab = json.load(f)
        
        return answer_vocab
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformation if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = image.resize((self.image_size, self.image_size))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - image: Image tensor [3, H, W]
                - question_input_ids: Question token IDs [seq_len]
                - question_attention_mask: Question attention mask [seq_len]
                - answers: Answer index
        """
        item = self.data[idx]
        
        # Load image
        image_path = item["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.data_path), image_path)
        
        image = self._load_image(image_path)
        
        # Process question
        question = item["question"]
        
        # In a real implementation, we would tokenize the question here
        # For this example, we'll just create dummy token IDs
        question_input_ids = torch.randint(0, 1000, (self.max_length,))
        question_attention_mask = torch.ones_like(question_input_ids)
        
        # Add random padding to question
        padding_length = np.random.randint(0, self.max_length // 4)
        if padding_length > 0:
            question_attention_mask[-padding_length:] = 0
        
        # Process answer
        if "answer" in item and self.split != "test":
            answer = item["answer"]
            answer_idx = self.answer_vocab.get(answer, 0)
            answers = torch.tensor(answer_idx, dtype=torch.long)
        else:
            # For test split, we don't have answers
            answers = torch.tensor(0, dtype=torch.long)
        
        return {
            "image": image,
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "answers": answers,
            "question": question,  # Include raw question for debugging
            "answer": item.get("answer", "")  # Include raw answer for debugging
        }


class CaptioningDataset(Dataset):
    """
    Dataset for image captioning tasks.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_length: int = 512,
        image_size: int = 224
    ):
        """
        Initialize captioning dataset.
        
        Args:
            data_path: Path to data directory or file
            split: Data split (train, val, test)
            transform: Image transformation function
            max_length: Maximum sequence length for text
            image_size: Size of images (height and width)
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.image_size = image_size
        
        # Load data
        self.data = self._load_data()
        
        # Log dataset information
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(self.data)} captioning samples for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Returns:
            List of data samples
        """
        # Determine file path based on split
        if os.path.isdir(self.data_path):
            file_path = os.path.join(self.data_path, f"{self.split}.json")
        else:
            file_path = self.data_path
        
        # Load data from file
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformation if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = image.resize((self.image_size, self.image_size))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - image: Image tensor [3, H, W]
                - caption_ids: Caption token IDs [seq_len]
                - caption_attention_mask: Caption attention mask [seq_len]
        """
        item = self.data[idx]
        
        # Load image
        image_path = item["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.data_path), image_path)
        
        image = self._load_image(image_path)
        
        # Process caption
        caption = item["caption"]
        
        # In a real implementation, we would tokenize the caption here
        # For this example, we'll just create dummy token IDs
        caption_ids = torch.randint(0, 1000, (self.max_length,))
        caption_attention_mask = torch.ones_like(caption_ids)
        
        # Add random padding to caption
        padding_length = np.random.randint(0, self.max_length // 4)
        if padding_length > 0:
            caption_attention_mask[-padding_length:] = 0
        
        return {
            "image": image,
            "caption_ids": caption_ids,
            "caption_attention_mask": caption_attention_mask,
            "caption": caption  # Include raw caption for debugging
        } 