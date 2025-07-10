import argparse
import json
import logging
import shutil
import cv2
import torch
from PIL import Image
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from utils import DataStatistics, DataTransformFactory, load_model

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ImageMatcher:
    """
    A class for performing match-mode inference on images.

    Attributes:
        inference_model (InferenceModel): Model for computing matches.
        mean (List[float]): Channel-wise mean for normalization.
        std (List[float]): Channel-wise std for normalization.
        unnormalize (UnNormalize): Utility to invert normalization.
        device (str): Device on which inference runs ('cpu' or 'cuda').
        save_dir (Path): Directory to save match results.
    """

    def __init__(
        self,
        model_path: str,
        model_structure: str,
        embedding_size: int,
        threshold: float,
        mean_std_file: str,
        save_dir: str,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the MatchMode inference system.

        Args:
            model_path (str): Path to the trained model weights file.
            model_structure (str): Name of the model class.
            embedding_size (int): Dimension of the embedding vector.
            faiss_index (str): Path to the FAISS index file.
            threshold (float): Matching threshold (0 < threshold < 1).
            mean_std_file (str): Path to a text file containing mean and std values.
            save_dir (str): Directory to save matched results.
            device (Optional[str]): Device for inference ('cpu' or 'cuda'). Auto-detected if None.

        Raises:
            FileNotFoundError: If mean_std_file does not exist.
        """
        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Initialize the inference model
        self.inference_model = self._create_inference_model(
            model_path,
            model_structure,
            embedding_size,
            threshold,
        )

        # Load normalization statistics
        mean_std_path = Path(mean_std_file)
        if not mean_std_path.is_file():
            raise FileNotFoundError(f"Mean/std file not found: {mean_std_file}")
        self.mean, self.std = DataStatistics.get_mean_std(mean_std_path)

        # Prepare save directory
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Results will be saved to: {self.save_dir}")

    def _create_inference_model(
        self,
        model_path: str,
        model_structure: str,
        embedding_size: int,
        threshold: float,
    ) -> InferenceModel:
        """
        Create and load an InferenceModel for match-finding.

        Args:
            model_path (str): Path to model weights.
            model_structure (str): Name of the model class to load.
            embedding_size (int): Size of the model's embedding output.
            faiss_index (str): Path to the FAISS index file.
            threshold (float): Cosine similarity threshold for a match.

        Returns:
            InferenceModel: Configured inference model.

        Raises:
            Exception: If model or index fail to load.
        """
        logging.info("Loading model and FAISS index...")
        model = load_model(model_structure, model_path, embedding_size)
        match_finder = MatchFinder(distance=CosineSimilarity(), threshold=threshold)
        inference_model = InferenceModel(model, match_finder=match_finder)
        logging.info("Model loaded successfully.")
        return inference_model

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from disk and apply the model's preprocessing pipeline.

        Args:
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: A 4D tensor of shape (1, C, H, W) for inference.

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        image_array = cv2.imread(str(image_path))
        if image_array is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        transforms = DataTransformFactory.create_transform("train", self.mean, self.std)
        tensor = transforms(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def _match_file(
        self,
        image_path: Path,
        query_tensor: torch.Tensor,
    ) -> Tuple[Path, Optional[bool]]:
        """
        Check if a single image matches the preprocessed query.

        Args:
            image_path (Path): Path to the image to test.
            query_tensor (torch.Tensor): Precomputed tensor of the query image.

        Returns:
            Tuple[Path, Optional[bool]]: The image path and match result (True, False, or None if error).
        """
        try:
            img_tensor = self._process_image(str(image_path))
            is_match = self.inference_model.is_match(img_tensor, query_tensor)
            return image_path, is_match
        except Exception as exc:
            logging.error(f"Error processing {image_path}: {exc}")
            return image_path, None
        
    def _match_file_score(
        self,
        image_path: Path,
        query_tensor: torch.Tensor,
    ) -> Tuple[Path, float]:
        """
        Get Score with a single image matches the preprocessed query.

        Args:
            image_path (Path): Path to the image to test.
            query_tensor (torch.Tensor): Precomputed tensor of the query image.

        Returns:
            Tuple[Path, float]: The image path and match result (float).
        """
        try:
            img_tensor = self._process_image(str(image_path))
            distance = CosineSimilarity().pairwise_distance(img_tensor, query_tensor)
            return image_path, distance
        except Exception as exc:
            logging.error(f"Error processing {image_path}: {exc}")
            return image_path, 0.0
    

    def run(
        self,
        data: str,
        query_image: str,
        extensions: List[str],
        max_workers: int,
    ) -> None:
        """
        Execute match-mode inference on files or directories.

        Args:
            data (str): Path to a single image or a directory of images.
            query_image (str): Path to the query image for matching.
            extensions (List[str]): List of file extensions to include (e.g., ['jpg', 'png']).
            max_workers (int): Number of threads for parallel processing.

        Side Effects:
            - Creates 'OK' and 'NG' subdirectories under save_dir.
            - Copies matched images into the appropriate folder.
            - Writes a summary JSON report at save_dir/match_summary.json.
        """
        logging.info(f"Preprocessing query image: {query_image}")
        query_tensor = self._process_image(query_image)

        data_path = Path(data)
        files: List[Path] = []
        if data_path.is_file():
            files = [data_path]
        else:
            for ext in extensions:
                files.extend(sorted(data_path.rglob(f"*.{ext.lower()}")))

        if not files:
            logging.warning("No images found to process.")
            return

        logging.info(f"Found {len(files)} files; using {max_workers} workers.")

        summary = {"OK": 0, "NG": 0, "Error": 0}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._match_file, img, query_tensor): img for img in files}
            for future in as_completed(futures):
                img_path = futures[future]
                img_name = img_path.name
                _, result = future.result()
                if result is True:
                    dest = self.save_dir / "OK"
                    summary["OK"] += 1
                elif result is False:
                    dest = self.save_dir / "NG"
                    summary["NG"] += 1
                else:
                    summary["Error"] += 1
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, dest / img_name)

        report_file = self.save_dir / "match_summary.json"
        with report_file.open("w") as rf:
            json.dump(summary, rf, indent=2)

        logging.info(f"Summary saved: OK={summary['OK']}, NG={summary['NG']}, Error={summary['Error']}.")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the MatchMode CLI.

    Returns:
        argparse.Namespace: Object containing CLI argument values.
    """
    parser = argparse.ArgumentParser(description="Match Mode CLI for image matching.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to an image file or directory of images to match",
    )
    parser.add_argument(
        "--query-image",
        required=True,
        help="Path to the query image to match against",
    )
    parser.add_argument(
        "--model-path",
        default="model.pt",
        help="Path to the trained model weights file",
    )
    parser.add_argument(
        "--model-structure",
        default="EfficientArcFaceModel",
        help="Name of the model class structure to load",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=512,
        help="Dimension size of the embedding vector",
    )
    parser.add_argument(
        "--faiss-index",
        required=True,
        help="Path to the FAISS index file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Matching threshold (0 < threshold < 1)",
    )
    parser.add_argument(
        "--mean-std-file",
        default="mean_std.txt",
        help="Path to text file containing mean and std values",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Directory to save matched results",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference ('cpu' or 'cuda'); auto-detected if absent",
    )
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png",
        help="Comma-separated list of file extensions to process",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for processing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ext_list = [e.strip().lower() for e in args.extensions.split(",") if e.strip()]
    matcher = ImageMatcher(
        model_path=args.model_path,
        model_structure=args.model_structure,
        embedding_size=args.embedding_size,
        faiss_index=args.faiss_index,
        threshold=args.threshold,
        mean_std_file=args.mean_std_file,
        save_dir=args.save_dir,
        device=args.device,
    )
    matcher.run(
        data=args.data,
        query_image=args.query_image,
        extensions=ext_list,
        max_workers=args.max_workers,
    )
