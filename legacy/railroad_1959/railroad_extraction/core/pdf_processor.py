import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List

class PDFProcessor:
    def __init__(self, pdf_path: str, output_dir: str):
        """
        Initialize the PDF Processor.
        
        Args:
            pdf_path: Path to the source PDF file.
            output_dir: Directory to save extracted images.
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_images(self, dpi: int = 300) -> List[str]:
        """
        Convert PDF pages to images.
        
        Args:
            dpi: Resolution for the output images.
            
        Returns:
            List of paths to the generated image files.
        """
        print(f"Opening PDF: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        image_paths = []
        
        print(f"Converting {len(doc)} pages to images at {dpi} DPI...")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Calculate zoom factor for DPI (72 is default PDF DPI)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            image_filename = f"page_{page_num + 1:03d}.png"
            image_path = self.output_dir / image_filename
            
            pix.save(str(image_path))
            image_paths.append(str(image_path))
            
            if (page_num + 1) % 10 == 0:
                print(f"Processed {page_num + 1}/{len(doc)} pages")
                
        print(f"Conversion complete. Saved {len(image_paths)} images to {self.output_dir}")
        return image_paths

if __name__ == "__main__":
    # Test run
    base_dir = Path(__file__).parent.parent.parent.parent
    pdf_file = base_dir / "Public" / "1959RailRoadCodeRL.pdf"
    output_folder = base_dir / "RailroadEngineer1959" / "data" / "processed_images"
    
    try:
        processor = PDFProcessor(str(pdf_file), str(output_folder))
        processor.convert_to_images(dpi=150) # Use 150 for speed in test
    except Exception as e:
        print(f"Error: {e}")
