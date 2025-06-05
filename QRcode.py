import qrcode
from pathlib import Path
from config import (
    OUTPUTS_DIR,
    ensure_dir,
    is_colab,
    is_kaggle,
    is_local,
    ENVIRONMENT
)

def generate_qr_code():
    """Generate QR code for the repository URL with environment-specific error handling."""
    print(f"Generating QR code in {ENVIRONMENT} environment...")
    
    try:
        repo_url = "https://github.com/Eric0801/NTCIR-18-CLIR-pipeline"
        ensure_dir(OUTPUTS_DIR)
        
        # Generate QR code
        qr = qrcode.make(repo_url)
        
        # Save QR code
        output_path = OUTPUTS_DIR / "github_repo_qrcode.png"
        qr.save(str(output_path))
        print(f"✅ QR code saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating QR code: {e}")
        if is_colab:
            print("Please ensure you have write permissions in your Google Drive.")
        elif is_kaggle:
            print("Please ensure you have write permissions in your Kaggle workspace.")
        else:
            print("Please check if you have write permissions in the output directory.")
        raise

if __name__ == "__main__":
    generate_qr_code()