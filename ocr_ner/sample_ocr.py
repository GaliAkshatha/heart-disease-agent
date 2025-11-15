import pytesseract
from PIL import Image
import re

# -------------------------------------------
# If Tesseract is not in PATH, UNCOMMENT THIS:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -------------------------------------------

# REQUIRED FIELDS FOR ML PREDICTION
REQUIRED_FIELDS = [
    "Age", "Sex", "Height", "Weight",
    "RestingBP", "DiastolicBP", "Cholesterol", "Glucose",
    "Smoking", "AlcoholIntake", "PhysicalActivity",
    "HeartDisease"
]

def extract_ocr_text(image_path):
    """Run OCR on image and return text."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


# -----------------------------------------------
# RULE-BASED NER EXTRACTION
# -----------------------------------------------
def extract_field(text, field):
    """Extract specific fields using regex keywords."""
    patterns = {
        "Age": r"Age[:\s]*([0-9]{2})",
        "Sex": r"(Male|Female|M|F)",
        "Height": r"Height[:\s]*([0-9]{2,3})",
        "Weight": r"Weight[:\s]*([0-9]{2,3})",
        "RestingBP": r"RestingBP[:\s]*([0-9]{2,3})",
        "DiastolicBP": r"DiastolicBP[:\s]*([0-9]{2,3})",
        "Cholesterol": r"Cholesterol[:\s]*([0-9]{2,3})",
        "Glucose": r"Glucose[:\s]*([0-9]{2,3})",
        "Smoking": r"Smoking[:\s]*(Yes|No|0|1)",
        "AlcoholIntake": r"Alcohol[:\s]*(Yes|No|0|1)",
        "PhysicalActivity": r"PhysicalActivity[:\s]*([A-Za-z0-9]+)",
        "HeartDisease": r"HeartDisease[:\s]*(Yes|No|0|1)"
    }

    regex = patterns.get(field)
    if not regex:
        return None

    match = re.search(regex, text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def run_extraction(image_path):
    print("\n📌 Running OCR...")
    text = extract_ocr_text(image_path)
    print("\n===== OCR RAW TEXT =====")
    print(text)

    print("\n📌 Extracting fields...\n")

    extracted = {}
    for field in REQUIRED_FIELDS:
        value = extract_field(text, field)
        extracted[field] = value
        print(f"{field}: {value}")

    # Ask for missing data
    print("\n===== MISSING DATA REQUIRED =====\n")
    for field, value in extracted.items():
        if value is None:
            user_input = input(f"Enter value for {field}: ")
            extracted[field] = user_input

    print("\n===== FINAL EXTRACTED DATA (READY FOR PREDICTION) =====\n")
    for k, v in extracted.items():
        print(f"{k}: {v}")

    return extracted


# -----------------------------------------------
# MAIN
# -----------------------------------------------
if __name__ == "__main__":
    img_path = input("Enter the image file path: ")
    data = run_extraction(img_path)
