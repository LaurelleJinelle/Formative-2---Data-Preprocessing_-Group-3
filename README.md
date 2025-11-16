# Formative 2: Multimodal Data Preprocessing - Group 3

## E-Commerce Multimodal Authentication & Product Recommendation System

A complete multimodal authentication pipeline combining facial recognition, dual-voice biometric authentication, and AI-powered product recommendations for secure e-commerce transactions.

---

## Project Overview

This project implements a **three-model integration system** for secure e-commerce authentication with the following workflow:

**Stage 1: FACIAL RECOGNITION (Gateway)**

- DeepFace with Facenet512 -> 512 features -> Random Forest

**Stage 2: VOICE APPROVAL ("Yes, approve")**

- librosa audio features -> 15 features -> Random Forest

**Stage 3: PRODUCT RECOMMENDATION (Internal)**

- Customer behavioral data -> 9 features -> Random Forest Pipeline

**Stage 4: VOICE CONFIRMATION ("Confirm transaction")**

- librosa audio features -> 15 features -> Random Forest

**Stage 5: DISPLAY RESULTS**

- Show personalized product recommendations

---

## Repository Structure

```
Formative-2---Data-Preprocessing_-Group-3/
|
+-- datasets/                           # Training datasets
|   +-- audio_features.csv              # Voice features (15 per sample)
|   +-- image_features.csv              # Facial features (512 per image)
|   +-- merged_customer_dataset.csv     # Customer behavioral data
|
+-- models/                             # Training notebooks & models
|   |
|   +-- facial-recognition-Model/
|   |   +-- facial_data/                # Original training images
|   |   +-- facial_data_augmented/      # Augmented images
|   |   +-- facial_recognition_pipeline.ipynb
|   |   +-- facial_recognition_model.pkl
|   |   +-- feature_columns.pkl
|   |   +-- image_features.csv
|   |   +-- requirements_python311.txt
|   |
|   +-- Product Recommendation Model/
|   |   +-- Formative_2_Group_3_Data_Preparation_and_Prediction_Model.ipynb
|   |   +-- customer_social_profiles - customer_social_profiles.csv
|   |   +-- customer_transactions - customer_transactions.csv
|   |   +-- merged_customer_dataset.csv
|   |
|   +-- Voiceprint Verification Model/
|       +-- Voiceprint_Verification_Model.ipynb
|
+-- python-script/                      # Production deployment
    +-- prediction script.py            # Main CLI application
    |
    +-- Models (3 trained Random Forests):
    |   +-- facial_recognition_model.pkl
    |   +-- voiceprint_rf_model.pkl
    |   +-- recommendation_rf_model.pkl
    |
    +-- Data Files:
    |   +-- customer_profiles.csv
    |   +-- recommendation_label_encoder.pkl
    |
    +-- Sample Test Data:
        +-- sample-audio/
        |   +-- Wilsons-approve.wav
        |   +-- Wilsons-Confirm.wav
        +-- sample-images/
            +-- neutral.jpg
            +-- smiling.jpg
            +-- stranger.jpg
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd models/facial-recognition-Model
pip install -r requirements_python311.txt
```

**OR install manually:**

```bash
pip install tensorflow==2.15.0 keras==2.15.0 deepface==0.0.79
pip install opencv-python==4.8.1.78 scikit-learn==1.3.2
pip install librosa==0.10.1 soundfile pandas numpy joblib
```

**Key packages:**

- TensorFlow 2.15.0 - Deep learning framework
- DeepFace 0.0.79 - Face recognition (Facenet512)
- librosa 0.10.1 - Audio feature extraction
- scikit-learn 1.3.2 - Machine learning
- OpenCV 4.8.1.78 - Image processing

**Python Version:** 3.11.x

### 2. Run the Prediction Script

Navigate to python-script folder:

```bash
cd python-script
```

**Option A: Interactive Mode**

```bash
python "prediction script.py"
```

The system will prompt you for:

1. Path to face image
2. Path to approval audio ("Yes, approve")
3. Path to confirmation audio ("Confirm transaction")
4. Then display results

**Option B: Batch Mode**

```bash
python "prediction script.py" sample-images/neutral.jpg sample-audio/Wilsons-approve.wav sample-audio/Wilsons-Confirm.wav
```

**With custom thresholds:**

```bash
python "prediction script.py" sample-images/neutral.jpg sample-audio/Wilsons-approve.wav sample-audio/Wilsons-Confirm.wav --face-threshold 0.7 --voice-threshold 0.65
```

---

## Consolidated requirements & model downloads

If you want a single place to install the project dependencies and pre-download large model weights (so first runs are fast), follow these steps.

1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

2) Pre-download DeepFace weights (optional but recommended):

DeepFace downloads model weights the first time a model is used. To avoid long waits during a demo, pre-download them:

```powershell
# This script will download the Facenet model weights to DeepFace's cache directory
python tools/download_deepface_models.py --models Facenet
```

DeepFace stores weights under the default cache path (usually `~/.deepface/weights/`). You can override the location by setting the `DEEPFACE_HOME` environment variable before running the downloader.

Example (PowerShell):

```powershell
$env:DEEPFACE_HOME = 'C:\deepface_cache'
python tools/download_deepface_models.py --models Facenet
```

This ensures the models are available offline and reduces cold-start time when you run the notebooks or the CLI.



## Three Models Explained

### Model 1: Facial Recognition

**File:** facial_recognition_model.pkl
**Training Notebook:** models/facial-recognition-Model/facial_recognition_pipeline.ipynb

| Property         | Value                                              |
| ---------------- | -------------------------------------------------- |
| Algorithm        | Random Forest Classifier                           |
| Input Features   | 512 (Facenet512 embeddings)                        |
| Classes          | 5 (Emmanuel, Leny, Wilsons, jinelle, unauthorized) |
| Training Samples | ~100 (augmented images)                            |
| Accuracy         | 90-100%                                            |
| Dataset          | datasets/image_features.csv                        |

**Feature Extraction:**

- Uses DeepFace library with Facenet512 model
- Extracts 512-dimensional facial embeddings
- Captures facial geometry, texture, and characteristics

### Model 2: Voice Authentication

**File:** voiceprint_rf_model.pkl
**Training Notebook:** models/Voiceprint Verification Model/Voiceprint_Verification_Model.ipynb

| Property       | Value                           |
| -------------- | ------------------------------- |
| Algorithm      | Random Forest (200 trees)       |
| Input Features | 15 audio features               |
| Classes        | 4 (Emmanuel, LJ, Leny, Wilsons) |
| Accuracy       | 100% (cross-validation)         |
| Dataset        | datasets/audio_features.csv     |

**Feature Extraction (15 features):**

- 13 MFCCs (Mel-Frequency Cepstral Coefficients) - Mean values
- 1 Spectral Rolloff - Mean value
- 1 RMS Energy - Mean value

Uses librosa library for audio processing.

### Model 3: Product Recommendation

**File:** recommendation_rf_model.pkl
**Training Notebook:** models/Product Recommendation Model/Formative_2_Group_3_Data_Preparation_and_Prediction_Model.ipynb

| Property       | Value                                               |
| -------------- | --------------------------------------------------- |
| Algorithm      | Random Forest in sklearn Pipeline                   |
| Input Features | 9 (7 numeric + 2 categorical)                       |
| Classes        | 5 (Books, Clothing, Electronics, Groceries, Sports) |
| Pipeline       | ColumnTransformer (StandardScaler + OneHotEncoder)  |
| Dataset        | datasets/merged_customer_dataset.csv                |

**Input Features (9 total):**

Numeric features (7):

- engagement_score - Social media engagement (0-100)
- purchase_interest_score - Browsing score (0-5)
- num_transactions - Total purchases
- total_spent - Lifetime spending ($)
- avg_spent - Average order value ($)
- avg_rating - Average rating (1-5)
- recency_days - Days since last purchase

Categorical features (2):

- social_media_platform (Facebook, Twitter, LinkedIn, Instagram)
- review_sentiment (Positive, Neutral, Negative)

---

## Dataset Information

### datasets/image_features.csv

- Rows: ~100 (augmented images)
- Columns: 513 (person + 512 facial features)
- Purpose: Training facial recognition model

### datasets/audio_features.csv

- Rows: Multiple recordings per person
- Columns: 16 (person + 15 audio features)
- Purpose: Training voice authentication model

### datasets/merged_customer_dataset.csv

- Rows: Customer transaction records
- Columns: 10 (9 behavioral features + product)
- Purpose: Training product recommendation model

### python-script/customer_profiles.csv

- Rows: 5 (one per registered user)
- Columns: 10 (person + 9 behavioral features)
- Purpose: Runtime user-to-features mapping

---

## Command-line Options

```
usage: prediction script.py [-h] [--interactive]
                             [--face-threshold FACE_THRESHOLD]
                             [--voice-threshold VOICE_THRESHOLD]
                             [--top-n TOP_N] [--quiet] [--json]
                             [image] [approval_audio] [confirmation_audio]

positional arguments:
  image                 Face image path (optional)
  approval_audio        Approval audio "Yes, approve" (optional)
  confirmation_audio    Confirmation audio "Confirm transaction" (optional)

optional arguments:
  -h, --help            Show help
  --interactive, -i     Interactive mode
  --face-threshold      Face confidence threshold (default: 0.6)
  --voice-threshold     Voice confidence threshold (default: 0.6)
  --top-n               Number of recommendations (default: 5)
  --quiet               Minimal output
  --json                Output in JSON format
```

---

## Security Features

### Multi-Layer Authentication

1. Biometric Layer 1: Facial recognition
2. Biometric Layer 2: Voice approval ("Yes, approve")
3. Biometric Layer 3: Voice confirmation ("Confirm transaction")

### Replay Attack Prevention

- Two separate voice recordings required
- Different phrases for approval vs confirmation
- Voice must match initially recognized face
- Sequential authentication (all must pass)

### Security Alerts

- Face/Voice mismatch detection
- Adjustable confidence thresholds (default: 0.6)
- Unauthorized user rejection
- Recommendations discarded if confirmation fails

---

## Exit Codes

```
0  # Success - All authentication passed
1  # Face authentication failed
2  # Voice approval failed
3  # Voice confirmation failed
4  # Other error
5  # Model files not found
6  # Unexpected error
```

---

## Testing

### Test 1: Authorized User

```bash
cd python-script
python "prediction script.py" sample-images/neutral.jpg sample-audio/Wilsons-approve.wav sample-audio/Wilsons-Confirm.wav
```

Expected: All stages pass, recommendations displayed

### Test 2: Unauthorized User

```bash
python "prediction script.py" sample-images/stranger.jpg sample-audio/Wilsons-approve.wav sample-audio/Wilsons-Confirm.wav
```

Expected: Stage 1 fails (facial recognition rejects)

### Test 3: Voice Mismatch

Use different person's audio than face image.

Expected: Stage 2 or 4 fails (voice mismatch)

---

## Performance Metrics

### End-to-End Test (user: wilsons)

| Stage              | Result      | Confidence | Time      |
| ------------------ | ----------- | ---------- | --------- |
| Facial Recognition | Passed      | 86.33%     | ~1.5s     |
| Voice Approval     | Passed      | 84.50%     | ~0.8s     |
| Recommendation     | Success     | -          | ~0.2s     |
| Voice Confirmation | Passed      | 78.00%     | ~0.8s     |
| **Total**          | **Success** | -          | **~3.3s** |

**Recommendations:**

1. Electronics (49.5%) - Top recommendation
2. Clothing (32.2%)
3. Sports (8.0%)
4. Books (7.5%)
5. Groceries (2.8%)

### Model Accuracies

- Facial Recognition: 90-100%
- Voice Authentication: 100%
- Recommendations: Probability-based

---

## Troubleshooting

### Issue 1: Model not found

```bash
# Make sure you're in python-script/ folder
cd python-script
ls -la  # Verify .pkl files exist
```

### Issue 2: DeepFace downloads models

First run takes 5-10 minutes:

- Downloads Facenet512 (~100MB)
- Requires internet connection
- Cached in ~/.deepface/weights/
- Subsequent runs are fast

### Issue 3: librosa installation

```bash
pip install librosa soundfile
pip install audioread  # Windows
```

### Issue 4: No recommendations

Check person exists in customer_profiles.csv:

```bash
cat customer_profiles.csv | grep "wilsons"
```

### Issue 5: Low confidence

Lower thresholds for testing:

```bash
python "prediction script.py" image.jpg approve.wav confirm.wav --face-threshold 0.4 --voice-threshold 0.4
```

---

## Tips & Best Practices

### Image Quality

- Clear, well-lit photos
- Face centered
- No sunglasses/masks
- Minimum 400x400 pixels
- JPEG or PNG format

### Audio Quality

- Quiet environment
- WAV format recommended
- Speak clearly
- Correct phrases: "Yes, approve" / "Confirm transaction"

### Security Tuning

- High security (banking): 0.8+ thresholds
- User-friendly (e-commerce): 0.6 default
- Testing/development: 0.4-0.5

---

## Technologies Used

| Technology   | Version  | Purpose              |
| ------------ | -------- | -------------------- |
| Python       | 3.11.x   | Programming language |
| TensorFlow   | 2.15.0   | Deep learning        |
| DeepFace     | 0.0.79   | Face recognition     |
| librosa      | 0.10.1   | Audio processing     |
| scikit-learn | 1.3.2    | Machine learning     |
| OpenCV       | 4.8.1.78 | Image processing     |
| pandas       | 2.1.4    | Data manipulation    |
| numpy        | 1.26.2   | Numerical computing  |

---

## Team Members - Group 3

- Jinelle - Voice authentication, model training
- Wilsons - Facial recognition model and script integration
- Emmanuel - Product recommendations model
- Leny - Testing, documentation

---

## Project Checklist

**Models:**

- [x] Facial recognition trained (90%+ accuracy)
- [x] Voice authentication trained (100% accuracy)
- [x] Product recommendation trained
- [x] All models integrated

**Features:**

- [x] Dual-voice authentication
- [x] Sequential workflow
- [x] Customer profiles
- [x] Interactive mode
- [x] Batch mode
- [x] Security features

**Testing:**

- [x] Authorized user testing
- [x] Unauthorized rejection
- [x] Voice mismatch detection
- [x] End-to-end testing
- [x] Sample data included

**Documentation:**

- [x] README complete
- [x] Code comments
- [x] Training notebooks documented
- [x] Usage examples

---

## Assignment Deliverables

**1. Repository Folder:**

- All models trained and saved
- Training notebooks with outputs
- Prediction script functional
- Sample test data

**2. Datasets (datasets/ folder):**

- image_features.csv
- audio_features.csv
- merged_customer_dataset.csv

**3. Training Notebooks (models/ folder):**

- Facial recognition pipeline
- Voiceprint verification
- Product recommendation

**4. Prediction Script (python-script/ folder):**

- Working CLI application
- Dependencies listed
- Sample data

**5. Documentation:**

- README.md
- Code comments
- Usage instructions

---

## Contact

For questions:

1. Review this README
2. Check training notebooks
3. Review code comments
4. Contact team members

---

**Built with Python 3.11 | DeepFace, librosa & scikit-learn**

**Happy Coding**
