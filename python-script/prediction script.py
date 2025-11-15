"""
Combined Multimodal Authentication & Recommendation System
Formative 2: Multimodal Authentication with Product Recommendation

This script provides a complete pipeline:
1. Facial Recognition (Gateway Authentication)
2. Voice Approval - "Yes, approve" (Approves prediction request)
3. Product Recommendation Generation (Personalized predictions)
4. Voice Confirmation - "Confirm transaction" (Confirms to display results)
5. Display Predictions (Only if all authentication steps pass)

The system uses TWO separate voice authentication steps for enhanced security:
- First voice validates the user's intent to request predictions
- Second voice confirms the user wants to view the generated recommendations
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import joblib
import argparse
from typing import Dict, List, Optional, Any

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass  # If it fails, continue without UTF-8 encoding


class FacialRecognitionSystem:
    """
    Facial Recognition System for user authentication
    """

    def __init__(self, model_path='facial_recognition_model.pkl',
                 feature_columns_path='feature_columns.pkl',
                 model_name='Facenet512'):
        """
        Initialize the facial recognition system

        Args:
            model_path: Path to trained model
            feature_columns_path: Path to feature columns
            model_name: DeepFace model name
        """
        self.model = None
        self.feature_columns = None
        self.model_name = model_name

        # Load model and feature columns
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✓ Facial recognition model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Facial model not found at {model_path}")

        if os.path.exists(feature_columns_path):
            self.feature_columns = joblib.load(feature_columns_path)
            print(f"✓ Facial feature columns loaded")
        else:
            print(f"⚠ Warning: Feature columns not found")


    def extract_features(self, image_path):
        """
        Extract facial features from an image

        Args:
            image_path: Path to image file

        Returns:
            Feature vector or None if extraction fails
        """
        try:
            # Extract embeddings using DeepFace
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )

            feature_vector = embedding[0]['embedding']
            return feature_vector

        except Exception as e:
            print(f"Error extracting facial features: {str(e)}")
            return None


    def recognize(self, image_path, threshold=0.6, verbose=True):
        """
        Recognize a face in the given image

        Args:
            image_path: Path to image file
            threshold: Minimum confidence threshold (0-1)
            verbose: Print detailed output

        Returns:
            Dictionary with recognition results
        """
        if self.model is None:
            return {
                'status': 'error',
                'message': 'Model not loaded',
                'person': None,
                'confidence': 0.0
            }

        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': f'Image not found: {image_path}',
                'person': None,
                'confidence': 0.0
            }

        if verbose:
            print(f"\n[STAGE 1] FACIAL RECOGNITION")
            print(f"Processing image: {os.path.basename(image_path)}")
            print("Extracting facial features...")

        # Extract features
        features = self.extract_features(image_path)

        if features is None:
            return {
                'status': 'unauthorized',
                'message': 'No face detected or feature extraction failed',
                'person': None,
                'confidence': 0.0
            }

        # Prepare for prediction
        X_input = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(X_input)[0]
        probabilities = self.model.predict_proba(X_input)[0]
        confidence = max(probabilities)

        if verbose:
            print(f"Recognition confidence: {confidence:.2%}")

        # Check threshold
        if confidence < threshold:
            if verbose:
                print(f"❌ UNAUTHORIZED: Confidence below threshold ({threshold:.2%})")

            return {
                'status': 'unauthorized',
                'message': f'Confidence {confidence:.2%} below threshold {threshold:.2%}',
                'person': None,
                'confidence': confidence
            }

        # Check if recognized as unauthorized
        if prediction == 'unauthorized':
            if verbose:
                print(f"❌ UNAUTHORIZED: User not in authorized database")

            return {
                'status': 'unauthorized',
                'message': 'Not an authorized user',
                'person': None,
                'confidence': confidence
            }

        # Authorized user
        if verbose:
            print(f"✓ Face recognized: {prediction}")
            print(f"✓ Confidence: {confidence:.2%}")

        return {
            'status': 'authorized',
            'message': 'Face recognized successfully',
            'person': prediction,
            'confidence': confidence
        }


class VoiceRecognitionSystem:
    """
    Voice Recognition System for voice validation
    """

    def __init__(self, model_path='voiceprint_rf_model.pkl'):
        """
        Initialize the voice recognition system

        Args:
            model_path: Path to trained voice model (voiceprint_rf_model.pkl)
        """
        self.model = None
        self.model_path = model_path

        # Load voice model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✓ Voice recognition model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Voice model not found at {model_path}")


    def extract_voice_features(self, audio_path):
        """
        Extract voice features from an audio file

        Features extracted (15 total):
        - 13 MFCCs (Mel-Frequency Cepstral Coefficients)
        - 1 Spectral Rolloff
        - 1 RMS Energy

        This matches the training done in Voiceprint_Verification_Model.ipynb

        Args:
            audio_path: Path to audio file

        Returns:
            Voice feature vector (15 features) or None if extraction fails
        """
        try:
            import librosa

            # Load audio file (sr=None keeps original sample rate, matching training)
            y, sr = librosa.load(audio_path, sr=None)

            # Extract 13 MFCCs and take mean across time (axis=1)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

            # Extract spectral rolloff and take mean
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

            # Extract RMS energy and take mean
            energy = np.mean(librosa.feature.rms(y=y))

            # Combine all features: 13 MFCCs + 1 rolloff + 1 energy = 15 features
            features = list(mfccs) + [rolloff, energy]

            return np.array(features)

        except ImportError:
            print("❌ Error: librosa library not installed!")
            print("Please install it with: pip install librosa soundfile")
            return None
        except Exception as e:
            print(f"Error extracting voice features: {str(e)}")
            return None


    def validate_approval(self, audio_path, expected_person, threshold=0.6, verbose=True):
        """
        Validate approval voice ("Yes, approve") against expected person

        Args:
            audio_path: Path to approval audio file
            expected_person: Person name from facial recognition
            threshold: Minimum confidence threshold (0-1)
            verbose: Print detailed output

        Returns:
            Dictionary with validation results
        """
        if verbose:
            print(f"\n[STAGE 2] VOICE APPROVAL VALIDATION")
            print(f"Expected phrase: 'Yes, approve'")

        return self._validate_voice_internal(
            audio_path=audio_path,
            expected_person=expected_person,
            threshold=threshold,
            verbose=verbose,
            stage_name="APPROVAL",
            expected_phrase="Yes, approve"
        )


    def validate_confirmation(self, audio_path, expected_person, threshold=0.6, verbose=True):
        """
        Validate confirmation voice ("Confirm transaction") against expected person

        Args:
            audio_path: Path to confirmation audio file
            expected_person: Person name from facial recognition
            threshold: Minimum confidence threshold (0-1)
            verbose: Print detailed output

        Returns:
            Dictionary with validation results
        """
        if verbose:
            print(f"\n[STAGE 4] VOICE CONFIRMATION VALIDATION")
            print(f"Expected phrase: 'Confirm transaction'")

        return self._validate_voice_internal(
            audio_path=audio_path,
            expected_person=expected_person,
            threshold=threshold,
            verbose=verbose,
            stage_name="CONFIRMATION",
            expected_phrase="Confirm transaction"
        )


    def _validate_voice_internal(self, audio_path, expected_person, threshold=0.6, verbose=True, stage_name="VOICE", expected_phrase=""):
        """
        Internal method to validate voice against expected person

        Args:
            audio_path: Path to audio file
            expected_person: Person name from facial recognition
            threshold: Minimum confidence threshold (0-1)
            verbose: Print detailed output
            stage_name: Name of the validation stage (APPROVAL or CONFIRMATION)
            expected_phrase: Expected phrase for this validation

        Returns:
            Dictionary with validation results
        """
        if self.model is None:
            return {
                'status': 'error',
                'message': 'Voice model not loaded',
                'match': False,
                'confidence': 0.0
            }

        if not os.path.exists(audio_path):
            return {
                'status': 'error',
                'message': f'Audio file not found: {audio_path}',
                'match': False,
                'confidence': 0.0
            }

        if verbose:
            print(f"Processing audio: {os.path.basename(audio_path)}")
            print(f"Expected person: {expected_person}")
            print("Extracting voice features...")

        # Extract voice features
        features = self.extract_voice_features(audio_path)

        if features is None:
            return {
                'status': 'error',
                'message': 'Voice feature extraction failed',
                'match': False,
                'confidence': 0.0
            }

        # Prepare for prediction
        X_input = np.array(features).reshape(1, -1)

        # Make prediction
        try:
            prediction = self.model.predict(X_input)[0]
            probabilities = self.model.predict_proba(X_input)[0]
            confidence = max(probabilities)

            if verbose:
                print(f"Voice recognition confidence: {confidence:.2%}")
                print(f"Predicted voice: {prediction}")

            # Check if voice matches expected person (case-insensitive)
            voice_matches = (prediction.lower() == expected_person.lower())

            # Check threshold
            if confidence < threshold:
                if verbose:
                    print(f"❌ {stage_name} FAILED: Confidence below threshold ({threshold:.2%})")

                return {
                    'status': 'unauthorized',
                    'message': f'{stage_name} confidence {confidence:.2%} below threshold',
                    'match': False,
                    'confidence': confidence
                }

            # Check if voice matches face
            if not voice_matches:
                if verbose:
                    print(f"❌ {stage_name} FAILED: Voice mismatch")
                    print(f"   Face identified: {expected_person}")
                    print(f"   Voice identified: {prediction}")
                    print(f"   SECURITY ALERT: Possible impersonation attempt!")

                return {
                    'status': 'unauthorized',
                    'message': f'{stage_name}: Voice ({prediction}) does not match face ({expected_person})',
                    'match': False,
                    'confidence': confidence
                }

            # Voice validation successful
            if verbose:
                print(f"✓ Voice validated: {prediction}")
                print(f"✓ Voice matches face identity")
                print(f"✓ Confidence: {confidence:.2%}")

            return {
                'status': 'authorized',
                'message': 'Voice validated successfully',
                'match': True,
                'confidence': confidence
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Voice prediction error: {str(e)}',
                'match': False,
                'confidence': 0.0
            }


class RecommendationEngine:
    """
    Product Recommendation Engine
    """

    def __init__(self, model_path='recommendation_rf_model.pkl',
                 label_encoder_path='recommendation_label_encoder.pkl',
                 customer_data_path='customer_profiles.csv',
                 product_catalog_path='product_catalog.csv'):
        """
        Initialize the recommendation engine

        Args:
            model_path: Path to recommendation model (recommendation_rf_model.pkl)
            label_encoder_path: Path to label encoder (recommendation_label_encoder.pkl)
            customer_data_path: Path to customer profiles/history
            product_catalog_path: Path to product catalog
        """
        self.model = None
        self.label_encoder = None
        self.customer_data = None
        self.product_catalog = None
        self.model_path = model_path

        # Load recommendation model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✓ Recommendation model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Recommendation model not found at {model_path}")

        # Load label encoder
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
            print(f"✓ Recommendation label encoder loaded")
        else:
            raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")

        # Load customer data (optional)
        if os.path.exists(customer_data_path):
            self.customer_data = pd.read_csv(customer_data_path)
            print(f"✓ Customer profiles loaded")
        else:
            print(f"⚠ Warning: Customer data not found, using model only")

        # Load product catalog (optional)
        if os.path.exists(product_catalog_path):
            self.product_catalog = pd.read_csv(product_catalog_path)
            print(f"✓ Product catalog loaded")
        else:
            print(f"⚠ Warning: Product catalog not found")


    def get_recommendations(self, person, facial_features=None, top_n=5, verbose=True):
        """
        Generate product recommendations for a person

        Args:
            person: Person name/ID
            facial_features: Facial feature vector (optional, will use customer_data if not provided)
            top_n: Number of recommendations to return
            verbose: Print detailed output

        Returns:
            Dictionary with recommendations
        """
        if self.model is None:
            return {
                'status': 'error',
                'message': 'Recommendation model not loaded',
                'recommendations': []
            }

        if verbose:
            print(f"\n[STAGE 3] PRODUCT RECOMMENDATION GENERATION")
            print(f"Generating personalized recommendations for: {person}")

        try:
            # Get facial features for the person
            features = None

            if facial_features is not None:
                # Use provided facial features (not currently used)
                features = facial_features
                feature_cols = [f'feature_{i}' for i in range(len(features))]
            elif self.customer_data is not None:
                # Look up person's features from customer data
                person_data = self.customer_data[self.customer_data['person'] == person]

                # Get all columns except 'person'
                feature_cols = [col for col in self.customer_data.columns if col != 'person']

                if not person_data.empty:
                    # Get the first matching record's features as a Series/dict
                    features = person_data[feature_cols].iloc[0]
                else:
                    if verbose:
                        print(f"⚠ Warning: No data found for {person} in customer database")
                        print("   Using average features for prediction")
                    # Use average features as fallback
                    features = self.customer_data[feature_cols].mean()
            else:
                if verbose:
                    print(f"⚠ Warning: No customer data available")
                    print("   Cannot generate personalized recommendations")
                return {
                    'status': 'error',
                    'message': 'No feature data available for recommendation',
                    'recommendations': []
                }

            # Prepare features for prediction as DataFrame (model expects column names)
            # Features is now a pandas Series with correct column names
            X_input = pd.DataFrame([features], columns=feature_cols)

            # Get predictions and probabilities from the model
            prediction = self.model.predict(X_input)[0]
            probabilities = self.model.predict_proba(X_input)[0]

            # Get all product categories from label encoder
            all_categories = self.label_encoder.classes_

            # Convert prediction index to actual product name
            predicted_product = self.label_encoder.classes_[prediction]

            # Create recommendations from all categories sorted by probability
            recommendations = []
            for category, probability in zip(all_categories, probabilities):
                recommendations.append({
                    'product': category,
                    'score': probability,
                    'reason': 'Top recommendation' if category == predicted_product else 'Also recommended'
                })

            # Sort by score (probability) in descending order
            recommendations.sort(key=lambda x: x['score'], reverse=True)

            # Return top N recommendations
            recommendations = recommendations[:top_n]

            if verbose:
                print(f"✓ Generated {len(recommendations)} recommendations")
                print(f"✓ Top recommendation: {recommendations[0]['product']}")
                print("✓ Recommendations prepared (awaiting voice confirmation)")

            return {
                'status': 'success',
                'message': f'Generated {len(recommendations)} recommendations',
                'recommendations': recommendations,
                'person': person
            }

        except Exception as e:
            if verbose:
                print(f"⚠ Error generating recommendations: {str(e)}")
            return {
                'status': 'error',
                'message': f'Recommendation generation error: {str(e)}',
                'recommendations': []
            }


class MultimodalPipeline:
    """
    Main pipeline orchestrator for multimodal authentication and recommendation
    """

    def __init__(self,
                 face_model='facial_recognition_model.pkl',
                 face_features='feature_columns.pkl',
                 voice_model='voiceprint_rf_model.pkl',
                 rec_model='recommendation_rf_model.pkl',
                 rec_label_encoder='recommendation_label_encoder.pkl',
                 customer_data='customer_profiles.csv',
                 product_catalog='product_catalog.csv'):
        """
        Initialize the complete multimodal pipeline

        Args:
            face_model: Path to facial recognition model
            face_features: Path to facial feature columns
            voice_model: Path to voice recognition model (voiceprint_rf_model.pkl)
            rec_model: Path to recommendation model (recommendation_rf_model.pkl)
            rec_label_encoder: Path to recommendation label encoder (recommendation_label_encoder.pkl)
            customer_data: Path to customer profiles/features (customer_profiles.csv)
            product_catalog: Path to product catalog
        """
        print("="*60)
        print("INITIALIZING MULTIMODAL AUTHENTICATION SYSTEM")
        print("="*60)

        # Initialize all subsystems
        try:
            self.face_system = FacialRecognitionSystem(
                model_path=face_model,
                feature_columns_path=face_features
            )
        except Exception as e:
            print(f"❌ Error initializing facial recognition: {str(e)}")
            raise

        try:
            self.voice_system = VoiceRecognitionSystem(model_path=voice_model)
        except Exception as e:
            print(f"❌ Error initializing voice recognition: {str(e)}")
            raise

        try:
            self.recommendation_engine = RecommendationEngine(
                model_path=rec_model,
                label_encoder_path=rec_label_encoder,
                customer_data_path=customer_data,
                product_catalog_path=product_catalog
            )
        except Exception as e:
            print(f"❌ Error initializing recommendation engine: {str(e)}")
            raise

        print("="*60)
        print("✓ ALL SYSTEMS INITIALIZED SUCCESSFULLY")
        print("="*60)


    def run_pipeline(self,
                     image_path,
                     approval_audio_path,
                     confirmation_audio_path,
                     face_threshold=0.6,
                     voice_threshold=0.6,
                     top_n_recommendations=5,
                     verbose=True):
        """
        Run the complete multimodal pipeline

        Workflow:
        1. Facial Recognition (Gateway Authentication)
        2. Voice Approval - "Yes, approve" (Approves prediction request)
        3. Product Recommendations (Generate but don't display)
        4. Voice Confirmation - "Confirm transaction" (Confirms to display)
        5. Display Predictions (Only if all auth steps passed)

        Args:
            image_path: Path to user image
            approval_audio_path: Path to approval audio ("Yes, approve")
            confirmation_audio_path: Path to confirmation audio ("Confirm transaction")
            face_threshold: Facial recognition confidence threshold
            voice_threshold: Voice validation confidence threshold
            top_n_recommendations: Number of recommendations to generate
            verbose: Print detailed output

        Returns:
            Complete pipeline result dictionary
        """
        if verbose:
            print("\n" + "="*60)
            print("MULTIMODAL AUTHENTICATION & RECOMMENDATION PIPELINE")
            print("="*60)

        # STAGE 1: Facial Recognition
        face_result = self.face_system.recognize(
            image_path=image_path,
            threshold=face_threshold,
            verbose=verbose
        )

        # Check if face authentication failed
        if face_result['status'] != 'authorized':
            if verbose:
                print("\n" + "="*60)
                print("❌ ACCESS DENIED: Facial Recognition Failed")
                print("="*60)

            return {
                'status': 'denied',
                'stage_failed': 'facial_recognition',
                'face_result': face_result,
                'approval_result': None,
                'confirmation_result': None,
                'recommendations': None,
                'person': None
            }

        # Face authentication passed - get identified person
        identified_person = face_result['person']

        if verbose:
            print(f"\n✓ STAGE 1 PASSED: Face recognized as {identified_person}")

        # STAGE 2: Voice Approval ("Yes, approve")
        approval_result = self.voice_system.validate_approval(
            audio_path=approval_audio_path,
            expected_person=identified_person,
            threshold=voice_threshold,
            verbose=verbose
        )

        # Check if approval failed
        if approval_result['status'] != 'authorized' or not approval_result.get('match', False):
            if verbose:
                print("\n" + "="*60)
                print("❌ ACCESS DENIED: Voice Approval Failed")
                print("="*60)
                print("⚠ SECURITY NOTE: Face recognized but approval voice did not match")
                print("   Prediction request denied")

            return {
                'status': 'denied',
                'stage_failed': 'voice_approval',
                'face_result': face_result,
                'approval_result': approval_result,
                'confirmation_result': None,
                'recommendations': None,
                'person': identified_person
            }

        if verbose:
            print(f"\n✓ STAGE 2 PASSED: Voice approval validated for {identified_person}")

        # STAGE 3: Generate Product Recommendations
        recommendation_result = self.recommendation_engine.get_recommendations(
            person=identified_person,
            top_n=top_n_recommendations,
            verbose=verbose
        )

        # Check if recommendation generation failed (non-critical)
        if recommendation_result['status'] != 'success':
            if verbose:
                print(f"⚠ Warning: Recommendation generation failed")
                print(f"   Continuing with authentication...")

        if verbose:
            print(f"\n✓ STAGE 3 PASSED: Recommendations generated (pending confirmation)")

        # STAGE 4: Voice Confirmation ("Confirm transaction")
        confirmation_result = self.voice_system.validate_confirmation(
            audio_path=confirmation_audio_path,
            expected_person=identified_person,
            threshold=voice_threshold,
            verbose=verbose
        )

        # Check if confirmation failed
        if confirmation_result['status'] != 'authorized' or not confirmation_result.get('match', False):
            if verbose:
                print("\n" + "="*60)
                print("❌ ACCESS DENIED: Voice Confirmation Failed")
                print("="*60)
                print("⚠ SECURITY NOTE: Confirmation voice did not match")
                print("   Recommendations have been discarded for security")

            return {
                'status': 'denied',
                'stage_failed': 'voice_confirmation',
                'face_result': face_result,
                'approval_result': approval_result,
                'confirmation_result': confirmation_result,
                'recommendations': None,  # Discard recommendations
                'person': identified_person
            }

        if verbose:
            print(f"\n✓ STAGE 4 PASSED: Voice confirmation validated for {identified_person}")

        # STAGE 5: Display Results (All authentications passed)
        if verbose:
            print("\n" + "="*60)
            print("✓ AUTHENTICATION SUCCESSFUL - ACCESS GRANTED")
            print("="*60)
            print(f"Authenticated user: {identified_person}")
            print(f"Face confidence: {face_result['confidence']:.2%}")
            print(f"Approval voice confidence: {approval_result['confidence']:.2%}")
            print(f"Confirmation voice confidence: {confirmation_result['confidence']:.2%}")

            # Display recommendations
            if recommendation_result['status'] == 'success':
                recommendations = recommendation_result['recommendations']
                print("\n" + "-"*60)
                print("PERSONALIZED PRODUCT RECOMMENDATIONS")
                print("-"*60)

                for idx, rec in enumerate(recommendations, 1):
                    print(f"\n{idx}. {rec['product']}")
                    print(f"   Score: {rec['score']:.1%}")
                    print(f"   Reason: {rec['reason']}")

                print("-"*60)
            else:
                print("\n⚠ No recommendations available")

        return {
            'status': 'authorized',
            'stage_failed': None,
            'face_result': face_result,
            'approval_result': approval_result,
            'confirmation_result': confirmation_result,
            'recommendations': recommendation_result,
            'person': identified_person
        }


def run_interactive_pipeline(pipeline, args):
    """
    Run the pipeline in interactive mode with step-by-step prompts

    Args:
        pipeline: Initialized MultimodalPipeline instance
        args: Parsed command-line arguments

    Returns:
        Complete pipeline result dictionary
    """
    print("\n" + "="*60)
    print("MULTIMODAL AUTHENTICATION SYSTEM - INTERACTIVE MODE")
    print("="*60)
    print("\nThis system will guide you through a secure authentication process:")
    print("  Step 1: Facial Recognition")
    print("  Step 2: Voice Approval ('Yes, approve')")
    print("  Step 3: Recommendation Generation")
    print("  Step 4: Voice Confirmation ('Confirm transaction')")
    print("  Step 5: Display Results")
    print("="*60)

    # STEP 1: Facial Recognition
    print("\n" + "="*60)
    print("[STEP 1/4] FACIAL RECOGNITION")
    print("="*60)
    print("Please provide your face image for authentication.")

    image_path = None
    while image_path is None:
        image_input = input("Enter path to your face image: ").strip()
        if os.path.exists(image_input):
            image_path = image_input
        else:
            print(f"❌ Error: File not found: {image_input}")
            print("Please try again.")

    print(f"\n▶ Processing facial recognition...")
    face_result = pipeline.face_system.recognize(
        image_path=image_path,
        threshold=args.face_threshold,
        verbose=False
    )

    if face_result['status'] != 'authorized':
        print("\n" + "="*60)
        print("❌ ACCESS DENIED: Facial Recognition Failed")
        print("="*60)
        print(f"Reason: {face_result['message']}")
        return {
            'status': 'denied',
            'stage_failed': 'facial_recognition',
            'face_result': face_result,
            'approval_result': None,
            'confirmation_result': None,
            'recommendations': None,
            'person': None
        }

    identified_person = face_result['person']
    print(f"✓ Face recognized: {identified_person}")
    print(f"✓ Confidence: {face_result['confidence']:.2%}")

    # STEP 2: Voice Approval
    print("\n" + "="*60)
    print("[STEP 2/4] VOICE APPROVAL")
    print("="*60)
    print(f"Welcome, {identified_person}!")
    print("Please approve the prediction request with your voice.")
    print("Expected phrase: 'Yes, approve'")

    approval_audio_path = None
    while approval_audio_path is None:
        approval_input = input("Enter path to approval audio file: ").strip()
        if os.path.exists(approval_input):
            approval_audio_path = approval_input
        else:
            print(f"❌ Error: File not found: {approval_input}")
            print("Please try again.")

    print(f"\n▶ Processing approval voice...")
    approval_result = pipeline.voice_system.validate_approval(
        audio_path=approval_audio_path,
        expected_person=identified_person,
        threshold=args.voice_threshold,
        verbose=False
    )

    if approval_result['status'] != 'authorized' or not approval_result.get('match', False):
        print("\n" + "="*60)
        print("❌ ACCESS DENIED: Voice Approval Failed")
        print("="*60)
        print(f"Reason: {approval_result['message']}")
        print("⚠ SECURITY NOTE: Face recognized but approval voice did not match")
        return {
            'status': 'denied',
            'stage_failed': 'voice_approval',
            'face_result': face_result,
            'approval_result': approval_result,
            'confirmation_result': None,
            'recommendations': None,
            'person': identified_person
        }

    print(f"✓ Approval validated!")
    print(f"✓ Voice confidence: {approval_result['confidence']:.2%}")

    # STEP 3: Generate Recommendations
    print("\n" + "="*60)
    print("[STEP 3/4] RECOMMENDATION GENERATION")
    print("="*60)
    print(f"▶ Generating personalized product recommendations for {identified_person}...")

    recommendation_result = pipeline.recommendation_engine.get_recommendations(
        person=identified_person,
        top_n=args.top_n,
        verbose=False
    )

    if recommendation_result['status'] == 'success':
        print(f"✓ Generated {len(recommendation_result['recommendations'])} recommendations")
        print("✓ Recommendations ready (awaiting confirmation)")
    else:
        print(f"⚠ Warning: Recommendation generation encountered an issue")
        print(f"   Continuing with authentication...")

    # STEP 4: Voice Confirmation
    print("\n" + "="*60)
    print("[STEP 4/4] VOICE CONFIRMATION")
    print("="*60)
    print("Please confirm to view your personalized recommendations.")
    print("Expected phrase: 'Confirm transaction'")

    confirmation_audio_path = None
    while confirmation_audio_path is None:
        confirmation_input = input("Enter path to confirmation audio file: ").strip()
        if os.path.exists(confirmation_input):
            confirmation_audio_path = confirmation_input
        else:
            print(f"❌ Error: File not found: {confirmation_input}")
            print("Please try again.")

    print(f"\n▶ Processing confirmation voice...")
    confirmation_result = pipeline.voice_system.validate_confirmation(
        audio_path=confirmation_audio_path,
        expected_person=identified_person,
        threshold=args.voice_threshold,
        verbose=False
    )

    if confirmation_result['status'] != 'authorized' or not confirmation_result.get('match', False):
        print("\n" + "="*60)
        print("❌ ACCESS DENIED: Voice Confirmation Failed")
        print("="*60)
        print(f"Reason: {confirmation_result['message']}")
        print("⚠ SECURITY NOTE: Confirmation voice did not match")
        print("   Recommendations have been discarded for security")
        return {
            'status': 'denied',
            'stage_failed': 'voice_confirmation',
            'face_result': face_result,
            'approval_result': approval_result,
            'confirmation_result': confirmation_result,
            'recommendations': None,
            'person': identified_person
        }

    print(f"✓ Confirmation validated!")
    print(f"✓ Voice confidence: {confirmation_result['confidence']:.2%}")

    # STEP 5: Display Results
    print("\n" + "="*60)
    print("✓ AUTHENTICATION SUCCESSFUL - ACCESS GRANTED")
    print("="*60)
    print(f"Authenticated user: {identified_person}")
    print(f"Face confidence: {face_result['confidence']:.2%}")
    print(f"Approval voice confidence: {approval_result['confidence']:.2%}")
    print(f"Confirmation voice confidence: {confirmation_result['confidence']:.2%}")

    # Display recommendations
    if recommendation_result['status'] == 'success':
        recommendations = recommendation_result['recommendations']
        print("\n" + "-"*60)
        print("YOUR PERSONALIZED PRODUCT RECOMMENDATIONS")
        print("-"*60)

        for idx, rec in enumerate(recommendations, 1):
            print(f"\n{idx}. {rec['product']}")
            print(f"   Score: {rec['score']:.1%}")
            print(f"   Reason: {rec['reason']}")

        print("-"*60)
    else:
        print("\n⚠ No recommendations available")

    return {
        'status': 'authorized',
        'stage_failed': None,
        'face_result': face_result,
        'approval_result': approval_result,
        'confirmation_result': confirmation_result,
        'recommendations': recommendation_result,
        'person': identified_person
    }


def main():
    """
    Main CLI function
    """
    parser = argparse.ArgumentParser(
        description='Multimodal Authentication & Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Modes:

1. INTERACTIVE MODE (Recommended for demonstrations):
   %(prog)s --interactive
   or simply:
   %(prog)s

   The system will guide you through each step with prompts.

2. BATCH MODE (For automation):
   %(prog)s image.jpg approval.wav confirmation.wav
   %(prog)s image.jpg approval.wav confirmation.wav --face-threshold 0.7
   %(prog)s image.jpg approval.wav confirmation.wav --top-n 10 --quiet

Note: The system requires TWO separate audio files:
  1. approval_audio: Voice saying "Yes, approve" (to approve prediction request)
  2. confirmation_audio: Voice saying "Confirm transaction" (to confirm and display results)
        """
    )

    # Input arguments (optional - will prompt if not provided)
    parser.add_argument('image', type=str, nargs='?',
                       help='Path to image file for facial recognition (optional, will prompt if not provided)')
    parser.add_argument('approval_audio', type=str, nargs='?',
                       help='Path to approval audio file ("Yes, approve") (optional, will prompt if not provided)')
    parser.add_argument('confirmation_audio', type=str, nargs='?',
                       help='Path to confirmation audio file ("Confirm transaction") (optional, will prompt if not provided)')

    # Interactive mode flag
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode with step-by-step prompts')

    # Threshold arguments
    parser.add_argument('--face-threshold', type=float, default=0.6,
                       help='Facial recognition confidence threshold (0-1), default: 0.6')
    parser.add_argument('--voice-threshold', type=float, default=0.6,
                       help='Voice validation confidence threshold (0-1), default: 0.6')

    # Model paths
    parser.add_argument('--face-model', type=str, default='facial_recognition_model.pkl',
                       help='Path to facial recognition model file')
    parser.add_argument('--face-features', type=str, default='feature_columns.pkl',
                       help='Path to facial feature columns file')
    parser.add_argument('--voice-model', type=str, default='voiceprint_rf_model.pkl',
                       help='Path to voice recognition model file (voiceprint_rf_model.pkl)')
    parser.add_argument('--rec-model', type=str, default='recommendation_rf_model.pkl',
                       help='Path to recommendation model file (recommendation_rf_model.pkl)')
    parser.add_argument('--rec-label-encoder', type=str, default='recommendation_label_encoder.pkl',
                       help='Path to recommendation label encoder file (recommendation_label_encoder.pkl)')

    # Data paths
    parser.add_argument('--customer-data', type=str, default='customer_profiles.csv',
                       help='Path to customer profiles/features CSV (customer_profiles.csv)')
    parser.add_argument('--product-catalog', type=str, default='product_catalog.csv',
                       help='Path to product catalog CSV')

    # Recommendation settings
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of product recommendations to generate, default: 5')

    # Output options
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (only final status)')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')

    args = parser.parse_args()

    # Determine if running in interactive mode
    # Interactive mode if: --interactive flag OR no arguments provided
    use_interactive_mode = args.interactive or (args.image is None)

    # Validate batch mode arguments
    if not use_interactive_mode:
        # In batch mode, all arguments must be provided
        if args.image is None or args.approval_audio is None or args.confirmation_audio is None:
            print("❌ Error: In batch mode, all arguments must be provided")
            print("Usage: python combined_recognition_cli.py <image> <approval_audio> <confirmation_audio>")
            print("Or run with --interactive flag for interactive mode")
            sys.exit(1)

        # Validate input files exist
        if not os.path.exists(args.image):
            print(f"❌ Error: Image file not found: {args.image}")
            sys.exit(1)

        if not os.path.exists(args.approval_audio):
            print(f"❌ Error: Approval audio file not found: {args.approval_audio}")
            sys.exit(1)

        if not os.path.exists(args.confirmation_audio):
            print(f"❌ Error: Confirmation audio file not found: {args.confirmation_audio}")
            sys.exit(1)

    try:
        # Initialize the multimodal pipeline (suppress initialization messages in interactive mode)
        if use_interactive_mode:
            # Temporarily suppress print statements during initialization
            import io
            import contextlib

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                pipeline = MultimodalPipeline(
                    face_model=args.face_model,
                    face_features=args.face_features,
                    voice_model=args.voice_model,
                    rec_model=args.rec_model,
                    rec_label_encoder=args.rec_label_encoder,
                    customer_data=args.customer_data,
                    product_catalog=args.product_catalog
                )
        else:
            pipeline = MultimodalPipeline(
                face_model=args.face_model,
                face_features=args.face_features,
                voice_model=args.voice_model,
                rec_model=args.rec_model,
                rec_label_encoder=args.rec_label_encoder,
                customer_data=args.customer_data,
                product_catalog=args.product_catalog
            )

        # Run the pipeline in appropriate mode
        if use_interactive_mode:
            # Interactive mode - prompt user step by step
            result = run_interactive_pipeline(pipeline, args)
        else:
            # Batch mode - use provided arguments
            result = pipeline.run_pipeline(
                image_path=args.image,
                approval_audio_path=args.approval_audio,
                confirmation_audio_path=args.confirmation_audio,
                face_threshold=args.face_threshold,
                voice_threshold=args.voice_threshold,
                top_n_recommendations=args.top_n,
                verbose=not args.quiet
            )

        # Output results
        if args.json:
            import json
            print("\n" + json.dumps(result, indent=2, default=str))
        elif args.quiet:
            print(result['status'])
            if result['status'] == 'authorized' and result['person']:
                print(result['person'])

        # Exit with appropriate code
        if result['status'] == 'authorized':
            sys.exit(0)  # Success
        elif result['stage_failed'] == 'facial_recognition':
            sys.exit(1)  # Face authentication failed
        elif result['stage_failed'] == 'voice_approval':
            sys.exit(2)  # Voice approval failed
        elif result['stage_failed'] == 'voice_confirmation':
            sys.exit(3)  # Voice confirmation failed
        else:
            sys.exit(4)  # Other error

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please ensure all model files are present:")
        print(f"  - Facial model: {args.face_model}")
        print(f"  - Voice model: {args.voice_model}")
        print(f"  - Recommendation model: {args.rec_model}")
        print(f"  - Recommendation label encoder: {args.rec_label_encoder}")
        sys.exit(5)

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(6)


if __name__ == '__main__':
    main()
