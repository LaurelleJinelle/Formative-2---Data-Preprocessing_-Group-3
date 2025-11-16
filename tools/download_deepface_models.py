"""Small helper to pre-download DeepFace model weights.

Usage:
  python tools/download_deepface_models.py --models Facenet,OpenFace

If no --models provided, defaults to Facenet.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Pre-download DeepFace models')
    parser.add_argument('--models', default='Facenet', help='Comma-separated model names (e.g. Facenet,OpenFace)')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]

    try:
        from deepface import DeepFace
    except Exception as e:
        print('DeepFace import failed. Make sure you installed dependencies (pip install deepface).', file=sys.stderr)
        print(e, file=sys.stderr)
        return 2

    deepface_home = os.environ.get('DEEPFACE_HOME')
    if deepface_home:
        print(f'Using DEEPFACE_HOME={deepface_home}')

    for model_name in models:
        print(f'Building/loading model: {model_name} (this will download weights if missing)')
        try:
            _ = DeepFace.build_model(model_name)
            print(f'Model {model_name} ready')
        except Exception as e:
            print(f'Failed to build/load model {model_name}: {e}', file=sys.stderr)

    print('Done. Model weights should be cached under DeepFace cache directory (e.g. ~/.deepface/weights/)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
