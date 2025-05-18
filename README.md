# Musical Instrument Detector

A web application that can detect musical instruments from images using computer vision techniques. The application uses OpenCV for feature extraction and a rule-based classification system to identify different musical instruments.

## Supported Instruments

- Accordion
- Banjo
- Cello
- Drum
- Guitar
- Piano
- Saxophone
- Trumpet
- Violin
- Xylophone

## Features

- Upload images through a modern web interface
- Real-time instrument detection
- Confidence score visualization
- Responsive design
- Beautiful instrument cards with icons

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Musical-Instrument-Detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python -m uvicorn app:app --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Upload an image of a musical instrument and click "Detect Instrument"

## Project Structure

```
Musical-Instrument-Detector/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── static/               # Static files
└── templates/            # HTML templates
    └── index.html        # Main web interface
```

## Technical Details

The application uses:
- FastAPI for the web server
- OpenCV for image processing and feature extraction
- Custom feature extraction for:
  - Shape analysis
  - Edge detection
  - Color analysis
  - Texture analysis
  - Symmetry detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Icons provided by Icons8
- Built with FastAPI and OpenCV 