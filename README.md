# Car Damage Detection API

An AI-powered FastAPI service that detects and classifies various types of car damage from images. This service uses a deep learning model based on the ResNet50 architecture to identify 14 different types of car damage.

## 🚀 Features
- Real-time car damage detection from images
- Multi-label classification for 14 damage types
- RESTful API endpoints
- Easy integration with frontend applications
- CORS support for cross-origin requests
- Optimized for production deployment

## 🛠️ Technologies Used
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for the damage detection model
- **ResNet50**: Pre-trained neural network architecture
- **Pillow**: Image processing library
- **Python 3.8+**: Core programming language

## 📝 Damage Types Detected
The model can detect the following types of damage:
- Bonnet Dent
- Boot Dent
- Door Outer Dent
- Fender Dent
- Front Bumper Dent
- Front Windscreen Damage
- Headlight Damage
- Quarter Panel Dent
- Rear Bumper Dent
- Rear Windscreen Damage
- Roof Dent
- Running Board Damage
- Side Mirror Damage
- Taillight Damage

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JanithIKahandaSumithra/Car-Damage-Identify-fastapi-app.git
   cd car-damage-detection-api

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Download the model weights and place car_damage_model.pth in the root directory.
5. Start the server:
   ```bash
   uvicorn main:app --reload
The API will be available at http://localhost:8000.




   
