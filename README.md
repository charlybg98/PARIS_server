# PARIS_server

This repository serves as the server-side component for PARIS (Plataforma de Aprendizaje con Reconocimiento Inteligente y Secuencial), an advanced learning assistant leveraging intelligent recognition and sequential processing to enrich the educational experience. By integrating artificial intelligence into teaching and learning, PARIS aims to offer an interactive and personalized approach for both students and educators. The server-side specifically handles image recognition tasks, allowing for real-time interaction between the client-side application and the server.

## Getting Started

### Prerequisites

- Python 3.7
- TensorFlow 3.4
- NumPy 1.19.2

### Installation

1. Clone this repository to your local machine.
2. Ensure Python 3.8 or higher is installed on your system.
3. Install the required Python packages by running:
4. Place your TensorFlow model in the specified directory (`~/Documents/PARIS/models/ImageToAction`).
5. Update the `server_config.json` file with your server's IP address and port settings.

### File Structure

- `server.py`: Main server script for image recognition and client communication.
- `server_config.json`: Configuration file with server IP and port.
- `warmup.png`: Image used for TensorFlow model warm-up to ensure faster inference.

## Usage

To launch the server, execute:

python server.py

Upon startup, the server initializes and listens for connections from the client-side of PARIS. It processes received images through the TensorFlow model and returns the recognition results to the client.

## Contributions

Your contributions to the server-side of PARIS are highly valued. If you have improvements or fixes you'd like to suggest, please fork the repository and submit a pull request.

## Acknowledgments

A heartfelt thank you to everyone who has contributed to PARIS's development, especially to family, friends, and advisors. Your support and inspiration have been pivotal in this journey.
