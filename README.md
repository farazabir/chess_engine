# chess_engine

![Chess Board Visualization](https://example.com/chess-board-image.jpg) 
*Example board state representation used in model training*

## Abstract
This paper presents a neural chess engine that combines traditional chess programming techniques with deep learning for move prediction. We implement a convolutional neural network (CNN) architecture trained on 2.5 million chess positions from Lichess.org games, achieving competitive move prediction accuracy while maintaining real-time performance. Our model demonstrates the effectiveness of using spatial board representations combined with legal move masking for chess AI development.

## 1. Introduction
Modern chess engines face the dual challenge of strategic evaluation and computational efficiency. This work explores a hybrid approach:

- **Spatial Board Encoding**: 13-channel matrix representation (12 piece layers + legal moves)
- **Deep Learning Architecture**: Custom CNN with optimized parameter initialization
- **Large-Scale Training**: Leveraging Lichess's open database of 10,000 games
- **Real-Time Prediction**: GPU-accelerated inference pipeline

## 2. Methodology

### 2.1 Data Processing Pipeline
```python
def board_to_matrix(board: Board):
    # 13-layer tensor representation:
    # Layers 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    # Layers 6-11: Black pieces
    # Layer 12: Legal move mask
    ...


Key Features:

PGN parsing with python-chess library
Move encoding using Universal Chess Interface (UCI) format
Batch processing with PyTorch DataLoader
Memory-efficient tensor storage (float32 precision)
2.2 Neural Architecture
python

Run

Copy
class ChessModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3)  # Input: 13x8x8
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, num_classes)
Architecture Details:

2 convolutional layers with ReLU activation
Kaiming initialization for convolutional layers
Xavier initialization for fully connected layers
Gradient clipping (max norm = 1.0)
Adam optimizer (lr=0.0001)
2.3 Training Protocol
Hardware: NVIDIA T4 GPU (Google Colab)
Batch Size: 64 positions
Epochs: 50
Loss Function: Cross-entropy with label smoothing
Regularization: Automatic legal move filtering
