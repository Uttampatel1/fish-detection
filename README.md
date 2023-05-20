# Fish Location Detection

Fish Location Detection is a project aimed at providing fish enthusiasts with a simple tool to retrieve the geographic coordinates (latitude and longitude) of a specific fish species. By entering the name of a fish, users can obtain the corresponding location information where that fish is commonly found.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with the Fish Location Detection project, follow the instructions below.

### Prerequisites

- Python 3.x
- Pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Uttampatel1/fish-detection.git
    ```

2. Navigate to the project directory:

    ```bash
    cd fish-detection
    ```
3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main Python script:

    ```bash
    python main.py
    ```
2. Enter the name of the fish species you want to find the location for.

3. The program will display the latitude and longitude coordinates for the specified fish species.

## API Documentation

The Fish Location Detection project provides a simple API endpoint for retrieving fish location information programmatically.

### Endpoint

```bash
POST http://127.0.0.1:5000/predict
```

### Parameters

The API endpoint accepts a JSON object with the following structure:

```json
[
    {
    "English Name":"Mud crab"
    }
]
```

### Response

The response will be a JSON object with the following structure:

```json
{
  "lat": 21.767,
  "lon": 78.901
}
```
