# HSapp

HSapp is a small Streamlit web app for detecting toxic Myanglish text.

It loads a trained TensorFlow model from `modelv3.h5` and a tokenizer from `tokenizer.pickle`, then predicts whether the entered text is toxic or non-toxic.

## Features

- Streamlit-based web interface
- TensorFlow model inference
- Pickled tokenizer loading
- Toxic vs non-toxic prediction for Myanglish input
- Simple result message shown in Burmese

## Project Structure

```text
HSapp/
|- app.py
|- modelv3.h5
|- tokenizer.pickle
|- logo.png
|- requirements.txt
|- README.md
```

## Requirements

- Python 3.9 or newer recommended
- `pip`

## Installation

Install project dependencies:

```bash
pip install -r requirements.txt
```

Current dependencies:

- `tensorflow==2.19.0`
- `streamlit`
- `keras-preprocessing`

TensorFlow version note:

- This project is pinned to TensorFlow `2.19.0`, which is the latest stable TensorFlow release on PyPI at the time this README was updated.

## Run the App

Use the hosted app directly:

`https://hatespeechpy.streamlit.app/`

Or start the app locally from the project folder:

```bash
streamlit run app.py
```

Streamlit will print a local address in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. The app loads the trained model from `modelv3.h5`.
2. It loads the tokenizer from `tokenizer.pickle`.
3. User input is converted into token IDs.
4. The sequence is padded to length `100`.
5. The model predicts whether the text is toxic.
6. The app displays one of several preset response messages.

## Files

- `app.py`: main Streamlit application and prediction logic
- `modelv3.h5`: trained Keras model
- `tokenizer.pickle`: saved tokenizer used for preprocessing
- `logo.png`: app image asset
- `requirements.txt`: Python dependencies

## Usage

1. Open the hosted app at `https://hatespeechpy.streamlit.app/` or run it locally with `streamlit run app.py`.
2. Type a Myanglish sentence into the input box.
3. The app predicts whether the text is toxic or not.

## Troubleshooting

### `ModuleNotFoundError: No module named 'keras_preprocessing'`

This happens when the tokenizer file was created with `keras_preprocessing`, but that package is not installed in the current environment.

Fix:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install keras-preprocessing
```

### TensorFlow install notes on Windows

- On Windows, install TensorFlow with `pip`, not `conda`.
- Native Windows GPU support only goes up to TensorFlow `2.10`. For newer TensorFlow GPU workflows, TensorFlow recommends WSL2.
- CPU usage on native Windows is still supported via the `tensorflow` pip package.

### Streamlit context warnings when using `python app.py`

Run the app with Streamlit, not plain Python:

```bash
streamlit run app.py
```

### Model-loading warnings

The app now loads the model with `compile=False` because it only needs inference, not training. This avoids extra warnings related to restoring older optimizer state from the saved `.h5` model.

You may still see some non-blocking compatibility warnings from older saved Keras model artifacts. These do not usually prevent the app from running.

## Current Notes

- The model and tokenizer files need to remain in the project root.
- The app expects Myanglish input.
- Some UI strings in `app.py` appear to have text encoding issues.
- The saved model may still show a few non-blocking compatibility warnings depending on your TensorFlow/Keras version.
- If you use a different TensorFlow major or minor version, older saved model artifacts may produce extra warnings during load.

## Future Improvements

- Improve the UI text and result labels
- Show prediction confidence
- Clean up text encoding in the source
- Add deployment instructions
- Add tests for model-loading and prediction flow
