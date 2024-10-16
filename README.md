## Description

### tl;dr

[18 Oct 2024: The setup instructions below are brand new. The pipeline descriptions below have also been brought up to date. Too much detail about development so far available at [stuff.danavery.com](https://stuff.danavery.com/tags/audio-tokens/], including a (temporary?) wrap-up post.]

Cluster STFT vectors to create an audio token vocabulary, then send token sequences for each example audio file through a transformer or LSTM for classification.

### The concept

Audio classification has typically been done by producing spectrograms (usually mel-scale amplitude plots) and then using visual ML techniques on the images generated to identify relevant audio features with a CNN. Given the success of transformers in classifying and generating language sequences, more recently there have been attempts to use transformers for image and audio processing. In the audio world, the Audio Spectrogram Transformer ([AST](https://arxiv.org/abs/2104.01778)) is the most well-known attempt. It is based on the Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929)), which chops an image into an overlapping grid, then uses the flattened grid elements as embeddings to train a transformer, skipping the tokenization process entirely. The AST does the same, but with audio spectrograms.

But I was thinking...unlike an image, an audio clip is really a sequence in the same way that a sentence is in a text document, and by slicing it up into 2D squares, it seems like some valuable time information might be lost, even with positional encodings added on--they're still 2D positions. So why not try to treat an audio clip as a sequence instead of an image? Then maybe you could train a transformer to classify audio without losing the essential time-based nature of sound?

This brings up a couple possible options.

* Skip tokenization.  Like AST does with its 2D patches, treat each individual STFT time-slice of an audio file as a pseudo-embedding and feed it into a transformer. That way you don't have the slightly awkward conversion of a 2D patch into a 1D embedding of the AST. Treat the mel-spectrum vectors as pre-generated token embeddings.
* Tokenize the audio.  Take all of the STFT time-slice vectors in all of your training data and feed them into a clustering algorithm to create a set of clusters of STFT vectors. Now you have a "vocabulary" of STFT clusters, and with a vocabulary, you can tokenize your audio. Number all of the cluster centroids, and for every STFT slice in a particular audio example in your training data, the nearest cluster centroid is the next "token" in the audio sequence. Covert every audio file into a linear sequence of centroid IDs, and use those sequences to train a transformer. That way your model can learn embeddings for each token instead of processing pseudo-embeddings which are all going to be unique.

To be honest, the second one seemed more interesting. It would be an attempt to directly adapt the power of text transformers to the audio space.

So I've started an experiment using AudioSet to test it. Tokenize the audio and train a transformer with those sequences of tokens and see if it can tell the difference between a dog barking and a door slamming.

## Details


### Training/Val Set Generation

DatasetSplitter is set up to use either a random sample of the AudioSet "balanced_train" set (20,000 examples), the "unbalanced_train" set (~2,000,000 examples), or a combination of the two. You can specify whether you want to include one or the other, and what portion of the combination you want to use for training and validation, as well as what portion of _that_ set you want to use as a validation set. The names of examples in the generated train/val sets are saved in a JSON file.

### Spectrogram Generation (SpectrogramGenerator)

This is a fairly straightforward mel-spectrogram generator. Take an audio file, make a picture of its amplitude per frequency bin per time period.

Given an instance of AudioTokensConfig, it uses:
* The names/YouTubeIDs of the train and val example files, pulled from the JSON file created above.
* The location of a top-level directory of AudioSet audio files. Default is "/media/davery/audioset" because that's where my AudioSet HDD is mounted.
* The "source set" of the dataset(s) to use. For AudioSet the possible values are "bal_train", "unbal_train". This is really just a list of search paths for where to look for the train and val files.

Output:

* An .npy file for each audio file containing the spectrogram. Default location is `spectrograms/{train, validation}'.

Hyperparameters:

* *sample_rate*: 22050. Files that aren't at this rate get converted to it. Why 22050? It's common, and why not?
* *normalization*: True. All of the spectrogram values are individually linearly normalized to be between 0 and 1.
* *n_fft*: 1024. A typical starting point for STFT window size
* *hop_length*: 512. Overlap the STFT windows by half their size
* *n_mels*: 64. Now this gets more interesting. 64 is a common starting point, but I can see this one needing to be adjusted. (I have conceptual issues with using the mel scale for sounds not generated by or for humans, given where it comes from, but we'll stick with it for now.)

### Generating the clusters/vocabulary (ClusterCreator)

This is where the audio token vocabulary is generated. Take each time-slice vector of every spectrogram in the training set, throw them all into a K-Means clustering algorithm (currently using FAISS for this), and come up with a list of centroids. Each of your centroids/clusters is now a token in the audio vocabulary of the training corpus.

Input:

* Spectrograms of the training set. By default in AudioTokensConfig they're assumed to be in the directory used to save them above: `spectrograms/train/`

Output:

* A numpy array of `vocab_size` cluster centroid locations. By default saved to "output/centroids.npy"

Hyperparameters:

* *vocab_size*: This is a big one. This is the number of clusters to generate, or how large we want our corpus vocabulary to be. Too many or too few and the clusters won't really be dense enough or distinct enough to convey any useful information. 50 is almost certainly too few, but we'll see.
* *niter*: How many iterations for FAISS to perform in the clustering operation.

### Tokenizing the training set (SpecTokenizer)

Now for every file in the training and validation sets, we classify each STFT time-slice vector into one of *n_cluster* clusters based on distance from the centroids. We convert our each spectrograms into a sequence of audio tokens.

Input:

* Spectrograms of the training and validation sets
* the cluster centroid locations just created

Output:

* Tokenized versions of every example audio file. Default location is "tokenized_audio/{train,validation}"

Hyperparameters:

* None.

### Training a model (ModelTrainer)

For `bert', we take a standard uninitialized BERT model and add layers to send the [CLS] token vector into a single linear layer to project it into 534 possible outputs, one for each used AudioSet label.
For 'lstm' we use an LSTM classifier

Input:

* Tokenized audio files

Output:

* Magic. Or maybe just a transformer that can classify AudioSet files.
* Train and validation mean average precision scores (mAP), since that's what AudioSet models are typically evaluated with.

Hyperparameters:

* *vocab_size*: Same as in ClusterCreator
* *num_layers*: How many layers to use for a BERT or LSTM model
* *epochs*: 20
* *batch_size*: I've been using whatever I can use without overloading on my RTX 3070, meaning something low. 16/32 or so.
* *learning_rate*: This should probably not be a constant, but haven't gotten around to learning rate schedules yet.

## Setup

### Get the code and dependencies

* `git@github.com:danavery/audio-tokens.git`
* `cd audio-tokens`
* `conda env create --name audio-tokens --file=environment.yml`

### Get AudioSet

AudioSet is a set of labeled 10-second audio clips from YouTube. Unfortunately Google doesn't make the audio clips easily available, so we have to go to a user-contributed Hugging Face repository.

You're going to need a lot of space. Something like 2TB for the initial tar download, and another 2TB for the expanded audio files.

* Install the Hugging Face CLI: `pipx install huggingface_hub[cli]` if you use pipx (like me) or `pip install -U "huggingface_hub[cli]"`
* `cd` to wherever you have the first 2TB of extra space. I'll call this [audioset].
* `huggingface-cli download agkphysics/AudioSet --local-dir . --repo-type dataset`. This will take a while.
* `cd` to the top of the project directory
* `mkdir metadata`
* `cp [audioset]/data/*json [audioset]/data/*.csv metadata/` to move the AudioSet ontology and index files to the project

Now you need to untar all those .tar files into individual audio files:

* `cd [audioset]/data`
* update BASE_DEST_DIR in tools/audioset_expander.py to wherever you have the second 2TB
* python tools/audio_expander.py. This will take a while. It's going to expand and spread out all the audio files into a series of subdirectories to keep from overwhelming the filesystem.

### Create train and dev datasets

* Edit audio_tokens_config.py. (This assumes you want to start by using the small balanced_train dataset to start.)
    * Change `audio_source_path` to your expanded audio file root directory.
    * Change `dataset_ratio` to the portion (0.0-1.0) of the balanced_training set you want to use for train+val.
    * Change `validation_ratio` to the portion (0.0-1.0) of the above dataset to use for validation


* `python -m processors.dataset_splitter` to create the train/val split csv in the `metadata` directory

### Run the basic task

* `python run_pipeline.py` to train an LSTM with tokenized audio and some basic settings

### More Configuration

All config is in `audio_tokens_config.py`

* `n_fft`, `n_fft`, `hop_length` to change how the spectrograms are generated. Set `normalize` to enable per-spectrogram normalization
* `vocab_size` to change the number of token cluster to use
* `use_convolution`, `num_kernels`, and `kernel_size` to run 1D convolution on the STFT inputs. I'm not sure this helps at all
* `model_type`: one of `baseline`, `cnn`, `bert`, `lstm`, `simple`. Models are in `models/`.
* For LSTM: `lstm_embed_dim` and `lstm_hidden_dim`
* For other models: `hidden_size`
* `dropout` is also an option for `lstm` and `bert`
* `dataset_type`: Dataset classes reside in `dataset/`
    * For models that use tokenization (`lstm`, `bert`, `simple`), use `TokenizedSpecDataset`
    * For models that use individual 1D STFT vectors (`baseline`), use `RawSTFTFlatDataset`
    * For models that use 2D spectrograms as input (`cnn`), use `RawSTFTDataset`
    * Special case: If you want to feed the raw 1D STFT vectors into BERT, using them as token embeddings, set `use_precomputed_embeddings` to True
    * `layers` is the number of layers to use for models that have layers as a parameter ('bert', 'lstm')
