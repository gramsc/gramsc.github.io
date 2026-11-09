---
title: "Understand modern Speech-to-text models"
date: 2026-11-09
category: lorem
summary: "An overview of how speech-to-text models works, and the evolutions of its architecture."
---

Automatic speech recognition (ASR), or *Speech-to-text*, is the task of turning
an audio into written text.

For decades, ASR systems were composed of: 
- an *acoustic model* that matched short slices of sound to speech
sounds (phonemes), 
- a *pronunciation dictionary* that turned those sounds into
words 
- a *language model* that picked the most likely sequence of words.


With the developments in Artificial Intelligence, the research for speech recognition explored different
models such as Multi-layer Perceptrons (MLP), Hidden Markov models (HMM), Gaussian-Mixture models (GMM)
and more refined neural networks such as Recursive Neural Networks (RNN), especially with RNN-CTC.

But, in 2017 Google released Transformers, which lead to a lot of new algorithms
and namely it took a bit of time but in speech recognition as well: **Whisper**, released by OpenAI. 

This post opens the hood of four of today's open models: from **Whisper**,
**Parakeet TDT v3**, **Voxtral**, and the family of **SALMs** (speech-augmented
language models). The goal is not to list their specifications, but to
understand the *ideas* behind each architecture and what each model brings.

The figures are based on the notes I drew in Excalidraw while studying these
models.

![Four models side by side: their encoder, the link, and their decoder](/images/speech-to-text/roadmap.svg)

*Figure 1. Four ways to build a speech-to-text model. Each one encodes the audio
into a representation (a sequence of embeddings), then decodes that
representation into text. Voxtral and Canary-Qwen follow the same recipe, called
SALM.*

## A few words before we start

You don't need a machine learning background to follow along, but five terms
come up again and again:

- **Token**: a small piece of text, often a word or part of a word. Language
  models read and write text one token at a time.
- **Embedding**: a list of numbers that represents something: a word, or a slice
  of sound. Similar things get similar lists of numbers.
- **Transformer**: the neural network architecture behind ChatGPT and almost
  every model in this post. Its key operation is **attention**: every element of
  the input looks at every other element and decides which ones matter for it.
- **Encoder**: the part of a model that turns the input (here, sound) into a
  *representation*: a sequence of embeddings that describe what is in it.
- **Decoder**: the part that turns that representation into the output (here,
  text).

Most of the history below is a story of keeping the encoder roughly the same and
reinventing the decoder.

## Step one, shared by everyone: turning sound into a picture

A microphone records sound as a long list of numbers: how far the air pressure
pushes the membrane, measured 16,000 times per second (16 kHz) for speech
models. Thirty seconds of audio is 480,000 numbers. That raw list is hard to
learn from, so every model in this post first converts it into something that
looks more like an image.

![Cutting a 30-second segment into overlapping 25 ms windows, one every 10 ms](/images/speech-to-text/windows.svg)

*Figure 2. The audio is cut into tiny overlapping windows: 25 milliseconds long,
starting every 10 milliseconds. That gives 100 windows per second, or 3,000 for 30
seconds.*

Each window is then analysed with a **Fourier transform**, a classic piece of
maths that answers the question *"which pitches are present in this slice of
sound, and how loud is each?"*. Finally, these pitches are grouped into 80 bands
(128 in newer models) spaced the way human hearing works: lots of detail in the
low pitches where speech lives, less in the high ones. This is called the
**mel scale**.

![Windows → Fourier transform → mel spectrogram](/images/speech-to-text/spectrogram.svg)

*Figure 3. From 3,000 windows of raw sound, to 3,000 columns of pitches, to 3,000
columns of 80 mel bands.*

The result is a **spectrogram**: a picture where time runs left to right, pitch
runs bottom to top, and brightness shows loudness. Vowels show up as horizontal
stripes, consonants as short bursts. From here on, recognising speech is not so
different from reading an image.

## Whisper (OpenAI, 2022): the template

Whisper was the model that made open speech recognition mainstream. Its design
is simple, and almost everything that came after it is a reaction to it, so it is
worth going through slowly.

### Everything happens in 30-second boxes

Whisper only ever looks at exactly 30 seconds of audio at a time. Shorter
recordings are padded with silence. Longer ones are cut into 30-second segments,
which common implementations overlap by a few seconds so that no word is cut in
half, before stitching the texts back together.

![Long audio cut into overlapping 30-second segments](/images/speech-to-text/chunking-audio.svg)

*Figure 4. Long audio is cut into 30-second segments with a small overlap.*

### Shrinking time: the convolutional stem

3,000 columns per segment is a lot. Before the main network, Whisper runs two
small **convolutions**: filters that slide along time and look at three
neighbouring columns at once, picking up very local patterns. The second filter
jumps two columns at a time instead of one, which halves the length: 3,000
columns become **1,500 frames, one every 20 milliseconds**.

![Two 1-D convolutions; the second one has stride 2 and halves the number of frames](/images/speech-to-text/conv-stem.svg)

*Figure 5. The convolutional stem. Each filter looks at three neighbouring time
steps; the second one moves by two steps, halving the sequence.*

### The encoder: listening

Transformers have no built-in sense of order: to them, the input is a bag of
frames. So Whisper adds a **positional encoding** to each frame, a fixed wave
pattern that acts like a timestamp written into the numbers. Then the frames go
through a stack of transformer layers (from 4 in the tiny model to 32 in the
large one). In each layer, every frame can look at every other frame in the 30
seconds. What comes out is 1,500 embeddings, each describing *what is being said
around this moment, given everything else in the segment*.

### The decoder: writing

The decoder works like a chatbot: it writes the transcript one token at a time.
At each step it looks at two things: the text it has written so far, and the
audio embeddings, through a mechanism called **cross-attention** (a window from
the writer to the listener).

What made Whisper clever is how it is *instructed*. The decoder doesn't start
from a blank page. It starts from a few special tokens that tell it what to do:

```text
<|startoftranscript|> <|fr|> <|transcribe|> <|notimestamps|>
```

This means "this is French, write down what is said, no timestamps". Swap
`<|transcribe|>` for `<|translate|>` and the same model translates into English.
Leave out `<|notimestamps|>` and it interleaves timing tokens with the text. If
the segment contains no speech, it can answer `<|nospeech|>`. For long audio,
the text of the previous segment is fed back in as a prompt, so the model keeps
names and style consistent from one box to the next.

![Instruction tokens, then the transcript written token by token; the decoder looks at the audio through cross-attention](/images/speech-to-text/whisper-decoder.svg)

*Figure 6. Whisper's decoder starts from instruction tokens (red), then writes the
transcript one token at a time, looking back at the audio at every step.*

### Training: quantity over polish

Whisper was trained on 680,000 hours of audio collected from the web, paired
with transcripts that already existed (subtitles, captions) and were never
checked by hand. This is called *weak supervision*. About two-thirds was
English, the rest covered close to 100 languages, including translation pairs.
The lesson was that a large amount of messy data makes a model robust to
accents, background noise and bad microphones, more than a small, clean dataset
does.

### The three limits

Whisper's design has three weaknesses, and they frame everything that follows:

1. **The 30-second box.** Every input is processed as 30 seconds, even a
   two-second voice command, and long recordings have to be cut and glued back
   together.
2. **The decoder can make things up.** Because it writes freely, like a language
   model, it sometimes produces text that was never said, especially on silence
   or music, or gets stuck repeating a sentence. These are called
   **hallucinations**.
3. **Speed.** Writing one token at a time with a large decoder is slow. OpenAI's
   later "turbo" version cut the decoder from 32 layers to 4 to speed it up.

## Parakeet TDT v3 (NVIDIA, 2025): a different way to write

NVIDIA's Parakeet TDT 0.6B v3 is small (600 million parameters, against 1.5
billion for Whisper large) and covers 25 European languages. It was trained on
about 660,000 hours of speech, largely from NVIDIA's open Granary dataset. It
changes both halves of the model.

### The encoder: fewer frames, longer audio

Parakeet's convolutional front end shrinks time **eight times** instead of two:
**one frame every 80 milliseconds**. A 30-second clip becomes 375 frames instead
of 1,500. That is still enough detail for speech, because a syllable lasts
around 200 milliseconds.

![One second of audio sliced every 10 ms, every 20 ms and every 80 ms](/images/speech-to-text/frames-per-second.svg)

*Figure 7. The same second of audio at three resolutions. Going from 50 to 12.5
frames per second still leaves two or three frames per syllable.*

Why does it matter? Attention compares every frame with every other frame, so
its cost grows with the *square* of the length: four times fewer frames means
roughly sixteen times less work. That is what lets Parakeet drop the 30-second
box: it can process around twenty minutes of audio in a single pass. For even
longer recordings it can switch to **local attention**, where each frame only
looks at its neighbours, and handle hours of audio.

The encoder itself is a **FastConformer**: transformer layers with convolution
modules added in between. Convolutions are good at short, local patterns, like
the shape of a syllable. Attention is good at long-range context, like the topic
of the conversation. Combining both works better for speech than either one
alone. And instead of stamping an absolute position on each frame, it uses
*relative* positions: what matters is how far apart two frames are, not where
they sit in the file.

### The decoder: a transducer, not a writer

This is the deepest change. Whisper's decoder writes freely and *glances* at the
audio when it needs to. Parakeet's decoder is a **transducer**: it walks through
the audio frames from left to right, and at each frame it decides *"is a new
token spoken here, or nothing?"*. A small network remembers the text written so
far, and a second one combines that memory with the current audio frame to make
the decision.

Because the decoder is forced to move forward through the audio, it cannot drift
off into text that isn't there, so it hallucinates much less than Whisper. And
since every token is tied to a frame, timestamps come for free.

**TDT** stands for *Token-and-Duration Transducer*. On top of the token, the
model also predicts **how many frames it can skip** before the next decision.
Silences, pauses and long vowels get jumped over instead of being examined frame
by frame, which makes Parakeet one of the fastest open transcription models.

![A transducer makes one decision per frame; TDT jumps over frames](/images/speech-to-text/transducer-tdt.svg)

*Figure 8. Top: a plain transducer looks at every frame and usually outputs
"nothing". Bottom: TDT also says how far to jump, and skips the frames in between.
(Simplified example.)*

### The trade-off

Parakeet does one thing: transcription, with punctuation and capital letters. It
takes no instructions, no prompt, and it doesn't "understand" what is said in
any broader sense. For transcription, that is a strength.

## Voxtral (Mistral AI, 2025): Whisper's encoder, an LLM as decoder

Voxtral took the opposite road. Instead of inventing a new decoder, Mistral asked:
*what if the decoder were a full large language model (LLM)?* Voxtral comes in
two sizes: Mini, built on the 3-billion-parameter Ministral 3B, and Small, built
on the 24-billion-parameter Mistral Small 3.1.

### Encoding: Whisper's encoder, chunk by chunk

Voxtral reuses Whisper's large-v3 encoder: no need to reinvent a part that
already works. It still processes audio in 30-second chunks, but now the chunks
are encoded **independently** (the position counter restarts at zero for each
one), and their outputs are simply placed end to end. This is **stitching**.
Nothing is lost at the boundaries, because the LLM that comes next sees all the
chunks at once.

### The adapter: compressing for the LLM

Whisper's encoder produces 50 frames per second. An LLM reads audio frames the
same way it reads words, and 30 minutes of audio at that rate would be 90,000
"words", far too many. So Voxtral adds a small **adapter** network that merges
every 4 consecutive frames into one: **12.5 frames per second, one every 80
milliseconds**. That is exactly Parakeet's rate, reached by a completely
different route. At this rate, 40 minutes of audio fits in the model's
32,000-token context window.

![Chunk outputs stitched end to end, then downsampled 4x by the adapter](/images/speech-to-text/voxtral-stitching-adapter.svg)

*Figure 9. Each 30-second chunk goes through the encoder on its own. The results
are stitched end to end, then the adapter divides their number by four: 40
minutes of audio becomes about 30,000 frames.*

### Decoding: an ordinary language model

The compressed audio embeddings are placed directly in the LLM's input, as if
they were words. There is no cross-attention any more. The LLM reads audio and
text with the same mechanism it uses to read a document.

To teach the LLM to "read" sound, Mistral trained it on two kinds of examples,
marked with special tokens:

- `<repeat>`: a piece of audio followed by its transcript. This teaches the
  model to transcribe.
- `<next>`: a piece of audio followed by the text that comes *after* it. This
  teaches the model to understand what was said well enough to continue the
  conversation.

### What you gain, and what it costs

The same model can transcribe a recording, summarise it, answer questions about
it, or translate it, all from a plain-language instruction. The cost: billions
of parameters instead of 600 million, and a decoder that writes freely again, so
the risk of hallucination is back.

## SALMs: the recipe becomes a family

Voxtral is one example of a broader family that NVIDIA calls **SALMs**,
*speech-augmented language models*. The recipe always has the same three parts:

1. a **speech encoder**,
2. a small **adapter** that converts its output into the LLM's "language",
3. a **pretrained LLM** as the decoder, often kept frozen and only lightly adjusted
   with small add-on weights (a technique called LoRA), so that it doesn't lose
   its text abilities.

![Audio → encoder → adapter → audio tokens next to text tokens → LLM → text](/images/speech-to-text/salm-recipe.svg)

*Figure 10. The SALM recipe. The adapter turns audio frames into "words" the LLM
can read, placed right next to the text of the instruction.*

NVIDIA's **Canary-Qwen-2.5B** is a good example: it plugs the FastConformer
encoder of NVIDIA's Canary model into Alibaba's Qwen3-1.7B LLM. Its encoder is
from the Parakeet side of the family, and its decoder is a general-purpose LLM,
like Voxtral's. Other labs have built similar models, such as Alibaba's
Qwen2-Audio and IBM's Granite Speech.

Encoders and LLMs have become interchangeable parts: you can pick an encoder
and an LLM and connect them with an adapter.

## The evolution in one table

|                         | Whisper             | Parakeet TDT v3          | Voxtral                 | Canary-Qwen              |
| ----------------------- | ------------------- | ------------------------ | ----------------------- | ------------------------ |
| Year                    | 2022                | 2025                     | 2025                    | 2025                     |
| Encoder                 | Transformer         | FastConformer            | Transformer (Whisper's) | FastConformer            |
| Decoder                 | Transformer decoder | Transducer (TDT)         | Full LLM                | Full LLM                 |
| How the decoder reads the audio | Cross-attention     | One frame at a time      | Audio as input "words"  | Audio as input "words"   |
| Audio frames per second | 50                  | 12.5                     | 12.5                    | 12.5                     |
| Takes instructions?     | A few fixed tokens  | No                       | Yes, in plain language  | Yes, in plain language   |

Three trends stand out:

1. **Encoders change slowly, decoders change fast.** Whisper's encoder lives
   on inside Voxtral and Qwen2-Audio. The real innovation happened in how text
   is produced.
2. **Fewer frames per second.** From 50 to 12.5. Attention and LLM context are
   expensive, and 80 milliseconds per frame turns out to be enough to capture
   speech.
3. **Two directions.** On one side, the specialists like Parakeet: small, fast,
   faithful to the audio, transcription only. On the other side, the
   generalists like Voxtral and the SALMs: large and flexible, able to
   understand and reason about what is said. Which one you want depends on the
   job.

## Why this matters

Every model in this post can be downloaded and run on your own hardware, under
permissive licences (MIT or Apache 2.0 for Whisper depending on where you get
it, Apache 2.0 for Voxtral, CC-BY-4.0 for NVIDIA's models). Transcribing a meeting, an interview or a union assembly no
longer requires handing the recording to someone else's cloud.

"Open" does not mean the same thing for each of them, though. Having the weights
is not the same as having the training data or the means to train them again.
That will be the subject of a future post.

## Sources

- Radford et al., [*Robust Speech Recognition via Large-Scale Weak Supervision*](https://arxiv.org/abs/2212.04356) (Whisper), 2022.
- Rekesh et al., [*Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition*](https://arxiv.org/abs/2305.05084), 2023.
- Xu et al., [*Efficient Sequence Transduction by Jointly Predicting Tokens and Durations*](https://arxiv.org/abs/2304.06795) (TDT), 2023.
- Chen et al., [*SALM: Speech-augmented Language Model with In-context Learning for Speech Recognition and Translation*](https://arxiv.org/abs/2310.09424), 2023.
- Mistral AI, [*Voxtral*](https://arxiv.org/abs/2507.13264), 2025.
- NVIDIA, [*Canary-1B-v2 & Parakeet-TDT-0.6B-v3*](https://arxiv.org/abs/2509.14128), 2025, and the [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) model cards.
