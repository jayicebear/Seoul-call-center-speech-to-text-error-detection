# Training file for korean call center speech to text error detection & Correction.

Token level detector Model:monologg/koelectra-base-v3-discriminator
</br>
Span level corrector (Encoder + Decoder) Model :'google/mt5-base','paust/pko-t5-base',hyunwoongko/kobart, Llama-3.1-8b-Instruct, Qwen-2.5-7b-Instruct, GPT4o

We set monologg/koelectra-base-v3-discriminator as the backbone encoder for our detector, leveraging its strong performance on token classification tasks. For the utterance-level baseline detector, we use
klue/roberta-base.
For the correctors and seq2seq-based baseline detectors, we employ google/mt5-base,paust/pko-t5-base, and hyunwoongko/kobart as
backbones. For the LLM-based baseline correctors, we fine-tune Llama-3.1-8b-Instruct and Qwen-2.5-7b-Instruct, while in-context learning relies on GPT-4o
## Result 



