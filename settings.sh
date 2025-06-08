conda create -n frame python==3.10 -y
conda activate frame
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib einops transformers datasets peft wandb sentencepiece decord pytorchvideo diffusers
pip install git+https://github.com/openai/CLIP.git opencv-python

cd pipelines/utils/image_gen_aux
pip install -e .