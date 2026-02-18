conda create -n frame python==3.10 -y
conda activate frame
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib einops datasets peft wandb sentencepiece decord pytorchvideo diffusers

# 26.02.12: there is an error in recent transformers version for loading Wan model -->
pip install transformers==4.57.3
pip install ftfy regex tqdm opencv-python

# Please refer to "https://github.com/openai/CLIP/issues/528"
# pip install --force-reinstall pip==25.2 setuptools==80.10.2 
pip install git+https://github.com/openai/CLIP.git 

cd pipelines/utils/image_gen_aux
pip install -e .

# download CSD model for stylized video generation
# Download https://huggingface.co/tomg-group-umd/CSD-ViT-L and put it under `.model/CSD-ViT-L`.