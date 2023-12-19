import argparse
import uuid
from typing import Optional

import torch
import uvicorn
import pathlib
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    BlipForConditionalGeneration,
    GPT2Tokenizer,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
)


class Api:
    def __init__(
        self, checkpoint_path: Optional[str], model_name: Optional[str]
    ) -> None:
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.blip_auto_processor = AutoProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.config = AutoConfig.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration(self.config)
        self.blip_model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
            )
        )

        self.vision_encoder_decoder_model = VisionEncoderDecoderModel.from_pretrained(
            model_name
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/blip")
        async def inference_blip(file: UploadFile):
            file_extension = pathlib.Path(file.filename).suffix
            if file_extension.lower() not in [".jpg", ".jpeg"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file format. Please upload a JPG or JPEG file.",
                )
            new_file_name = str(uuid.uuid4())
            file_location = f"./images/{new_file_name}.jpg"

            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())

            generated_caption = self.generate_caption_blip(image_path=file_location)
            return {"caption": generated_caption}

        @self.app.post("/vision-encoder-decoder")
        async def inference_vision_encoder_decoder(file: UploadFile):
            file_extension = pathlib.Path(file.filename).suffix
            if file_extension.lower() not in [".jpg", ".jpeg"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file format. Please upload a JPG or JPEG file.",
                )
            new_file_name = str(uuid.uuid4())

            file_location = f"./images/{new_file_name}.jpg"

            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())

            generated_caption = self.generate_caption_vision_encoder_decoder(
                image_path=file_location
            )
            return {"caption": generated_caption}

    def generate_caption_blip(self, image_path: Optional[str]):
        image = Image.open(image_path)
        pixel_values = self.blip_auto_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        input_ids = self.blip_model.generate(pixel_values=pixel_values, max_length=82)
        return self.blip_auto_processor.decode(input_ids[0], skip_special_tokens=True)

    def generate_caption_vision_encoder_decoder(self, image_path: Optional[str]):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values
        input_ids = self.vision_encoder_decoder_model.generate(
            pixel_values, max_length=82
        )
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def run(self, port: int):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name", type=str, default="hungsvdut2k2/sport-image-captioning"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/Users/viethungnguyen/Sport-Image-Captioning/weights/checkpoint_4.pth",
    )

    arguments = parser.parse_args()

    app = Api(
        checkpoint_path=arguments.checkpoint_path, model_name=arguments.model_name
    )

    app.run(port=arguments.port)
