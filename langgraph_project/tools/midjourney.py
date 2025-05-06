import os
import requests
import re
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@tool
def midjourney_image_generator(prompt: str) -> str:
    """
    Generates an image using the Stability AI API based on the provided prompt
    and saves it locally. Requires STABILITY_API_KEY environment variable.
    """
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        return "Error: STABILITY_API_KEY environment variable not set."
    if not prompt:
        return "Error: Prompt cannot be empty for image generation."

    print(f"🎨 [Stability AI Tool] Received request for prompt: '{prompt}'")

    # Sanitize prompt for use in filename
    safe_prompt = re.sub(r'[^\w\-]+', '_', prompt.lower())
    output_filename = f"./generated_image_{safe_prompt[:50]}.webp" # Limit filename length

    try:
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "image/*"
            },
            files={"none": ''}, # Required by the API
            data={
                "prompt": prompt,
                "output_format": "webp", # Or png, jpeg
                # Add other parameters like aspect_ratio, style_preset etc. if needed
                # "aspect_ratio": "16:9"
            },
            timeout=60 # Add a timeout
        )

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Save the image
        with open(output_filename, 'wb') as file:
            file.write(response.content)

        success_message = f"Successfully generated image for '{prompt}'. Saved as: {output_filename}"
        print(f"🖼️ [Stability AI Tool] {success_message}")
        return success_message

    except requests.exceptions.RequestException as e:
        error_message = f"Error calling Stability AI API: {e}"
        print(f"❌ [Stability AI Tool] {error_message}")
        # Attempt to get more details from response if available
        try:
            error_details = response.json()
            error_message += f" - Details: {error_details}"
        except: # Handle cases where response is not JSON or doesn't exist
             pass
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"❌ [Stability AI Tool] {error_message}")
        return error_message
