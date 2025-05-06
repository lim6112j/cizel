import os
import requests
import re
import tempfile # Added for temporary file creation
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

    # Create a temporary file to save the image.
    # delete=False is used because Gradio needs to access this file path after the function returns.
    # Gradio will copy the file to its own cache.
    # Note: These temporary files (from delete=False) might accumulate if not managed.
    try:
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp_file:
            output_filename = tmp_file.name
        
        # Ensure the temporary file is closed before writing to it again with open()
        # The 'with' statement for NamedTemporaryFile already closes it upon exiting the block.
        # If it wasn't closed, response.content might write to an open handle incorrectly.

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

        # Save the image to the temporary file path obtained
        with open(output_filename, 'wb') as file:
            file.write(response.content)

        success_message = f"Successfully generated image for '{prompt}'. Saved as: {output_filename}"
        print(f"🖼️ [Stability AI Tool] {success_message}")
        return success_message

    except requests.exceptions.RequestException as e:
        error_message = f"Error calling Stability AI API: {e}"
        print(f"❌ [Stability AI Tool] {error_message}")
        try:
            error_details = response.json() # response might not be defined if request failed early
            error_message += f" - Details: {error_details}"
        except:
             pass
        # Clean up the temporary file if an error occurs before Gradio gets to it
        if 'output_filename' in locals() and os.path.exists(output_filename):
            os.remove(output_filename)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"❌ [Stability AI Tool] {error_message}")
        # Clean up the temporary file if an error occurs
        if 'output_filename' in locals() and os.path.exists(output_filename):
            os.remove(output_filename)
        return error_message
