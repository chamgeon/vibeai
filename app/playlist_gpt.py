import base64
from openai import OpenAI
import json
from jsonschema import validate, ValidationError
import os
import re
import mimetypes


vibe_extraction_schema = {
    "type": "object",
    "properties": {
        "description": { "type": "string", "minLength": 1 },
        "imagination": { "type": "string", "minLength": 1 },
        "vibes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 50
                    },
                    "explanation": {
                        "type": "string",
                        "minLength": 5
                    }
                },
                "required": ["label", "explanation"],
                "additionalProperties": False
            },
            "minItems": 1
        }
    },
    "required": ["description", "imagination", "vibes"],
    "additionalProperties": False
}

playlist_schema = {
  "type": "object",
  "properties": {
    "name": { "type": "string", "minLength": 1 },
    "description": { "type": "string", "minLength": 5 },
    "tracks": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "song": { "type": "string", "minLength": 1 },
          "artist": { "type": "string", "minLength": 1 },
          "vibe": {
            "type": "string",
            "minLength": 5
          }
        },
        "required": ["song", "artist", "vibe"],
        "additionalProperties": False
      }
    }
  },
  "required": ["name", "description", "tracks"],
  "additionalProperties": False
}

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

def gpt_calling_with_file(file_obj, prompt, temperature=0.9):
    mime_type, _ = mimetypes.guess_type(getattr(file_obj, 'name', 'image.jpg'))
    mime_type = mime_type or "image/jpeg"

    base64_image = base64.b64encode(file_obj.read()).decode("utf-8")
    file_obj.seek(0)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        temperature=temperature,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def gpt_calling(prompt, model = "gpt-5", temperature=0.9):

    if model == "gpt-5":
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={ "effort": "low"},
            text = {"verbosity": "medium"}
        )

        return response.output_text

    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=1000
        )
        
        return response.choices[0].message.content


def verify_response(response, schema):

    # catch backtick case
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", response, re.DOTALL)
    if match:
        response = match.group(1)

    # Parse string to Python obj
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return False, "Invalid JSON format"
    
    # Validate against JSON schema
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        return False, f"JSON Schema validation error: {e.message}"
    
    return True, data



def build_playlist_generation_prompt(vibe_result, base_prompt_template):
    """
    Build the input prompt for playlist generation based on a vibe extraction result.

    Args:
        vibe_result (dict): Output from the first GPT call, containing 'description', 'imagination', and 'vibes'.
        base_prompt_template (str): A template prompt string that contains '[INPUT]' as a placeholder.
    
    Returns:
        str: The final prompt ready for GPT playlist generation.
    """
    description = vibe_result['description']
    imagination = vibe_result['imagination']
    vibes = vibe_result['vibes']

    formatted_vibes = "\n".join(
        f"{vibe['label']} - {vibe['explanation']}"
        for vibe in vibes
    )

    input_block = (
        f"description:\n{description}\n\n"
        f"imagination:\n{imagination}\n\n"
        f"vibes:\n{formatted_vibes}"
    )

    return base_prompt_template.replace('[INPUT]', input_block)


def call_gpt_and_verify(prompt, schema, temperature = 0.9, max_try = 3, file_obj = None):

    """
    Call gpt with prompt and verify it. If verification failed, try again and return error if it failed again.

    Input:
        prompt (str): input prompt
        schema (dict): output JSON schema
        file_obj (file object, optional): file object the user uploaded
        temparature (float): gpt calling temparature
        max_try (int): maximum number of trial
    
    Returns:
        dict or list: Parsed and validated result.
    
    Raises:
        RuntimeError: If all GPT calls fail.
        ValueError: If GPT returns invalid JSON after all retries.
    """

    for attempt in range(1, max_try + 1):
        try:
            if file_obj:
                file_obj.seek(0)
                response = gpt_calling_with_file(file_obj, prompt, temperature)
            else:
                response = gpt_calling(prompt = prompt, temperature=temperature)
        except Exception as e:
            if attempt == max_try:
                raise RuntimeError(f"GPT call failed on attempt {attempt}: {e}")
            continue  # try again

        valid, result = verify_response(response, schema)
        if valid:
            return result  # Success

        if attempt == max_try:
            raise ValueError(f"GPT response invalid after {max_try} tries: {result}")
        # otherwise, retry

    # This line should never be reached
    raise RuntimeError("Unexpected failure in call_gpt_and_verify.")
