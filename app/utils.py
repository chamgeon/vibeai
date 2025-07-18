import uuid, os, json
import boto3
from PIL import Image
import io
import requests
import base64

def upload_to_s3(file):
    s3 = boto3.client("s3", region_name="us-east-2")
    file.seek(0)
    filename = str(uuid.uuid4()) + ".jpg"

    s3.upload_fileobj(
        file,
        os.environ["AWS_BUCKET_NAME"],
        filename
    )

    return filename


def generate_presigned_url(filename, expiration=36000):
    s3 = boto3.client("s3", region_name="us-east-2")
    return s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': os.environ["AWS_BUCKET_NAME"], 'Key': filename},
        ExpiresIn=expiration
    )


def resize_image_by_longest_side(file_obj, target_longest_side=512):
    """
    Resize an image (from a file object) so that the longest side equals `target_longest_side`,
    preserving the aspect ratio. Returns a new BytesIO object containing the resized image in JPEG format.
    
    Args:
        file_obj (file-like): An image file object (e.g., from `open(file, 'rb')` or an upload).
        target_longest_side (int): The desired size for the longest side (width or height).
    
    Returns:
        BytesIO: A file-like object containing the resized image (JPEG format).
    """
    image = Image.open(file_obj)
    original_width, original_height = image.size
    
    # Determine the scale factor
    if original_width >= original_height:
        scale = target_longest_side / float(original_width)
    else:
        scale = target_longest_side / float(original_height)

    new_size = (int(original_width * scale), int(original_height * scale))
    resized_image = image.resize(new_size, Image.LANCZOS)

    # Save to BytesIO object
    output = io.BytesIO()
    resized_image.save(output, format='JPEG')
    output.seek(0)
    return output

def download_image(url):
    """
    download image from url and return base64 image
    """
    response = requests.get(url)
    response.raise_for_status()
    image_bytes = response.content
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64