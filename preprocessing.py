from PIL import Image
import os

def resize_images(directory, target_size):
    for filename in os.listdir(directory):
        if filename.endswith(".heic"):
            heic_path = os.path.join(directory, filename)
            jpeg_path = os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")
            try:
                with Image.open(heic_path) as image:
                    image.save(jpeg_path)
                print(f"Converted {filename} to JPEG")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".jpeg"):
            image_path = os.path.join(directory, filename)
            try:
                with Image.open(image_path) as image:
                    resized_image = image.resize(target_size)
                    resized_image.save(image_path)
                    print(f"Resized {filename}")
            except IOError:
                print(f"Unable to resize {filename}")

# Example usage:
mummy_path = "images\\mummy"
papa_path = "images\\papa"
raghav_path = "images\\raghav"
zac_path = "images\\zac"
atharav_path = "images\\atharav"
target_size = (224, 224)
resize_images(mummy_path, target_size)
resize_images(raghav_path, target_size)
resize_images(papa_path, target_size)
resize_images(zac_path, target_size)
resize_images(atharav_path, target_size)
