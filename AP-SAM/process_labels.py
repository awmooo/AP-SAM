from PIL import Image
import os

def convert_low_gray_to_black(image):
    """

    """
    mode = image.mode
    width, height = image.size
    pixels = image.load()

    for x in range(width):
        for y in range(height):
            if mode in ('L', '1'):
                if pixel < 128:
                    pixels[x, y] = 0
            elif mode in ('RGB', 'RGBA'):
                r, g, b = pixels[x, y][:3]
                if r < 128 and g < 128 and b < 128:
                    if mode == 'RGB':
                        pixels[x, y] = (0, 0, 0)
                    else:
                        a = pixels[x, y][3]
                        pixels[x, y] = (0, 0, 0, a)
    return image

def process_images_in_folder(input_folder, output_folder):
    """

    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)


        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            try:

                image = Image.open(input_path)

                processed_image = convert_low_gray_to_black(image)


                if filename.lower().endswith(('.jpg', '.jpeg')):
                    processed_image.save(output_path, quality=100)
                else:
                    processed_image.save(output_path)

                print(f"Processed: {input_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

# original mudstone label dir
input_folder = "E:\\259\mudstone\labels"
# conveted results dir(only contain pore and background)
output_folder = "E:\\259\mudstone\process_labels"
process_images_in_folder(input_folder, output_folder)