from PIL import Image
import numpy as np

def load_image_as_features(image_path):
    # load image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)  # shape: (H, W, 3)
    H, W, _ = image.shape

    # create coordinates grid
    coords = np.indices((H, W)).transpose(1, 2, 0).reshape(-1, 2)  # shape: (H*W, 2)

    # get RGB colors
    colors = image.reshape(-1, 3)  # shape: (H*W, 3)

    # merge [x, y, R, G, B]
    features = np.hstack((coords, colors)).astype(np.float32)  # shape: (10000, 5)
    return features, (H, W)

if __name__ == "__main__":
    feature1, size1 = load_image_as_features("images/image1.png")
    feature2, size2 = load_image_as_features("images/image2.png")
    #print("Feature1 shape:", feature1.shape)
    #print("Image1 size:", size1)
    #print("Feature2 shape:", feature2.shape)
    #print("Image2 size:", size2)