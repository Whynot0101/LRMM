from PIL import Image
from torchvision import transforms
import torchvision
class ImageCropAndResize:
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=16, width_division_factor=16,
                max_side=3072, no_hflip=False): 
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.max_side = max_side 
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.final_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x) if no_hflip else transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size

            if self.max_pixels is not None and width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)

            if self.max_side is not None:
                long_side = max(width, height)
                if long_side > self.max_side:
                    scale = long_side / self.max_side
                    height, width = int(height / scale), int(width / scale)

            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return self.final_transform(image)