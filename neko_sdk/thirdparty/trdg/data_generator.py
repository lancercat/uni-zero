import os
import random as rnd

import numpy as np;
from PIL import Image, ImageFilter, ImageColor

from neko_sdk.thirdparty.trdg import computer_text_generator, background_generator, distorsion_generator

# try:
    # from trdg_ import handwritten_text_generator
# except ImportError as e:
print("Missing ocr_modules for handwritten text generation.")


class FakeTextDataGenerator(object):
    @classmethod
    def transform( cls,
                   image,mask,margins,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            width,
            alignment,
            orientation,
            image_dir):
        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_img.size[0]
                * (float(size - vertical_margin) / float(distorted_img.size[1]))
            )
            resized_img = distorted_img.resize(
                (new_width, size - vertical_margin), Image.LANCZOS
            )
            resized_mask = distorted_mask.resize((new_width, size - vertical_margin))
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            try:
                background_img = background_generator.image(
                    background_height, background_width, image_dir
                ).convert("RGBA")
            except:
                print(image_dir);
                return None, None
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size
        resized_img = resized_img.convert("RGBA");

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        ##################################
        # Apply gaussian blur #
        ##################################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.randint(0, blur)
        )
        final_image = background_img.filter(gaussian_filter).convert("RGB")
        final_mask = background_mask.filter(gaussian_filter)
        return final_image,final_mask;
    @classmethod
    def gen_bg(cls,background_type,background_height,background_width,background_img):
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            try:
                if (type(background_img) == str):
                    background_img = background_generator.image(
                        background_height, background_width, background_img
                    ).convert("RGBA")
                else:
                    background_img = background_generator.image_pic(
                        background_height, background_width, background_img
                    ).convert("RGBA")
            except:
                # print(image_dir);
                background_img = background_generator.quasicrystal(
                    background_height, background_width
                )

        return background_img;

    @classmethod
    def transform2(cls,
                  image, mask, margins,
                  size,
                  skewing_angle,
                  random_skew,
                  blur,
                  random_blur,
                  background_type,
                  distorsion_type,
                  distorsion_orientation,
                  width,
                  alignment,
                  orientation,
                  background_img):
        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################
        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_img.size[0]
                * (float(size - vertical_margin) / float(distorted_img.size[1]))
            )
            resized_img = distorted_img.resize(
                (new_width, size - vertical_margin), Image.ANTIALIAS
            )
            resized_mask = distorted_mask.resize((new_width, size - vertical_margin))
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################

        background_img=cls.gen_bg(background_type,background_height,background_width,background_img)
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )
        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size
        resized_img=np.concatenate([np.array(resized_img)[:,:,:3],np.array(resized_mask)[:,:,0:1]],-1);
        resized_img=Image.fromarray(resized_img);

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        ##################################
        # Apply gaussian blur #
        ##################################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.randint(0, blur)
        )
        final_image = background_img.filter(gaussian_filter).convert("RGB")
        final_mask = background_mask.filter(gaussian_filter)
        return final_image, final_mask;

    @classmethod
    def generate_core(
            cls,
            text,
            font,
            size,
            is_handwritten,
            text_color,
            orientation,
            space_width,
            character_spacing,
            fit,
            word_split
    ):
        image = None
        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image, mask = handwritten_text_generator.generate(text, text_color)
        else:
            image, mask = computer_text_generator.generate(
                text,
                font,
                text_color,
                size,
                orientation,
                space_width,
                character_spacing,
                fit,
                word_split,
            )
        return image,mask
    @classmethod
    def generate_and_transform(
            cls,
            text,
            font,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            is_handwritten,
            width,
            alignment,
            text_color,
            orientation,
            space_width,
            character_spacing,
            margins,
            fit,
            word_split,
            image_dir,
    ):
        image,mask=cls.generate_core(
            text,
            font,
            size,
            is_handwritten,
            text_color,
            orientation,
            space_width,
            character_spacing,
            fit,
            word_split
        );
        return  cls.transform(
                      image, mask, margins,
                      size,
                      skewing_angle,
                      random_skew,
                      blur,
                      random_blur,
                      background_type,
                      distorsion_type,
                      distorsion_orientation,
                      width,
                      alignment,
                      orientation,
                      image_dir);
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    def generate(
            cls,
            index,
            text,
            font,
            out_dir,
            size,
            extension,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            is_handwritten,
            name_format,
            width,
            alignment,
            text_color,
            orientation,
            space_width,
            character_spacing,
            margins,
            fit,
            output_mask,
            word_split,
            image_dir,
    ):
        final_image,final_mask=cls.generate_and_transform(
            text,
            font,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            is_handwritten,
            width,
            alignment,
            text_color,
            orientation,
            space_width,
            character_spacing,
            margins,
            fit,
            word_split,
            image_dir,
        )
        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))
        elif name_format == 1:
            image_name = "{}_{}.{}".format(str(index), text, extension)
            mask_name = "{}_{}_mask.png".format(str(index), text)
        elif name_format == 2:
            image_name = "{}.{}".format(str(index), extension)
            mask_name = "{}_mask.png".format(str(index))
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))

        # Save the image
        if out_dir is not None:
            final_image.convert("RGB").save(os.path.join(out_dir, image_name))
            if output_mask == 1:
                final_mask.convert("RGB").save(os.path.join(out_dir, mask_name))
        else:
            if output_mask == 1:
                return final_image.convert("RGB"), final_mask.convert("RGB")
            return final_image.convert("RGB")

    @classmethod
    def generate_from_mask_core(
            cls,
            mask,
            text_color
    ):
        text_color = [ImageColor.getrgb(c) for c in text_color.split(",")]
        c1, c2 = text_color[0], text_color[-1]
        fill = (
            rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
            rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
            rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
        )
        image = Image.fromarray((mask / 255 * fill).astype(np.uint8));
        mask = Image.fromarray(mask);
        return image,mask
        #####################################


    # add style and noise to mask. Hence we can save space by only caching rendered masks
    @classmethod
    def generate_from_mask_and_transform(
            cls,
            mask,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            width,
            alignment,
            text_color,
            orientation,
            margins,
            output_mask,
            bgim,
    ):
        image,mask=cls.generate_from_mask_core(   mask,
            text_color);

        final_image,final_mask=cls.transform2(
            image, mask, margins,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            width,
            alignment,
            orientation,
            bgim
        )
        #####################################
        # Generate name for resulting image #
        #####################################



        if output_mask == 1:
            return final_image.convert("RGB"), final_mask.convert("RGB")
        return final_image.convert("RGB")
