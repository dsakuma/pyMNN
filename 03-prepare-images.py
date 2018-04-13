import math
import os
import string

from PIL import Image, ImageDraw, ImageFont, ImageChops

chars = ('ABCDEFGHJKLMNOPQRSTUVWYZ' + '123456789'
         + 'abcdefghjkmnopqrstuvwxyz'
         + 'ĄĆĘŁÓŚŻŹ' + 'ąćęóśżź'
         + 'αβδζηθκλμξπρσψω'
         + 'ÄÖÜäöüß')
font_name = 'fonts/Arial-Bold.ttf'


def font_factory(size):
    return ImageFont.truetype(font_name, size=size)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def generate_char_img(c, size, fg, bg, font_factory):
    font_size = int(math.log2(size)) * 4
    font = font_factory(font_size)

    c_size = font.getsize(c)
    img = Image.new('1', c_size, color=bg)

    draw = ImageDraw.Draw(img)
    draw.text((0, 0), c, font=font, fill=fg)

    return trim(img).resize((size, size))


if __name__ == '__main__':
    for size in range(10, 81):
        path = 'data_gen/{0}'.format(size)
        os.makedirs(path, exist_ok=True)
        for i, char in enumerate(chars):
            img = generate_char_img(char, size, 'black', 'white', font_factory)
            img.save('{0}/{1}.png'.format(path, i + 1))
