import exifread

with open('IMG_3732.jpg', 'rb') as f:
    exif_dict = exifread.process_file(f)
    for key in exif_dict:
        print("%s: %s" % (key, exif_dict[key]))