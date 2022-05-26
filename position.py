import exifread

with open('IMG_3729.jpg', 'rb') as f:
    exif_dict = exifread.process_file(f)
    for key in exif_dict:
        if 'MakerNote' not in key:
            print(f"{key}: {exif_dict[key]}")