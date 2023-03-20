#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com

import os
from datetime import datetime


def create_file(filename):
    if os.path.exists(filename):
        # create new file with datetime suffix
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        basename, ext = os.path.splitext(filename)
        new_filename = f"{basename}_{timestamp}{ext}"
        return new_filename
    return filename
