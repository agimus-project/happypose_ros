from vision_msgs.msg import Detection2DArray


def createAABB2D(left: int, top: int, width: int, height: int) -> Detection2DArray:
    bbox = Detection2DArray()
    bbox.center.position.x = left + width / 2.0
    bbox.center.position.y = top + height / 2.0
    bbox.size_x = width
    bbox.size_y = height

    return bbox
