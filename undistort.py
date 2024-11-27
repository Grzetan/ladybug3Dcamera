import cv2
import numpy as np
import sys

DIM=(1232, 1024)
K=np.array([[610.2580761337351, 0.0, 604.2690061132831], [0.0, 610.4400113688544, 518.3692515772925], [0.0, 0.0, 1.0]])
D=np.array([[0.006872700150132997], [-0.05977840726726811], [0.11494007264061318], [-0.06120828960421955]])


def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.imwrite("ok.png", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)