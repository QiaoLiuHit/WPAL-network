#!/usr/bin/env python

# --------------------------------------------------------------------
# This file is part of
# Weakly-supervised Pedestrian Attribute Localization Network.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Weakly-supervised Pedestrian Attribute Localization Network.
# If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------

"""Test localization of a WPAL Network."""

import math
import os
import cv2
import numpy as np

from recog import recognize_attr, ResizedImageTooLargeException, ResizedSideTooShortException
from config import cfg
from utils.kmeans import weighted_kmeans

colors = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
]


def gaussian_filter(shape, center_y, center_x, var=1):
    filter_map = np.ndarray(shape)
    for i in xrange(0, shape[0]):
        for j in xrange(0, shape[1]):
            filter_map[i][j] = math.exp(-(math.pow(i - center_y, 2) + math.pow(j - center_x, 2)) / 2 / var)
    return filter_map


def zero_mask(size, area):
    mask = np.zeros(size)
    for i in xrange(int(math.floor(area['y'])), min(size[0], int(math.ceil(area['y'] + area['h'])))):
        mask[i][int(math.floor(area['x'])):min(size[1], int(math.ceil(area['x'] + area['w'])))] += 1
    return mask


def cluster_heat(img, k, stepsX, max_round=1000):
    """Return centroids of heat clusters (in x-y order)."""
    stepsY = stepsX * img.shape[0] / img.shape[1]

    thresh = (np.max(img) + max(np.mean(img), np.median(img), 0)) / 2

    dy = img.shape[0] / stepsY
    dx = img.shape[1] / stepsX

    act_points = []
    for y in xrange(stepsY):
        for x in xrange(stepsX):
            score = img[y * dy: (y + 1) * dy, x * dx: (x + 1) * dx]
            if score > thresh:
                act_points.append([x, y, score])
    act_points = np.array(act_points)

    centroids, _ = weighted_kmeans(act_points, k, max_round)
    return centroids


def locate(xa1, ya1, pw, ph, img_ind, scaled_img,
           attr_id,
           db,
           attr,
           heat_maps,
           display=True,
           vis_img_dir=None):
    canvas = np.array(scaled_img)
    feature_heat_map = np.array(scaled_img)
    feature_heat_map_bbox = np.array(scaled_img)
    img_height = scaled_img.shape[0]
    img_width = scaled_img.shape[1]
    img_area = img_height * img_width
    cross_len = math.sqrt(img_area) * 0.05

    # if vis_img_dir is not None:
    #    print 'Saving to:', os.path.join(vis_img_dir, 'heat{}.jpg'.format(j))
    #    cv2.imwrite(os.path.join(vis_img_dir, 'heat{}.jpg'.format(j)),
    #                heat_vis)

    # Center of the feature.
    # center_y = sum([w_func(j) / w_sum * target[j][0] / bin2heat[j].shape[0]
    #                for j in xrange(len(score))])
    # center_x = sum([w_func(j) / w_sum * target[j][1] / bin2heat[j].shape[1]
    #                for j in xrange(len(score))])
    # Superposition of the heat maps.
    superposition = heat_maps[attr_id]
    superposition = cv2.resize(superposition, (img_width, img_height))
    #    thresh = min(np.median(superposition), np.mean(superposition))
    #    val_range = superposition.max() - superposition.min()
    #    superposition = (superposition - thresh) / val_range

    #    expected_num_centroids = db.expected_loc_centroids[attr_id]
    #    centroids = cluster_heat(superposition,
    #                             expected_num_centroids + 2,
    #                             scaled_img.shape[1],
    #                             max_round=10)

    if display or vis_img_dir is not None:
        # for c in centroids[:expected_num_centroids]:
        # cv2.line(canvas,
        #         (int(c[0] - cross_len), int(c[1])),
        #         (int(c[0] + cross_len), int(c[1])),
        #         (0, 255, 255),
        #         thickness=4)
        # cv2.line(canvas,
        #         (int(c[0]), int(c[1] - cross_len)),
        #         (int(c[0]), int(c[1] + cross_len)),
        #         (0, 255, 255),
        #         thickness=4)

        act_map = superposition * 256
        print "height %d" %img_height
        print "width %d" %img_width
        print "act:heigh %d" %len(act_map)
        print "act: Width %d" %len(act_map[0])

        for j in xrange(img_height):
            for k in xrange(img_width):
                canvas[j][k][2] = min(255, max(0, canvas[j][k][2] + max(0, act_map[j][k])))
                canvas[j][k][1] = min(255, max(0, canvas[j][k][1]))
                canvas[j][k][0] = min(255, max(0, canvas[j][k][0]))
                feature_heat_map[j][k][2] = min(255, max(0, max(0, act_map[j][k])))
                feature_heat_map[j][k][1] = 0
                feature_heat_map[j][k][0] = 0
        canvas = canvas.astype('uint8')

        feature_heat_map_bbox = np.array(canvas)
        feature_heat_map = feature_heat_map.astype('uint8')
        feature_heat_map_bbox = feature_heat_map_bbox.astype('uint8')

        pos_loc_img = 0
        iou = 0
        feature_heat_map_gray = cv2.cvtColor(feature_heat_map, cv2.COLOR_BGR2GRAY)
        retval, feature_heat_map_binary = cv2.threshold(feature_heat_map_gray, 35, 255, cv2.THRESH_BINARY)
        binary = feature_heat_map_binary
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suitable_contours = []
        for j in range(0, len(contours)):
            featurex, featurey, featurew, featureh = cv2.boundingRect(contours[j])
            suitable_contours.append(contours[j])
            cv2.rectangle(feature_heat_map_bbox,
                          (featurex, featurey), (featurex + featurew, featurey + featureh),
                          (0, 0, 255))
        if attr_id != -1:
            if 8 < attr_id < 35 or attr_id == 43:
                if len(suitable_contours) != 0:

                    xa2 = xa1 + pw
                    ya2 = ya1 + ph
                    print "xa1 = %d, ya1 = %d, pw = %d, ph = %d" % (xa1, ya1, pw, ph)
                    cv2.rectangle(feature_heat_map_bbox,
                                  (xa1, ya1), (xa1 + pw, ya1 + ph),
                                  (0, 255, 0))
                    overlap = 0.0
                    findarea = 0.0
                    originarea = 0.0
                    for z in range(0, len(suitable_contours)):
                        xb1, yb1, cw, ch = cv2.boundingRect(suitable_contours[z])
                        xb2 = xb1 + cw
                        yb2 = yb1 + ch
                        findarea += cw * ch
                        print "xb1 = %d, yb1 = %d, cw = %d, ch = %d" % (xb1, yb1, cw, ch)
                        if (abs(xb2 + xb1 - xa2 - xa1) <= (xa2 - xa1 + xb2 - xb1)) and (
                                    abs(yb2 + yb1 - ya2 - ya1) <= (ya2 - ya1 + yb2 - yb1)):
                            #   xc1 = max(xa1, xb1)
                            #   yc1 = min(ya1, yb1)
                            #   xc2 = min(xa2, xb2)
                            #   yc2 = max(ya2, yb2)
                            #   print "xc1 = %d, yc1 = %d, xc2 = %d, yc2 = %d" % (xc1, yc1, xc2, yc2)
                            recw = (xa2 - xa1) + (xb2 - xb1) - ((max(xa2, xb2)) - min(xa1, xb1))
                            rech = (ya2 - ya1) + (yb2 - yb1) - ((max(ya2, yb2)) - min(ya1, yb1))
                            overlap += float(rech) * float(recw)
                            pos_loc_img = 1
                            print "Overlap in process = %f" % overlap

                    iou = float(overlap) / float(findarea)
                    print "The area of findarea is %d " % findarea
                    print "The area of overlap is %d " % overlap
                    print "iou of attribute %d in img %d is %f" % (attr_id, img_ind, iou)
                else:
                    print "The localization of this attribute failed."
                    pos_loc_img = 1
            else:
                print "This attribute is not in the scope of statistics."
    if display:
        cv2.imshow("img", canvas)
        cv2.waitKey(0)
        if len(suitable_contours) != 0:
            cv2.imshow("feature bounding boxes", feature_heat_map_bbox)
            cv2.waitKey(0)
            # if vis_img_dir is not None:
            #    print 'Saving to:', os.path.join(vis_img_dir, 'final.jpg')
            #    cv2.imwrite(os.path.join(vis_img_dir, 'final.jpg'), canvas)
            #     if len(suitable_contours) != 0:
    # print 'Saving to:', os.path.join(vis_img_dir, 'image_with_feature_bounding_boxes.jpg')
    #        cv2.imwrite(os.path.join(vis_img_dir, 'image_with_feature_bounding_boxes.jpg'), feature_heat_map_bbox)
    cv2.destroyWindow("heat")
    cv2.destroyWindow("img")
    cv2.destroyWindow("feature bounding boxes")

    return superposition, iou, pos_loc_img


def test_localization(net,
                      db,
                      output_dir,
                      attr_id=-1,
                      display=True,
                      max_count=-1):
    """Test localization of a WPAL Network."""
    iou_all = []
    for i in range(0, 51):
        iou_all.append([])

    cfg.TEST.MAX_AREA = cfg.TEST.MAX_AREA * 7 / 8

    num_images = len(db.test_ind)
    if (max_count == -1):
        max_count = num_images

    threshold = np.ones(db.num_attr) * 0.5

    if attr_id == -1:
        # locate whole body outline
        attr_list = xrange(db.num_attr)
    else:
        # locate only one attribute
        attr_list = []
        attr_list.append(attr_id)

    cnt = 0
    for img_ind in db.test_ind:
        img_path = db.get_img_path(img_ind)
        name = os.path.split(img_path)[1]
        if attr_id != -1 and db.labels[img_ind][attr_id] == 0:
            print 'Image {} skipped for it is a negative sample for attribute {}!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        # prepare the image
        img = cv2.imread(img_path)
        print img.shape[0], img.shape[1]

        # pass the image throught the test net.
        try:
            attr, heat_maps, img_scale = recognize_attr(net,
                                                        img,
                                                        db.attr_group,
                                                        threshold,
                                                        neglect=False)
        except ResizedImageTooLargeException:
            print 'Skipped for too large resized image.'
            continue
        except ResizedSideTooShortException:
            print 'Skipped for too short side.'
            continue

        if attr_id != -1 and attr[attr_id] != 1:
            print 'Image {} skipped for failing to be recognized attribute {} from!' \
                .format(name, db.attr_eng[attr_id][0][0])
            continue

        img_height = int(img.shape[0] * img_scale)
        img_width = int(img.shape[1] * img_scale)
        img = cv2.resize(img, (img_width, img_height))

        if display:
            cv2.imshow("img", img)

        if attr_id == -1:
            total_superposition = np.zeros(img.shape[0:2], dtype=float)
            all_centroids = []

        for a in attr_list:
            # check directory for saving visualization images
            vis_img_dir = os.path.join(output_dir, 'display', db.attr_eng[a][0][0], name)
            if not os.path.exists(vis_img_dir):
                os.makedirs(vis_img_dir)
            low = (4 * int(db.attr_position_ind[attr_id]))
            up = low + 4
            bbxx, bbxy, bbxw, bbxh = db.position[int(img_ind)][0:4]
            xa1, ya1, pw, ph = db.position[int(img_ind)][low:up]
            xa1 = xa1 - bbxx
            xa1 = int(xa1 * img_scale)
            ya1 = ya1 - bbxy
            ya1 = int(ya1 * img_scale)
            pw = int(pw * img_scale)
            if a == 9:
                ph /= 2
            if a == 12:
                ph = ph * 3 / 4
            if a == 13:
                ya1 += ph / 3
                ph /= 3
            if a == 14:
                ya1 += ph / 2
                ph /= 2
            if 15 <= a <= 23:
                ph = ph * 4 / 5
            if 30 <= a <= 34:
                ya1 += 3 * ph / 4
                ph /= 4
            act_map, iou_single, pos_loc_img = locate(xa1, ya1, pw, ph, img_ind, img,
                                                                 a,
                                                                 db,
                                                                 attr, heat_maps,
                                                                 False and display and attr_id != -1,
                                                                 vis_img_dir)
            if pos_loc_img == 1:
                iou_all[a].append(iou_single)
            if attr_id == -1:
                total_superposition += act_map * 256 / len(attr_list)
            print 'Localized attribute {}: {}!'.format(a, db.attr_eng[a][0][0])

        if attr_id == -1:
            img_area = img_height * img_width
            cross_len = math.sqrt(img_area) * 0.05

            canvas = np.array(img)
            for j in xrange(img_height):
                for k in xrange(img_width):
                    canvas[j][k][2] = min(255, max(0, canvas[j][k][2] + max(0, total_superposition[j][k])))
                    canvas[j][k][1] = min(255, max(0, canvas[j][k][1]))
                    canvas[j][k][0] = min(255, max(0, canvas[j][k][0]))
            canvas = canvas.astype('uint8')

            for c in all_centroids:
                cv2.line(canvas,
                         (int(c[0] - cross_len), int(c[1])),
                         (int(c[0] + cross_len), int(c[1])),
                         (0, 255, 255),
                         thickness=4)
                cv2.line(canvas,
                         (int(c[0]), int(c[1] - cross_len)),
                         (int(c[0]), int(c[1] + cross_len)),
                         (0, 255, 255),
                         thickness=4)

            vis_img_dir = os.path.join(output_dir, 'display', 'body', name)
            if not os.path.exists(vis_img_dir):
                os.makedirs(vis_img_dir)

            if display:
                cv2.imshow("img", canvas)
                cv2.waitKey(0)
                cv2.destroyWindow("img")
            print 'Saving to:', os.path.join(vis_img_dir, 'final.jpg')
            cv2.imwrite(os.path.join(vis_img_dir, 'final.jpg'), canvas)

        cnt += 1
        print 'Localized {} targets!'.format(cnt)
        if cnt >= max_count:
            break
    if attr_id != -1:
        # Count mean IoU:
        if len(iou_all[attr_id]) != 0:
            iou_single_attr_sum = 0.0
            for x in iou_all[attr_id]:
                iou_single_attr_sum += x
            iou_single_attr_sum /= len(iou_all[attr_id])
            return iou_single_attr_sum
        else:
            return -1


def locate_in_video(net,
                    db,
                    video_path, tracking_res_path,
                    output_dir,
                    pos_ave, neg_ave, dweight,
                    attr_id_list):
    """Locate attributes of pedestrians in a video using a WPAL-network.
    The tracking results should be provided in a text file.
    """

    cfg.TEST.MAX_AREA = cfg.TEST.MAX_AREA * 3 / 4

    attr_ids = [int(s) for s in attr_id_list.split(',')]
    if len(attr_ids) > len(colors):
        print 'Cannot locate more than {} attributes in one video!'.format(len(colors))
        return

    name_comb = db.attr_eng[attr_ids[0]][0][0]
    for attr_id in attr_ids[1:]:
        name_comb += db.attr_eng[attr_id][0][0]
    vid_path = os.path.join(output_dir, 'display', name_comb, os.path.basename(video_path))
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    # Read tracks
    with open(tracking_res_path) as f:
        num_tracklets = int(f.readline())
        tracklets = []
        for i in xrange(num_tracklets):
            f.readline()
            tracklet = {'start_frame_ind': int(f.readline())}
            num_bbox = int(f.readline())
            bbox_seq = []
            for j in xrange(num_bbox):
                line = f.readline()
                x, y, h, w = line.split()
                bbox_seq.append([int(x), int(y), int(h), int(w)])
            tracklet['bbox_seq'] = bbox_seq
            tracklets.append(tracklet)

    threshold = np.ones(db.num_attr) * 0.5
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

    writer = None
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        canvas = np.array(frame)

        for i in xrange(len(attr_ids)):
            cv2.rectangle(canvas,
                          (frame.shape[1] - 300, 30 + 60 * i),
                          (frame.shape[1] - 280, 50 + 60 * i),
                          colors[i],
                          thickness=20)
            cv2.putText(canvas,
                        db.attr_eng[attr_ids[i]][0][0],
                        (frame.shape[1] - 260, 50 + 60 * i),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        colors[i],
                        thickness=3)

        has_pedestrian = False
        for tracklet in tracklets:
            if tracklet['start_frame_ind'] \
                    <= frame_cnt \
                    < tracklet['start_frame_ind'] + len(tracklet['bbox_seq']):
                has_pedestrian = True

                bbox_seq = tracklet['bbox_seq']
                bbox = bbox_seq[frame_cnt - tracklet['start_frame_ind']]

                cropped = frame[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]

                # pass the image throught the test net.
                try:
                    attr, heat_maps, score, img_scale = recognize_attr(net,
                                                                       cropped,
                                                                       db.attr_group,
                                                                       threshold,
                                                                       neglect=False)
                except ResizedSideTooShortException:
                    print 'Skipped for too short side.'
                    continue

                msg = ''
                for i in xrange(len(attr_ids)):
                    if attr[attr_ids[i]] == 1:
                        msg += db.attr_eng[attr_ids[i]][0][0] + ' '
                print 'Recognized {}from Frame {}'.format(msg, frame_cnt)
                msg = ''
                for i in xrange(len(attr)):
                    if attr[i] == 1 and not attr_ids.__contains__(i):
                        msg += db.attr_eng[i][0][0] + ' '
                print 'Unshown attributes: ' + msg

                cv2.imshow("cropped", cropped)
                cv2.waitKey(1)

                cropped_height = int(cropped.shape[0] * img_scale)
                cropped_width = int(cropped.shape[1] * img_scale)
                cropped = cv2.resize(cropped, (cropped_width, cropped_height))

                for i in xrange(len(attr_ids)):
                    attr_id = attr_ids[i]
                    if attr[attr_id] != 1:
                        continue
                    act_map, centroids = locate(cropped, pos_ave, neg_ave, dweight, attr_id, db,
                                                attr, heat_maps, score, display=False)
                    act_map = cv2.resize(act_map, (bbox[2], bbox[3]))
                    for x in xrange(bbox[2]):
                        for y in xrange(bbox[3]):
                            fx = x + bbox[0]
                            fy = y + bbox[1]
                            canvas[fy][fx][0] = np.uint8(min(255, canvas[fy][fx][0]
                                                             + max(0, act_map[y][x]) * colors[i][0]))
                            canvas[fy][fx][1] = np.uint8(min(255, canvas[fy][fx][1]
                                                             + max(0, act_map[y][x]) * colors[i][1]))
                            canvas[fy][fx][2] = np.uint8(min(255, canvas[fy][fx][2]
                                                             + max(0, act_map[y][x]) * colors[i][2]))
                    centroids = centroids[:, :2] / img_scale + (bbox[0], bbox[1])
                    cross_len = math.sqrt(frame.shape[0] * frame.shape[1]) * 0.02

                    thickness = len(centroids) * 2
                    for c in centroids:
                        cv2.line(canvas,
                                 (int(c[0] - cross_len), int(c[1])),
                                 (int(c[0] + cross_len), int(c[1])),
                                 colors[i],
                                 thickness=thickness)
                        cv2.line(canvas,
                                 (int(c[0]), int(c[1] - cross_len)),
                                 (int(c[0]), int(c[1] + cross_len)),
                                 colors[i],
                                 thickness=thickness)
                        thickness -= 2

        if has_pedestrian:
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(vid_path, str(frame_cnt) + '.avi'),
                                         fourcc=cv2.cv.FOURCC('M', 'J', 'P', 'G'),
                                         fps=fps / 2,
                                         frameSize=(frame.shape[1], frame.shape[0]),
                                         isColor=True)
            cv2.imshow("Vis", canvas)
            cv2.waitKey(1)
            writer.write(canvas)
        elif writer is not None:
            writer = None
            cv2.destroyWindow("Vis")
        frame_cnt += 1


if __name__ == '__main__':
    print gaussian_filter((8, 3), 2, 1)
