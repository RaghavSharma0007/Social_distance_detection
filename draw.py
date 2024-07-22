import cv2
###################################social distancing and camera parameters############################################
CONFIDENCE_THRESHOLD = 0
NMS_THRESHOLD = 0.4
ref_obj_width_in_cm = 25
ref_obj_width_in_pixel = 18
pixel_per_unit_cm = 18 / 25
ref_distance_from_camera_in_cm = 2000
avg_person_width = 30
focal_length = pixel_per_unit_cm * ref_distance_from_camera_in_cm
#########################################model_parameters##############################################################
COLORS = [(0, 0, 255)]
class_names = ['person']
#######################################DRAWING BOX###################################
def drawing(classes, scores, boxes,frame,min_dist):
    box_centres = {}
    box_new = {}
    i = 0
    for (classid, score, box) in zip(classes, scores, boxes):
        if classid == 0:
            # print(classid,box,"inside")
            box_center_x = int(box[0] + box[2] / 2)
            box_center_y = int(box[1] + box[3] / 2)
            dist1 = (focal_length * avg_person_width) / box[2]
            box_centres[i] = [box_center_x, box_center_y, dist1]
            box_new[i] = box
            # cv2.circle(frame, (box_center_x, box_center_y), 10, (0, 255,),-1)
            # print(box_centres, 'inside')
            color = COLORS[int(classid) % len(COLORS)]
            color = (0, 255, 0)
            label = "%s : %f" % (class_names[0], score)
            cv2.rectangle(frame, box, color, 2)
            i += 1
            # cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            pass
    # print(box_centres)
    import math
    set1 = set()
    set2 = set()
    if len(box_centres) <= 1:
        pass
    else:
        for obj1 in range(len(box_centres)):
            # print(obj1,box_centres[obj1])
            for obj2 in range(obj1 + 1, len(box_centres)):
                dist1 = box_centres[obj1][2]
                dist2 = box_centres[obj2][2]
                x1 = box_centres[obj1][0] * dist1 / focal_length
                y1 = box_centres[obj1][1] * dist1 / focal_length
                x2 = box_centres[obj2][0] * dist1 / focal_length
                y2 = box_centres[obj2][1] * dist1 / focal_length

                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (dist1 - dist2) ** 2)
                # print(dist)
                if dist <= min_dist:
                    set1.add(tuple(box_centres[obj1]))
                    set1.add(tuple(box_centres[obj2]))
                    set2.add(tuple(box_new[obj1]))
                    set2.add(tuple(box_new[obj2]))
        # print(set1)
        # for i in set1:
        #     cv2.circle(frame, (i[0], i[1]), radius=10, color=(0, 0, 255), thickness=-1)
        for j in set2:
            cv2.rectangle(frame, j, (0,0,255), 2)
#######################################DRAWING BOX ON FRAME###################################