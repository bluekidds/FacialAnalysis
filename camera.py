#!/usr/bin/env python
import cv2
import dlib
import face_recognition
import collections
import numpy as np
from scipy.spatial.distance import euclidean as dist
from imutils import face_utils
from imutils import resize


detector = dlib.get_frontal_face_detector()
shape_path = 'models/shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(shape_path)
trackingFace = 0
skipFrames = 30
rectangleColor = (0, 165, 255)
threshold = 0.4


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def enhance_image(image):
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    Y, Cr, Cb = cv2.split(image_YCrCb)
    Y = cv2.equalizeHist(Y)
    image_YCrCb = cv2.merge([Y, Cr, Cb])
    image = cv2.cvtColor(image_YCrCb, cv2.COLOR_YCR_CB2BGR)
    return image


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def compute_blurrness(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return variance_of_laplacian(gray)


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.totalFrame = 0
        self.currentFaceID = 0
        self.faceNames = {}
        self.trackers = []
        self.trackableObjects = {}
        self.faceTrackers = {}
        self.encodings = collections.OrderedDict()

    def __del__(self):
        self.video.release()

    def create_face(self, encoding, fid):
        self.encoding[fid] = encoding

    def compare_faces(self, encoding):
        # If faces encoding match, then return fid, otherwise, create one
        edists = []
        if len(self.encodings) == 0:
            return None
        else:

            for old_id, old_encoding in self.encodings.items():
                edist = dist(encoding, old_encoding)
                print(edist, old_id)
                edists.append(edist)

            mindist = min(edists)
            if mindist > float(threshold):
                return 'unknown'
            else:
                names = list(self.encodings.keys())
                minID = names[edists.index(mindist)]

            return minID

    def get_frame_shape(self):
        success, image = self.video.read()
        image = resize(image, width=900)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for (i, rect) in enumerate(faces):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = shape_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        ret, jpeg = cv2.imencode('.jpg', image)
        self.totalFrame += 1
        return jpeg.tobytes()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image = enhance_image(image)
        image = adjust_gamma(image)

        image = resize(image, width=900)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        fidsToDelete = []
        for fid in self.faceTrackers.keys():
            trackingQuality = self.faceTrackers[fid].update(gray)

            # If the tracking quality is not good enough, we must delete
            # this tracker
            if trackingQuality < 7:
                fidsToDelete.append(fid)
        for fid in fidsToDelete:
            print("Removing fid " + str(fid) + " from list of trackers")
            self.faceTrackers.pop(fid, None)

        if (self.totalFrame % 30) == 0:
            blur = compute_blurrness(image)

            #if blur < 70:
            #    print('blurness: ', blur)
            #    ret, jpeg = cv2.imencode('.jpg', image)
            #    return jpeg.tobytes()

            #faces = detector(gray, 0)
            faces = face_recognition.face_locations(gray)

            for face in faces:
                #bb = face_utils.rect_to_bb(face)
                #(_x, _y, _w, _h) = bb
                (top, right, bottom, left) = face
                (_x, _y, _w, _h) = (left, top, right - left, bottom - top)
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                # calculate the centerpoint
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                # Variable holding information which faceid we
                # matched with
                matchedFid = None

                # Now loop over all the trackers and check if the
                # centerpoint of the face is within the box of a
                # tracker
                for fid in self.faceTrackers.keys():
                    tracked_position = self.faceTrackers[fid].get_position()

                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())

                    # calculate the centerpoint
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    # check if the centerpoint of the face is within the
                    # rectangleof a tracker region. Also, the centerpoint
                    # of the tracker region must be within the region
                    # detected as a face. If both of these conditions hold
                    # we have a match
                    if ((t_x <= x_bar <= (t_x + t_w)) and
                            (t_y <= y_bar <= (t_y + t_h)) and
                            (x <= t_x_bar <= (x + w)) and
                            (y <= t_y_bar <= (y + h))):
                        matchedFid = fid

                # If no matched fid, then we will check if it is one of our
                # existing ids, otherwise we create a new ID and encoding.

                if matchedFid is None:
                    print("Creating new tracker " + str(self.currentFaceID))

                    encoding = face_recognition.face_encodings(image,
                                                               [face],
                                                               num_jitters=10)
                    response = self.compare_faces(encoding)
                    print('Response is:', response)
                    if (response is None) or (response == 'unknown'):

                        # Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(gray,
                                            dlib.rectangle(x - 10,
                                                           y - 20,
                                                           x + w + 10,
                                                           y + h + 20))

                        self.faceTrackers[self.currentFaceID] = tracker
                        self.faceNames[self.currentFaceID] = "Person " \
                                                             + str(self.currentFaceID)
                        self.encodings[self.currentFaceID] = encoding
                        self.currentFaceID += 1
                    else:

                        # Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(gray,
                                            dlib.rectangle(x - 10,
                                                           y - 20,
                                                           x + w + 10,
                                                           y + h + 20))

                        self.faceTrackers[response] = tracker
                        self.faceNames[response] = "Person " + str(response)
                        print('Find existed person: ', response)

        # Keep tracking

        for fid in self.faceTrackers.keys():
            tracked_position = self.faceTrackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            cv2.rectangle(image, (t_x, t_y),
                          (t_x + t_w, t_y + t_h),
                          rectangleColor, 2)

            if fid in self.faceNames.keys():
                cv2.putText(image, self.faceNames[fid],
                            (int(t_x + t_w / 2), int(t_y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
            else:
                cv2.putText(image, "Detecting...",
                            (int(t_x + t_w / 2), int(t_y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
        text = "total frames{}".format(self.totalFrame)
        cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', image)

        self.totalFrame += 1

        return jpeg.tobytes()
