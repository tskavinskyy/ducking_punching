# ducking_punching
## Making an AI Gesture Recognition Boxing Game
### Using computer vision AI to make interactive video games
#### Introduction
During the pandemic, I picked up playing video games again in years. While playing Ryse I wondered if I could combine playing with working out to stay more fit. There are a few systems out there that you can move and play, but most are expensive and require special equipment. But all we really need to do is just keep track of the body movement with the webcam and create a way to translate those movements into something a computer will understand. There are a few amazing videos on YouTube that try writing code to play games with body gestures. I wanted to add to them and hopefully add to the quality of the games. Check out how the game looks:


How the Game Works
The first game I attempted was a “boxing” game, where I could punch something that appeared on the screen and get points for it:

https://www.youtube.com/watch?v=hsSwy80Rm64
https://medium.com/@skavinskyy/making-an-ai-gesture-recognition-boxing-game-a3d1b99e1a01

First, with OpenCV and MediaPipe and a tutorial from Nicholas Renotte, I set up some code that uses my webcam and keeps track of my body movement.

cap = cv2.VideoCapture(0)
Initiate holistic model
with mp_holistic.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        start=time.time()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,  mp_drawing.DrawingSpec(color=(0,230,230), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(230,0,0), thickness=2, circle_radius=2))

            mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(0,230,230), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(230,0,0), thickness=2, circle_radius=2)
                                      )
Then I had to add something to punch on a screen. This page had some neat code on how to add two images together. I wanted to punch something evil-looking so with the help of Dal-E I created a villain.


function based on https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
def image_add(img1,img2,y,x): 
    
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    x=int(x-rows/2)
    y=int(y-cols/2)
    roi = img1[x:rows+x, y:cols+y]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[x:rows+x, y:cols+y ] = dst
    return img1
Now I just had to calculate the distance between either of my hands that are tracked by mediapipe and the evil face. When that distance was sufficiently small enough then impact occurred, I got a point, and the evil face reset. I added a punching sound to it to make the game more engaging.

        try:
            #### Coding in movement: Calculating distance
            point=mp_pose.PoseLandmark.RIGHT_INDEX.value
            point_l=mp_pose.PoseLandmark.LEFT_INDEX.value
            point_eye=mp_pose.PoseLandmark.LEFT_EYE.value

            visibility =results.pose_landmarks.landmark[point].visibility
            visibility_left =results.pose_landmarks.landmark[point_l].visibility

## Hand positions
            x=((results.pose_landmarks.landmark[point].x))
            y=((results.pose_landmarks.landmark[point].y))
            hand =([x,y])
            speed= int(5*abs(np.sqrt((hand[0]-hand_previous[0])*(hand[0]-hand_previous[0])+(hand[1]-hand_previous[1])*(hand[1]-hand_previous[1]))))
            hand_previous =hand
            xl=((results.pose_landmarks.landmark[point_l].x))
            yl=((results.pose_landmarks.landmark[point_l].y))
            xeye=((results.pose_landmarks.landmark[point_eye].x))
            yeye=((results.pose_landmarks.landmark[point_eye].y))
            hand_left =([xl,yl])
            speed_left= int(10*abs(np.sqrt((hand[0]-hand_previous[0])*(hand[0]-hand_previous[0])+(hand[1]-hand_previous[1])*(hand[1]-hand_previous[1]))))
            hand_left_previous =hand_left
            speed=max(speed,speed_left)

###Distance calculation
            distr= np.sqrt((dot[0]-x)*(dot[0]-x)+(dot[1]-y)*(dot[1]-y))
            distl= np.sqrt((dot[0]-xl)*(dot[0]-xl)+(dot[1]-yl)*(dot[1]-yl))
            dist=min(distr,distl)

            
Sometimes villains punch back. After a while testing this game, I found that just punching a face gets a bit repetitive. So once in a while villain's glove appears on the screen headed toward your face and you have to duck away from it. Here I was able to move the glove and increase its size on the screen. At this point, we keep track of the player’s head and when they duck down or to the side the glove misses and the fight resets. If you don't duck fast enough points are reduced.

Now we set it all in a loop for one round or 60 seconds. Let's see how many points we can get. You can quit by pressing Q. Time and points are tracked on the top of the screen. at the end of the round, you can start another round by pressing r on the keyboard.

I added the link to Github where you can find the code for this game. I will be building other games using this method so please follow me if you want to see other games. Also, I’ve used this method to play almost any game where your body gestures become the controller. Check out the original implementation for Ryse here.
