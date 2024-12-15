'''
Adjust and compare HSV threshold for media file with tkinter GUI.
'''

from math import ceil
from time import perf_counter_ns
import tkinter as tk
from tkinter import filedialog
import cv2
from numpy import array
from PIL import Image, ImageTk

class BaseAdjuster:
    '''
    Base class for adjusting HSV thresholds of media files.
    '''
    def __init__(self, tk_root):
        self.tk_root = tk_root
        self.fields = ('Hue', 'Saturation', 'Value')
        self.image = None
        self.init_ui()
        tk_root.bind("<Configure>", self.update)

    def clean_img(self, image, lower_bound, upper_bound):
        '''
        Clean image based on HSV thresholds.

        Args:
            image (numpy.ndarray): Image with cv2 BGR format.
            lower_bound (np.ndarray): The lower HSV boundaries.
            upper_bound (np.ndarray): The upper HSV boundaries.

        Returns:
            Cleaned image with cv2 BGR format.
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result

    def display_image(self, image, label):
        '''
        Display image to label. Image would be resized if width/height exceed limit.

        Args:
            image (numpy.ndarray): Image with cv2 BGR format.
            label: Tkinter label.
        '''
        max_width = self.tk_root.winfo_width() * 0.4
        max_height = self.tk_root.winfo_height()
        img_height, img_width = image.shape[:2]
        scale_width = max_width / img_width
        scale_height = max_height / img_height
        scale = min(scale_width, scale_height)

        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image # keep reference

    def init_ui(self):
        '''
        Initialize UI.
        '''
        self.tk_root.geometry('1200x600')

        right_frame = tk.Frame(self.tk_root)
        self.original_label = tk.Label(self.tk_root)
        self.cleaned_label = tk.Label(self.tk_root)

        self.original_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.cleaned_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        min_value, max_value = 0, 255
        self.controls = {}
        for field in self.fields:
            for initial_pos, type_text in [(min_value, 'Min'), (max_value, 'Max')]:
                label_text = field + ' ' + type_text
                scale = tk.Scale(
                    right_frame,
                    from_=min_value,
                    to=max_value,
                    orient=tk.HORIZONTAL,
                    label=label_text
                )
                scale.set(initial_pos)
                scale.pack(side=tk.TOP, fill=tk.X)
                scale.bind('<B1-Motion>', self.update)
                scale.bind('<ButtonPress-1>', self.update)
                self.controls[label_text] = scale

    def update(self, _=None):
        '''
        Callback function to update image. There are two kinds of callback:
        1. <configure> means the size of the window changed
        2. <B1-Motion> or <ButtonPress-1> means the threshold changed
        '''
        if self.image is not None:
            lower_bound = array([self.controls[field + ' Min'].get() for field in self.fields])
            upper_bound = array([self.controls[field + ' Max'].get() for field in self.fields])
            cleaned_img = self.clean_img(self.image, lower_bound, upper_bound)
            self.display_image(self.image, self.original_label)
            self.display_image(cleaned_img, self.cleaned_label)


class ImageAdjuster(BaseAdjuster):
    '''
    Adjuster class for image files.
    '''
    def __init__(self, tk_root, image_path):
        tk_root.title('Image HSV Threshold Adjuster')
        super().__init__(tk_root)
        self.image = cv2.imread(image_path)


class VideoAdjuster(BaseAdjuster):
    '''
    Adjuster class for video files.

    Note: the speed of the video would be slower,
        because tk.after has unstable delays at the millisecond precision level.
    '''
    def __init__(self, tk_root, video_path):
        tk_root.title('Video HSV Threshold Adjuster')
        super().__init__(tk_root)
        tk_root.bind('<space>', self.pause)
        tk_root.bind('<Left>', self.backward) # 5 seconds
        tk_root.bind('<Right>', self.forward) # 5 seconds

        self.video_capture = cv2.VideoCapture(video_path)
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.update_interval = 1000 / fps
        self.playing = True
        self.after_id = self.tk_root.after(ceil(self.update_interval), self.update_next_frame)

    def __del__(self):
        self.video_capture.release()

    def backward(self, _):
        '''
        Callback function for <Left> button, moving video to 5 seconds backward.
        '''
        self.video_capture.set(
            cv2.CAP_PROP_POS_MSEC,
            max(0, self.video_capture.get(cv2.CAP_PROP_POS_MSEC) - 5000)
        )
        self.update_next_frame()

    def get_frame(self):
        '''
        Try to get frame from video

        Returns:
            Union[image, None]: frame with cv2 BGR format, None if finished reading the final frame.
        '''
        ret, frame = self.video_capture.read()
        if ret:
            return frame
        self.playing = False
        return None

    def get_time_ms(self):
        '''
        Get time in millisecond.

        Returns:
            float: Time in millisecond.
        '''
        return perf_counter_ns() * 1e-6

    def forward(self, _):
        '''
        Callback function for <Right> button, moving video to 5 seconds forward.
        '''
        self.video_capture.set(
            cv2.CAP_PROP_POS_MSEC,
            self.video_capture.get(cv2.CAP_PROP_POS_MSEC) + 5000
        )
        self.update_next_frame()

    def update_next_frame(self):
        '''
        Get next frame of the video and update.
        Use tk.after to schedule next update, keeping the video to play.
        This function may be called from various place,
            so after_calcel is needed to ensure only one after event is scheduled.
        '''
        if self.playing:
            next_update_t = self.get_time_ms() + self.update_interval
            self.image = self.get_frame()
            super().update()
            after_t = ceil(next_update_t - self.get_time_ms())
            self.tk_root.after_cancel(self.after_id)
            self.after_id = self.tk_root.after(after_t, self.update_next_frame)

    def pause(self, _):
        '''
        Callback function for <Space> button, pausing the video.
        '''
        self.playing = not self.playing
        self.update_next_frame()


class MediaHsvThresholdAdjuster:
    '''
    Interface class to load media resource.
    '''
    def __init__(self, tk_root):
        self.init_ui(tk_root)

    def init_ui(self, tk_root):
        '''
        Initialize UI.
        '''
        tk_root.title('Media HSV Threshold Adjuster')
        tk_root.geometry('200x200')
        pack_kwargs = {'side': tk.TOP, 'expand': True}
        tk.Button(tk_root, text='Load Image', command=self.load_image).pack(**pack_kwargs)
        tk.Button(tk_root, text='Load Video', command=self.load_video).pack(**pack_kwargs)

    def load_image(self):
        '''
        Use filedialog to open image file.
        https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html lists the supported formats as below.
        '''
        filetypes = ' '.join([
            '*.bmp', '*.dib',
            '*.jpeg', '*.jpg', '*.jpe',
            '*.jp2',
            '*.png',
            '*.webp',
            '*.pbm', '*.pgm', '*.ppm', '*.pxm', '*.pnm',
            '*.sr', '*.ras',
            '*.tiff', '*.tif',
            '*.exr',
            '*.hdr', '*.pic'
        ])
        path = filedialog.askopenfilename(
            title='Open Image',
            filetypes=(('Image files', filetypes),)
        )
        if path:
            ImageAdjuster(tk.Toplevel(), path)

    def load_video(self):
        '''
        Use filedialog to open video file.
        No documentation found has stated all formats supported by cv2.
        '''
        filetypes = ' '.join([
            '*.mp4', '*.avi', '*.mov', '*.mpeg', '*.flv', '*.wmv'
        ])
        path = filedialog.askopenfilename(
            title='Open Video',
            filetypes=(('Video files', filetypes),)
        )
        if path:
            VideoAdjuster(tk.Toplevel(), path)


if __name__ == '__main__':
    root = tk.Tk()
    app = MediaHsvThresholdAdjuster(root)
    root.mainloop()
